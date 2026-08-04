#!/usr/bin/env python
"""Bit-exact A/B check for the opt-in AMX decode dependency graph."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="default")

try:
    import kt_kernel
    import torch

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    HAS_AMX_BF16 = hasattr(kt_kernel_ext.moe, "AMXBF16_MOE")
except ImportError:
    HAS_AMX_BF16 = False


@pytest.mark.cpu
@pytest.mark.parametrize("num_threads", [2, 8])
@pytest.mark.parametrize("token_count", [1, 2, 4])
def test_fine_grained_decode_matches_staged_decode(
    num_threads: int, token_count: int
) -> None:
    if not HAS_AMX_BF16:
        pytest.skip("AMX BF16 extension is unavailable")

    expert_num = 8
    experts_per_token = 4
    hidden_size = 256
    intermediate_size = 256

    generator = torch.Generator(device="cpu").manual_seed(20260725)
    gate_proj = torch.randn(
        (expert_num, intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    up_proj = torch.randn(
        (expert_num, intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    down_proj = torch.randn(
        (expert_num, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    base_expert_ids = torch.tensor([7, 1, 4, 2], dtype=torch.int64)
    expert_ids = torch.stack(
        [torch.roll(base_expert_ids, shifts=token) for token in range(token_count)]
    )
    weights = torch.tensor(
        [[0.125, 0.25, 0.375, 0.25]] * token_count, dtype=torch.float32
    )
    input_data = (
        torch.randn(
            (token_count, hidden_size),
            dtype=torch.bfloat16,
            generator=generator,
        )
        / 100
    ).contiguous()
    batch_size = torch.tensor([token_count], dtype=torch.int64)
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)

    cpu_infer = kt_kernel_ext.CPUInfer(num_threads)
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num,
        experts_per_token,
        hidden_size,
        intermediate_size,
        0,
    )
    config.max_len = 32
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    config.gate_scale = 0
    config.pool = cpu_infer.backend_
    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpu_infer.sync()

    def run_decode(mode: str) -> torch.Tensor:
        if mode == "staged":
            os.environ.pop("KT_AMX_FINE_GRAINED_DECODE", None)
            os.environ.pop("KT_AMX_FUSED_ACTIVATION", None)
        elif mode == "fine_grained":
            os.environ["KT_AMX_FINE_GRAINED_DECODE"] = "1"
            os.environ.pop("KT_AMX_FUSED_ACTIVATION", None)
        elif mode == "fused":
            os.environ["KT_AMX_FINE_GRAINED_DECODE"] = "1"
            os.environ["KT_AMX_FUSED_ACTIVATION"] = "1"
        else:
            raise ValueError(f"unknown decode mode: {mode}")
        output = torch.empty((token_count, hidden_size), dtype=torch.bfloat16)
        cpu_infer.submit(
            moe.forward_task(
                batch_size.data_ptr(),
                experts_per_token,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_data.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpu_infer.sync()
        return output

    try:
        staged = run_decode("staged")
        fine_grained = run_decode("fine_grained")
        fine_grained_reused = run_decode("fine_grained")
        fused = run_decode("fused")
        fused_reused = run_decode("fused")
    finally:
        os.environ.pop("KT_AMX_FINE_GRAINED_DECODE", None)
        os.environ.pop("KT_AMX_FUSED_ACTIVATION", None)

    assert torch.equal(fine_grained, staged)
    assert torch.equal(fine_grained_reused, staged)
    assert torch.equal(fused, staged)
    assert torch.equal(fused_reused, staged)
