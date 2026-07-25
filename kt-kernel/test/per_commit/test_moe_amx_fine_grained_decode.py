#!/usr/bin/env python
"""Bit-exact A/B check for the opt-in AMX decode dependency graph."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="default")

try:
    import torch
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    HAS_AMX_BF16 = hasattr(kt_kernel_ext.moe, "AMXBF16_MOE")
except ImportError:
    HAS_AMX_BF16 = False


@pytest.mark.cpu
@pytest.mark.parametrize("num_threads", [2, 8])
def test_fine_grained_decode_matches_staged_decode(num_threads: int) -> None:
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
    expert_ids = torch.tensor([[7, 1, 4, 2]], dtype=torch.int64)
    weights = torch.tensor([[0.125, 0.25, 0.375, 0.25]], dtype=torch.float32)
    input_data = (
        torch.randn((1, hidden_size), dtype=torch.bfloat16, generator=generator) / 100
    ).contiguous()
    batch_size = torch.tensor([1], dtype=torch.int64)
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

    def run_decode(fine_grained: bool) -> torch.Tensor:
        if fine_grained:
            os.environ["KT_AMX_FINE_GRAINED_DECODE"] = "1"
        else:
            os.environ.pop("KT_AMX_FINE_GRAINED_DECODE", None)
        output = torch.empty((1, hidden_size), dtype=torch.bfloat16)
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
        staged = run_decode(fine_grained=False)
        fine_grained = run_decode(fine_grained=True)
    finally:
        os.environ.pop("KT_AMX_FINE_GRAINED_DECODE", None)

    assert torch.equal(fine_grained, staged)
