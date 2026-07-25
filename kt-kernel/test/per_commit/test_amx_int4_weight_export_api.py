"""Mock regression test for the AMXINT4 stream-prefill export surface."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="default")


def _load_amx_module():
    """Load utils/amx.py with only its import-time dependencies stubbed."""

    module_path = Path(__file__).parents[2] / "python" / "utils" / "amx.py"

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object

    fake_extension = types.ModuleType("kt_kernel_ext")
    fake_moe = types.ModuleType("kt_kernel_ext.moe")
    fake_moe.MOEConfig = object
    fake_extension.moe = fake_moe

    fake_package = types.ModuleType("kt_kernel")
    fake_package.__path__ = []
    fake_utils = types.ModuleType("kt_kernel.utils")
    fake_utils.__path__ = []
    fake_base = types.ModuleType("kt_kernel.experts_base")
    fake_base.BaseMoEWrapper = object

    fake_loader = types.ModuleType("kt_kernel.utils.loader")
    for name in (
        "BF16SafeTensorLoader",
        "CompressedSafeTensorLoader",
        "FP8SafeTensorLoader",
        "GPTQSafeTensorLoader",
        "MXFP4SafeTensorLoader",
        "MXFP8SafeTensorLoader",
        "SafeTensorLoader",
        "SharedSafeTensorLoader",
    ):
        setattr(fake_loader, name, type(name, (), {}))

    fake_shared = types.ModuleType("kt_kernel.utils.shared_host_weights")
    fake_shared.SharedHostWeightError = RuntimeError
    fake_shared.SharedHostWeightLease = type("SharedHostWeightLease", (), {})

    module_name = "kt_kernel.utils.amx_export_api_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    dependencies = {
        "torch": fake_torch,
        "kt_kernel_ext": fake_extension,
        "kt_kernel_ext.moe": fake_moe,
        "kt_kernel": fake_package,
        "kt_kernel.utils": fake_utils,
        "kt_kernel.experts_base": fake_base,
        "kt_kernel.utils.loader": fake_loader,
        "kt_kernel.utils.shared_host_weights": fake_shared,
        module_name: module,
    }
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def test_amx_int4_wrapper_forwards_export_submit_and_sync() -> None:
    amx = _load_amx_module()
    wrapper = object.__new__(amx.AMXMoEWrapper)
    wrapper.moe = MagicMock()
    wrapper.cpu_infer = MagicMock()
    task = object()
    wrapper.moe.write_weight_scale_to_buffer_task.return_value = task

    arguments = (2, 17, [101, 102], [0, 0], [201, 202], [0, 0])
    wrapper.submit_write_weight_scale_to_buffer(*arguments)
    wrapper.sync_write_weight_scale_to_buffer()

    wrapper.moe.write_weight_scale_to_buffer_task.assert_called_once_with(*arguments)
    wrapper.cpu_infer.submit.assert_called_once_with(task)
    wrapper.cpu_infer.sync.assert_called_once_with()
