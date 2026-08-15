from __future__ import annotations

import runpy
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

KT_KERNEL_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = KT_KERNEL_ROOT.parent
INFERENCE_TORCH = Requirement("torch==2.11.0")
SFT_TORCH = Requirement("torch==2.9.1")

sys.path.insert(0, str(KT_KERNEL_ROOT / "test"))
from ci.ci_register import register_cpu_ci  # noqa: E402


register_cpu_ci(est_time=0.1, suite="default")


def _requirement(requirements: list[str], name: str) -> Requirement:
    canonical_name = canonicalize_name(name)
    for raw_requirement in requirements:
        if not raw_requirement.strip() or raw_requirement.lstrip().startswith("#"):
            continue
        parsed = Requirement(raw_requirement)
        if canonicalize_name(parsed.name) == canonical_name:
            return parsed
    raise AssertionError(f"{name} is missing from {requirements}")


def _root_setup_metadata() -> dict[str, object]:
    captured: dict[str, object] = {}

    def capture_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    with patch("setuptools.setup", capture_setup):
        runpy.run_path(str(REPOSITORY_ROOT / "setup.py"), run_name="__main__")
    return captured


def test_kt_kernel_torch_range_accepts_both_validated_cohorts() -> None:
    with (KT_KERNEL_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    torch_requirement = _requirement(metadata["project"]["dependencies"], "torch")

    assert Version("2.9.1") in torch_requirement.specifier
    assert Version("2.11.0") in torch_requirement.specifier
    assert Version("2.9.0") not in torch_requirement.specifier
    assert Version("2.12.0") not in torch_requirement.specifier

    requirements_torch = _requirement(
        (KT_KERNEL_ROOT / "requirements.txt").read_text().splitlines(), "torch"
    )
    assert requirements_torch.specifier == torch_requirement.specifier


def test_sglang_install_resolves_torch_211_without_a_downgrade() -> None:
    with (KT_KERNEL_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    kernel_torch = _requirement(metadata["project"]["dependencies"], "torch")

    assert Version("2.11.0") in kernel_torch.specifier
    assert Version("2.11.0") in INFERENCE_TORCH.specifier
    assert Version("2.9.1") not in INFERENCE_TORCH.specifier

    install_script = (REPOSITORY_ROOT / "install.sh").read_text()
    install_all = install_script[install_script.index("install_all()") :]
    assert install_all.index('install_sglang "$editable"') < install_all.index(
        'install_kt_kernel "${kt_args[@]}"'
    )


def test_sft_extra_stays_on_its_separate_torch_29_cohort() -> None:
    metadata = _root_setup_metadata()
    extras = metadata["extras_require"]
    assert isinstance(extras, dict)
    sft_requirements = extras["sft"]
    sglang_requirements = extras["sglang"]
    assert isinstance(sft_requirements, list)
    assert isinstance(sglang_requirements, list)

    sft_torch = _requirement(sft_requirements, "torch")
    _requirement(sft_requirements, "transformers-kt")
    _requirement(sft_requirements, "accelerate-kt")
    _requirement(sglang_requirements, "sglang-kt")

    assert sft_torch.specifier == SFT_TORCH.specifier
    assert all(
        canonicalize_name(Requirement(requirement).name) != "transformers-kt"
        for requirement in sglang_requirements
    )


def test_sft_and_sglang_torch_cohorts_are_intentionally_disjoint() -> None:
    validated_versions = (Version("2.9.1"), Version("2.11.0"))
    compatible = [
        version
        for version in validated_versions
        if version in SFT_TORCH.specifier and version in INFERENCE_TORCH.specifier
    ]
    assert compatible == []


def test_autosetup_defaults_to_the_inference_torch_abi() -> None:
    autosetup = (KT_KERNEL_ROOT / "autosetup.sh").read_text()
    assert 'TORCH_LIST=${TORCH_LIST:-"2.11.0"}' in autosetup


def test_cuda_reductions_use_cub_version_agnostic_functors() -> None:
    source = (
        KT_KERNEL_ROOT / "cuda" / "moe" / "moe_topk_softmax_kernels.cu"
    ).read_text()
    assert "cub::Sum" not in source
    assert "cub::Max" not in source
    assert "KTReduceSum{}" in source
    assert "KTReduceMax{}" in source


def test_cuda_extension_build_enables_its_cuda_only_exports() -> None:
    setup_source = (KT_KERNEL_ROOT / "cuda" / "setup.py").read_text()
    assert "-DKTRANSFORMERS_USE_CUDA=1" in setup_source
