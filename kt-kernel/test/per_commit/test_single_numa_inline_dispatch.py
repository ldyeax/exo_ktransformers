"""Parity tests for one-subpool inline dispatch and worker affinity telemetry."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="default")

try:
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
except ImportError:
    kt_kernel_ext = None


INLINE_KEYS = {
    "environment_enabled",
    "eligible_single_numa_subpool",
    "requested",
    "active",
    "status",
    "physical_numa_id",
    "configured_worker_count",
    "distributor_worker_count",
    "distributor_thread_elided",
    "dispatch_count",
    "exception_count",
    "last_native_thread_id",
    "last_cpu_id",
    "last_worker_pool_thread_id",
    "task_queue_native_thread_id",
    "task_queue_cpu_id",
    "task_queue_affinity_active",
    "last_dispatch_on_task_queue_thread",
    "last_dispatch_on_task_queue_cpu",
    "logical_worker_zero_proven",
    "collision_free_worker_zero",
}


def _require_inline_build() -> None:
    if kt_kernel_ext is None:
        pytest.skip("kt_kernel_ext not built or available")
    required = {
        "single_numa_inline_dispatch",
        "worker_pool_affinity",
        "probe_single_numa_inline_dispatch",
    }
    if not required.issubset(dir(kt_kernel_ext.CPUInfer)):
        pytest.skip("installed kt_kernel_ext predates inline-dispatch telemetry")


def _parse_cpu_list(text: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in text.strip().split(","):
        bounds = part.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        result.extend(range(start, end + 1))
    return tuple(result)


def _numa_nodes() -> tuple[tuple[int, tuple[int, ...]], ...]:
    result = []
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        numa_id = int(path.name.removeprefix("node"))
        result.append((numa_id, _parse_cpu_list((path / "cpulist").read_text())))
    return tuple(result)


def _deterministic_worker_cpu_order(node_cpus: tuple[int, ...]) -> tuple[int, ...]:
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu_id in node_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
        package_id = int((topology / "physical_package_id").read_text())
        core_id = int((topology / "core_id").read_text())
        cores.setdefault((package_id, core_id), []).append(cpu_id)
    ordered_cores = sorted(
        (sorted(cpus) for cpus in cores.values()), key=lambda cpus: cpus[0]
    )
    result = [cpus[0] for cpus in ordered_cores]
    sibling_index = 1
    while True:
        layer = [
            cpus[sibling_index] for cpus in ordered_cores if len(cpus) > sibling_index
        ]
        if not layer:
            break
        result.extend(layer)
        sibling_index += 1
    return tuple(result)


def _cpuinfer(numa_id: int, worker_count: int):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [numa_id]
    config.subpool_thread_count = [worker_count]
    return kt_kernel_ext.CPUInfer(config)


@pytest.mark.cpu
def test_inline_dispatch_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_inline_build()
    nodes = _numa_nodes()
    if not nodes:
        pytest.skip("machine exposes no NUMA nodes")
    numa_id, _ = nodes[0]
    monkeypatch.delenv("KT_SINGLE_NUMA_INLINE_DISPATCH", raising=False)
    monkeypatch.delenv("KT_TASK_QUEUE_PIN_FIRST_CORE", raising=False)

    cpuinfer = _cpuinfer(numa_id, 2)
    try:
        telemetry = cpuinfer.single_numa_inline_dispatch()
        assert set(telemetry) == INLINE_KEYS
        assert telemetry == {
            "environment_enabled": False,
            "eligible_single_numa_subpool": True,
            "requested": False,
            "active": False,
            "status": "not_requested",
            "physical_numa_id": numa_id,
            "configured_worker_count": 2,
            "distributor_worker_count": 1,
            "distributor_thread_elided": False,
            "dispatch_count": 0,
            "exception_count": 0,
            "last_native_thread_id": -1,
            "last_cpu_id": -1,
            "last_worker_pool_thread_id": -1,
            "task_queue_native_thread_id": -1,
            "task_queue_cpu_id": -1,
            "task_queue_affinity_active": False,
            "last_dispatch_on_task_queue_thread": False,
            "last_dispatch_on_task_queue_cpu": False,
            "logical_worker_zero_proven": False,
            "collision_free_worker_zero": False,
        }
    finally:
        del cpuinfer
        gc.collect()


@pytest.mark.cpu
def test_ineligible_inline_request_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_inline_build()
    monkeypatch.setenv("KT_SINGLE_NUMA_INLINE_DISPATCH", "1")
    with pytest.raises(ValueError, match="exactly one physical NUMA subpool"):
        kt_kernel_ext.CPUInfer(4)


@pytest.mark.cpu
def test_inline_task_queue_is_worker_zero_and_72_bindings_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_inline_build()
    candidates = [node for node in _numa_nodes() if len(node[1]) >= 72]
    if not candidates:
        pytest.skip("machine has no NUMA node with at least 72 processing units")
    numa_id, node_cpus = candidates[0]
    expected_cpu_ids = _deterministic_worker_cpu_order(node_cpus)[:72]
    monkeypatch.setenv("KT_SINGLE_NUMA_INLINE_DISPATCH", "1")
    monkeypatch.delenv("KT_TASK_QUEUE_PIN_FIRST_CORE", raising=False)

    cpuinfer = _cpuinfer(numa_id, 72)
    try:
        task_queue = cpuinfer.task_queue_affinity()
        assert set(task_queue) == {
            "environment_enabled",
            "eligible_single_numa_subpool",
            "requested",
            "active",
            "status",
            "numa_id",
            "cpu_id",
            "native_thread_id",
        }
        assert task_queue["environment_enabled"] is True
        assert task_queue["active"] is True
        assert task_queue["numa_id"] == numa_id
        assert os.sched_getaffinity(task_queue["native_thread_id"]) == {
            task_queue["cpu_id"]
        }

        initial = cpuinfer.single_numa_inline_dispatch()
        assert set(initial) == INLINE_KEYS
        assert initial["environment_enabled"] is True
        assert initial["eligible_single_numa_subpool"] is True
        assert initial["requested"] is True
        assert initial["active"] is True
        assert initial["status"] == "active"
        assert initial["physical_numa_id"] == numa_id
        assert initial["configured_worker_count"] == 72
        assert initial["distributor_worker_count"] == 0
        assert initial["distributor_thread_elided"] is True
        assert initial["dispatch_count"] == 0
        assert initial["collision_free_worker_zero"] is True

        probe = cpuinfer.probe_single_numa_inline_dispatch(72)
        assert probe["executed_task_count"] == 72
        after = cpuinfer.single_numa_inline_dispatch()
        assert after["dispatch_count"] == 1
        assert after["exception_count"] == 0
        assert after["last_native_thread_id"] == task_queue["native_thread_id"]
        assert after["last_cpu_id"] == task_queue["cpu_id"]
        assert after["last_worker_pool_thread_id"] == 0
        assert after["last_dispatch_on_task_queue_thread"] is True
        assert after["last_dispatch_on_task_queue_cpu"] is True
        assert after["logical_worker_zero_proven"] is True

        affinity = cpuinfer.worker_pool_affinity()
        assert affinity["subpool_count"] == 1
        assert affinity["configured_worker_count"] == 72
        assert affinity["all_worker_bindings_active"] is True
        assert affinity["all_worker_cpu_ids_unique"] is True
        assert affinity["all_workers_on_expected_numa"] is True
        subpool = affinity["subpools"][0]
        assert subpool["logical_subpool_index"] == 0
        assert subpool["physical_numa_id"] == numa_id
        assert subpool["configured_worker_count"] == 72
        assert subpool["active_worker_count"] == 72
        assert tuple(subpool["worker_cpu_ids"]) == expected_cpu_ids
        assert subpool["worker_affinity_statuses"] == ["active"] * 72
        assert subpool["worker_roles"][0] == "inline_task_queue_worker0"
        assert subpool["worker_roles"][1:] == ["background_worker"] * 71
        assert subpool["last_caller_native_thread_id"] == task_queue["native_thread_id"]
        assert subpool["last_caller_cpu_id"] == task_queue["cpu_id"]
        assert subpool["worker_native_thread_ids"] == probe["worker_native_thread_ids"]
        assert subpool["worker_cpu_ids"] == probe["worker_cpu_ids"]
        for native_thread_id, cpu_id in zip(
            subpool["worker_native_thread_ids"],
            subpool["worker_cpu_ids"],
            strict=True,
        ):
            assert os.sched_getaffinity(native_thread_id) == {cpu_id}

        for _ in range(4):
            repeated = cpuinfer.probe_single_numa_inline_dispatch(72)
            assert repeated["executed_task_count"] == 72
        assert cpuinfer.single_numa_inline_dispatch()["dispatch_count"] == 5
    finally:
        del cpuinfer
        gc.collect()
