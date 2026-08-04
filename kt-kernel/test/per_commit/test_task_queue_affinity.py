"""Focused tests for the opt-in CPUInfer TaskQueue affinity contract."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

try:
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
except ImportError:
    kt_kernel_ext = None


def _require_affinity_build() -> None:
    if kt_kernel_ext is None:
        pytest.skip("kt_kernel_ext not built or available")
    if not hasattr(kt_kernel_ext.CPUInfer, "task_queue_affinity"):
        pytest.skip("installed kt_kernel_ext predates TaskQueue affinity telemetry")


def _numa_ids() -> tuple[int, ...]:
    nodes = []
    for candidate in Path("/sys/devices/system/node").glob("node[0-9]*"):
        suffix = candidate.name.removeprefix("node")
        if suffix.isdigit():
            nodes.append(int(suffix))
    return tuple(sorted(nodes))


def _single_subpool_cpuinfer(numa_id: int):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [numa_id]
    config.subpool_thread_count = [1]
    return kt_kernel_ext.CPUInfer(config)


@pytest.mark.cpu
def test_task_queue_first_core_pin_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_affinity_build()
    nodes = _numa_ids()
    if not nodes:
        pytest.skip("machine exposes no NUMA node in sysfs")

    monkeypatch.delenv("KT_TASK_QUEUE_PIN_FIRST_CORE", raising=False)
    cpuinfer = _single_subpool_cpuinfer(nodes[0])
    try:
        assert cpuinfer.task_queue_affinity() == {
            "environment_enabled": False,
            "eligible_single_numa_subpool": True,
            "requested": False,
            "active": False,
            "status": "not_requested",
            "numa_id": -1,
            "cpu_id": -1,
            "native_thread_id": -1,
        }
    finally:
        del cpuinfer
        gc.collect()


@pytest.mark.cpu
def test_single_subpool_task_queue_is_pinned_and_verifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_affinity_build()
    nodes = _numa_ids()
    if not nodes:
        pytest.skip("machine exposes no NUMA node in sysfs")

    monkeypatch.setenv("KT_TASK_QUEUE_PIN_FIRST_CORE", "1")
    for numa_id in nodes:
        cpuinfer = _single_subpool_cpuinfer(numa_id)
        try:
            telemetry = cpuinfer.task_queue_affinity()
            assert telemetry["environment_enabled"] is True
            assert telemetry["eligible_single_numa_subpool"] is True
            assert telemetry["requested"] is True
            assert telemetry["active"] is True
            assert telemetry["status"] == "active"
            assert telemetry["numa_id"] == numa_id
            assert isinstance(telemetry["cpu_id"], int)
            assert telemetry["cpu_id"] >= 0
            assert isinstance(telemetry["native_thread_id"], int)
            assert telemetry["native_thread_id"] > 0
            assert os.sched_getaffinity(telemetry["native_thread_id"]) == {
                telemetry["cpu_id"]
            }
        finally:
            del cpuinfer
            gc.collect()


@pytest.mark.cpu
def test_environment_does_not_pin_default_worker_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_affinity_build()
    monkeypatch.setenv("KT_TASK_QUEUE_PIN_FIRST_CORE", "1")

    cpuinfer = kt_kernel_ext.CPUInfer(4)
    try:
        telemetry = cpuinfer.task_queue_affinity()
        assert telemetry["environment_enabled"] is True
        assert telemetry["eligible_single_numa_subpool"] is False
        assert telemetry["requested"] is False
        assert telemetry["active"] is False
        assert telemetry["status"] == "not_requested"
        assert telemetry["numa_id"] == -1
        assert telemetry["cpu_id"] == -1
        assert telemetry["native_thread_id"] == -1
    finally:
        del cpuinfer
        gc.collect()
