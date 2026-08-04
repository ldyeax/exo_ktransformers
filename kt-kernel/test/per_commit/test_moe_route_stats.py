#!/usr/bin/env python
"""CPU-only checks for graph-safe KT route evidence."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci
from kt_kernel.experts_base import (
    BaseMoEWrapper,
    _RouteTelemetryGeneration,
    _RouteTelemetryState,
)

register_cpu_ci(est_time=1, suite="default")


class ControllableCompletion:
    def __init__(self, completed: bool) -> None:
        self.completed = completed

    def query(self) -> bool:
        return self.completed

    def record(self, _stream=None) -> None:
        return None


class ExplodingCompletion:
    def query(self) -> bool:
        raise AssertionError("capture-time telemetry queried a CUDA event")

    def record(self, _stream=None) -> None:
        return None


def route_stats_wrapper(
    immediate_ids: torch.Tensor,
    deferred_ids: torch.Tensor | None,
    *,
    completed: bool = True,
    generation: int = 1,
    num_experts: int = 3,
    method: str = "BF16",
) -> tuple[SimpleNamespace, ControllableCompletion]:
    completion = ControllableCompletion(completed)
    telemetry_state = _RouteTelemetryState()
    telemetry_state.add(
        _RouteTelemetryGeneration(
            generation=generation,
            immediate_ids=immediate_ids,
            deferred_ids=deferred_ids,
            completion=completion,
        )
    )
    wrapper = SimpleNamespace(
        _route_telemetry_state=telemetry_state,
        _route_telemetry_error=None,
        num_experts=num_experts,
        method=method,
        hidden_size=4096 if method == "MXFP4" else 8,
        moe_intermediate_size=2048 if method == "MXFP4" else 4,
    )
    return wrapper, completion


def test_completed_route_stats_merge_immediate_and_deferred_slots() -> None:
    wrapper, _ = route_stats_wrapper(
        torch.tensor([[0, -1, 1], [0, 1, -1]], dtype=torch.int64),
        torch.tensor([[-1, 2, -1], [-1, -1, -1]], dtype=torch.int64),
        method="MXFP4",
    )

    stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)

    assert stats == {
        "cpu_route_generation": 1,
        "cpu_active_experts": 3,
        "cpu_route_rows": 5,
        "cpu_rows_per_expert": {"1": 1, "2": 2},
        "estimated_expert_weight_bytes": 13_369_344,
        "estimated_cpu_weight_stream_bytes": 40_108_032,
    }


def test_completed_route_stats_handle_an_empty_cpu_complement() -> None:
    wrapper, _ = route_stats_wrapper(
        torch.full((2, 3), -1, dtype=torch.int64),
        None,
    )

    stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)

    assert stats == {
        "cpu_route_generation": 1,
        "cpu_active_experts": 0,
        "cpu_route_rows": 0,
        "cpu_rows_per_expert": {},
        "estimated_expert_weight_bytes": None,
        "estimated_cpu_weight_stream_bytes": None,
    }


def test_completed_route_stats_flatten_large_int32_prefill_with_sentinels() -> None:
    row_count = 65_536
    top_k = 6
    num_experts = 128
    expected_routes = (
        torch.arange(row_count * top_k, dtype=torch.int64).reshape(row_count, top_k)
        % num_experts
    )
    immediate_ids = expected_routes.to(torch.int32)
    deferred_ids = torch.full_like(immediate_ids, -1)
    deferred_ids[:, 3:] = immediate_ids[:, 3:]
    immediate_ids[:, 3:] = -1

    # Exercise both negative and upper-bound sentinels.  Neither may reach
    # bincount, and the two-dimensional prefill population must be flattened.
    immediate_ids[0, 0] = num_experts
    deferred_ids[0, 0] = num_experts
    expected_routes[0, 0] = -1
    expected_valid_routes = expected_routes.reshape(-1)
    expected_valid_routes = expected_valid_routes[
        (expected_valid_routes >= 0) & (expected_valid_routes < num_experts)
    ]
    expected_counts = torch.bincount(
        expected_valid_routes,
        minlength=num_experts,
    )
    expected_histogram = torch.bincount(expected_counts[expected_counts > 0])

    wrapper, _ = route_stats_wrapper(
        immediate_ids,
        deferred_ids,
        num_experts=num_experts,
    )

    stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)

    assert stats is not None
    assert stats["cpu_active_experts"] == num_experts
    assert stats["cpu_route_rows"] == row_count * top_k - 1
    assert stats["cpu_rows_per_expert"] == {
        str(rows): int(expert_count)
        for rows, expert_count in enumerate(expected_histogram.tolist())
        if rows > 0 and expert_count > 0
    }


@pytest.mark.parametrize("invalid_dtype", [torch.bool, torch.float32])
def test_completed_route_stats_reject_nonintegral_route_dtype(
    invalid_dtype: torch.dtype,
) -> None:
    wrapper, _ = route_stats_wrapper(
        torch.zeros((2, 3), dtype=invalid_dtype),
        None,
    )

    with pytest.raises(TypeError, match="must use an integral dtype"):
        BaseMoEWrapper.get_last_forward_route_stats(wrapper)


def test_pending_generation_returns_none_then_becomes_completed() -> None:
    wrapper, completion = route_stats_wrapper(
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
        None,
        completed=False,
        generation=7,
    )

    assert BaseMoEWrapper.get_last_forward_route_stats(wrapper) is None
    completion.completed = True
    stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)

    assert stats is not None
    assert stats["cpu_route_generation"] == 7
    assert stats["cpu_route_rows"] == 3


def test_pending_current_generation_returns_last_completed_generation() -> None:
    wrapper, _ = route_stats_wrapper(
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
        None,
        generation=11,
    )
    first_stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)
    assert first_stats is not None

    current_completion = ControllableCompletion(False)
    wrapper._route_telemetry_state.add(
        _RouteTelemetryGeneration(
            generation=12,
            immediate_ids=torch.tensor([[2, 2, 2]], dtype=torch.int64),
            deferred_ids=None,
            completion=current_completion,
        )
    )

    pending_stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)
    assert pending_stats == first_stats
    assert pending_stats["cpu_route_generation"] == 11


def test_completed_snapshot_is_stable_after_source_mutation() -> None:
    source = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    wrapper, _ = route_stats_wrapper(source, None, generation=21)
    completed_stats = BaseMoEWrapper.get_last_forward_route_stats(wrapper)
    source.fill_(2)

    assert BaseMoEWrapper.get_last_forward_route_stats(wrapper) == completed_stats
    assert completed_stats is not None
    assert completed_stats["cpu_rows_per_expert"] == {"1": 3}


def test_distinct_wrappers_do_not_share_layer_parity_route_slots(monkeypatch) -> None:
    real_empty = torch.empty
    real_empty_like = torch.empty_like

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    def unpinned_empty_like(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned_empty)
    monkeypatch.setattr(torch, "empty_like", unpinned_empty_like)

    def new_wrapper() -> SimpleNamespace:
        return SimpleNamespace(
            _route_telemetry_enabled=True,
            _route_telemetry_generation=0,
            _route_telemetry_state=_RouteTelemetryState(),
            _route_telemetry_buffers={},
            _route_telemetry_waiting_for_sync=None,
            _route_telemetry_error=None,
        )

    source = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    layer_zero = new_wrapper()
    layer_two = new_wrapper()
    first = BaseMoEWrapper._prepare_route_telemetry_generation(
        layer_zero, source, None
    )
    second = BaseMoEWrapper._prepare_route_telemetry_generation(
        layer_two, source, None
    )

    assert first is not None and second is not None
    assert first.immediate_ids.data_ptr() != second.immediate_ids.data_ptr()


def test_capture_skips_route_generation_without_query_or_allocation(monkeypatch) -> None:
    telemetry_state = _RouteTelemetryState()
    telemetry_state.add(
        _RouteTelemetryGeneration(
            generation=1,
            immediate_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            deferred_ids=None,
            completion=ExplodingCompletion(),
        )
    )
    wrapper = SimpleNamespace(
        _route_telemetry_enabled=True,
        _route_telemetry_generation=1,
        _route_telemetry_state=telemetry_state,
        _route_telemetry_buffers={},
        _route_telemetry_waiting_for_sync=object(),
        _route_telemetry_error=None,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture-time telemetry allocated pinned memory")
        ),
    )

    route_generation = BaseMoEWrapper._prepare_route_telemetry_generation(
        wrapper,
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
        None,
    )

    assert route_generation is None
    assert wrapper._route_telemetry_generation == 1
    assert wrapper._route_telemetry_waiting_for_sync is None
    assert wrapper._route_telemetry_error is None


def test_capture_skips_route_stats_event_query_without_consuming_error(
    monkeypatch,
) -> None:
    telemetry_state = _RouteTelemetryState()
    telemetry_state.add(
        _RouteTelemetryGeneration(
            generation=3,
            immediate_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            deferred_ids=None,
            completion=ExplodingCompletion(),
        )
    )
    telemetry_error = RuntimeError("preserve until ordinary execution")
    wrapper = SimpleNamespace(
        _route_telemetry_state=telemetry_state,
        _route_telemetry_error=telemetry_error,
        num_experts=3,
        method="BF16",
        hidden_size=8,
        moe_intermediate_size=4,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert BaseMoEWrapper.get_last_forward_route_stats(wrapper) is None
    assert wrapper._route_telemetry_error is telemetry_error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_cuda_graph_capture_skips_pending_route_event_query() -> None:
    source = torch.arange(6, device="cuda", dtype=torch.int64)
    destination = torch.empty_like(source)
    completion = torch.cuda.Event(
        enable_timing=False,
        blocking=False,
        interprocess=False,
    )
    completion.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    telemetry_state = _RouteTelemetryState()
    telemetry_state.add(
        _RouteTelemetryGeneration(
            generation=9,
            immediate_ids=torch.empty((1, 6), dtype=torch.int64, pin_memory=True),
            deferred_ids=None,
            completion=completion,
        )
    )
    wrapper = SimpleNamespace(
        _route_telemetry_enabled=True,
        _route_telemetry_generation=9,
        _route_telemetry_state=telemetry_state,
        _route_telemetry_buffers={},
        _route_telemetry_waiting_for_sync=None,
        _route_telemetry_error=None,
    )
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        route_generation = BaseMoEWrapper._prepare_route_telemetry_generation(
            wrapper,
            source.view(1, 6),
            None,
        )
        destination.copy_(source)

    assert route_generation is None
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)
