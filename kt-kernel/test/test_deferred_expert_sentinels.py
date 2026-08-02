from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer


class _ForwardTask(NamedTuple):
    expert_ids_pointer: int
    output_pointer: int
    incremental: bool


class _RecordingMoE:
    def forward_task(
        self,
        _batch_size_pointer: int,
        _top_k: int,
        expert_ids_pointer: int,
        _weights_pointer: int,
        _input_pointer: int,
        output_pointer: int,
        incremental: bool,
    ) -> _ForwardTask:
        return _ForwardTask(expert_ids_pointer, output_pointer, incremental)


class _RecordingCPUInfer:
    def __init__(self) -> None:
        self.pending = 0
        self.submitted: list[_ForwardTask] = []
        self.sync_allowances: list[int] = []

    def submit_with_cuda_stream(self, _cuda_stream, task: _ForwardTask) -> None:
        self.pending += 1
        self.submitted.append(task)

    def sync_with_cuda_stream(self, _cuda_stream, allow_n_pending: int = 0) -> None:
        self.sync_allowances.append(allow_n_pending)
        self.pending = allow_n_pending


def _make_buffers(batch_size: int, hidden_size: int, top_k: int):
    depth = KExpertsCPUBuffer.buffer_depth
    return (
        [
            torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16)
            for _ in range(depth)
        ],
        [torch.zeros((batch_size, top_k), dtype=torch.long) for _ in range(depth)],
        [torch.full((batch_size, top_k), -1, dtype=torch.long) for _ in range(depth)],
        [torch.zeros((batch_size, top_k), dtype=torch.float32) for _ in range(depth)],
        [
            torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16)
            for _ in range(depth)
        ],
        [torch.full((1,), batch_size, dtype=torch.int32) for _ in range(depth)],
        [None for _ in range(depth)],
    )


def _make_wrapper(
    layer_idx: int,
    cpu_infer: _RecordingCPUInfer,
    max_deferred_experts_per_token: int,
) -> SimpleNamespace:
    wrapper = SimpleNamespace(
        layer_idx=layer_idx,
        num_experts=4,
        num_experts_per_tok=3,
        max_deferred_experts_per_token=max_deferred_experts_per_token,
        cpu_infer=cpu_infer,
        moe=_RecordingMoE(),
    )
    wrapper.select_deferred_experts = lambda ids, scores, protected_k: (
        BaseMoEWrapper.select_deferred_experts(wrapper, ids, scores, protected_k)
    )
    return wrapper


def test_select_deferred_experts_ignores_unowned_sentinels() -> None:
    wrapper = SimpleNamespace(num_experts=4)
    expert_ids = torch.tensor([[0, -1, 2], [-1, 1, 3]])
    expert_scores = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])

    immediate_ids, deferred_ids = BaseMoEWrapper.select_deferred_experts(
        wrapper, expert_ids, expert_scores, protected_k=2
    )

    assert torch.equal(immediate_ids, torch.tensor([[0, -1, -1], [-1, 1, -1]]))
    assert deferred_ids is not None
    assert torch.equal(deferred_ids, torch.tensor([[-1, -1, 2], [-1, -1, 3]]))


def test_invalid_protected_route_does_not_alias_expert_zero() -> None:
    wrapper = SimpleNamespace(num_experts=4)
    expert_ids = torch.tensor([[-1, 0, 2]])
    expert_scores = torch.tensor([[0.9, 0.1, 0.8]])

    immediate_ids, deferred_ids = BaseMoEWrapper.select_deferred_experts(
        wrapper, expert_ids, expert_scores, protected_k=1
    )

    assert torch.equal(immediate_ids, torch.tensor([[-1, -1, -1]]))
    assert deferred_ids is not None
    assert torch.equal(deferred_ids, torch.tensor([[-1, 0, 2]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_deferred_expert_selection_is_cuda_graph_safe() -> None:
    wrapper = SimpleNamespace(num_experts=4)
    expert_ids = torch.tensor([[0, -1, 2], [-1, 1, 3]], device="cuda", dtype=torch.long)
    expert_scores = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], device="cuda")

    for _ in range(2):
        BaseMoEWrapper.select_deferred_experts(
            wrapper, expert_ids, expert_scores, protected_k=2
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        immediate_ids, deferred_ids = BaseMoEWrapper.select_deferred_experts(
            wrapper, expert_ids, expert_scores, protected_k=2
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(immediate_ids.cpu(), torch.tensor([[0, -1, -1], [-1, 1, -1]]))
    assert deferred_ids is not None
    assert torch.equal(deferred_ids.cpu(), torch.tensor([[-1, -1, 2], [-1, -1, 3]]))


def test_deferred_tail_is_drained_into_its_own_layer(monkeypatch) -> None:
    buffers = _make_buffers(batch_size=2, hidden_size=4, top_k=3)
    monkeypatch.setattr(
        KExpertsCPUBuffer,
        "get_buffer",
        classmethod(lambda _cls, _hidden_states, _top_k: buffers),
    )
    cpu_infer = _RecordingCPUInfer()
    first_layer = _make_wrapper(
        layer_idx=0,
        cpu_infer=cpu_infer,
        max_deferred_experts_per_token=1,
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
    expert_scores = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])

    BaseMoEWrapper.submit_forward(
        first_layer, hidden_states, expert_ids, expert_scores, cuda_stream=object()
    )

    immediate_task, deferred_task = cpu_infer.submitted
    first_output_pointer = buffers[4][0].data_ptr()
    assert immediate_task.output_pointer == first_output_pointer
    assert not immediate_task.incremental
    assert deferred_task.output_pointer == first_output_pointer
    assert deferred_task.incremental
    assert cpu_infer.pending == 2

    BaseMoEWrapper.sync_forward(first_layer, hidden_states, cuda_stream=object())

    assert cpu_infer.sync_allowances == [0]
    assert cpu_infer.pending == 0

    second_layer = _make_wrapper(
        layer_idx=1,
        cpu_infer=cpu_infer,
        max_deferred_experts_per_token=0,
    )
    BaseMoEWrapper.submit_forward(
        second_layer, hidden_states, expert_ids, expert_scores, cuda_stream=object()
    )

    second_layer_task = cpu_infer.submitted[-1]
    assert second_layer_task.output_pointer == buffers[4][1].data_ptr()
    assert not second_layer_task.incremental
