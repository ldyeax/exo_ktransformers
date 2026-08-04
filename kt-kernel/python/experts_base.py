# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import torch
from kt_kernel import kt_kernel_ext


def _cuda_graph_capture_active() -> bool:
    """Return whether this thread's current CUDA stream is being captured.

    CUDA event queries and first-use pinned-host allocations both invalidate an
    active capture.  Keep the check behind ``is_initialized`` so CPU-only
    imports and tests never initialize the CUDA runtime merely for optional
    route diagnostics.
    """
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


class _RouteCompletion(Protocol):
    """A nonblocking completion fence used by CPU-route diagnostics."""

    def query(self) -> bool: ...

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None: ...


@dataclass
class _RouteTelemetryGeneration:
    """One wrapper-local route population and its stream completion fence."""

    generation: int
    immediate_ids: torch.Tensor
    deferred_ids: Optional[torch.Tensor]
    completion: Optional[_RouteCompletion] = None


@dataclass(frozen=True)
class _CompletedRouteSnapshot:
    """Immutable CPU snapshot retained after a generation has completed."""

    generation: int
    immediate_ids: torch.Tensor
    deferred_ids: Optional[torch.Tensor]


class _RouteTelemetryState:
    """Publish only stable, completed route generations without waiting.

    CUDA work records a completion fence after CPUInfer's stream-ordered sync.
    ``completed_snapshot`` merely queries that fence; it never synchronizes a
    CUDA stream or waits for the CPU worker.  The source tensors are copied
    exactly once when the fence reports completion, so later reuse cannot
    mutate a receipt that has already been published.
    """

    def __init__(self) -> None:
        self._pending: Dict[int, _RouteTelemetryGeneration] = {}
        self._last_completed: Optional[_CompletedRouteSnapshot] = None

    def add(self, route_generation: _RouteTelemetryGeneration) -> None:
        self._pending[route_generation.generation] = route_generation

    def completed_snapshot(self) -> Optional[_CompletedRouteSnapshot]:
        completed_generations: List[int] = []
        for generation, route_generation in sorted(self._pending.items()):
            completion = route_generation.completion
            if completion is None or not completion.query():
                continue
            immediate_snapshot = _snapshot_route_buffer(
                route_generation.immediate_ids,
                "immediate",
            )
            deferred_snapshot = (
                _snapshot_route_buffer(route_generation.deferred_ids, "deferred")
                if route_generation.deferred_ids is not None
                else None
            )
            if (
                self._last_completed is None
                or generation > self._last_completed.generation
            ):
                self._last_completed = _CompletedRouteSnapshot(
                    generation=generation,
                    immediate_ids=immediate_snapshot,
                    deferred_ids=deferred_snapshot,
                )
            completed_generations.append(generation)

        for generation in completed_generations:
            del self._pending[generation]
        return self._last_completed

    def source_is_pending(self, route_buffer: torch.Tensor) -> bool:
        """Return whether a live generation still owns ``route_buffer``."""
        data_pointer = route_buffer.data_ptr()
        return any(
            route_generation.immediate_ids.data_ptr() == data_pointer
            or (
                route_generation.deferred_ids is not None
                and route_generation.deferred_ids.data_ptr() == data_pointer
            )
            for route_generation in self._pending.values()
        )

    def completion_is_pending(self, completion: _RouteCompletion) -> bool:
        """Return whether an older live generation still owns a fence."""
        return any(
            route_generation.completion is completion
            for route_generation in self._pending.values()
        )


def _snapshot_route_buffer(
    route_buffer: torch.Tensor, buffer_name: str
) -> torch.Tensor:
    if route_buffer.device.type != "cpu":
        raise ValueError(
            f"{buffer_name} route buffer must remain on CPU, got "
            f"{route_buffer.device}"
        )
    if route_buffer.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            f"{buffer_name} route buffer must use an integral dtype, got "
            f"{route_buffer.dtype}"
        )
    return route_buffer.detach().reshape(-1).to(dtype=torch.int64, copy=True)


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> torch.Tensor:
    """
    Generate GPU experts masks based on activation frequency.

    Selects the top `num_gpu_experts` experts with highest activation frequency
    across all layers to be placed on GPU.

    Args:
        activation_freq: Activation frequency table of shape (num_layers, num_experts).
                         Higher values indicate more frequently activated experts.
        num_gpu_experts: Total number of experts to place on GPU across all layers.

    Returns:
        gpu_experts_masks: Boolean mask of shape (num_layers, num_experts) on CPU.
                           True means the expert should be on GPU.

    Example:
        >>> activation_freq = torch.tensor([
        ...     [0.1, 0.5, 0.3, 0.8],  # layer 0
        ...     [0.2, 0.4, 0.9, 0.1],  # layer 1
        ... ])
        >>> masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)
        >>> # Top 3: layer0-expert3 (0.8), layer1-expert2 (0.9), layer0-expert1 (0.5)
        >>> masks
        tensor([[False,  True, False,  True],
                [False, False,  True, False]])
    """
    num_layers, num_experts_per_layer = activation_freq.shape
    total_experts = num_layers * num_experts_per_layer

    # Clamp num_gpu_experts to valid range
    num_gpu_experts = min(num_gpu_experts, total_experts)
    num_gpu_experts = max(num_gpu_experts, 0)

    if num_gpu_experts == 0:
        return torch.zeros(
            num_layers, num_experts_per_layer, dtype=torch.bool, device="cpu"
        )

    # Flatten and find top-k indices
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(
        flat_freq, k=num_gpu_experts, largest=True, sorted=False
    )

    # Create mask
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True

    # Reshape to (num_layers, num_experts)
    gpu_experts_masks = gpu_experts_masks.view(num_layers, num_experts_per_layer)

    return gpu_experts_masks


class KExpertsCPUBuffer:
    """
    CPU buffer management for expert computation.

    Manages pinned memory buffers for efficient GPU-CPU data transfer.
    """

    capture_bs: List = list()
    capture_buffers: Dict = dict()
    temp_bs: int = 0
    temp_buffer: tuple = tuple()
    buffer_depth: int = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        pin_memory = True

        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            if batch_size in cls.capture_bs and cls.temp_buffer:
                cls.capture_buffers[batch_size] = cls.temp_buffer
            return cls.temp_buffer

        # A new ragged-verify size can first reach this allocator while the
        # model runner is inside torch.inference_mode().  Such tensors cannot
        # later be updated by a breakable-CUDA-graph host callback running
        # outside inference mode.  The staging ring is deliberately mutable
        # and outlives either context, so always allocate ordinary tensors.
        with torch.inference_mode(False):
            input_tensor_cpu = [
                torch.zeros(
                    (batch_size, hidden_size),
                    device="cpu",
                    pin_memory=pin_memory,
                    dtype=torch.bfloat16,
                )
                for _ in range(cls.buffer_depth)
            ]
            immediate_experts_ids_cpu = [
                torch.zeros(
                    (batch_size, num_experts_per_tok),
                    device="cpu",
                    dtype=torch.long,
                    pin_memory=pin_memory,
                )
                for _ in range(cls.buffer_depth)
            ]
            deferred_experts_ids_cpu = [
                torch.full(
                    (batch_size, num_experts_per_tok),
                    -1,
                    device="cpu",
                    dtype=torch.long,
                    pin_memory=pin_memory,
                )
                for _ in range(cls.buffer_depth)
            ]
            weights_cpu = [
                torch.zeros(
                    (batch_size, num_experts_per_tok),
                    device="cpu",
                    dtype=torch.float32,
                    pin_memory=pin_memory,
                )
                for _ in range(cls.buffer_depth)
            ]
            output_cpu = [
                torch.zeros(
                    (batch_size, hidden_size),
                    device="cpu",
                    pin_memory=pin_memory,
                    dtype=torch.bfloat16,
                )
                for _ in range(cls.buffer_depth)
            ]
            bsz_tensor_cpu = [
                torch.full(
                    (1,),
                    batch_size,
                    device="cpu",
                    dtype=torch.int32,
                    pin_memory=pin_memory,
                )
                for _ in range(cls.buffer_depth)
            ]
            # Allocate device outputs lazily in sync_forward. SGLang supplies
            # its now-dead shared input staging tensor as the copy-out target,
            # avoiding two persistent [chunk, hidden] GPU ring slots.
            output_gpu: List[Optional[torch.Tensor]] = [
                None for _ in range(cls.buffer_depth)
            ]

        cur_buffer = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = cur_buffer
        return cur_buffer


class _MoEBase:
    """
    Shared base class for inference and SFT MoE wrappers.

    Provides:
    - CPUInfer singleton management
    - Basic configuration validation

    This class is shared between BaseMoEWrapper (inference) and BaseSFTMoEWrapper (SFT).
    """

    _cpu_infer_instance = None

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        """
        Get or create the CPUInfer singleton instance.

        Args:
            cpuinfer_threads: Total number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            numa_nodes: Explicit list of NUMA node IDs. If None, defaults to sequential.

        Returns:
            CPUInfer singleton instance
        """
        if cls._cpu_infer_instance is None:
            worker_config = kt_kernel_ext.WorkerPoolConfig()

            if numa_nodes is not None:
                if len(numa_nodes) != threadpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match "
                        f"threadpool_count ({threadpool_count})"
                    )
                subpool_numa_map = list(numa_nodes)
            else:
                subpool_numa_map = list(range(threadpool_count))
            subpool_thread_count = [
                cpuinfer_threads // threadpool_count
                + (1 if i < cpuinfer_threads % threadpool_count else 0)
                for i in range(threadpool_count)
            ]

            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            cls._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)

        return cls._cpu_infer_instance

    @staticmethod
    def _validate_base_config(
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
    ) -> None:
        """
        Validate basic configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if moe_intermediate_size <= 0:
            raise ValueError(
                f"moe_intermediate_size must be positive, got {moe_intermediate_size}"
            )
        if num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be positive, got {num_experts_per_tok}"
            )
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed "
                f"num_experts ({num_experts})"
            )


class BaseMoEWrapper(_MoEBase, ABC):
    """
    Base class for MoE CPU inference operations.
    Provides common functionality for all backend implementations.
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
    ):
        """
        Initialize base MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            gpu_experts_mask: Boolean mask indicating which experts are on GPU.
                              Shape: [num_experts], dtype: torch.bool.
                              mask[i] = True means expert i is on GPU.
                              If None, all experts are on CPU.
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of lower-scored experts per
                token to schedule as a tail task on this layer. The tail is
                completed before this layer returns. Defaults to 0 (no split).
            method: Backend method string
            numa_nodes: Explicit list of NUMA node IDs for subpool mapping.
                        If None, defaults to [0, 1, ..., threadpool_count-1].
        """
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # A CPU-only expert sidecar never copies this mask to a GPU.  Allow it
        # to run with CUDA hidden so the sidecar does not consume a CUDA
        # context on the serving GPU.
        pin_expert_mask = os.environ.get("KT_CPU_ONLY_SIDECAR") != "1"

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = torch.zeros(
                num_experts,
                dtype=torch.bool,
                device="cpu",
                pin_memory=pin_expert_mask,
            )
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = torch.empty(
                num_experts,
                dtype=torch.bool,
                device="cpu",
                pin_memory=pin_expert_mask,
            )
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token)
            if max_deferred_experts_per_token is not None
            else 0
        )

        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)

        # Initialize CPU inference engine (singleton via shared base class)
        self.cpu_infer = self._get_cpu_infer(
            cpuinfer_threads, threadpool_count, numa_nodes=numa_nodes
        )

        # Backend-specific initialization happens in subclasses
        self.moe = None
        # Route telemetry is diagnostic-only and therefore disabled unless KT
        # timing is explicitly requested.  When enabled, each wrapper/layer
        # owns two tiny pinned route slots per captured shape.  CPUInfer reads
        # those same slots, so telemetry adds no duplicate D2H route copy and
        # never aliases the global layer-parity ring.
        self._route_telemetry_enabled = (
            os.environ.get("SGLANG_KT_HYBRID_TIMING") == "1"
        )
        self._route_telemetry_generation = 0
        self._route_telemetry_state = _RouteTelemetryState()
        self._route_telemetry_buffers: Dict[
            Tuple[int, int, bool],
            Tuple[
                Tuple[torch.Tensor, Optional[torch.Tensor]],
                Tuple[torch.Tensor, Optional[torch.Tensor]],
            ],
        ] = {}
        self._route_telemetry_events: List[Optional[_RouteCompletion]] = [None, None]
        self._route_telemetry_waiting_for_sync: Optional[
            _RouteTelemetryGeneration
        ] = None
        self._route_telemetry_error: Optional[BaseException] = None

    @abstractmethod
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Load and quantize weights from BF16/FP16 tensors (online quantization).

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """

    def select_deferred_experts(
        self,
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, topk = expert_ids.shape
        device = expert_ids.device

        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:
            deferred_ids = expert_ids.clone()
            immediate_ids = torch.full_like(expert_ids, -1)
            return immediate_ids, deferred_ids

        topk_result = torch.topk(
            expert_scores, k=protected_k, dim=-1, largest=True, sorted=False
        )
        protected_indices = topk_result.indices
        protected_ids = torch.gather(expert_ids, -1, protected_indices)

        # EP wrappers map experts that are not owned by the CPU complement to
        # -1.  Keep those sentinel entries out of scatter/gather: CUDA treats
        # a negative scatter index as a device-side assertion rather than the
        # Python-style wraparound used by ordinary tensor indexing.
        valid_expert_ids = (expert_ids >= 0) & (expert_ids < self.num_experts)
        valid_protected_ids = (protected_ids >= 0) & (protected_ids < self.num_experts)
        # Reserve one fixed sentinel slot instead of compacting the valid IDs
        # with boolean indexing.  Boolean indexing has a data-dependent output
        # shape and is therefore illegal while SGLang captures a full decode
        # CUDA graph.  The extra slot keeps every scatter/gather shape static
        # while preventing invalid routes from aliasing logical expert zero.
        sentinel_expert_id = self.num_experts
        safe_protected_ids = protected_ids.masked_fill(
            ~valid_protected_ids, sentinel_expert_id
        )
        protected_flag = torch.zeros(
            (self.num_experts + 1,), dtype=torch.int32, device=device
        )
        protected_flag.scatter_(0, safe_protected_ids.reshape(-1), 1)

        safe_expert_ids = expert_ids.masked_fill(~valid_expert_ids, sentinel_expert_id)
        protected_mask_flat = torch.gather(
            protected_flag, 0, safe_expert_ids.reshape(-1)
        ).ne(0)
        protected_mask_flat &= valid_expert_ids.reshape(-1)
        protected_mask = protected_mask_flat.view(batch, topk)

        immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
        deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)

        return immediate_ids, deferred_ids

    def _prepare_route_telemetry_generation(
        self,
        immediate_ids: torch.Tensor,
        deferred_ids: Optional[torch.Tensor],
    ) -> Optional[_RouteTelemetryGeneration]:
        """Select wrapper-local pinned route storage for one generation.

        This helper is fail-open for serving: any diagnostic allocation or
        fence-query failure makes this generation use the original global
        staging ring instead.  No CUDA wait is introduced.
        """
        if not getattr(self, "_route_telemetry_enabled", False):
            return None
        # Route telemetry is diagnostic-only.  In particular, do not call
        # completed_snapshot() here while a graph is being captured: its
        # nonblocking Event.query() maps to cudaEventQuery, which CUDA forbids
        # during capture and reports asynchronously at the following D2H copy.
        # A missing shape would also allocate pinned host memory, another
        # capture-invalidating operation.  The original KT staging ring remains
        # the graph's serving path when diagnostics are skipped.
        if _cuda_graph_capture_active():
            self._route_telemetry_waiting_for_sync = None
            return None
        try:
            # Retire any prior generation whose nonblocking fence has fired
            # before selecting a slot that may be reused by this submission.
            self._route_telemetry_state.completed_snapshot()
            next_generation = self._route_telemetry_generation + 1
            slot_index = next_generation % 2
            rows, columns = immediate_ids.shape
            buffer_key = (int(rows), int(columns), deferred_ids is not None)
            route_slots = self._route_telemetry_buffers.get(buffer_key)
            if route_slots is None:
                with torch.inference_mode(False):
                    allocated_slots = []
                    for _ in range(2):
                        immediate_buffer = torch.empty(
                            (rows, columns),
                            dtype=torch.int64,
                            device="cpu",
                            pin_memory=True,
                        )
                        deferred_buffer = (
                            torch.empty_like(
                                immediate_buffer,
                                device="cpu",
                                pin_memory=True,
                            )
                            if deferred_ids is not None
                            else None
                        )
                        allocated_slots.append((immediate_buffer, deferred_buffer))
                route_slots = (allocated_slots[0], allocated_slots[1])
                self._route_telemetry_buffers[buffer_key] = route_slots

            immediate_buffer, deferred_buffer = route_slots[slot_index]
            if self._route_telemetry_state.source_is_pending(immediate_buffer):
                raise RuntimeError(
                    "route telemetry staging depth exhausted before completion"
                )
            route_generation = _RouteTelemetryGeneration(
                generation=next_generation,
                immediate_ids=immediate_buffer,
                deferred_ids=deferred_buffer,
            )
            self._route_telemetry_generation = next_generation
            self._route_telemetry_state.add(route_generation)
            self._route_telemetry_waiting_for_sync = route_generation
            return route_generation
        except Exception as error:
            self._route_telemetry_error = error
            self._route_telemetry_waiting_for_sync = None
            return None

    def _record_route_telemetry_completion(self, hidden_states: torch.Tensor) -> None:
        """Record a nonblocking fence after the complete CPU result copy."""
        route_generation = getattr(self, "_route_telemetry_waiting_for_sync", None)
        self._route_telemetry_waiting_for_sync = None
        if route_generation is None:
            return
        try:
            slot_index = route_generation.generation % 2
            completion = self._route_telemetry_events[slot_index]
            if completion is None or self._route_telemetry_state.completion_is_pending(
                completion
            ):
                completion = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                self._route_telemetry_events[slot_index] = completion
            # sync_with_cuda_stream() and the H2D output copy were enqueued on
            # this same current stream immediately before this call.  Querying
            # the event later is therefore an explicit completion test for the
            # whole CPU forward, not merely for the earlier route D2H copy.
            completion.record(torch.cuda.current_stream(hidden_states.device))
            route_generation.completion = completion
        except Exception as error:
            self._route_telemetry_error = error

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
        """
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        topk_ids_long = topk_ids.to(torch.long)
        immediate_ids: torch.Tensor
        deferred_ids: Optional[torch.Tensor]
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token

            immediate_ids, deferred_ids = self.select_deferred_experts(
                topk_ids_long, topk_weights, protected_k
            )
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        route_generation = BaseMoEWrapper._prepare_route_telemetry_generation(
            self,
            immediate_ids,
            deferred_ids,
        )
        if route_generation is None:
            immediate_ids_staging = immediate_experts_ids_cpu[current_slot]
            deferred_ids_staging = (
                deferred_experts_ids_cpu[current_slot]
                if deferred_ids is not None
                else None
            )
        else:
            immediate_ids_staging = route_generation.immediate_ids
            deferred_ids_staging = route_generation.deferred_ids

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_ids_staging.copy_(immediate_ids, non_blocking=True)

        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                immediate_ids_staging.size(-1),
                immediate_ids_staging.data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[current_slot].data_ptr(),
                False,
            ),
        )

        if deferred_ids is not None:
            assert deferred_ids_staging is not None
            deferred_ids_staging.copy_(deferred_ids, non_blocking=True)
            # The old scheduler wrote this tail into the next layer's slot and
            # let sync_forward return while it was still pending.  The next
            # layer then folded that stale contribution into its own MoE
            # output.  Besides making the result depend on cross-stream task
            # timing, that is not a legal transformer reschedule: layer L+1
            # must consume layer L's complete result.
            #
            # Keep the useful high-score/tail partition, but finish the tail
            # in FIFO order into this layer's output.  All kt-kernel backends
            # implement incremental merge as ``output = output + task``.
            self.cpu_infer.submit_with_cuda_stream(
                cuda_stream,
                self.moe.forward_task(
                    bsz_slot_tensor.data_ptr(),
                    deferred_ids_staging.size(-1),
                    deferred_ids_staging.data_ptr(),
                    weights_cpu[current_slot].data_ptr(),
                    input_tensor_cpu[current_slot].data_ptr(),
                    output_cpu[current_slot].data_ptr(),
                    True,
                ),
            )

    def get_last_forward_route_stats(self) -> Optional[Dict[str, object]]:
        """Return the newest completed generation, never the live generation.

        ``Event.query`` is nonblocking.  If the current CPUInfer work is still
        pending, this returns the prior immutable completed snapshot (or
        ``None`` before the first completion) rather than synchronizing CUDA.
        """
        # cudaEventQuery is nonblocking in ordinary execution but is still an
        # unsupported API during stream capture.  Timing receipts are optional,
        # so omit route evidence for capture-time forwards instead of poisoning
        # the model graph.
        if _cuda_graph_capture_active():
            return None
        telemetry_error = self._route_telemetry_error
        self._route_telemetry_error = None
        completed_snapshot = self._route_telemetry_state.completed_snapshot()
        if completed_snapshot is None:
            if telemetry_error is not None:
                raise RuntimeError(
                    "route telemetry generation failed"
                ) from telemetry_error
            return None

        immediate_snapshot = completed_snapshot.immediate_ids
        route_ids = immediate_snapshot
        if completed_snapshot.deferred_ids is not None:
            deferred_snapshot = completed_snapshot.deferred_ids
            if deferred_snapshot.numel() != immediate_snapshot.numel():
                raise ValueError(
                    "immediate and deferred route buffers must contain the same "
                    "number of entries"
                )
            immediate_valid = (immediate_snapshot >= 0) & (
                immediate_snapshot < self.num_experts
            )
            route_ids = torch.where(
                immediate_valid,
                immediate_snapshot,
                deferred_snapshot,
            )

        valid_route_ids = route_ids[(route_ids >= 0) & (route_ids < self.num_experts)]
        if valid_route_ids.numel() == 0:
            active_expert_count = 0
            route_row_count = 0
            rows_per_expert_histogram: Dict[str, int] = {}
        else:
            expert_row_counts = torch.bincount(
                valid_route_ids,
                minlength=self.num_experts,
            )
            active_row_counts = expert_row_counts[expert_row_counts > 0]
            active_expert_count = int(active_row_counts.numel())
            route_row_count = int(valid_route_ids.numel())
            histogram = torch.bincount(active_row_counts)
            rows_per_expert_histogram = {
                str(row_count): int(count)
                for row_count, count in enumerate(histogram.tolist())
                if row_count > 0 and count > 0
            }

        estimated_expert_weight_bytes: Optional[int] = None
        if self.method == "MXFP4":
            weight_element_count = 3 * self.hidden_size * self.moe_intermediate_size
            estimated_expert_weight_bytes = (weight_element_count + 1) // 2 + (
                weight_element_count + 31
            ) // 32
        return {
            "cpu_route_generation": completed_snapshot.generation,
            "cpu_active_experts": active_expert_count,
            "cpu_route_rows": route_row_count,
            "cpu_rows_per_expert": rows_per_expert_histogram,
            "estimated_expert_weight_bytes": estimated_expert_weight_bytes,
            "estimated_cpu_weight_stream_bytes": (
                None
                if estimated_expert_weight_bytes is None
                else active_expert_count * estimated_expert_weight_bytes
            ),
        }

    def sync_forward(
        self,
        hidden_states: torch.Tensor,
        cuda_stream,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU. If output_tensor is supplied,
                its storage is reused after the CPU task has consumed the
                staged input.
        """
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        # Returning before the tail task completes changes model semantics and
        # also exposes the shared ring slot to the following layer.  Drain the
        # queue before copying this layer's complete CPU contribution out.
        self.cpu_infer.sync_with_cuda_stream(cuda_stream)
        if output_tensor is None:
            current_output = output_gpu[current_slot]
            if (
                current_output is None
                or current_output.shape != flat_hidden_states.shape
                or current_output.dtype != flat_hidden_states.dtype
                or current_output.device != flat_hidden_states.device
            ):
                current_output = torch.empty_like(flat_hidden_states)
                output_gpu[current_slot] = current_output
        else:
            current_output = output_tensor.view_as(flat_hidden_states)
            if (
                current_output.dtype != flat_hidden_states.dtype
                or current_output.device != flat_hidden_states.device
            ):
                raise ValueError(
                    "KExperts CPU output reuse requires matching dtype and device"
                )
        current_output.copy_(output_cpu[current_slot], non_blocking=True)
        BaseMoEWrapper._record_route_telemetry_completion(self, hidden_states)
        return current_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        """
        Execute forward inference synchronously (submit + sync).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization

        Returns:
            Output tensor on GPU
        """
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        return self.sync_forward(hidden_states, cuda_stream)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
        """
        KExpertsCPUBuffer.capture_bs = capture_bs
        temp_batch_size = KExpertsCPUBuffer.temp_bs
        if temp_batch_size in capture_bs and KExpertsCPUBuffer.temp_buffer:
            KExpertsCPUBuffer.capture_buffers[temp_batch_size] = (
                KExpertsCPUBuffer.temp_buffer
            )

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return KExpertsCPUBuffer.capture_bs

    @staticmethod
    def clear_buffer_cache():
        """
        Clear all cached buffers.

        This frees up memory by clearing the buffer cache. Useful when you want
        to reset the buffer state or free memory.
        """
        KExpertsCPUBuffer.capture_buffers.clear()
        KExpertsCPUBuffer.temp_bs = 0
        KExpertsCPUBuffer.temp_buffer = tuple()
