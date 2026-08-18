# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MoonEP resource, symmetric-weight, and autograd integration.

Imports from :mod:`moonep` are deliberately delayed until a MoonEP dispatcher
is selected. This keeps installations that do not use MoonEP independent of
its optional accelerator extension.
"""

from __future__ import annotations

import importlib.metadata
import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from fsdp_turbo.ops.grad_weight_sink import GradWeightSink

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh


_SUPPORTED_MOONEP_VERSION = "0.0.1"


def _is_cuda() -> bool:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return False
    current = accelerator.current_accelerator()
    return current is not None and current.type == "cuda"


def _current_stream():
    if _is_cuda():
        return torch.cuda.current_stream()
    return torch.accelerator.current_stream()


def _stream_ctx(stream):
    """Activate ``stream`` with the native CUDA/NPU context manager."""
    if _is_cuda():
        return torch.cuda.stream(stream)
    return torch.accelerator.stream(stream)


def _wait_event(event, stream=None) -> None:
    """Wait with ``Stream.wait_event``, not ``Event.wait(accelerator stream)``.

    ``Event.wait(torch.accelerator.current_stream())`` SIGSEGVs on PyTorch 2.9.
    """
    if event is None:
        return
    if stream is None:
        stream = _current_stream()
    wait_event = getattr(stream, "wait_event", None)
    if callable(wait_event):
        wait_event(event)
        return
    if _is_cuda():
        torch.cuda.current_stream().wait_event(event)
        return
    event.wait(stream)


def _dispatch_async_finish(runtime) -> bool:
    """CUDA dispatch/combine stay on the compute stream.

    ``Buffer.dispatch(async_finish=True)`` compiles CuTe cooperative grids on
    the high-priority comm stream. On H20 that is an unspecified launch
    failure; other ranks then time out in ``cross_rank_barrier``. Prefetch is
    a non-cooperative copy and may still use ``config.async_finish``.
    """
    if getattr(runtime, "accelerator_type", None) == "cuda":
        return False
    return bool(runtime.config.async_finish)


class MoonEPRuntimeConfig(Protocol):
    """Configuration values consumed by the MoonEP runtime."""

    num_sms: int
    token_padding: int
    comm_stream_priority: int
    enable_pdl: bool
    async_finish: bool
    tokens_per_rank: int | None
    top_k: int | None


@dataclass
class _MoonEPImports:
    Buffer: Any
    launch_prefetch: Any
    launch_grad_reduce: Any
    launch_inter_rank_sync: Any
    create_nvl_dist_tensor: Any
    exchange_ipc_fds: Any
    nvl_dist_alloc: Any
    nvl_release_mem_handle: Any
    nvl_dist_map: Any
    get_vmm_granularity: Any
    nvl_multicast_supported: Any


def _load_moonep() -> _MoonEPImports:
    try:
        version = importlib.metadata.version("moonep")
        from moonep import Buffer
        from moonep.buffer import create_nvl_dist_tensor, _exchange_ipc_fds
        from moonep.prefetch import launch_prefetch
        from moonep.grad_reduce import launch_grad_reduce
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep._C import (
            get_vmm_granularity,
            nvl_dist_alloc,
            nvl_dist_map,
            nvl_multicast_supported,
            nvl_release_mem_handle,
        )
    except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "MoonEP dispatcher requires moonep==0.0.1. Install the sibling "
            "repository with `pip install -e ../MoonEP` or install "
            "`fsdp-turbo[moonep]`."
        ) from exc

    if version != _SUPPORTED_MOONEP_VERSION:
        raise RuntimeError(
            f"Unsupported MoonEP version {version!r}; this adapter requires "
            f"{_SUPPORTED_MOONEP_VERSION!r} because it uses the versioned "
            "VMM and kernel-launcher interfaces."
        )

    return _MoonEPImports(
        Buffer=Buffer,
        launch_prefetch=launch_prefetch,
        launch_grad_reduce=launch_grad_reduce,
        launch_inter_rank_sync=launch_inter_rank_sync,
        create_nvl_dist_tensor=create_nvl_dist_tensor,
        exchange_ipc_fds=_exchange_ipc_fds,
        nvl_dist_alloc=nvl_dist_alloc,
        nvl_release_mem_handle=nvl_release_mem_handle,
        nvl_dist_map=nvl_dist_map,
        get_vmm_granularity=get_vmm_granularity,
        nvl_multicast_supported=nvl_multicast_supported,
    )


def _validate_ep_host_group(group) -> None:
    hosts = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise RuntimeError(
            "MoonEP only supports an intra-node EP group; group members were "
            f"found on hosts {sorted(set(hosts))}."
        )


def _validate_projection_shape(
    name: str,
    tensor: torch.Tensor,
    ep_size: int,
    granularity: int,
    accelerator_type: str,
) -> None:
    if tensor.device.type != accelerator_type:
        raise RuntimeError(
            f"MoonEP {name} must be initialized on the current {accelerator_type} "
            f"accelerator, got {tensor.device}."
        )
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"MoonEP {name} must be BF16, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise RuntimeError(f"MoonEP {name} must be contiguous.")
    if tensor.ndim != 3:
        raise RuntimeError(
            f"MoonEP {name} must have [experts, out_features, in_features] shape, "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] % ep_size:
        raise RuntimeError(
            f"MoonEP {name} expert dimension {tensor.shape[0]} is not divisible "
            f"by EP size {ep_size}."
        )
    if tensor.shape[-2] % 128 or tensor.shape[-1] % 128:
        raise RuntimeError(
            f"MoonEP {name} matrix dimensions must be multiples of 128 for "
            f"prefetch/gradient kernels, got {tuple(tensor.shape[-2:])}."
        )
    local_rows = tensor.shape[0] // ep_size
    chunk_bytes = local_rows * tensor.shape[-2] * tensor.shape[-1] * tensor.element_size()
    if chunk_bytes % granularity:
        raise RuntimeError(
            f"MoonEP {name} local expert chunk is {chunk_bytes} bytes, which is "
            f"not aligned to the VMM granularity {granularity}."
        )


class _ProjectionPool:
    """Process-local prefetch slots and the rank-mapped FP32 gradient buffer.

    The gradient side follows MoonEP's documented layout: one contiguous
    ``[E+B, out, in]`` FP32 mapping whose rows ``[0, E)`` are every rank's
    parameter gradients and whose rows ``[E, E+B)`` alias this rank's slice of
    the reduce buffer. Handing that mapping to the grouped matmul lets the
    backward pass accumulate in place instead of allocating a second full-size
    gradient tensor and copying it in afterwards.

    One pool is shared by every layer with the same projection shape, so the
    extra memory is a fixed cost per shape rather than per layer.
    """

    def __init__(self, runtime: "MoonEPRuntime", role: str, shape: tuple[int, int]):
        self.runtime = runtime
        self.role = role
        self.shape = shape
        self.closed = False
        experts_per_rank = runtime.experts_per_rank
        out_features, in_features = shape
        chunk_shape = [experts_per_rank, out_features, in_features]
        imports = runtime.imports
        ranks = list(range(runtime.size))

        allocation, export_handle, owned_handle = imports.nvl_dist_alloc(
            shape=chunk_shape, dtype=torch.bfloat16
        )
        imports.nvl_release_mem_handle(owned_handle)
        self.prefetch_allocation = allocation
        self.prefetch_export_handle = export_handle

        # Reduce slots hold duplicated experts' gradients. They are allocated
        # explicitly rather than through ``create_nvl_dist_tensor`` so the local
        # chunk's handle can also back rows [E, E+B) of the main-grad mapping.
        reduce_allocation, reduce_handle, reduce_owned = imports.nvl_dist_alloc(
            shape=chunk_shape, dtype=torch.float32
        )
        imports.nvl_release_mem_handle(reduce_owned)
        self.reduce_allocation = reduce_allocation
        self.reduce_export_handle = reduce_handle
        self.reduce_full = self._map_across_ranks(
            chunk_shape, torch.float32, os.dup(reduce_handle), ranks
        )
        self.reduce_buffers = self.reduce_full.view(
            runtime.size, experts_per_rank, out_features, in_features
        )

        owner_allocation, owner_handle, owner_owned = imports.nvl_dist_alloc(
            shape=chunk_shape, dtype=torch.float32
        )
        imports.nvl_release_mem_handle(owner_owned)
        self.owner_grad_allocation = owner_allocation
        self.main_grad = self._map_across_ranks(
            chunk_shape,
            torch.float32,
            owner_handle,
            ranks,
            extra_handle=os.dup(reduce_handle),
        )
        self.owner_grad_full = self.main_grad[: runtime.num_experts]
        self.owner_grad_buffers = self.owner_grad_full.view(
            runtime.size, experts_per_rank, out_features, in_features
        )
        self.owner_grad_buffers[runtime.rank].zero_()
        self.main_grad[runtime.num_experts:].zero_()
        dist.barrier(group=runtime.group)

    def _map_across_ranks(
        self,
        chunk_shape: list[int],
        dtype: torch.dtype,
        local_handle: int,
        ranks: list[int],
        extra_handle: int | None = None,
    ) -> torch.Tensor:
        """Map every rank's chunk, plus an optional local one, as one tensor."""
        runtime = self.runtime
        imports = runtime.imports
        exchanged = imports.exchange_ipc_fds(
            local_handle, ranks, runtime.rank, runtime.size, runtime.group
        )
        os.close(local_handle)
        handles = [exchanged[rank] for rank in ranks]
        if extra_handle is not None:
            handles.append(extra_handle)
        try:
            return imports.nvl_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                fds=handles,
                local_rank=runtime.rank,
                world_size=len(handles),
            )
        finally:
            for handle in handles:
                os.close(handle)

    def duplicate_prefetch_handle(self) -> int:
        if self.closed:
            raise RuntimeError("MoonEP projection pool is already closed.")
        return os.dup(self.prefetch_export_handle)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.prefetch_export_handle is not None:
            os.close(self.prefetch_export_handle)
            self.prefetch_export_handle = None
        if self.reduce_export_handle is not None:
            os.close(self.reduce_export_handle)
            self.reduce_export_handle = None
        self.owner_grad_buffers = None
        self.owner_grad_full = None
        self.main_grad = None
        self.owner_grad_allocation = None
        self.reduce_buffers = None
        self.reduce_full = None
        self.reduce_allocation = None
        self.prefetch_allocation = None


class MoonEPSymmetricProjection:
    """A layer projection backed by a MoonEP ``[E+B]`` VMM mapping."""

    def __init__(
        self,
        runtime: "MoonEPRuntime",
        role: str,
        source: torch.Tensor,
        ep_mesh: "DeviceMesh",
    ):
        self.runtime = runtime
        self.role = role
        self.closed = False
        self._grad_sink: GradWeightSink | None = None
        self.global_shape = tuple(source.shape)
        self.out_features = int(source.shape[-2])
        self.in_features = int(source.shape[-1])
        self.pool = runtime.get_projection_pool(role, (self.out_features, self.in_features))

        experts_per_rank = runtime.experts_per_rank
        start = runtime.rank * experts_per_rank
        end = start + experts_per_rank
        imports = runtime.imports

        allocation, local_export_handle, owned_handle = imports.nvl_dist_alloc(
            shape=[experts_per_rank, self.out_features, self.in_features], dtype=torch.bfloat16
        )
        imports.nvl_release_mem_handle(owned_handle)
        self.expert_allocation = allocation

        exchanged = imports.exchange_ipc_fds(
            local_export_handle, list(range(runtime.size)), runtime.rank, runtime.size, runtime.group
        )
        os.close(local_export_handle)
        expert_handles = [exchanged[rank] for rank in range(runtime.size)]
        prefetch_handle = self.pool.duplicate_prefetch_handle()
        try:
            self.full_weight = imports.nvl_dist_map(
                chunk_shape=[experts_per_rank, self.out_features, self.in_features],
                dtype=torch.bfloat16,
                fds=expert_handles + [prefetch_handle],
                local_rank=runtime.rank,
                world_size=runtime.size + 1,
            )
        finally:
            for expert_handle in expert_handles:
                os.close(expert_handle)
            os.close(prefetch_handle)

        source_local = source.to_local() if isinstance(source, DTensor) else source[start:end]
        self.full_weight[start:end].copy_(source_local.detach())
        self.full_weight[runtime.num_experts:].zero_()
        dist.barrier(group=runtime.group)

        local_view = self.full_weight[start:end]
        local_dtensor = DTensor.from_local(
            local_view,
            ep_mesh,
            (Shard(0),),
            run_check=False,
        )
        self.parameter = torch.nn.Parameter(local_dtensor, requires_grad=source.requires_grad)

    def prefetch(self, plan) -> None:
        experts = plan.experts_to_copy[self.runtime.rank]
        if not experts.is_contiguous():
            experts = experts.contiguous()
        self.runtime.imports.launch_prefetch(
            self.full_weight[:self.runtime.num_experts],
            self.full_weight[self.runtime.num_experts:],
            experts,
            num_sms=self.runtime.config.num_sms,
        )

    def grad_weight_sink(self) -> GradWeightSink:
        """Expose the ``[E+B]`` FP32 buffer the grouped matmul writes into.

        Only the rows this rank physically owns are writable: its own experts
        and the local prefetch slots. Every other row reaches a remote rank
        through the symmetric mapping. The sink is cached so that any staging
        buffer it allocates is reused for the lifetime of the projection.
        """
        if self._grad_sink is None:
            num_experts = self.runtime.num_experts
            experts_per_rank = self.runtime.experts_per_rank
            start = self.runtime.rank * experts_per_rank
            self._grad_sink = GradWeightSink(
                buffer=self.pool.main_grad,
                row_ranges=(
                    (start, start + experts_per_rank),
                    (num_experts, num_experts + experts_per_rank),
                ),
            )
        return self._grad_sink

    def reduce_gradient(self, plan, comm_context: dict, dtype: torch.dtype) -> torch.Tensor:
        rank = self.runtime.rank
        experts_per_rank = self.runtime.experts_per_rank
        start = rank * experts_per_rank
        end = start + experts_per_rank

        # The grouped matmul already wrote this step's FP32 gradients into the
        # [E+B] mapping: rows [0, E) are the owner ranks' gradients and rows
        # [E, E+B) alias this rank's reduce slots.
        #
        # ``launch_grad_reduce`` reads every rank's reduce slots before its
        # internal cross-rank barrier. Publish the local slot writes and wait
        # until every peer has done the same before any rank starts reading.
        # This is required when autograd reaches the weight bridge at slightly
        # different times on different ranks.
        self.runtime.imports.launch_inter_rank_sync(comm_context)
        self.runtime.imports.launch_grad_reduce(
            self.pool.owner_grad_full,
            self.pool.reduce_buffers,
            plan.experts_to_copy,
            rank=rank,
            num_sms=self.runtime.config.num_sms,
            meta_buf=comm_context["meta_buf"],
            meta_stride=int(comm_context["meta_chunk_padded"]),
            barrier_off=int(comm_context["BARRIER_OFF"]),
            grid_sync_bar=comm_context["grid_sync_bar"],
        )
        # The pool is reused by other layers with the same projection shape.
        # Casting back to the parameter dtype also gives autograd an
        # independent, local-sized result before the pool is overwritten.
        return self.pool.main_grad[start:end].to(dtype=dtype)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._grad_sink = None
        self.full_weight = None
        self.expert_allocation = None


class _MoonEPWeightGradBridge(torch.autograd.Function):
    """Reduce both projections' weight gradients and hand them to autograd.

    The node is applied to the dispatched activations, i.e. upstream of both
    grouped matmuls, so in the backward pass it runs only once every expert
    gradient has been written into the ``[E+B]`` buffers. A single multi-output
    node also launches the cross-rank reductions in the same order on every
    rank; independent nodes could be scheduled differently and deadlock
    MoonEP's internal barriers.
    """

    @staticmethod
    def forward(
        ctx,
        dispatched,
        local_gate_up,
        local_down,
        gate_up_projection,
        down_projection,
        plan,
        comm_context,
    ):
        ctx.gate_up_projection = gate_up_projection
        ctx.down_projection = down_projection
        ctx.plan = plan
        ctx.comm_context = comm_context
        ctx.gate_up_dtype = local_gate_up.dtype
        ctx.down_dtype = local_down.dtype
        return dispatched.as_strided(dispatched.shape, dispatched.stride())

    @staticmethod
    def backward(ctx, grad_dispatched):
        # Fixed ordering is part of the distributed protocol.
        with torch.profiler.record_function("moonep.grad_reduce"):
            local_down = ctx.down_projection.reduce_gradient(
                ctx.plan, ctx.comm_context, ctx.down_dtype
            )
            local_gate_up = ctx.gate_up_projection.reduce_gradient(
                ctx.plan, ctx.comm_context, ctx.gate_up_dtype
            )
        return (
            grad_dispatched,
            local_gate_up,
            local_down,
            None,
            None,
            None,
            None,
        )


class MoonEPCallState:
    def __init__(
        self,
        runtime: "MoonEPRuntime",
        buffer,
        tokens_per_rank: int,
        hidden_dim: int,
        top_k: int,
    ):
        self.runtime = runtime
        self.buffer = buffer
        self.tokens_per_rank = tokens_per_rank
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.plan = None
        self.cu_seqlens = None
        self.dispatch_event = None

    @property
    def comm_context(self) -> dict:
        return self.buffer._require_ctx()

    def wait(self, event) -> None:
        _wait_event(event)

    def prefetch(self, *projections: MoonEPSymmetricProjection) -> None:
        if self.plan is None:
            raise RuntimeError("MoonEP dispatch plan is not initialized.")
        with torch.profiler.record_function("moonep.prefetch"):
            if self.runtime.config.async_finish:
                comm_stream = self.buffer._comm_stream
                if comm_stream is None:
                    raise RuntimeError("MoonEP communication stream is unavailable.")
                main_stream = _current_stream()
                recorded = [self.plan.experts_to_copy]
                recorded.extend(projection.full_weight for projection in projections)
                for tensor in recorded:
                    if tensor is not None:
                        tensor.record_stream(comm_stream)
                comm_stream.wait_event(main_stream.record_event())
                with _stream_ctx(comm_stream):
                    for projection in projections:
                        projection.prefetch(self.plan)
                    done = comm_stream.record_event()
                _wait_event(done, main_stream)
                self.dispatch_event = None
            else:
                for projection in projections:
                    projection.prefetch(self.plan)


class MoonEPRuntime:
    """Own all MoonEP resources associated with one FSDPTurbo model."""

    def __init__(
        self,
        group,
        ep_mesh: "DeviceMesh",
        num_experts: int,
        config: MoonEPRuntimeConfig,
    ):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None or accelerator.type not in {"cuda", "npu"}:
            raise RuntimeError(
                "MoonEP requires a CUDA or NPU accelerator; use another dispatcher "
                "on unsupported devices."
            )
        if torch.accelerator.current_device_index() < 0:
            raise RuntimeError("MoonEP requires a current accelerator device.")

        self.imports = _load_moonep()
        self.accelerator_type = accelerator.type
        self.group = group
        self.ep_mesh = ep_mesh
        self.rank = dist.get_rank(group=group)
        self.size = dist.get_world_size(group=group)
        self.num_experts = int(num_experts)
        self.experts_per_rank = self.num_experts // self.size
        self.config = config
        self._tokens_per_rank = config.tokens_per_rank
        self._top_k = config.top_k
        self.closed = False
        self._buffers: dict[tuple[int, int, int], Any] = {}
        self._projection_pools: dict[tuple[str, int, int], _ProjectionPool] = {}
        self._projections: list[MoonEPSymmetricProjection] = []

        _validate_ep_host_group(group)
        if not self.imports.nvl_multicast_supported():
            raise RuntimeError(
                "MoonEP requires accelerator multicast support. Use another EP "
                "dispatcher on unsupported hardware or interconnects."
            )
        if self.num_experts % self.size:
            raise RuntimeError(
                f"MoonEP experts ({self.num_experts}) must be divisible by EP size ({self.size})."
            )

    def get_projection_pool(self, role: str, shape: tuple[int, int]) -> _ProjectionPool:
        key = (role, *shape)
        pool = self._projection_pools.get(key)
        if pool is None:
            pool = _ProjectionPool(self, role, shape)
            self._projection_pools[key] = pool
        return pool

    def distribute_projection(
        self, role: str, source: torch.Tensor, ep_mesh: "DeviceMesh"
    ) -> MoonEPSymmetricProjection:
        granularity = int(self.imports.get_vmm_granularity())
        _validate_projection_shape(
            role, source, self.size, granularity, self.accelerator_type
        )
        projection = MoonEPSymmetricProjection(self, role, source, ep_mesh)
        self._projections.append(projection)
        return projection

    def new_call(
        self,
        tokens_per_rank: int,
        hidden_dim: int,
        top_k: int,
    ) -> MoonEPCallState:
        if self._tokens_per_rank is None:
            self._tokens_per_rank = tokens_per_rank
        elif self._tokens_per_rank != tokens_per_rank:
            raise RuntimeError(
                "MoonEP uses a static token shape: expected "
                f"S={self._tokens_per_rank}, got S={tokens_per_rank}. "
                "Pad/drop the batch or use another dispatcher."
            )
        if self._top_k is None:
            self._top_k = top_k
        elif self._top_k != top_k:
            raise RuntimeError(
                f"MoonEP uses a static top-k: expected K={self._top_k}, got K={top_k}."
            )

        key = (tokens_per_rank, hidden_dim, top_k)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self.imports.Buffer(
                S=tokens_per_rank,
                H=hidden_dim,
                K=top_k,
                E=self.num_experts,
                num_ep_ranks=self.size,
                B=self.experts_per_rank,
                num_sms=self.config.num_sms,
                token_padding=self.config.token_padding,
                group=self.group,
                comm_stream_priority=self.config.comm_stream_priority,
                enable_pdl=self.config.enable_pdl,
                explicitly_destroy=True,
            )
            self._buffers[key] = buffer
        return MoonEPCallState(
            self, buffer, tokens_per_rank, hidden_dim, top_k
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for buffer in self._buffers.values():
            buffer.destroy()
        self._buffers.clear()
        if dist.is_initialized():
            dist.barrier(group=self.group)
        for projection in self._projections:
            projection.close()
        self._projections.clear()
        for pool in self._projection_pools.values():
            pool.close()
        self._projection_pools.clear()


class _MoonEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, route_weights, topk_experts, tokens_per_expert, state):
        with torch.profiler.record_function("moonep.dispatch.forward"):
            async_finish = _dispatch_async_finish(state.runtime)
            result = state.buffer.dispatch(
                hidden,
                route_weights,
                topk_experts,
                tokens_per_expert,
                async_finish=async_finish,
                zero_copy=False,
            )
        if async_finish:
            dispatched_hidden, dispatched_route_weights, cu_seqlens, plan, event = result
            state.dispatch_event = event
        else:
            dispatched_hidden, dispatched_route_weights, cu_seqlens, plan = result
        state.plan = plan
        state.cu_seqlens = cu_seqlens
        ctx.state = state
        return dispatched_hidden, dispatched_route_weights

    @staticmethod
    def backward(ctx, grad_dispatched_hidden, grad_dispatched_route_weights):
        state = ctx.state
        grad_dispatched_hidden = grad_dispatched_hidden.contiguous()
        grad_dispatched_route_weights = (
            grad_dispatched_route_weights.float().contiguous()
            if grad_dispatched_route_weights is not None
            else None
        )
        with torch.profiler.record_function("moonep.dispatch.backward"):
            grad_hidden, grad_route, event = state.buffer.combine(
                plan=state.plan,
                hidden_nvsh=grad_dispatched_hidden,
                route_weights_nvs=grad_dispatched_route_weights,
                async_finish=_dispatch_async_finish(state.runtime),
                zero_copy=False,
            )
        state.wait(event)
        return grad_hidden, grad_route, None, None, None


class _MoonEPWeightedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, dispatched_route_weights, state):
        weighted = expert_output * dispatched_route_weights.to(
            expert_output.dtype
        ).unsqueeze(-1)
        with torch.profiler.record_function("moonep.combine.forward"):
            output, _, event = state.buffer.combine(
                plan=state.plan,
                hidden_nvsh=weighted.contiguous(),
                async_finish=_dispatch_async_finish(state.runtime),
                zero_copy=False,
            )
        state.wait(event)
        ctx.state = state
        ctx.save_for_backward(expert_output, dispatched_route_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.state
        with torch.profiler.record_function("moonep.combine.backward"):
            async_finish = _dispatch_async_finish(state.runtime)
            result = state.buffer.dispatch(
                grad_output.contiguous(),
                plan=state.plan,
                async_finish=async_finish,
                zero_copy=False,
            )
        if async_finish:
            grad_weighted, _, _, _, event = result
            state.wait(event)
        else:
            grad_weighted, _, _, _ = result
        expert_output, dispatched_route_weights = ctx.saved_tensors
        active_rows = (
            torch.arange(grad_weighted.shape[0], device=grad_weighted.device)
            < state.cu_seqlens[-1]
        )
        grad_weighted = grad_weighted * active_rows.unsqueeze(-1).to(grad_weighted.dtype)
        grad_expert = grad_weighted * dispatched_route_weights.to(
            grad_weighted.dtype
        ).unsqueeze(-1)
        grad_route = (grad_weighted.float() * expert_output.float()).sum(dim=-1)
        return grad_expert, grad_route, None


def moonep_dispatch(
    state: MoonEPCallState,
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
):
    return _MoonEPDispatch.apply(
        hidden, route_weights, topk_experts, tokens_per_expert, state
    )


def moonep_weighted_combine(
    state: MoonEPCallState,
    expert_output: torch.Tensor,
    dispatched_route_weights: torch.Tensor,
) -> torch.Tensor:
    return _MoonEPWeightedCombine.apply(
        expert_output, dispatched_route_weights, state
    )


def moonep_bind_weight_grads(
    dispatched: torch.Tensor,
    gate_up_parameter: torch.Tensor,
    down_parameter: torch.Tensor,
    gate_up_projection: MoonEPSymmetricProjection,
    down_projection: MoonEPSymmetricProjection,
    plan,
    comm_context: dict,
) -> torch.Tensor:
    """Route both projections' weight gradients back to their parameters.

    Returns ``dispatched`` unchanged; the returned tensor must be the one fed
    to the expert grouped matmuls so that the reduction runs after them.
    """
    local_gate_up = (
        gate_up_parameter.to_local()
        if isinstance(gate_up_parameter, DTensor)
        else gate_up_parameter
    )
    local_down = (
        down_parameter.to_local()
        if isinstance(down_parameter, DTensor)
        else down_parameter
    )
    return _MoonEPWeightGradBridge.apply(
        dispatched,
        local_gate_up,
        local_down,
        gate_up_projection,
        down_projection,
        plan,
        comm_context,
    )
