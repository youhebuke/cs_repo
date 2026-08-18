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
    """Process-local prefetch slots and rank-mapped FP32 gradient slots.

    NPU uses one contiguous ``[E+B, out, in]`` FP32 mapping: rows ``[0, E)``
    are every rank's parameter gradients and rows ``[E, E+B)`` alias this
    rank's reduce slice. CUDA keeps the original MR#47 owner/reduce VMMs
    (no extra ``[E+B]`` tensor) so the wrap matches the tree that already
    trains on H20.
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

        if getattr(runtime, "accelerator_type", None) == "cuda":
            self._init_cuda_grad_buffers(chunk_shape, experts_per_rank, out_features, in_features)
        else:
            self._init_aliased_grad_buffers(
                chunk_shape, experts_per_rank, out_features, in_features, ranks
            )
        self.owner_grad_buffers[runtime.rank].zero_()
        self.reduce_buffers[runtime.rank].zero_()
        if self.main_grad is not None:
            self.main_grad[runtime.num_experts:].zero_()
        dist.barrier(group=runtime.group)

    def _init_aliased_grad_buffers(
        self,
        chunk_shape: list[int],
        experts_per_rank: int,
        out_features: int,
        in_features: int,
        ranks: list[int],
    ) -> None:
        """NPU: one ``[E+B]`` VMM whose tail aliases this rank's reduce pages."""
        runtime = self.runtime
        imports = runtime.imports
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

    def _init_cuda_grad_buffers(
        self,
        chunk_shape: list[int],
        experts_per_rank: int,
        out_features: int,
        in_features: int,
    ) -> None:
        """CUDA keeps original MR#47 ``create_nvl_dist_tensor`` owner/reduce maps.

        The ``[E+B]`` alias and a regular ``torch.zeros([E+B])`` sink both
        change wrap-time allocations versus the tree that already trains on
        H20 with ``MOONEP_ASYNC_FINISH=0``. GPU is only a functional check.
        """
        runtime = self.runtime
        imports = runtime.imports
        self.main_grad = None
        self.reduce_allocation = None
        self.reduce_export_handle = None
        self.owner_grad_allocation = None
        self.reduce_full = imports.create_nvl_dist_tensor(
            chunk_shape,
            torch.float32,
            runtime.rank,
            runtime.size,
            group=runtime.group,
        )
        self.reduce_buffers = self.reduce_full.view(
            runtime.size, experts_per_rank, out_features, in_features
        )
        self.owner_grad_full = imports.create_nvl_dist_tensor(
            chunk_shape,
            torch.float32,
            runtime.rank,
            runtime.size,
            group=runtime.group,
        )
        self.owner_grad_buffers = self.owner_grad_full.view(
            runtime.size, experts_per_rank, out_features, in_features
        )

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
        self.runtime.imports.launch_prefetch(
            self.full_weight[:self.runtime.num_experts],
            self.full_weight[self.runtime.num_experts:],
            experts,
            num_sms=self.runtime.config.num_sms,
        )

    def grad_weight_sink(self) -> GradWeightSink:
        """Expose the NPU ``[E+B]`` FP32 buffer the grouped matmul writes into.

        CUDA uses the original MR#47 weight bridge and has no sink buffer.
        """
        if self.pool.main_grad is None:
            raise RuntimeError(
                "MoonEP CUDA wrap uses the original weight-gradient bridge; "
                "there is no [E+B] sink buffer."
            )
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

    def reduce_gradient(self, first, second, third):
        """Reduce local expert gradients across ranks.

        CUDA (original MR#47)::
            reduce_gradient(full_grad, plan, comm_context)
        NPU (``[E+B]`` sink)::
            reduce_gradient(plan, comm_context, dtype)
        """
        if isinstance(first, torch.Tensor):
            return self._reduce_gradient_from_full_grad(first, second, third)
        return self._reduce_gradient_from_sink(first, second, third)

    def _reduce_gradient_from_full_grad(
        self, full_grad: torch.Tensor, plan, comm_context: dict
    ) -> torch.Tensor:
        num_experts = self.runtime.num_experts
        rank = self.runtime.rank
        experts_per_rank = self.runtime.experts_per_rank
        start = rank * experts_per_rank
        end = start + experts_per_rank

        self.pool.owner_grad_buffers[rank].copy_(full_grad[start:end])
        self.pool.reduce_buffers[rank].copy_(
            full_grad[num_experts:num_experts + experts_per_rank]
        )
        self._launch_grad_reduce(plan, comm_context)
        return self.pool.owner_grad_buffers[rank].to(dtype=full_grad.dtype)

    def _reduce_gradient_from_sink(
        self, plan, comm_context: dict, dtype: torch.dtype
    ) -> torch.Tensor:
        rank = self.runtime.rank
        experts_per_rank = self.runtime.experts_per_rank
        start = rank * experts_per_rank
        end = start + experts_per_rank
        self._launch_grad_reduce(plan, comm_context)
        return self.pool.main_grad[start:end].to(dtype=dtype)

    def _launch_grad_reduce(self, plan, comm_context: dict) -> None:
        # ``launch_grad_reduce`` reads every rank's reduce slots before its
        # internal cross-rank barrier. Publish the local slot writes and wait
        # until every peer has done the same before any rank starts reading.
        self.runtime.imports.launch_inter_rank_sync(comm_context)
        self.runtime.imports.launch_grad_reduce(
            self.pool.owner_grad_full,
            self.pool.reduce_buffers,
            plan.experts_to_copy,
            rank=self.runtime.rank,
            num_sms=self.runtime.config.num_sms,
            meta_buf=comm_context["meta_buf"],
            meta_stride=int(comm_context["meta_chunk_padded"]),
            barrier_off=int(comm_context["BARRIER_OFF"]),
            grid_sync_bar=comm_context["grid_sync_bar"],
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._grad_sink = None
        self.full_weight = None
        self.expert_allocation = None


class _MoonEPWeightsBridge(torch.autograd.Function):
    """Connect both registered parameters to their full VMM mappings.

    CUDA uses this original MR#47 bridge. A single multi-output autograd node
    receives both projection gradients and therefore launches the cross-rank
    reductions in the same order on every rank.
    """

    @staticmethod
    def forward(
        ctx,
        local_gate_up,
        local_down,
        full_gate_up,
        full_down,
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
        return (
            full_gate_up.as_strided(full_gate_up.shape, full_gate_up.stride()),
            full_down.as_strided(full_down.shape, full_down.stride()),
        )

    @staticmethod
    def backward(ctx, grad_full_gate_up, grad_full_down):
        if grad_full_gate_up is None or grad_full_down is None:
            raise RuntimeError(
                "MoonEP requires gradients for both gate_up_proj and down_proj."
            )
        with torch.profiler.record_function("moonep.grad_reduce"):
            local_down = ctx.down_projection.reduce_gradient(
                grad_full_down, ctx.plan, ctx.comm_context
            )
            local_gate_up = ctx.gate_up_projection.reduce_gradient(
                grad_full_gate_up, ctx.plan, ctx.comm_context
            )
        return (
            local_gate_up.to(ctx.gate_up_dtype),
            local_down.to(ctx.down_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
        if event is not None:
            event.wait(torch.accelerator.current_stream())

    def prefetch(self, *projections: MoonEPSymmetricProjection) -> None:
        if self.plan is None:
            raise RuntimeError("MoonEP dispatch plan is not initialized.")
        with torch.profiler.record_function("moonep.prefetch"):
            if self.runtime.config.async_finish:
                comm_stream = self.buffer._comm_stream
                if comm_stream is None:
                    raise RuntimeError("MoonEP communication stream is unavailable.")
                # The asynchronous dispatch is already queued on this stream;
                # enqueue both projection prefetches behind it and make the
                # compute stream wait for the combined event.
                with torch.accelerator.stream(comm_stream):
                    for projection in projections:
                        projection.prefetch(self.plan)
                    done = comm_stream.record_event()
                done.wait(torch.accelerator.current_stream())
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
        self._first_dispatch_synced = False
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

    def _finish_overlapping_fsdp_collectives(self) -> None:
        """Wait for in-flight FSDP allgathers before MoonEP NCCL or CuTe JIT.

        ``test_moonep.py`` sets ``num_to_forward_prefetch=1``. That launches the
        next layer's FSDP allgather on a side stream at the start of the current
        layer. Creating a Buffer (EP ``dist.barrier`` / 1-element allreduce) or
        JIT-compiling the first dispatch kernel holds the GIL while that
        allgather still needs it, so the watchdog reports a 600s
        ``_ALLGATHER_BASE`` timeout on ``mesh_fsdp`` and a 1-element allreduce
        timeout on ``mesh_ep``. This wait is a functional guard, not a memory
        optimization.
        """
        torch.accelerator.synchronize()

    def _make_buffer(self, tokens_per_rank: int, hidden_dim: int, top_k: int):
        self._finish_overlapping_fsdp_collectives()
        return self.imports.Buffer(
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

    def ensure_comm_buffer(self, hidden_dim: int) -> None:
        """Allocate the MoonEP Buffer during wrap, before FSDP forward prefetch."""
        if self._tokens_per_rank is None or self._top_k is None:
            return
        key = (self._tokens_per_rank, hidden_dim, self._top_k)
        if key not in self._buffers:
            self._buffers[key] = self._make_buffer(
                self._tokens_per_rank, hidden_dim, self._top_k
            )

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
            buffer = self._make_buffer(tokens_per_rank, hidden_dim, top_k)
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
        config = state.runtime.config
        if not state.runtime._first_dispatch_synced:
            state.runtime._finish_overlapping_fsdp_collectives()
            state.runtime._first_dispatch_synced = True
        with torch.profiler.record_function("moonep.dispatch.forward"):
            result = state.buffer.dispatch(
                hidden,
                route_weights,
                topk_experts,
                tokens_per_expert,
                async_finish=config.async_finish,
                zero_copy=False,
            )
        if config.async_finish:
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
        config = state.runtime.config
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
                async_finish=config.async_finish,
                zero_copy=False,
            )
        state.wait(event)
        return grad_hidden, grad_route, None, None, None


class _MoonEPWeightedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, dispatched_route_weights, state):
        config = state.runtime.config
        weighted = expert_output * dispatched_route_weights.to(
            expert_output.dtype
        ).unsqueeze(-1)
        with torch.profiler.record_function("moonep.combine.forward"):
            output, _, event = state.buffer.combine(
                plan=state.plan,
                hidden_nvsh=weighted.contiguous(),
                async_finish=config.async_finish,
                zero_copy=False,
            )
        state.wait(event)
        ctx.state = state
        ctx.save_for_backward(expert_output, dispatched_route_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state = ctx.state
        config = state.runtime.config
        with torch.profiler.record_function("moonep.combine.backward"):
            result = state.buffer.dispatch(
                grad_output.contiguous(),
                plan=state.plan,
                async_finish=config.async_finish,
                zero_copy=False,
            )
        if config.async_finish:
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


def moonep_weights_for_step(
    gate_up_parameter: torch.Tensor,
    down_parameter: torch.Tensor,
    gate_up_projection: MoonEPSymmetricProjection,
    down_projection: MoonEPSymmetricProjection,
    plan,
    comm_context: dict,
):
    """CUDA original MR#47: expose full VMM weights inside autograd."""
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
    return _MoonEPWeightsBridge.apply(
        local_gate_up,
        local_down,
        gate_up_projection.full_weight,
        down_projection.full_weight,
        gate_up_projection,
        down_projection,
        plan,
        comm_context,
    )
