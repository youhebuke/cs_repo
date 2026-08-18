# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Gradient norm computation and clipping utilities.

This module provides decoupled, FSDP-aware gradient norm operations:
- ``compute_grad_norm``: Compute the total gradient norm (L2 or L-inf) without clipping.
- ``clip_grad_by_norm``: Clip gradients in-place given a pre-computed total norm.
- ``clip_grad_norm``: Compute the gradient norm and clip in one call.

DTensor (``torch.distributed.tensor.DTensor``) gradients are handled
transparently: they are converted to local shards via ``.to_local()``
before computation, and the total norm is synchronized across the
data-parallel group via ``all_reduce``.
"""

from typing import Optional, Union

import torch
from torch import inf

try:
    from torch.distributed._tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard
    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


# ------------------------------------------------------------------
# DTensor helpers
# ------------------------------------------------------------------

def _to_local_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard if *tensor* is a DTensor, else return as-is."""
    if _HAS_DTENSOR and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _get_dp_group_if_dtensor(
    tensor: torch.Tensor,
    current_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Optional[torch.distributed.ProcessGroup]:
    """Extract the data-parallel group from a DTensor's device mesh.

    For regular tensors, returns *current_group* unchanged.

    For DTensors, inspects the tensor's ``placements`` to find which
    mesh dimensions are **sharded**.  The process group for the first
    sharded dimension is returned (this is the dimension along which
    gradient norms must be all-reduced).

    For 1D meshes this is equivalent to ``tensor.device_mesh.get_group()``.
    For multi-dimensional meshes (e.g. HSDP with ``(replicate, shard)``),
    this correctly picks the sharded dimension instead of returning the
    flattened group.
    """
    if not (_HAS_DTENSOR and isinstance(tensor, DTensor)):
        return current_group

    mesh = tensor.device_mesh

    # Find the first mesh dimension that is Shard in the placements.
    # This is the dimension along which the tensor is split across ranks,
    # and therefore the dimension that requires an all-reduce for norm sync.
    for dim_idx, placement in enumerate(tensor.placements):
        if isinstance(placement, Shard):
            group = mesh.get_group(dim_idx)
            if current_group is not None:
                assert current_group == group, (
                    f"Inconsistent data-parallel groups: {current_group} vs {group}"
                )
            return group

    # All placements are Replicate -- no sharding, no group needed.
    return current_group

def compute_grad_norm(
    parameters: Union[list, torch.Tensor],
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Compute the total p-norm of gradients across parameters.

    Gradients are cast to FP32 for the norm computation to avoid
    precision loss with BF16/FP16 gradients.  DTensor gradients are
    converted to their local shards before computation.

    When *group* is provided (or DTensor gradients are found), the
    partial norm is **all-reduced** across the group so that every
    rank sees the same total norm.

    Args:
        parameters: Model parameters whose ``.grad`` will be read.
        norm_type: Type of the p-norm. Use ``float('inf')`` for the
            infinity norm. Defaults to 2.0 (L2 norm).
        group: Process group for all-reducing the norm across
            data-parallel ranks.  If ``None`` and DTensor gradients
            are present, the group is inferred from the DTensors.

    Returns:
        The total gradient norm as a Python float.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Collect local gradient shards, inferring the DP group from DTensors.
    grads = []
    dp_group = group
    for p in parameters:
        if p.grad is not None:
            dp_group = _get_dp_group_if_dtensor(p.grad, group)
            grads.append(_to_local_if_dtensor(p.grad).detach().float())

    if not grads:
        return 0.0

    norm_type = float(norm_type)

    if norm_type == inf:
        # Compute local max, then all-reduce MAX across the group.
        local_max = max(g.abs().max() for g in grads)
        total_norm = torch.tensor([local_max], dtype=torch.float, device=grads[0].device)
        if dp_group is not None:
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        return total_norm.item()

    # General p-norm: sum of local ||grad||^p, then all-reduce SUM, then take ^(1/p).
    if norm_type == 2.0:
        local_sum = sum(g.norm().item() ** 2 for g in grads)
    else:
        local_sum = sum(g.norm(norm_type).item() ** norm_type for g in grads)

    total_norm = torch.tensor([local_sum], dtype=torch.float, device=grads[0].device)
    if dp_group is not None:
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    return total_norm.item() ** (1.0 / norm_type)


def clip_grad_by_norm(
    parameters: Union[list, torch.Tensor],
    max_norm: float,
    total_norm: float,
) -> None:
    """Clip gradients in-place so that their total norm <= ``max_norm``.

    This is a **no-op** when ``total_norm <= max_norm``.
    DTensor gradients are clipped on their local shards directly
    (the clip coefficient is the same on every rank after all-reduce).

    Args:
        parameters: Model parameters whose ``.grad`` will be scaled.
        max_norm: Maximum allowed total norm.
        total_norm: The current total gradient norm (as returned by
            :func:`compute_grad_norm`).
    """
    if total_norm == 0.0 or max_norm <= 0.0:
        return

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff >= 1.0:
        return

    for p in parameters:
        if p.grad is not None:
            _to_local_if_dtensor(p.grad).detach().mul_(clip_coeff)


def clip_grad_norm(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Compute gradient norm and clip in one call.

    For FSDP-wrapped models (that expose ``model.clip_grad_norm_``),
    this delegates to the FSDP-native implementation which is both
    correct (handles sharded parameters) and efficient (avoids
    redundant all-gathers).

    For regular models (including those with DTensor gradients),
    this falls back to :func:`compute_grad_norm` +
    :func:`clip_grad_by_norm`, using *group* for cross-rank
    norm synchronization.

    Args:
        model: The model whose parameters will be clipped.
        max_norm: Maximum allowed total gradient norm.
        norm_type: Type of the p-norm. Defaults to 2.0.
        group: Process group for all-reducing the gradient norm
            across data-parallel ranks.  If ``None`` and DTensor
            gradients are present, the group is inferred automatically.

    Returns:
        The total gradient norm **before** clipping (as a float),
        useful for logging / monitoring.
    """
    # FSDP models provide their own clip_grad_norm_ which correctly
    # handles sharded parameters and avoids unnecessary all-gathers.
    if hasattr(model, "clip_grad_norm_"):
        total_norm = model.clip_grad_norm_(max_norm, norm_type)
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)

    params = list(model.parameters())
    total_norm = compute_grad_norm(params, norm_type, group=group)
    clip_grad_by_norm(params, max_norm, total_norm)
    return total_norm
