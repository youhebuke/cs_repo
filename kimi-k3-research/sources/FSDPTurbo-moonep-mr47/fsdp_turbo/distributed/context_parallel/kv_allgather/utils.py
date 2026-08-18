# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_tensor_autograd


def kv_allgather(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        gather_dim: int = 2,
) -> torch.Tensor:
    """All-gather a tensor along ``gather_dim`` across ``process_group``.

    Each rank contributes its local shard and receives the concatenation of
    all shards.  Gradients are propagated across workers via
    ``all_gather_tensor_autograd``.

    Args:
        input_: Local tensor shard.
        process_group: Process group for the collective operation.
        gather_dim: Dimension along which to concatenate gathered shards.

    Returns:
        Tensor with all shards concatenated along ``gather_dim``.
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    return all_gather_tensor_autograd(input_.contiguous(), gather_dim=gather_dim, group=process_group)


def kv_allgather_split_batch_data(batch, group, split_dim: int = 1):
    """Recursively traverse batch data and split tensors in-place along ``split_dim``.

    Used to shard the input batch across CP ranks before the forward pass.
    Supports common batch structures: dicts, lists, tuples, and plain tensors.

    Args:
        batch: Input batch data (dict, list, tuple, or tensor).
        group: CP process group.
        split_dim: Dimension along which to split tensors (default: 1, the
                   sequence dimension for typical ``[batch, seq_len, ...]`` data).

    Returns:
        The batch with all tensors split along ``split_dim``.
    """
    if isinstance(batch, torch.Tensor):
        return kv_allgather_split_tensor(batch, group, split_dim)

    if isinstance(batch, dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = kv_allgather_split_tensor(batch[key], group, split_dim)
            else:
                batch[key] = kv_allgather_split_batch_data(batch[key], group, split_dim)
        return batch

    if isinstance(batch, (list, tuple)):
        split_items = [kv_allgather_split_batch_data(item, group, split_dim) for item in batch]
        return type(batch)(split_items)

    return batch


def kv_allgather_split_tensor(tensor, group, split_dim: int = 1):
    """Split a tensor along ``split_dim`` and return the local rank's chunk."""
    if group is None or dist.get_world_size(group) == 1:
        return tensor

    cp_size = dist.get_world_size(group)
    cp_rank = dist.get_rank(group)

    chunk_size = tensor.size(split_dim) // cp_size
    start = cp_rank * chunk_size
    return tensor.narrow(split_dim, start, chunk_size).contiguous()
