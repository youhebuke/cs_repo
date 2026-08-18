import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd


def ulysses_all_to_all(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
) -> torch.Tensor:
    """
    All-to-all: scatter one dimension and gather another across all processes.

    Each rank splits its ``scatter_dim`` into ``world_size`` chunks (one per
    peer) and receives ``world_size`` chunks concatenated along
    ``gather_dim``.

    Args:
        input_: Input tensor.
        process_group: Process group for the collective operation.
        scatter_dim: Dimension to scatter (split and send to other ranks).
        gather_dim: Dimension to gather (receive and concatenate from other ranks).

    Returns:
        Tensor after all-to-all operation.
    """
    world_size = dist.get_world_size(process_group)

    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if scatter_size % world_size != 0:
        raise ValueError(
            f"scatter_dim size ({scatter_size}) must be divisible by world_size ({world_size})."
        )

    ndim = input_.ndim

    # Build a permutation that puts scatter_dim at position 0 and gather_dim
    # at position 1, preserving the relative order of all other dims.
    perm = [0] * ndim
    perm[0] = scatter_dim
    perm[1] = gather_dim
    pos = 2
    for i in range(ndim):
        if i != scatter_dim and i != gather_dim:
            perm[pos] = i
            pos += 1

    # Inverse permutation (for permuting back after the all-to-all)
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    # 1. Permute: bring scatter and gather dims to positions 0 and 1
    x = input_.permute(*perm).contiguous()

    # 2. Merge dims 0 and 1 into a single dim for the 1-D all-to-all
    rest_shape = x.shape[2:]
    x = x.reshape(scatter_size * input_.size(gather_dim), *rest_shape)

    # 3. All-to-all along dim 0
    x = all_to_all_single_autograd(x, output_split_sizes=None, input_split_sizes=None, group=process_group)

    # 4. Un-merge: split dim 0 back into (scatter_per_rank, gather_size)
    scatter_per_rank = scatter_size // world_size
    gather_size = input_.size(gather_dim) * world_size
    x = x.reshape(scatter_per_rank, gather_size, *rest_shape)

    # 5. Permute back to the original dimension order
    output = x.permute(*inv_perm).contiguous()

    return output

def ulysses_split_batch_data(batch, group, split_dim: int = 1):
    """Recursively traverse batch data and split tensors in-place via ulysses_split_tensor.

    Supports common batch structures: dicts, lists, tuples, and plain tensors.
    Tensors found at any nesting depth are replaced with their split counterpart.

    Args:
        batch: Input batch data (dict, list, tuple, or tensor).
        group: Ulysses process group.
        split_dim: Dimension along which to split tensors (default: 1).

    Returns:
        The batch with all tensors split along split_dim.
    """
    if isinstance(batch, torch.Tensor):
        return ulysses_split_tensor(batch, group, split_dim)

    if isinstance(batch, dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = ulysses_split_tensor(batch[key], group, split_dim)
            else:
                batch[key] = ulysses_split_batch_data(batch[key], group, split_dim)
        return batch

    if isinstance(batch, (list, tuple)):
        split_items = [ulysses_split_batch_data(item, group, split_dim) for item in batch]
        return type(batch)(split_items)

    return batch

def ulysses_split_tensor(tensor, group, split_dim: int = 1):
    if group is None or dist.get_world_size(group) == 1:
        return tensor

    ulysses_size = dist.get_world_size(group)
    ulysses_rank = dist.get_rank(group)

    chunk_size = tensor.size(split_dim) // ulysses_size
    start = ulysses_rank * chunk_size
    return tensor.narrow(split_dim, start, chunk_size).contiguous()