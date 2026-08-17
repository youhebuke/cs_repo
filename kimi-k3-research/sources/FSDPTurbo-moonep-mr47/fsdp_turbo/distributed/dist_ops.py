# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch


def gather_along_first_dim_expert_parallel(input_, group, async_op=False):
    """Gather tensors and concatenate along the first dimension."""
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.accelerator.current_device_index())
    handle = torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group, async_op=async_op)

    return output, handle
