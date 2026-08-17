# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


@register_op('grouped_matmul', 'cpu')
def grouped_matmul_cpu(inputs, m_split, weights):
    """
    Grouped matrix multiplication.

    Weights follow the linear-layer convention: [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). They are always transposed to
    [num_groups, input_dim, output_dim] before the matmul. This avoids the
    unreliable shape-based detection that breaks when input_dim == output_dim.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of shape [num_groups, output_dim, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    batch_size, input_dim = inputs.shape

    # Always transpose: [num_groups, output_dim, input_dim] -> [num_groups, input_dim, output_dim]
    output_dim = weights.shape[1]
    weights = weights.transpose(1, 2)

    # Initialize output tensor
    output_shape = (batch_size, output_dim)
    final_hidden_states = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)

    # Calculate group boundaries from cumulative sum
    group_list = [0] + torch.cumsum(m_split, dim=0).tolist()

    # Process each group separately
    # inputs[start_idx:end_idx, :] has shape [group_size, input_dim]
    # weights[i] has shape [input_dim, output_dim] (after transpose)
    for i in range(len(group_list) - 1):
        start_idx = group_list[i]
        end_idx = group_list[i + 1]

        final_hidden_states[start_idx:end_idx, :] = torch.matmul(
            inputs[start_idx:end_idx, :],
            weights[i]
        )

    return final_hidden_states
