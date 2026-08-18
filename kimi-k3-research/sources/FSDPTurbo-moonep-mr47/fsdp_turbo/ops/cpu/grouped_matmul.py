# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.grad_weight_sink import validate_sink, write_group_grads
from fsdp_turbo.ops.registry import register_op


class GroupedMatmulSinkCPU(torch.autograd.Function):
    """Reference grouped matmul whose weight gradient lands in a caller buffer."""

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split, sink):
        offs = torch.cumsum(m_split, dim=0).to(torch.int64)
        # Rows past offs[-1] are MoonEP's static tail. Keep the full NvS
        # shape so combine/autograd stay aligned with the communication buffer.
        output = input_tensor.new_zeros(input_tensor.shape[0], weights.shape[1])
        group_start = 0
        for group_index, group_end in enumerate(offs.tolist()):
            if group_end > group_start:
                output[group_start:group_end] = (
                    input_tensor[group_start:group_end]
                    @ weights[group_index].transpose(0, 1)
                )
            group_start = group_end
        ctx.save_for_backward(input_tensor, weights, offs)
        ctx.sink = sink
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights, offs = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = input_tensor.new_zeros(input_tensor.shape)
            group_start = 0
            for group_index, group_end in enumerate(offs.tolist()):
                if group_end > group_start:
                    grad_input[group_start:group_end] = (
                        grad_output[group_start:group_end] @ weights[group_index]
                    )
                group_start = group_end

        write_group_grads(ctx.sink, grad_output, input_tensor, offs.tolist())
        return grad_input, None, None, None


@register_op('grouped_matmul', 'cpu')
def grouped_matmul_cpu(inputs, m_split, weights, grad_weight_sink=None, vmm_safe=False):
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
        grad_weight_sink: Optional pre-allocated weight-gradient destination.

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    if grad_weight_sink is not None:
        validate_sink(grad_weight_sink, weights)
        return GroupedMatmulSinkCPU.apply(inputs, weights, m_split, grad_weight_sink)

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
