import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type) -> torch.Tensor:
        # Due to ascend gmm kernel k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type

        # Get weight chunks
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Weights follow the linear-layer convention [output_dim, input_dim] per expert.
        # Transpose to [input_dim, output_dim] so that input [M, K] @ weight.T [K, N] = output [M, N].
        # Always transpose rather than guessing from shape, since shape-based detection is
        # unreliable when input_dim == output_dim (square weight matrices).
        weights_for_matmul = [w.T for w in weight_chunks]

        ctx.save_for_backward(input_tensor, weights)

        fwd_output = torch_npu.npu_grouped_matmul([input_tensor], weights_for_matmul, bias=weights_bias,
                                                  group_list=ctx.group_list, split_item=2, group_type=0,
                                                  group_list_type=ctx.group_list_type)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        group_list = ctx.group_list
        inp, weights = ctx.saved_tensors
        group_list_type = ctx.group_list_type

        # Get weight chunks (original [output_dim, input_dim] format)
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Calculate input gradient: grad_output [M, N] @ weight [N, K] = grad_input [M, K].
        # Forward used transposed weights, so the original weights are used directly here.
        grad = torch_npu.npu_grouped_matmul([grad_output], weight_chunks, bias=None,
                                            group_list=group_list, split_item=2, group_type=0,
                                            group_list_type=group_list_type)[0]

        # Calculate weight gradient (K split gmm): grad_weight = inp^T @ grad_output = [K, N]
        grad_weight = torch_npu.npu_grouped_matmul([inp.T], [grad_output], bias=None,
                                                   group_list=group_list, split_item=3,
                                                   group_type=2, group_list_type=group_list_type)[0]

        # Transpose to match original weight format [output_dim, input_dim]
        grad_weight_chunks = [w.T for w in grad_weight]

        return grad, torch.stack(grad_weight_chunks), None, None, None


@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights):
    return GroupedMatmul.apply(inputs, weights, None, m_split, 1)
