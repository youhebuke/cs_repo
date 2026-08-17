import torch
from fsdp_turbo.ops.grad_weight_sink import validate_sink
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


# Whether ``npu_grouped_matmul`` accepts ``output_dtype`` for BF16/FP16 inputs.
# Probed once on the first weight-gradient call and cached, because the answer
# depends on the installed CANN and torch_npu versions.
_SUPPORTS_OUTPUT_DTYPE = None



def _weight_grad(inputs, grad_output, group_list, group_list_type, output_dtype):
    """Compute the per-group weight gradient as a ``[groups, in, out]`` tensor.

    ``output_dtype`` requests the accumulation dtype directly from the kernel so
    that an FP32 gradient buffer does not need a separate cast afterwards.
    """
    global _SUPPORTS_OUTPUT_DTYPE

    def launch(**extra):
        return torch_npu.npu_grouped_matmul(
            [inputs.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
            **extra,
        )[0]

    if output_dtype is None or output_dtype == grad_output.dtype:
        return launch()
    if _SUPPORTS_OUTPUT_DTYPE is False:
        return launch()
    try:
        result = launch(output_dtype=output_dtype)
    except (RuntimeError, TypeError):
        _SUPPORTS_OUTPUT_DTYPE = False
        return launch()
    _SUPPORTS_OUTPUT_DTYPE = True
    return result


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type,
                sink=None) -> torch.Tensor:
        # Due to ascend gmm kernel k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type
        ctx.sink = sink

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
        sink = ctx.sink

        # Get weight chunks (original [output_dim, input_dim] format)
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Calculate input gradient: grad_output [M, N] @ weight [N, K] = grad_input [M, K].
        # Forward used transposed weights, so the original weights are used directly here.
        grad = torch_npu.npu_grouped_matmul([grad_output], weight_chunks, bias=None,
                                            group_list=group_list, split_item=2, group_type=0,
                                            group_list_type=group_list_type)[0]

        # Calculate weight gradient (K split gmm): grad_weight = inp^T @ grad_output = [K, N].
        # ``transpose`` restores the [output_dim, input_dim] convention as a view; stacking
        # transposed chunks instead would copy the whole gradient a second time.
        grad_weight = _weight_grad(
            inp,
            grad_output,
            group_list,
            group_list_type,
            sink.buffer.dtype if sink is not None else None,
        ).transpose(1, 2)

        if sink is None:
            return grad, grad_weight, None, None, None, None

        # Only this rank's own rows are copied into the symmetric buffer; the
        # remaining rows are remote ranks' gradients that must stay untouched.
        for start, end in sink.row_ranges:
            sink.buffer[start:end].copy_(grad_weight[start:end])
        return grad, None, None, None, None, None


@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights, grad_weight_sink=None):
    if grad_weight_sink is not None:
        validate_sink(grad_weight_sink, weights)
    return GroupedMatmul.apply(inputs, weights, None, m_split, 1, grad_weight_sink)
