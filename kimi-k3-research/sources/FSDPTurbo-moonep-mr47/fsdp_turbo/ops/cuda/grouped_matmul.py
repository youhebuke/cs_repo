# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from fsdp_turbo.ops.grad_weight_sink import (
    assert_local_groups,
    token_span,
    validate_sink,
    write_group_grads,
)
from fsdp_turbo.ops.registry import register_op


# Whether ``grouped_mm`` can produce the K-grouped weight gradient with an
# explicit ``out_dtype``. Probed once and cached, because the answer depends on
# the PyTorch build, the GPU architecture and the operand alignment.
_SUPPORTS_FUSED_WEIGHT_GRAD = None


def _fused_weight_grads(sink, grad_output, inputs, group_ends) -> bool:
    """Compute each owned row range's weight gradient with one grouped GEMM.

    ``grouped_mm`` accepts ``out_dtype``, so the FP32 gradient comes straight
    out of the kernel instead of going through a BF16 staging tensor and a cast.
    Only the ranges this rank owns are computed, which keeps the temporary at
    ``[E/R, out, in]`` per range rather than the full ``[E+B, out, in]``.
    """
    global _SUPPORTS_FUSED_WEIGHT_GRAD

    if _SUPPORTS_FUSED_WEIGHT_GRAD is False:
        return False
    try:
        for row_range in sink.row_ranges:
            token_start, token_end = token_span(row_range, group_ends)
            if token_end == token_start:
                continue
            row_start, row_end = row_range
            offsets = torch.tensor(
                [end - token_start for end in group_ends[row_start:row_end]],
                dtype=torch.int32,
                device=grad_output.device,
            )
            # mat_a [out, tokens] x mat_b [tokens, in] -> [groups, out, in]
            partial = F.grouped_mm(
                grad_output[token_start:token_end].transpose(0, 1),
                inputs[token_start:token_end],
                offs=offsets,
                out_dtype=sink.buffer.dtype,
            )
            sink.buffer[row_start:row_end].copy_(partial)
    except (NotImplementedError, RuntimeError, TypeError):
        # Any range may already have been written; the per-group fallback
        # overwrites every owned row, so a partial result is harmless.
        _SUPPORTS_FUSED_WEIGHT_GRAD = False
        return False
    _SUPPORTS_FUSED_WEIGHT_GRAD = True
    return True


def write_weight_grads_cuda(sink, grad_output, inputs, group_ends) -> None:
    assert_local_groups(sink, group_ends)
    if _fused_weight_grads(sink, grad_output, inputs, group_ends):
        return
    write_group_grads(sink, grad_output, inputs, group_ends)


class GroupedMatmulCUDA(torch.autograd.Function):
    """
    CUDA fused grouped matmul backed by torch.nn.functional.grouped_mm.

    Weights follow the linear-layer convention [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). The fused kernel expects the
    right operand in [num_groups, K, N] layout, so weights are transposed before
    the forward call.
    """

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split, sink=None):
        # offs[i] marks the end of group i; the kernel requires int32 offsets.
        offs = torch.cumsum(m_split, dim=0).to(torch.int32)

        # out = input @ weight.T  ->  mat_b = [num_groups, K, N]
        mat_b = weights.transpose(-2, -1)
        output = F.grouped_mm(input_tensor, mat_b, offs=offs)

        ctx.save_for_backward(input_tensor, weights, offs)
        ctx.sink = sink
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights, offs = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        sink = ctx.sink

        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ weight (per group).
            grad_input = F.grouped_mm(grad_output, weights, offs=offs)

        if sink is not None:
            # The weight gradient lands straight in MoonEP's [E+B] buffer, so no
            # full-size tensor is allocated and no rank writes a remote row.
            write_weight_grads_cuda(
                sink, grad_output, input_tensor, offs.detach().cpu().tolist()
            )
        elif ctx.needs_input_grad[1]:
            # Compute each weight gradient with a regular GEMM. This avoids the
            # grouped mat2-backward path for MoonEP's VMM-backed expert weights.
            grad_weight = torch.empty_like(weights)
            group_ends = offs.detach().cpu().tolist()
            group_start = 0
            for group_index, group_end in enumerate(group_ends):
                if group_end == group_start:
                    grad_weight[group_index].zero_()
                else:
                    torch.mm(
                        grad_output[group_start:group_end].transpose(0, 1),
                        input_tensor[group_start:group_end],
                        out=grad_weight[group_index],
                    )
                group_start = group_end

        return grad_input, grad_weight, None, None


@register_op('grouped_matmul', 'cuda')
def grouped_matmul_cuda(inputs, m_split, weights, grad_weight_sink=None):
    if grad_weight_sink is not None:
        validate_sink(grad_weight_sink, weights)
    return GroupedMatmulCUDA.apply(inputs, weights, m_split, grad_weight_sink)
