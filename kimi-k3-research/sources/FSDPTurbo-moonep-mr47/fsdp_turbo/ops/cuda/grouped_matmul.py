# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from fsdp_turbo.ops.grad_weight_sink import validate_sink, write_group_grads
from fsdp_turbo.ops.registry import register_op


def _grouped_mm(mat_a, mat_b, offs):
    """Grouped GEMM. PT 2.9 has ``torch._grouped_mm``, not ``F.grouped_mm``.

    Offsets stay on device. A ``.cpu()`` here deadlocks FSDP prefetch.
    """
    offs = offs.to(device=mat_a.device, dtype=torch.int32)
    if offs.stride(0) != 1:
        offs = offs.contiguous()

    functional = getattr(F, "grouped_mm", None)
    if functional is not None:
        try:
            return functional(mat_a, mat_b, offs=offs)
        except TypeError:
            return functional(mat_a, mat_b, offs)

    torch_op = getattr(torch, "_grouped_mm", None)
    if torch_op is not None:
        if not mat_b.is_contiguous():
            mat_b = mat_b.contiguous()
        try:
            return torch_op(mat_a, mat_b, offs=offs)
        except TypeError:
            return torch_op(mat_a, mat_b, offs)

    aten = getattr(torch.ops, "aten", None)
    aten_op = getattr(aten, "_grouped_mm", None) if aten is not None else None
    if aten_op is not None:
        if not mat_b.is_contiguous():
            mat_b = mat_b.contiguous()
        try:
            return aten_op(mat_a, mat_b, offs=offs)
        except TypeError:
            return aten_op(mat_a, mat_b, offs)

    raise AttributeError(
        "No grouped GEMM on this PyTorch build: F.grouped_mm, "
        "torch._grouped_mm and torch.ops.aten._grouped_mm are all missing"
    )


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
        output = _grouped_mm(input_tensor, mat_b, offs)

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
            grad_input = _grouped_mm(grad_output, weights, offs)

        if sink is not None:
            # The weight gradient lands straight in MoonEP's [E+B] buffer, so no
            # full-size tensor is allocated and no rank writes a remote row.
            write_group_grads(
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
