# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from fsdp_turbo.ops.grad_weight_sink import validate_sink, write_group_grads
from fsdp_turbo.ops.registry import register_op


def _invoke_grouped_mm(fn, mat_a, mat_b, offs, out_dtype):
    """Call grouped-mm across PyTorch 2.9 / 2.10 signatures.

    ``F.grouped_mm`` (2.10+) takes keyword ``offs``. ``torch._grouped_mm``
    (2.9, H20 training stack) takes ``offs`` as the third positional argument.
    """
    if out_dtype is None:
        attempts = (
            lambda: fn(mat_a, mat_b, offs=offs),
            lambda: fn(mat_a, mat_b, offs),
        )
    else:
        attempts = (
            lambda: fn(mat_a, mat_b, offs=offs, out_dtype=out_dtype),
            lambda: fn(mat_a, mat_b, offs, out_dtype=out_dtype),
            lambda: fn(mat_a, mat_b, offs, None, out_dtype),
        )
    last_type_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_type_error = exc
    raise last_type_error


def _grouped_mm(mat_a, mat_b, offs, out_dtype=None):
    """Grouped GEMM that stays on device: never ``.cpu()`` the offset tensor.

    PyTorch 2.9 has no ``F.grouped_mm``; it exposes ``torch._grouped_mm`` and
    ``aten._grouped_mm``. GPU ``int32`` offsets are required in FSDP forward:
    a host sync on the default stream deadlocks prefetch all-gather (NCCL
    timeout 600s). Do not replace this with a Python ``torch.mm`` loop.
    """
    offs = offs.to(device=mat_a.device, dtype=torch.int32)
    if offs.stride(0) != 1:
        offs = offs.contiguous()

    functional = getattr(F, "grouped_mm", None)
    if functional is not None:
        return _invoke_grouped_mm(functional, mat_a, mat_b, offs, out_dtype)

    torch_op = getattr(torch, "_grouped_mm", None)
    if torch_op is not None:
        return _invoke_grouped_mm(torch_op, mat_a, mat_b, offs, out_dtype)

    aten = getattr(torch.ops, "aten", None)
    aten_op = getattr(aten, "_grouped_mm", None) if aten is not None else None
    if aten_op is not None:
        return _invoke_grouped_mm(aten_op, mat_a, mat_b, offs, out_dtype)

    raise AttributeError(
        "No grouped GEMM on this PyTorch build: F.grouped_mm, "
        "torch._grouped_mm and torch.ops.aten._grouped_mm are all missing"
    )


def _exclusive_group_ends(m_split, row_count):
    """Device-side group ends whose last value is exactly ``row_count``.

    Do not ``.item()`` / ``.cpu()`` the offset tensor: that host sync deadlocks
    FSDP prefetch all-gather. MoonEP's static dispatch buffer is ``[NvS, H]``;
    ``torch._grouped_mm`` on PT 2.9 / H20 launches unspecified if
    ``offs[-1] < NvS``. Always write the last end on device.
    """
    offs = torch.cumsum(m_split, dim=0).to(dtype=torch.int32)
    if offs.numel() == 0:
        return offs
    offs = offs.clone()
    offs[-1] = row_count
    return offs


class GroupedMatmulCUDA(torch.autograd.Function):
    """CUDA fused grouped matmul via ``_grouped_mm``.

    Weights follow the linear-layer convention [num_groups, output_dim, input_dim].
    The fused kernel expects the right operand in [num_groups, K, N] layout.

    When ``sink`` is set (OPT-2), the weight gradient is written into MoonEP's
    ``[E+B]`` FP32 buffer with per-group ``torch.mm``; activations live in the
    VMM dispatch buffer so fused K-grouped wgrad is not used here.
    """

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split, sink=None):
        # offs[i] marks the end of group i. Last offset must equal M (NvS):
        # stopping early (offs[-1] < NvS) crashes PT 2.9 grouped_mm on H20.
        offs = _exclusive_group_ends(m_split, input_tensor.shape[0])

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
            # OPT-2: write local rows into MoonEP's [E+B] buffer. Host group
            # ends are OK in backward; FSDP prefetch is not in flight.
            write_group_grads(
                sink, grad_output, input_tensor, offs.detach().cpu().tolist()
            )
        elif ctx.needs_input_grad[1]:
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
