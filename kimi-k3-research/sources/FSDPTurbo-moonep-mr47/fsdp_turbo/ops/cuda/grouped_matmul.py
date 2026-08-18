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


def _invoke_grouped_mm(fn, mat_a, mat_b, offs, out_dtype):
    """Call a grouped-mm implementation across PyTorch 2.9 / 2.10 signatures.

    ``F.grouped_mm`` (2.10+) takes keyword ``offs``. ``torch._grouped_mm``
    (2.9) takes ``offs`` as the third positional argument and ``out_dtype`` as
    a keyword, matching the CUTLASS grouped GEMM wrapper.
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

    PyTorch 2.9 (the H20 / Qwen3-30B training stack) exposes this as
    ``torch._grouped_mm`` and ``aten._grouped_mm``. ``F.grouped_mm`` only
    exists on newer builds (2.10+). Calling any of these with GPU ``int32``
    offsets is required during FSDP forward: a host sync on the default
    stream deadlocks the prefetch all-gather (NCCL timeout 600s).
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


def _zero_rows_past(tensor, exclusive_end):
    """In-place zero rows ``[exclusive_end, T)`` without a host scalar read.

    ``torch._grouped_mm`` allocates the full ``[M, N]`` output with
    ``empty_strided`` and only writes ``[:offs[-1]]``. Combine still sees the
    NvS-sized tensor, so the static tail must be zeros rather than allocator
    garbage. The mask is ``[M]``, not another ``[M, H]`` activation clone.
    """
    if tensor is None or tensor.numel() == 0:
        return tensor
    rows = torch.arange(
        tensor.shape[0], device=tensor.device, dtype=exclusive_end.dtype
    )
    tensor.mul_((rows < exclusive_end).unsqueeze(-1).to(dtype=tensor.dtype))
    return tensor


def _group_ends(m_split) -> list[int]:
    """Host-side exclusive ends for each group. One D2H sync per call.

    Only the explicit ``vmm_safe=True`` escape hatch uses this, and only
    because the per-expert ``torch.mm`` loop needs Python integers. Do not
    call it from the fused forward: that path runs under FSDP prefetch.
    """
    return torch.cumsum(m_split, dim=0).detach().cpu().tolist()


def _per_group_forward(inputs, weights, group_ends):
    """``out[g] = inputs[g] @ weights[g].T`` for non-empty groups.

    Each expert is a regular ``torch.mm`` against a 2-D view of ``weights``.
    That uses cuBLAS strided GEMM. This is the ``vmm_safe=True`` escape hatch
    only — the training path uses ``_grouped_mm`` so forward never D2Hs.
    """
    output = inputs.new_zeros(inputs.shape[0], weights.shape[1])
    start = 0
    for group_index, group_end in enumerate(group_ends):
        if group_end > start:
            torch.mm(
                inputs[start:group_end],
                weights[group_index].transpose(0, 1),
                out=output[start:group_end],
            )
        start = group_end
    return output


def _per_group_dgrad(grad_output, weights, group_ends):
    """``grad_input[g] = grad_output[g] @ weights[g]`` for non-empty groups."""
    grad_input = grad_output.new_zeros(grad_output.shape[0], weights.shape[2])
    start = 0
    for group_index, group_end in enumerate(group_ends):
        if group_end > start:
            torch.mm(
                grad_output[start:group_end],
                weights[group_index],
                out=grad_input[start:group_end],
            )
        start = group_end
    return grad_input


def _fused_weight_grads(sink, grad_output, inputs, group_ends) -> bool:
    """Compute each owned row range's weight gradient with one grouped GEMM.

    ``grouped_mm`` accepts ``out_dtype``, so the FP32 gradient comes straight
    out of the kernel instead of going through a BF16 staging tensor and a cast.
    Only the ranges this rank owns are computed, which keeps the temporary at
    ``[E/R, out, in]`` per range rather than the full ``[E+B, out, in]``.

    This path is safe only when ``grad_output`` / ``inputs`` live in ordinary
    CUDA memory. MoonEP's dispatch buffer is VMM-mapped, so the training
    backward uses :func:`write_group_grads` instead.
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
            partial = _grouped_mm(
                grad_output[token_start:token_end].transpose(0, 1),
                inputs[token_start:token_end],
                offsets,
                out_dtype=sink.buffer.dtype,
            )
            sink.buffer[row_start:row_end].copy_(partial)
    except (NotImplementedError, RuntimeError, TypeError, AttributeError):
        # Any range may already have been written; the per-group fallback
        # overwrites every owned row, so a partial result is harmless.
        _SUPPORTS_FUSED_WEIGHT_GRAD = False
        return False
    _SUPPORTS_FUSED_WEIGHT_GRAD = True
    return True


def write_weight_grads_cuda(sink, grad_output, inputs, group_ends, use_fused=True) -> None:
    assert_local_groups(sink, group_ends)
    if use_fused and _fused_weight_grads(sink, grad_output, inputs, group_ends):
        return
    write_group_grads(sink, grad_output, inputs, group_ends)


class GroupedMatmulCUDA(torch.autograd.Function):
    """CUDA fused grouped matmul via ``_grouped_mm``.

    Weights follow the linear-layer convention [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). The fused kernel expects the
    right operand in [num_groups, K, N] layout, so weights are transposed before
    the forward call.

    Forward offsets stay on device. A ``.cpu()`` here would sync the default
    stream while FSDP's prefetch all-gather is in flight on another stream,
    and ranks then split across ``mesh_fsdp`` ALLGATHER vs ``mesh_ep`` ALLREDUCE.
    """

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split, sink=None):
        # offs[i] marks the end of group i; the kernel requires int32 offsets.
        offs = torch.cumsum(m_split, dim=0).to(dtype=torch.int32)

        # out = input @ weight.T  ->  mat_b = [num_groups, K, N]
        mat_b = weights.transpose(-2, -1)
        output = _grouped_mm(input_tensor, mat_b, offs)
        # grouped_mm leaves rows past offs[-1] uninitialized. Combine may
        # still see the full NvS tensor, so zero that tail in place.
        _zero_rows_past(output, offs[-1])

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
            _zero_rows_past(grad_input, offs[-1])

        if sink is not None:
            # OPT-2: write local rows into MoonEP's [E+B] buffer. Per-group
            # ``torch.mm`` (not fused grouped_mm) because activations live in
            # the VMM dispatch buffer. Host group ends are OK in backward.
            write_weight_grads_cuda(
                sink,
                grad_output,
                input_tensor,
                offs.detach().cpu().tolist(),
                use_fused=False,
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


class GroupedMatmulVmmSafeCUDA(torch.autograd.Function):
    """Per-expert ``torch.mm`` escape hatch. Not used by MoonEP training.

    Forward D2Hs group ends, which deadlocks FSDP prefetch. Keep this behind
    an explicit ``vmm_safe=True`` for debugging; the training dispatcher must
    not set that flag.
    """

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split, sink=None):
        group_ends = _group_ends(m_split)
        output = _per_group_forward(input_tensor, weights, group_ends)
        ctx.save_for_backward(input_tensor, weights)
        ctx.group_ends = group_ends
        ctx.sink = sink
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        group_ends = ctx.group_ends
        sink = ctx.sink

        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = _per_group_dgrad(grad_output, weights, group_ends)

        if sink is not None:
            write_weight_grads_cuda(
                sink, grad_output, input_tensor, group_ends, use_fused=False
            )
        elif ctx.needs_input_grad[1]:
            grad_weight = torch.empty_like(weights)
            start = 0
            for group_index, group_end in enumerate(group_ends):
                if group_end == start:
                    grad_weight[group_index].zero_()
                else:
                    torch.mm(
                        grad_output[start:group_end].transpose(0, 1),
                        input_tensor[start:group_end],
                        out=grad_weight[group_index],
                    )
                start = group_end

        return grad_input, grad_weight, None, None


@register_op('grouped_matmul', 'cuda')
def grouped_matmul_cuda(inputs, m_split, weights, grad_weight_sink=None, vmm_safe=False):
    if grad_weight_sink is not None:
        validate_sink(grad_weight_sink, weights)
    # ``vmm_safe`` is an explicit escape hatch. A sink alone must not force
    # it: MoonEP always passes a sink, and the per-expert path D2Hs in
    # forward, which deadlocks FSDP all-gather.
    if vmm_safe:
        return GroupedMatmulVmmSafeCUDA.apply(
            inputs, weights, m_split, grad_weight_sink
        )
    return GroupedMatmulCUDA.apply(inputs, weights, m_split, grad_weight_sink)
