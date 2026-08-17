# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import torch
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

try:
    from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd
    from transformer_engine.pytorch.constants import TE_DType

    _HAS_TRANSFORMER_ENGINE = True
except ImportError:
    _HAS_TRANSFORMER_ENGINE = False


class RMSNormCUDA(torch.autograd.Function):
    """
    CUDA fused RMSNorm backed by the TransformerEngine fused rmsnorm kernels.

    The forward pass uses ``rmsnorm_fwd`` which returns the normalized output
    together with an inverse standard deviation used by the backward pass.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        input_dtype = x.dtype
        if input_dtype not in TE_DType:
            raise NotImplementedError(f"RMSNorm CUDA kernel does not support dtype '{input_dtype}'")

        inner_dim = weight.numel()
        x_2d = x.contiguous().view(-1, inner_dim)
        w_1d = weight.contiguous().view(inner_dim)

        y, _, rstd = rmsnorm_fwd(x_2d, w_1d, eps, None, None, TE_DType[input_dtype], 0, False)

        ctx.save_for_backward(x_2d, rstd, w_1d)
        ctx.input_shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, rstd, w_1d = ctx.saved_tensors
        dy = grad_output.contiguous().view(x_2d.shape)
        dx, dw = rmsnorm_bwd(dy, x_2d, rstd, w_1d, 0, False)
        return dx.view(ctx.input_shape), dw.view(w_1d.shape), None


@register_op('rms_norm', 'cuda')
def rms_norm_cuda(x, weight, eps):
    """
    CUDA implementation of RMSNorm backed by the TransformerEngine fused
    ``rmsnorm_fwd``/``rmsnorm_bwd`` kernels.

    Falls back to the CPU implementation when TransformerEngine is not installed.

    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Epsilon for numerical stability

    Returns:
        Normalized output tensor
    """
    if not _HAS_TRANSFORMER_ENGINE:
        log_warning_once(
            logger, "TransformerEngine is not installed, falling back to CPU implementation for 'rms_norm'"
        )
        from fsdp_turbo.ops.cpu.rms_norm import rms_norm_cpu

        return rms_norm_cpu(x, weight, eps)

    return RMSNormCUDA.apply(x, weight, eps)
