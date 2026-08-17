# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import torch
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

try:
    from transformer_engine.pytorch.permutation import (
        moe_permute as te_moe_permute,
        moe_unpermute as te_moe_unpermute,
    )

    _HAS_TRANSFORMER_ENGINE = True
except ImportError:
    _HAS_TRANSFORMER_ENGINE = False


@register_op('permute', 'cuda')
def permute_cuda(tokens, indices):
    """
    CUDA implementation of token permutation for MoE (Mixture of Experts),
    backed by the TransformerEngine fused ``moe_permute`` kernel.

    Falls back to the CPU implementation when TransformerEngine is not installed.

    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]

    Returns:
        permuted_tokens: Permuted tokens
        row_id_map: Row ID map used for unpermutation
    """
    if not _HAS_TRANSFORMER_ENGINE:
        log_warning_once(logger, "TransformerEngine is not installed, falling back to CPU implementation for 'permute'")
        from fsdp_turbo.ops.cpu.permute import permute_cpu

        return permute_cpu(tokens, indices)

    # TransformerEngine expects routing indices in [num_tokens, topk] int32 layout.
    topk = 1 if indices.dim() == 1 else indices.size(1)
    routing_map = indices.reshape(-1, topk).to(torch.int32)
    num_out_tokens = tokens.size(0) * topk

    permuted_tokens, row_id_map = te_moe_permute(tokens, routing_map, num_out_tokens, map_type="index")
    return permuted_tokens, row_id_map


@register_op('unpermute', 'cuda')
def unpermute_cuda(permuted_tokens, sorted_indices, probs=None):
    """
    CUDA implementation of token unpermutation for MoE (Mixture of Experts),
    backed by the TransformerEngine fused ``moe_unpermute`` kernel.

    Falls back to the CPU implementation when TransformerEngine is not installed.

    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Row ID map produced by ``permute``
        probs: Optional probability weights [num_tokens, topk]

    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    if not _HAS_TRANSFORMER_ENGINE:
        log_warning_once(
            logger, "TransformerEngine is not installed, falling back to CPU implementation for 'unpermute'"
        )
        from fsdp_turbo.ops.cpu.permute import unpermute_cpu

        return unpermute_cpu(permuted_tokens, sorted_indices, probs)

    return te_moe_unpermute(permuted_tokens, sorted_indices, merging_probs=probs, map_type="index")
