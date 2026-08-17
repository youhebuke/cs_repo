# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
from typing import Optional

import torch

from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all


def get_ulysses_attention_fn(fn, group: Optional[torch.distributed.ProcessGroup] = None):
    """Decorator: wraps an attention function with Ulysses all-to-all on heads/sequence.

    Before the wrapped function: scatter heads (dim=1), gather sequence (dim=2).
    After the wrapped function: scatter sequence (dim=2), gather heads (dim=1).

    Args:
        fn: Attention function to wrap.
        group: Ulysses process group. If None, all-to-all is skipped.

    Usage::

        decorated = get_ulysses_attention_fn(my_attention, group=ulysses_group)
    """

    @wraps(fn)
    def wrapper(module, query, key, value, *args, **kwargs):
        ulysses_size = torch.distributed.get_world_size(group) if group is not None else 1

        # GQA: repeat KV heads when num_groups > 1 or heads are sharded
        num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
        if (num_groups > 1) or (ulysses_size > module.config.num_key_value_heads):
            key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
            value = torch.repeat_interleave(value, dim=1, repeats=num_groups)

        # Scatter heads, gather sequence
        if ulysses_size > 1:
            query = ulysses_all_to_all(query, group, scatter_dim=1, gather_dim=2)
            key = ulysses_all_to_all(key, group, scatter_dim=1, gather_dim=2)
            value = ulysses_all_to_all(value, group, scatter_dim=1, gather_dim=2)

        # Call the original attention function
        attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)

        # Scatter sequence, gather heads
        if ulysses_size > 1:
            attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=2, gather_dim=1)

        return attn_output, attn_weights

    return wrapper

def get_ulysses_compressor_attention_fn(
    fn,
    group: Optional[torch.distributed.ProcessGroup] = None,
    attn_output_transposed: bool = False,
):
    """Decorator: wraps a compressor attention function with Ulysses all-to-all.

    Specialised for DeepSeekV4-style attention where:
    * The KV tensor is a concatenation of [sliding_kv, compressed_kv] along
      dim=2.  Both portions need all-to-all, but with different gather_sizes
      because compressed_kv has a different sequence length.
    * When key and value are the **same** tensor (shared KV, detected via
      ``id(key) == id(value)``), value's all-to-all is skipped and the
      gathered key is reused, saving one all-to-all.  When they differ,
      value goes through its own all-to-all independently.

    Before the wrapped function: scatter heads (dim=1), gather sequence (dim=2)
    on query, sliding_key, and compressed_key (and value if distinct).
    After the wrapped function: scatter sequence, gather heads on attn_output.
    The scatter/gather dims for attn_output depend on ``attn_output_transposed``:
    * False (default): attn_output is [B, H, S, D] → scatter_dim=2, gather_dim=1
    * True:  attn_output is [B, S, H, D] → scatter_dim=1, gather_dim=2

    Args:
        fn: Attention function to wrap (e.g. eager_attention_forward).
        group: Ulysses process group. If None, all-to-all is skipped.
        attn_output_transposed: Whether fn returns attn_output with S and H dims
            transposed (i.e. [B, S, H, D] instead of [B, H, S, D]).
    """

    @wraps(fn)
    def wrapper(module, query, key, value, *args, **kwargs):
        ulysses_size = torch.distributed.get_world_size(group) if group is not None else 1
        shared_kv = id(key) == id(value)

        # GQA: repeat KV heads when num_groups > 1 or heads are sharded
        num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
        if (num_groups > 1) or (ulysses_size > module.config.num_key_value_heads):
            key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
            value = key if shared_kv else torch.repeat_interleave(value, dim=1, repeats=num_groups)

        if ulysses_size > 1:
            # Split KV into sliding and compressed portions.
            # sliding_kv has the same seq_len as query; compressed_kv is the rest.
            sliding_len = query.shape[2]
            compressed_len = key.shape[2] - sliding_len
            sliding_key, compressed_key = key.split([sliding_len, compressed_len], dim=2)

            # Scatter heads, gather sequence on query and key (both portions)
            query = ulysses_all_to_all(query, group, scatter_dim=1, gather_dim=2)
            sliding_key = ulysses_all_to_all(sliding_key, group, scatter_dim=1, gather_dim=2)
            compressed_key = ulysses_all_to_all(compressed_key, group, scatter_dim=1, gather_dim=2)

            # Recombine key: [gathered_sliding_key | gathered_compressed_key]
            key = torch.cat([sliding_key, compressed_key], dim=2)

            if shared_kv:
                # value is the same tensor as key, reuse gathered key directly
                value = key
            else:
                # value is distinct — do its own all-to-all on both portions
                sliding_value, compressed_value = value.split([sliding_len, compressed_len], dim=2)
                sliding_value = ulysses_all_to_all(sliding_value, group, scatter_dim=1, gather_dim=2)
                compressed_value = ulysses_all_to_all(compressed_value, group, scatter_dim=1, gather_dim=2)
                value = torch.cat([sliding_value, compressed_value], dim=2)

        # Call the original attention function
        attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)

        # Scatter sequence, gather heads on attn_output
        if ulysses_size > 1:
            if attn_output_transposed:
                # attn_output is [B, S, H, D]: scatter S (dim=1), gather H (dim=2)
                attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=1, gather_dim=2)
            else:
                # attn_output is [B, H, S, D]: scatter S (dim=2), gather H (dim=1)
                attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=2, gather_dim=1)

        return attn_output, attn_weights

    return wrapper
