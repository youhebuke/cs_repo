# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
from typing import Optional

import torch

from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather


def get_kv_allgather_attention_fn(fn, group: Optional[torch.distributed.ProcessGroup] = None):
    """Decorator: wraps an attention function with KV all-gather on the sequence dim.

    Before the wrapped function: all-gather ``key`` and ``value`` along the
    sequence dimension (dim=2) so every rank holds the full KV history.
    ``query`` remains local — only the local sequence chunk is attended to.
    After the wrapped function: no operation is needed because the output
    already corresponds to the local query chunk.

    Args:
        fn: Attention function to wrap.
        group: CP process group. If None, all-gather is skipped.

    Usage::

        decorated = get_kv_allgather_attention_fn(my_attention, group=cp_group)
    """

    @wraps(fn)
    def wrapper(module, query, key, value, *args, **kwargs):
        cp_size = torch.distributed.get_world_size(group) if group is not None else 1

        # All-gather KV along the sequence dimension (dim=2)
        if cp_size > 1:
            key = kv_allgather(key, group, gather_dim=2)
            value = kv_allgather(value, group, gather_dim=2)

        # Call the original attention function with local Q and full KV
        attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)

        return attn_output, attn_weights

    return wrapper


def get_kv_allgather_compressor_attention_fn(
    fn,
    group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Decorator: wraps a compressor attention function with KV all-gather.

    Specialised for DeepSeekV4-style attention where:
    * The KV tensor is a concatenation of ``[sliding_kv, compressed_kv]`` along
      dim=2.  Both portions are all-gathered independently because they have
      different sequence lengths (compressed_kv is the long-range compressor
      output).
    * When key and value are the **same** tensor (shared KV, detected via
      ``id(key) == id(value)``), value's all-gather is skipped and the
      gathered key is reused, saving one all-gather.

    Before the wrapped function: all-gather both portions of key (and value if
    distinct) along the sequence dimension (dim=2).
    After the wrapped function: no operation is needed — the output corresponds
    to the local query chunk.

    Args:
        fn: Attention function to wrap (e.g. eager_attention_forward).
        group: CP process group. If None, all-gather is skipped.
    """

    @wraps(fn)
    def wrapper(module, query, key, value, *args, **kwargs):
        cp_size = torch.distributed.get_world_size(group) if group is not None else 1
        shared_kv = id(key) == id(value)

        if cp_size > 1:
            # Split KV into sliding and compressed portions.
            # sliding_kv has the same seq_len as query; compressed_kv is the rest.
            sliding_len = query.shape[2]
            compressed_len = key.shape[2] - sliding_len

            if compressed_len > 0:
                sliding_key, compressed_key = key.split([sliding_len, compressed_len], dim=2)

                # All-gather both portions along the sequence dimension
                sliding_key = kv_allgather(sliding_key, group, gather_dim=2)
                compressed_key = kv_allgather(compressed_key, group, gather_dim=2)

                # Recombine key: [gathered_sliding_key | gathered_compressed_key]
                key = torch.cat([sliding_key, compressed_key], dim=2)

                if shared_kv:
                    value = key
                else:
                    sliding_value, compressed_value = value.split([sliding_len, compressed_len], dim=2)
                    sliding_value = kv_allgather(sliding_value, group, gather_dim=2)
                    compressed_value = kv_allgather(compressed_value, group, gather_dim=2)
                    value = torch.cat([sliding_value, compressed_value], dim=2)
            else:
                # No compressed portion — simple all-gather
                key = kv_allgather(key, group, gather_dim=2)
                if shared_kv:
                    value = key
                else:
                    value = kv_allgather(value, group, gather_dim=2)

        # Call the original attention function with local Q and full KV
        attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)

        return attn_output, attn_weights

    return wrapper
