# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Cached per-group views of a grouped-matmul weight tensor.

Kernels that take the weights as one Python list per expert force the caller to
rebuild ``2 * num_groups`` views on every forward and backward. At MoonEP's
group counts that is thousands of host-side view constructions per layer per
step, right in front of the kernel launches. The weights themselves are
persistent, so the views are built once and reused.

Caching views keeps the weight tensor alive: a view holds its base through
``_base``, and PyTorch's garbage-collector traversal does not break that cycle.
The cache therefore has an explicit :func:`release_weight_views` that owners
call before dropping a weight tensor -- MoonEP needs it so that closing a
projection actually unmaps its symmetric memory -- plus a bounded FIFO so a
forgotten release cannot grow without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import torch

# Enough for every projection of a large MoE model; entries are one tuple of
# view lists each, and views carry no data of their own.
_MAX_ENTRIES = 256

_CACHE: "OrderedDict[tuple, Tuple[List[torch.Tensor], List[torch.Tensor]]]" = (
    OrderedDict()
)


def _key(weights: torch.Tensor) -> tuple:
    return (
        weights.data_ptr(),
        tuple(weights.shape),
        tuple(weights.stride()),
        weights.dtype,
        str(weights.device),
    )


def grouped_weight_views(
    weights: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return per-group ``[out, in]`` views and their ``[in, out]`` transposes."""
    key = _key(weights)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]
    cached = (chunks, [w.T for w in chunks])
    _CACHE[key] = cached
    while len(_CACHE) > _MAX_ENTRIES:
        # Evicting while the weights are still alive is safe: the next call
        # rebuilds the views. Evicting only after they are freed would risk
        # handing out views into memory that another tensor has since reused.
        _CACHE.popitem(last=False)
    return cached


def release_weight_views(weights: torch.Tensor | None) -> None:
    """Drop the cached views so the weight tensor's storage can be freed."""
    if weights is not None:
        _CACHE.pop(_key(weights), None)


def cached_entry_count() -> int:
    return len(_CACHE)
