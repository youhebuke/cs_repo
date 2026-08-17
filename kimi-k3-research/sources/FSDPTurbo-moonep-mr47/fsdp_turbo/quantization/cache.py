# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Callable, Any

import torch

_weight_cache: dict[int, Any] = {}
_num_quantized: int = 0
_num_cache_hits: int = 0
_opt_hooked: bool = False


def cached_weight_quant(
    weight: torch.Tensor,
    quantizer: Callable[[torch.Tensor, Any], Any],
    *args,
    **kwargs,
) -> Any:
    global _num_quantized, _num_cache_hits

    key = weight.untyped_storage().data_ptr()

    if key in _weight_cache:
        _num_cache_hits += 1
        return _weight_cache[key]

    result = quantizer(weight, *args, **kwargs)
    _weight_cache[key] = result
    _num_quantized += 1
    return result


def clear_weight_cache() -> None:
    global _num_quantized, _num_cache_hits
    for val in _weight_cache.values():
        if isinstance(val, torch.Tensor):
            val.untyped_storage().resize_(0)
        elif isinstance(val, tuple):
            for t in val:
                if isinstance(t, torch.Tensor):
                    t.untyped_storage().resize_(0)
    _weight_cache.clear()
    _num_quantized = 0
    _num_cache_hits = 0


def hook_optimizer_step(model, optimizer) -> None:
    global _opt_hooked
    if _opt_hooked:
        return

    original_step = optimizer.step

    def step_with_cache_clear(*args, **kwargs):
        clear_weight_cache()
        return original_step(*args, **kwargs)

    optimizer.step = step_with_cache_clear
    _opt_hooked = True
