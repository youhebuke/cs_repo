# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import importlib
import logging

import torch
from torch.distributed import ProcessGroup

from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_attention import (
    get_kv_allgather_attention_fn,
    get_kv_allgather_compressor_attention_fn,
)
from fsdp_turbo.fsdp_turbo_config import CPPlanConfig
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def kv_allgather_context_parallelize_modules(
        model: torch.nn.Module,
        cp_group: ProcessGroup,
        cp_plan: CPPlanConfig,
) -> torch.nn.Module:
    """Apply KV all-gather context parallelism by wrapping core attention functions.

    For each fully-qualified function path in
    ``cp_plan.kv_allgather_core_attention_function``, the function is imported,
    wrapped with the KV all-gather decorator selected by
    ``cp_plan.kv_allgather_attention_type``, and replaced in-place in its
    source module.

    Args:
        model: The model (kept for API consistency; wrapping is applied at function level)
        cp_group: Process group for CP communication
        cp_plan: Context parallel plan configuration specifying which
                 attention functions to wrap and which decorator type to use

    Returns:
        The model (unchanged reference; attention functions are replaced in-place)
    """
    cp_size = torch.distributed.get_world_size(cp_group)
    if cp_size == 1:
        return model

    if cp_plan is None or cp_plan.kv_allgather_core_attention_function is None:
        raise RuntimeError(
            "[Context Parallel/KVAllGather] cp_plan.kv_allgather_core_attention_function is not specified. "
            "Please provide the attention function paths in cp_plan.kv_allgather_core_attention_function."
        )

    # Resolve the decorator based on kv_allgather_attention_type
    attention_type = cp_plan.kv_allgather_attention_type
    if attention_type == "default":
        wrap_fn = get_kv_allgather_attention_fn
        wrap_kwargs = {}
    elif attention_type == "with_compressor":
        wrap_fn = get_kv_allgather_compressor_attention_fn
        wrap_kwargs = {}
    elif callable(attention_type):
        wrap_fn = attention_type
        wrap_kwargs = {}
    else:
        raise ValueError(
            f"[Context Parallel/KVAllGather] Invalid kv_allgather_attention_type: {attention_type!r}. "
            f"Expected 'default', 'with_compressor', or a callable."
        )

    for fn_path in cp_plan.kv_allgather_core_attention_function:
        _import_and_replace_fn(fn_path, cp_group, wrap_fn, wrap_kwargs)

    return model


def _import_and_replace_fn(fn_path: str, cp_group: ProcessGroup, wrap_fn, wrap_kwargs: dict = None):
    """Import a function by its fully qualified path, wrap it with the given
    KV all-gather decorator, and replace it in-place in its source module.

    Args:
        fn_path: Fully qualified function path, e.g.
            "transformers.models.deepseek_v4.modeling_deepseek_v4.eager_attention_forward"
        cp_group: Process group for CP communication
        wrap_fn: KV all-gather decorator function
        wrap_kwargs: Additional keyword arguments to pass to wrap_fn
    """
    parts = fn_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f"[Context Parallel/KVAllGather] Invalid function path '{fn_path}'. "
            f"Expected format: 'module.submodule.function_name'"
        )

    module_path, func_name = parts
    mod = importlib.import_module(module_path)
    original_fn = getattr(mod, func_name, None)
    if original_fn is None:
        raise AttributeError(
            f"[Context Parallel/KVAllGather] Function '{func_name}' not found in module '{module_path}'."
        )

    kwargs = wrap_kwargs or {}
    wrapped_fn = wrap_fn(original_fn, cp_group, **kwargs)
    setattr(mod, func_name, wrapped_fn)
    print_rank(logger.info, f'[Context Parallel/KVAllGather]: Replaced function <{fn_path}> with KVAllGather-wrapped version (type={wrap_fn.__name__})')
