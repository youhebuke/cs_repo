# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import importlib
import logging

import torch
from torch.distributed import ProcessGroup

from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import (
    get_ulysses_attention_fn,
    get_ulysses_compressor_attention_fn,
)
from fsdp_turbo.fsdp_turbo_config import CPPlanConfig
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def ulysses_context_parallelize_modules(
        model: torch.nn.Module,
        ulysses_group: ProcessGroup,
        cp_plan: CPPlanConfig,
) -> torch.nn.Module:
    """Apply Ulysses context parallelism by wrapping core attention functions.

    For each fully-qualified function path in cp_plan.ulysses_core_attention_function,
    the function is imported, wrapped with the Ulysses all-to-all decorator
    selected by ``cp_plan.ulysses_attention_type``, and replaced in-place in
    its source module.

    Args:
        model: The model (kept for API consistency; Ulysses is applied at function level)
        ulysses_group: Process group for Ulysses communication
        cp_plan: Context parallel plan configuration specifying which
                 attention functions to wrap and which decorator type to use

    Returns:
        The model (unchanged reference; attention functions are replaced in-place)
    """
    ulysses_size = torch.distributed.get_world_size(ulysses_group)
    if ulysses_size == 1:
        return model

    if cp_plan is None or cp_plan.ulysses_core_attention_function is None:
        raise RuntimeError(
            "[Context Parallel] cp_plan.ulysses_core_attention_function is not specified. "
            "Please provide the attention function paths in cp_plan.ulysses_core_attention_function."
        )

    # Resolve the decorator based on ulysses_attention_type
    attention_type = cp_plan.ulysses_attention_type
    attn_output_transposed = cp_plan.ulysses_attn_output_transposed
    if attention_type == "default":
        wrap_fn = get_ulysses_attention_fn
        wrap_kwargs = {}
    elif attention_type == "with_compressor":
        wrap_fn = get_ulysses_compressor_attention_fn
        wrap_kwargs = {"attn_output_transposed": attn_output_transposed}
    elif callable(attention_type):
        wrap_fn = attention_type
        wrap_kwargs = {}
    else:
        raise ValueError(
            f"[Context Parallel] Invalid ulysses_attention_type: {attention_type!r}. "
            f"Expected 'default', 'with_compressor', or a callable."
        )

    for fn_path in cp_plan.ulysses_core_attention_function:
        _import_and_replace_fn(fn_path, ulysses_group, wrap_fn, wrap_kwargs)

    return model


def _import_and_replace_fn(fn_path: str, ulysses_group: ProcessGroup, wrap_fn, wrap_kwargs: dict = None):
    """Import a function by its fully qualified path, wrap it with the given
    Ulysses decorator, and replace it in-place in its source module.

    Args:
        fn_path: Fully qualified function path, e.g.
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward"
        ulysses_group: Process group for Ulysses communication
        wrap_fn: Ulysses decorator function (e.g. get_ulysses_attention_fn,
                 get_ulysses_compressor_attention_fn, or a custom callable)
        wrap_kwargs: Additional keyword arguments to pass to wrap_fn
    """
    # Split into module path and function name
    # e.g. "a.b.c.func" -> module="a.b.c", func_name="func"
    parts = fn_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f"[Context Parallel] Invalid function path '{fn_path}'. "
            f"Expected format: 'module.submodule.function_name'"
        )

    module_path, func_name = parts
    mod = importlib.import_module(module_path)
    original_fn = getattr(mod, func_name, None)
    if original_fn is None:
        raise AttributeError(
            f"[Context Parallel] Function '{func_name}' not found in module '{module_path}'."
        )

    kwargs = wrap_kwargs or {}
    wrapped_fn = wrap_fn(original_fn, ulysses_group, **kwargs)
    setattr(mod, func_name, wrapped_fn)
    print_rank(logger.info, f'[Context Parallel/Ulysses]: Replaced function <{fn_path}> with Ulysses-wrapped version (type={wrap_fn.__name__})')
