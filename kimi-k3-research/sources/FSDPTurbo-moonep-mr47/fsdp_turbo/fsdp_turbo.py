# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import fully_shard_parallel_modules
from fsdp_turbo.distributed.parallel_state import init_parallel_state
from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel import ulysses_context_parallelize_modules
from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_context_parallel import (
    kv_allgather_context_parallelize_modules,
)
from fsdp_turbo.memory.recompute.recompute import recompute_modules
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.utils.patch import apply_module_patches


class FSDPTurbo(torch.nn.Module):
    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module):
        super().__init__()
        self.config = config
        self.model = model

        self.parallel_state = init_parallel_state(self.config)
        self.apply_quantization_modules()
        self.apply_tp_modules()
        self._capture_tp_info_on_mx_linear()
        self.apply_cp_modules()
        self.apply_module_patches()
        self.apply_ep_modules()
        self.apply_recompute_modules()
        self.apply_chunk_batch_modules()
        self.apply_fsdp_modules()

    def apply_fsdp_modules(self):
        # Use 2D (dp, fsdp) mesh so FSDP2 treats dense params as HSDP and
        # all-reduces gradients across the dp (replicate) dim.  Using the 1D
        # fsdp mesh skips the dp all-reduce, yielding wrong gradients when the
        # fsdp domain is not filled (data_parallel_size > 1).
        # The gradient divide factor is derived in the config validation
        # (from the launcher-provided WORLD_SIZE), so it is simply read here.
        self.model = fully_shard_parallel_modules(
            self.model, self.parallel_state.get_device_mesh('dp_fsdp'), self.config.distributed.fsdp_plan
        )

    def apply_tp_modules(self):
        if self.config.distributed.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(
            self.model, self.parallel_state.get_tp_device_mesh(), self.config.distributed.tp_plan
        )

    def apply_cp_modules(self):
        cp_plan = self.config.distributed.cp_plan
        if cp_plan is None:
            return
        # KV all-gather context parallelism
        if self.config.distributed.kv_allgather_parallel_size > 1 and cp_plan.kv_allgather_core_attention_function:
            self.model = kv_allgather_context_parallelize_modules(
                self.model,
                self.parallel_state.get_kvallgather_group(),
                cp_plan,
            )

        # Ulysses context parallelism
        if self.config.distributed.ulysses_parallel_size > 1 and cp_plan.ulysses_core_attention_function:
            self.model = ulysses_context_parallelize_modules(
                self.model,
                self.parallel_state.get_ulysses_group(),
                cp_plan,
            )

    def apply_ep_modules(self):
        if self.config.distributed.expert_parallel_size > 1:
            self.model = expert_parallelize_modules(
                self.model, self.parallel_state.get_ep_device_mesh(), self.config.distributed.ep_plan
            )
            if self.config.distributed.ep_plan.dispatcher != "moonep":
                self.model = expert_fully_shard_modules(
                    self.model,
                    self.parallel_state.get_device_mesh('edp_efsdp'),
                    self.config.distributed.ep_plan,
                    self.config.distributed.fsdp_plan,
                )

    def apply_module_patches(self):
        """Apply configured replacements before FSDP wraps the model.

        Each configured target must match either a module-level callable or at
        least one module in ``self.model.named_modules()``. Failing early keeps
        a misspelled or architecture-specific target from silently disabling a
        required patch.
        """
        for spec in self.config.module_patches:
            self.model, matched = apply_module_patches(self.model, spec)
            if not matched:
                raise RuntimeError(f"Configured module patch did not match the model: target={spec['target']!r}.")

    def apply_recompute_modules(self):
        if not self.config.memory.recompute:
            return
        self.model = recompute_modules(self.model, self.config.memory.recompute_plan)

    def _capture_tp_info_on_mx_linear(self):
        if self.config.distributed.tensor_parallel_size <= 1:
            return
        from fsdp_turbo.quantization.mx_formats.mx_linear import MXLinear
        from torch.distributed.tensor import Shard, Partial
        from fsdp_turbo.utils.str_match import module_name_match

        tp_mesh = self.parallel_state.get_device_mesh('tp')
        colwise = self.config.distributed.tp_plan.colwise_parallel or []
        rowwise = self.config.distributed.tp_plan.rowwise_parallel or []
        for name, mod in self.model.named_modules():
            if not isinstance(mod, MXLinear):
                continue
            if any(module_name_match(p, name) for p in colwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Shard(-1)]
            elif any(module_name_match(p, name) for p in rowwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Partial()]

    def apply_chunk_batch_modules(self):
        if not self.config.memory.chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.memory.chunk_batch_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization.quantization_plan.quant_recipe:
            return
        try:
            # When recompute is enabled, forward may be called during backward.
            # Force "all" mode so both fwd and bwd quantized tensors are available.
            if self.config.memory.recompute:
                self.config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode = "all"

            from fsdp_turbo.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError("Failed to convert quantization plan") from e

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def close(self):
        """Release optional backend resources before process-group teardown."""
        runtime = getattr(self.model, "_moonep_runtime", None)
        if runtime is not None:
            runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
