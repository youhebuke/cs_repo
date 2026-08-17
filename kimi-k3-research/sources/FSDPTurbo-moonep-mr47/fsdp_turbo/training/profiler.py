# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Training profiler creation utilities.

Creates a PyTorch profiler (or NPU profiler) from a ``ProfileConfig``,
or a no-op context manager when profiling is disabled.
"""

from contextlib import nullcontext

import torch


def create_profiler(profile_cfg):
    """Create a profiler context manager from a ``ProfileConfig``.

    When ``profile_cfg.enabled`` is ``False``, returns a no-op context.

    Automatically selects the NPU (HCCL) profiler when ``torch_npu`` is
    available, otherwise falls back to the standard CUDA/CPU profiler.

    Args:
        profile_cfg: A ``ProfileConfig`` dataclass instance.

    Returns:
        A context manager that yields an active profiler (or None).
    """
    if not profile_cfg.enabled:
        return nullcontext()

    schedule = torch.profiler.schedule(
        wait=profile_cfg.wait_steps,
        warmup=profile_cfg.warmup_steps,
        active=profile_cfg.active_steps,
        repeat=profile_cfg.repeat,
        skip_first=profile_cfg.skip_first,
    )

    try:
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
        )
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
            record_shapes=profile_cfg.record_shapes,
            profile_memory=profile_cfg.profile_memory,
            with_stack=profile_cfg.with_stack,
            experimental_config=experimental_config,
            schedule=schedule,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_cfg.output_dir),
        )
    except ImportError:
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=profile_cfg.record_shapes,
            profile_memory=profile_cfg.profile_memory,
            with_stack=profile_cfg.with_stack,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_cfg.output_dir),
        )
