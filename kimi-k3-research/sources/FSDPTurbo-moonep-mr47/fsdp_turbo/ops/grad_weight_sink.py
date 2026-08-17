# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""In-place destination for grouped-matmul weight gradients.

MoonEP maps one contiguous ``[E+B, out, in]`` FP32 gradient buffer per expert
projection: rows ``[0, E)`` reach every rank's parameter gradients through
symmetric memory and rows ``[E, E+B)`` alias this rank's reduce-buffer slots.
Handing that buffer to the grouped matmul lets the backward pass write its
result in place instead of allocating a second full-size tensor that is then
copied into the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class GradWeightSink:
    """Pre-allocated destination for a grouped matmul's weight gradient.

    ``row_ranges`` lists the half-open group ranges backed by this rank's own
    memory. Every other row is a remote rank's gradient reachable through the
    symmetric mapping, and writing there would corrupt that rank's state.
    """

    buffer: torch.Tensor
    row_ranges: Tuple[Tuple[int, int], ...]
    scratch: Optional[torch.Tensor] = field(default=None, repr=False)

    def owns(self, group_index: int) -> bool:
        return any(start <= group_index < end for start, end in self.row_ranges)

    def staging(self, dtype: torch.dtype, shape: Sequence[int]) -> torch.Tensor:
        """Return a reusable staging buffer for kernels that cannot emit FP32.

        One buffer is reused for every group, so the backward pass allocates a
        single ``[out, in]`` tensor instead of a full ``[E+B, out, in]`` one.
        """
        shape = tuple(shape)
        if (
            self.scratch is None
            or self.scratch.dtype != dtype
            or tuple(self.scratch.shape) != shape
        ):
            self.scratch = torch.empty(shape, dtype=dtype, device=self.buffer.device)
        return self.scratch


def validate_sink(sink: GradWeightSink, weights: torch.Tensor) -> None:
    if sink.buffer.shape != weights.shape:
        raise RuntimeError(
            "Grad weight sink shape "
            f"{tuple(sink.buffer.shape)} does not match the weight shape "
            f"{tuple(weights.shape)}."
        )
    if not sink.row_ranges:
        raise RuntimeError("Grad weight sink must own at least one row range.")


def assert_local_groups(sink: GradWeightSink, group_ends: Sequence[int]) -> None:
    """Fail if a non-empty group would write a row owned by a remote rank."""
    group_start = 0
    for group_index, group_end in enumerate(group_ends):
        if group_end != group_start and not sink.owns(group_index):
            raise RuntimeError(
                f"Grouped matmul group {group_index} received "
                f"{group_end - group_start} tokens but its weight gradient row "
                "is owned by a remote rank; MoonEP planning should only route "
                "tokens to local or prefetched experts."
            )
        group_start = group_end


def token_span(row_range: Tuple[int, int], group_ends: Sequence[int]) -> Tuple[int, int]:
    """Return the token range covered by a half-open range of groups."""
    row_start, row_end = row_range
    return (group_ends[row_start - 1] if row_start else 0), group_ends[row_end - 1]


def write_group_grads(
    sink: GradWeightSink,
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    group_ends: Sequence[int],
) -> None:
    """Write ``grad_weight[g] = grad_output[g]^T @ inputs[g]`` into the sink.

    Empty groups are skipped rather than zeroed: their rows usually belong to a
    remote rank, and MoonEP's planner guarantees this rank only receives tokens
    for the experts it owns or prefetched.
    """
    assert_local_groups(sink, group_ends)
    group_start = 0
    for group_index, group_end in enumerate(group_ends):
        if group_end == group_start:
            continue
        destination = sink.buffer[group_index]
        left = grad_output[group_start:group_end].transpose(0, 1)
        right = inputs[group_start:group_end]
        if destination.dtype == grad_output.dtype:
            torch.mm(left, right, out=destination)
        else:
            staging = sink.staging(grad_output.dtype, destination.shape)
            torch.mm(left, right, out=staging)
            destination.copy_(staging)
        group_start = group_end
