"""The grouped matmul must write expert gradients straight into MoonEP's buffer."""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep_dispatcher import (
    _grouped_matmul_with_static_tail,
)
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink


def _sink(num_groups, out_features, in_features, row_ranges, dtype=torch.float32):
    return GradWeightSink(
        buffer=torch.zeros(num_groups, out_features, in_features, dtype=dtype),
        row_ranges=row_ranges,
    )


def test_sink_receives_the_same_gradient_autograd_would_produce():
    offsets = torch.tensor([0, 4, 7], dtype=torch.int32)
    inputs = torch.randn(9, 8, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(3, 16, 8, dtype=torch.float32)

    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference = _grouped_matmul_with_static_tail(
        reference_inputs, offsets, reference_weights
    )
    grad = torch.randn_like(reference)
    reference.backward(grad)

    sink = _sink(3, 16, 8, row_ranges=((0, 3),))
    actual = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    actual.backward(grad)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(sink.buffer, reference_weights.grad)
    # The weight tensor stays outside the autograd graph.
    assert weights.grad is None


def test_sink_accepts_a_gradient_dtype_that_differs_from_the_weights():
    offsets = torch.tensor([2, 5], dtype=torch.int32)
    inputs = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(2, 6, 4, dtype=torch.bfloat16)

    sink = _sink(2, 6, 4, row_ranges=((0, 2),), dtype=torch.float32)
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    output.backward(torch.randn_like(output))

    assert sink.buffer.dtype == torch.float32
    assert sink.buffer.abs().sum() > 0
    # A single staging tensor is reused instead of a full [E+B] allocation.
    assert sink.scratch is not None
    assert tuple(sink.scratch.shape) == (6, 4)


def test_sink_refuses_to_write_a_row_owned_by_a_remote_rank():
    offsets = torch.tensor([3, 6], dtype=torch.int32)
    inputs = torch.randn(6, 4, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(2, 6, 4, dtype=torch.float32)

    # Only group 1 is local; group 0 belongs to another rank's memory.
    sink = _sink(2, 6, 4, row_ranges=((1, 2),))
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    with pytest.raises(RuntimeError, match="owned by a remote rank"):
        output.backward(torch.randn_like(output))


def test_empty_groups_leave_remote_rows_untouched():
    # Group 0 is empty, so its row must keep the sentinel value instead of
    # being zeroed through the symmetric mapping.
    offsets = torch.tensor([0, 5], dtype=torch.int32)
    inputs = torch.randn(5, 4, dtype=torch.float32, requires_grad=True)
    weights = torch.randn(2, 6, 4, dtype=torch.float32)

    sink = _sink(2, 6, 4, row_ranges=((1, 2),))
    sink.buffer[0].fill_(7.0)
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    output.backward(torch.randn_like(output))

    torch.testing.assert_close(
        sink.buffer[0], torch.full((6, 4), 7.0), rtol=0, atol=0
    )
    assert sink.buffer[1].abs().sum() > 0


def test_sink_shape_must_match_the_weights():
    sink = _sink(2, 6, 4, row_ranges=((0, 2),))
    with pytest.raises(RuntimeError, match="does not match the weight shape"):
        _grouped_matmul_with_static_tail(
            torch.randn(4, 4),
            torch.tensor([2, 4], dtype=torch.int32),
            torch.randn(2, 8, 4),
            grad_weight_sink=sink,
        )
