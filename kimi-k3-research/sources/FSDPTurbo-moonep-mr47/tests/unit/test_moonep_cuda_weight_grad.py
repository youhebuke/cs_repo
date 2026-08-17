"""The CUDA weight-gradient path must agree with the per-group reference."""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.ops import cuda as cuda_ops
from fsdp_turbo.ops.cuda import grouped_matmul as cuda_grouped_matmul
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink, token_span

# groups 0 and 3 are empty: their rows belong to remote ranks.
GROUP_ENDS = [0, 4, 9, 9, 13]
ROW_RANGES = ((1, 3), (4, 5))
OUT_FEATURES, IN_FEATURES = 6, 4


@pytest.fixture(autouse=True)
def _reset_probe():
    cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD = None
    yield
    cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD = None


def _inputs():
    torch.manual_seed(0)
    grad_output = torch.randn(GROUP_ENDS[-1], OUT_FEATURES)
    inputs = torch.randn(GROUP_ENDS[-1], IN_FEATURES)
    sink = GradWeightSink(
        buffer=torch.zeros(len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES),
        row_ranges=ROW_RANGES,
    )
    return grad_output, inputs, sink


def _reference(grad_output, inputs):
    expected = torch.zeros(len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES)
    group_start = 0
    for group_index, group_end in enumerate(GROUP_ENDS):
        if group_end != group_start:
            expected[group_index] = (
                grad_output[group_start:group_end].transpose(0, 1)
                @ inputs[group_start:group_end]
            )
        group_start = group_end
    return expected


def _emulate_grouped_mm(mat_a, mat_b, *, offs, out_dtype=None, bias=None):
    """CPU stand-in for the K-grouped ``grouped_mm`` used by the fused path."""
    outputs = []
    start = 0
    for end in offs.tolist():
        outputs.append(mat_a[:, start:end] @ mat_b[start:end])
        start = end
    result = torch.stack(outputs)
    return result if out_dtype is None else result.to(out_dtype)


def test_fused_path_matches_the_per_group_reference(monkeypatch):
    grad_output, inputs, sink = _inputs()
    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", _emulate_grouped_mm, raising=False
    )

    cuda_grouped_matmul.write_weight_grads_cuda(
        sink, grad_output, inputs, GROUP_ENDS
    )

    assert cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD is True
    torch.testing.assert_close(sink.buffer, _reference(grad_output, inputs))


def test_fused_path_falls_back_and_is_probed_only_once(monkeypatch):
    grad_output, inputs, sink = _inputs()
    attempts = []

    def unsupported(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("grouped_mm is unsupported on this device")

    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", unsupported, raising=False
    )

    cuda_grouped_matmul.write_weight_grads_cuda(
        sink, grad_output, inputs, GROUP_ENDS
    )
    cuda_grouped_matmul.write_weight_grads_cuda(
        sink, grad_output, inputs, GROUP_ENDS
    )

    assert cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD is False
    assert len(attempts) == 1
    torch.testing.assert_close(sink.buffer, _reference(grad_output, inputs))


def test_fused_path_produces_fp32_from_bf16_without_a_staging_buffer(monkeypatch):
    grad_output, inputs, _ = _inputs()
    grad_output = grad_output.to(torch.bfloat16)
    inputs = inputs.to(torch.bfloat16)
    sink = GradWeightSink(
        buffer=torch.zeros(
            len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES, dtype=torch.float32
        ),
        row_ranges=ROW_RANGES,
    )
    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", _emulate_grouped_mm, raising=False
    )

    cuda_grouped_matmul.write_weight_grads_cuda(
        sink, grad_output, inputs, GROUP_ENDS
    )

    assert sink.buffer.dtype == torch.float32
    assert sink.scratch is None
    torch.testing.assert_close(
        sink.buffer, _reference(grad_output.float(), inputs.float()), rtol=2e-2, atol=2e-2
    )


def test_remote_rows_are_rejected_before_any_gemm_runs(monkeypatch):
    grad_output, inputs, sink = _inputs()
    # Claim only the second range, leaving groups 1 and 2 remote.
    sink.row_ranges = ((4, 5),)
    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", _emulate_grouped_mm, raising=False
    )

    with pytest.raises(RuntimeError, match="owned by a remote rank"):
        cuda_grouped_matmul.write_weight_grads_cuda(
            sink, grad_output, inputs, GROUP_ENDS
        )
    assert sink.buffer.abs().sum() == 0


def test_token_span_covers_exactly_the_range_of_groups():
    assert token_span((1, 3), GROUP_ENDS) == (0, 9)
    assert token_span((4, 5), GROUP_ENDS) == (9, 13)
    assert token_span((0, 1), GROUP_ENDS) == (0, 0)


def test_cuda_grouped_matmul_writes_the_sink_on_a_real_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA accelerator is unavailable")

    device = torch.device("cuda")
    offsets = torch.tensor([0, 3, 3, 9], dtype=torch.int32, device=device)
    inputs = torch.randn(
        9, 128, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    weights = torch.randn(4, 256, 128, dtype=torch.bfloat16, device=device)
    sink = GradWeightSink(
        buffer=torch.zeros(4, 256, 128, dtype=torch.float32, device=device),
        row_ranges=((1, 2), (3, 4)),
    )

    output = cuda_ops.grouped_matmul_cuda(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    output.float().sum().backward()

    assert inputs.grad is not None
    assert weights.grad is None
    assert sink.buffer[1].abs().sum() > 0
    assert sink.buffer[3].abs().sum() > 0
    torch.testing.assert_close(
        sink.buffer[0], torch.zeros_like(sink.buffer[0]), rtol=0, atol=0
    )
