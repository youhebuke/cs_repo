"""CUDA grouped_mm fallback used for GPU functional checks (PT 2.9)."""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.ops.cuda import grouped_matmul as cuda_grouped_matmul
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink


M_SPLIT = torch.tensor([4, 0, 5], dtype=torch.int32)
OUT_FEATURES, IN_FEATURES = 8, 4
GROUP_ENDS = [4, 4, 9]


def _reference(inputs, weights, m_split):
    outputs = []
    start = 0
    for group, size in enumerate(m_split.tolist()):
        end = start + size
        outputs.append(inputs[start:end] @ weights[group].transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


def _emulate_2d_3d(mat_a, mat_b, offs=None, *args, **kwargs):
    if offs is None and args:
        offs = args[0]
    outputs = []
    start = 0
    for group, end in enumerate(offs.tolist()):
        outputs.append(mat_a[start:end] @ mat_b[group])
        start = end
    return torch.cat(outputs, dim=0)


def test_grouped_mm_uses_torch_when_functional_missing(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    monkeypatch.delattr(cuda_grouped_matmul.F, "grouped_mm", raising=False)
    calls = []

    def torch_gmm(mat_a, mat_b, offs=None, *args, **kwargs):
        if offs is None and args:
            offs = args[0]
        calls.append(offs.tolist())
        return _emulate_2d_3d(mat_a, mat_b, offs)

    monkeypatch.setattr(torch, "_grouped_mm", torch_gmm, raising=False)

    actual = cuda_grouped_matmul.grouped_matmul_cuda(inputs, M_SPLIT, weights)

    assert calls == [GROUP_ENDS]
    torch.testing.assert_close(actual, _reference(inputs, weights, M_SPLIT))


def test_sink_writes_local_rows_without_weight_grad(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES, requires_grad=True)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    sink = GradWeightSink(
        buffer=torch.zeros(3, OUT_FEATURES, IN_FEATURES),
        row_ranges=((0, 1), (2, 3)),
    )
    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", _emulate_2d_3d, raising=False
    )

    output = cuda_grouped_matmul.grouped_matmul_cuda(
        inputs, M_SPLIT, weights, grad_weight_sink=sink
    )
    output.sum().backward()

    assert weights.grad is None
    assert sink.buffer[0].abs().sum() > 0
    assert sink.buffer[1].abs().sum() == 0
    assert sink.buffer[2].abs().sum() > 0
