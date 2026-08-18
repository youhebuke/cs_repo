"""CUDA grouped matmul: fused _grouped_mm vs explicit vmm_safe escape hatch."""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.ops.cuda import grouped_matmul as cuda_grouped_matmul
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink


M_SPLIT = torch.tensor([4, 0, 5], dtype=torch.int32)
OUT_FEATURES, IN_FEATURES = 8, 4
GROUP_ENDS = [4, 4, 9]


@pytest.fixture(autouse=True)
def _reset_probe():
    cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD = None
    yield
    cuda_grouped_matmul._SUPPORTS_FUSED_WEIGHT_GRAD = None


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


def test_vmm_safe_forward_backward_never_calls_grouped_mm(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES, requires_grad=True)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES, requires_grad=True)
    inputs_ref = inputs.detach().clone().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("F.grouped_mm must not run on the VMM-safe path")

    monkeypatch.setattr(cuda_grouped_matmul.F, "grouped_mm", forbidden, raising=False)

    actual = cuda_grouped_matmul.grouped_matmul_cuda(
        inputs, M_SPLIT, weights, vmm_safe=True
    )
    expected = _reference(inputs_ref, weights_ref, M_SPLIT)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert not calls
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs.grad, inputs_ref.grad)
    torch.testing.assert_close(weights.grad, weights_ref.grad)


def test_sink_without_vmm_safe_uses_fused_grouped_mm(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES, requires_grad=True)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    sink = GradWeightSink(
        buffer=torch.zeros(3, OUT_FEATURES, IN_FEATURES),
        row_ranges=((0, 1), (2, 3)),
    )
    calls = []

    def emulate(mat_a, mat_b, offs=None, *args, **kwargs):
        calls.append(None if offs is None and not args else (offs if offs is not None else args[0]).tolist())
        return _emulate_2d_3d(mat_a, mat_b, offs, *args, **kwargs)

    monkeypatch.setattr(cuda_grouped_matmul.F, "grouped_mm", emulate, raising=False)

    output = cuda_grouped_matmul.grouped_matmul_cuda(
        inputs, M_SPLIT, weights, grad_weight_sink=sink
    )
    output.sum().backward()

    assert calls == [GROUP_ENDS, GROUP_ENDS]
    assert weights.grad is None
    assert sink.buffer[0].abs().sum() > 0
    assert sink.buffer[1].abs().sum() == 0
    assert sink.buffer[2].abs().sum() > 0


def test_sink_forward_does_not_materialize_host_group_ends(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES, requires_grad=True)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    sink = GradWeightSink(
        buffer=torch.zeros(3, OUT_FEATURES, IN_FEATURES),
        row_ranges=((0, 1), (2, 3)),
    )
    host_reads = []
    original = cuda_grouped_matmul._group_ends

    def spy(m_split):
        host_reads.append(True)
        return original(m_split)

    monkeypatch.setattr(cuda_grouped_matmul, "_group_ends", spy)
    monkeypatch.setattr(
        cuda_grouped_matmul.F, "grouped_mm", _emulate_2d_3d, raising=False
    )

    output = cuda_grouped_matmul.grouped_matmul_cuda(
        inputs, M_SPLIT, weights, grad_weight_sink=sink
    )
    assert not host_reads
    output.sum().backward()
    assert not host_reads


def test_fused_dispatcher_path_still_calls_grouped_mm(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    calls = []

    def emulate(mat_a, mat_b, *, offs, out_dtype=None, bias=None):
        calls.append(offs.tolist())
        return _emulate_2d_3d(mat_a, mat_b, offs)

    monkeypatch.setattr(cuda_grouped_matmul.F, "grouped_mm", emulate, raising=False)

    actual = cuda_grouped_matmul.grouped_matmul_cuda(inputs, M_SPLIT, weights)

    assert calls == [GROUP_ENDS]
    torch.testing.assert_close(actual, _reference(inputs, weights, M_SPLIT))


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


def test_grouped_mm_uses_aten_when_torch_and_functional_missing(monkeypatch):
    torch.manual_seed(0)
    inputs = torch.randn(9, IN_FEATURES)
    weights = torch.randn(3, OUT_FEATURES, IN_FEATURES)
    monkeypatch.delattr(cuda_grouped_matmul.F, "grouped_mm", raising=False)
    monkeypatch.setattr(torch, "_grouped_mm", None, raising=False)
    calls = []

    def aten_gmm(mat_a, mat_b, offs=None, *args, **kwargs):
        if offs is None and args:
            offs = args[0]
        calls.append(offs.tolist())
        return _emulate_2d_3d(mat_a, mat_b, offs)

    monkeypatch.setattr(torch.ops.aten, "_grouped_mm", aten_gmm, raising=False)

    actual = cuda_grouped_matmul.grouped_matmul_cuda(inputs, M_SPLIT, weights)

    assert calls == [GROUP_ENDS]
    torch.testing.assert_close(actual, _reference(inputs, weights, M_SPLIT))
