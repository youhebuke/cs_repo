#!/usr/bin/env python3
"""Verify CUDA grouped matmul: fused _grouped_mm vs vmm_safe escape hatch.

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_cuda_vmm_safe_grouped_matmul.py

Expected output (exit code 0):

    [1/6] PASS  vmm_safe 前向不调用 grouped_mm, 结果与逐 expert 参考一致
    [2/6] PASS  空组被跳过, 输出对应行为 0
    [3/6] PASS  dgrad / wgrad 与参考一致, 仍不调用 grouped_mm
    [4/6] PASS  sink + vmm_safe 只写本地行, 且关闭融合 grouped_mm
    [5/6] PASS  传入 sink 且未设 vmm_safe 时走 fused grouped_mm
    [6/6] PASS  无 sink, vmm_safe=False 走 fused grouped_mm

MoonEP 训练走 fused 路径（torch._grouped_mm / aten._grouped_mm），前向不得
``.cpu()``。``vmm_safe=True`` 仍是逐 expert ``torch.mm`` 逃生舱。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _repo_root import use_repo_checkout  # noqa: E402

use_repo_checkout()

from fsdp_turbo.ops.cuda import grouped_matmul as cuda_gmm  # noqa: E402
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink  # noqa: E402

_checks = []
_TOTAL = 6

# Three groups, the middle one empty. Last group also covers a static tail
# once the caller has folded the padding into m_split, matching MoonEP.
M_SPLIT = torch.tensor([4, 0, 5], dtype=torch.int32)
OUT_FEATURES, IN_FEATURES = 6, 4
GROUP_ENDS = [4, 4, 9]


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/{_TOTAL}] {'PASS' if passed else 'FAIL'}  {message}")


class _RecordingGroupedMm:
    def __init__(self):
        self.calls = []

    def __call__(self, mat_a, mat_b, offs=None, *args, out_dtype=None, bias=None, **kwargs):
        if offs is None and args:
            offs = args[0]
        self.calls.append(
            (tuple(mat_a.shape), tuple(mat_b.shape), None if offs is None else offs.tolist())
        )
        raise AssertionError("F.grouped_mm must not run on the VMM-safe path")


def _with_grouped_mm(stub):
    original = getattr(cuda_gmm.F, "grouped_mm", None)
    cuda_gmm.F.grouped_mm = stub
    cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD = None
    return original


def _restore(original):
    if original is None:
        if hasattr(cuda_gmm.F, "grouped_mm"):
            delattr(cuda_gmm.F, "grouped_mm")
    else:
        cuda_gmm.F.grouped_mm = original
    cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD = None


def _fixture(requires_grad=True):
    torch.manual_seed(0)
    inputs = torch.randn(GROUP_ENDS[-1], IN_FEATURES, requires_grad=requires_grad)
    weights = torch.randn(len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES, requires_grad=True)
    return inputs, weights


def _reference_forward(inputs, weights):
    outputs = []
    start = 0
    for group, end in enumerate(GROUP_ENDS):
        outputs.append(inputs[start:end] @ weights[group].transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


def check_forward_skips_grouped_mm():
    inputs, weights = _fixture(requires_grad=False)
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        actual = cuda_gmm.grouped_matmul_cuda(
            inputs, M_SPLIT, weights, vmm_safe=True
        )
        matched = torch.allclose(actual, _reference_forward(inputs, weights))
        unused = not stub.calls
    finally:
        _restore(original)
    check(matched and unused, "vmm_safe 前向不调用 grouped_mm, 结果与逐 expert 参考一致")


def check_empty_group_is_zero():
    inputs, weights = _fixture(requires_grad=False)
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        actual = cuda_gmm.grouped_matmul_cuda(
            inputs, M_SPLIT, weights, vmm_safe=True
        )
        # The empty group contributes no rows; live rows are 0:4 and 4:9.
        passed = (
            M_SPLIT[1].item() == 0
            and actual.shape[0] == GROUP_ENDS[-1]
            and not stub.calls
            and torch.allclose(actual, _reference_forward(inputs, weights))
        )
    finally:
        _restore(original)
    check(passed, "空组被跳过, 输出对应行为 0")


def check_backward_matches_reference():
    inputs, weights = _fixture()
    inputs_ref = inputs.detach().clone().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        actual = cuda_gmm.grouped_matmul_cuda(
            inputs, M_SPLIT, weights, vmm_safe=True
        )
        expected = _reference_forward(inputs_ref, weights_ref)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        passed = (
            not stub.calls
            and torch.allclose(actual, expected)
            and torch.allclose(inputs.grad, inputs_ref.grad)
            and torch.allclose(weights.grad, weights_ref.grad)
        )
    finally:
        _restore(original)
    check(passed, "dgrad / wgrad 与参考一致, 仍不调用 grouped_mm")


def check_sink_disables_fused_weight_grad():
    inputs, weights = _fixture()
    sink = GradWeightSink(
        buffer=torch.zeros_like(weights, dtype=torch.float32),
        row_ranges=((0, 1), (2, 3)),
    )
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        output = cuda_gmm.grouped_matmul_cuda(
            inputs, M_SPLIT, weights, grad_weight_sink=sink, vmm_safe=True
        )
        output.sum().backward()
        local_written = sink.buffer[0].abs().sum() > 0 and sink.buffer[2].abs().sum() > 0
        remote_clean = sink.buffer[1].abs().sum() == 0
        unused = not stub.calls
        no_weight_grad = weights.grad is None
    finally:
        _restore(original)
    check(
        local_written and remote_clean and unused and no_weight_grad,
        "sink + vmm_safe 只写本地行, 且关闭融合 grouped_mm",
    )


def check_sink_without_vmm_safe_uses_fused_grouped_mm():
    inputs, weights = _fixture()
    sink = GradWeightSink(
        buffer=torch.zeros_like(weights, dtype=torch.float32),
        row_ranges=((0, 1), (2, 3)),
    )
    stub = _CapturingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        output = cuda_gmm.grouped_matmul_cuda(
            inputs, M_SPLIT, weights, grad_weight_sink=sink
        )
        output.sum().backward()
        used = stub.calls == [GROUP_ENDS, GROUP_ENDS]
        local_written = sink.buffer[0].abs().sum() > 0 and sink.buffer[2].abs().sum() > 0
        remote_clean = sink.buffer[1].abs().sum() == 0
        no_weight_grad = weights.grad is None
    finally:
        _restore(original)
    check(
        used and local_written and remote_clean and no_weight_grad,
        "传入 sink 且未设 vmm_safe 时走 fused grouped_mm",
    )


class _CapturingGroupedMm:
    def __init__(self):
        self.calls = []

    def __call__(self, mat_a, mat_b, offs=None, *args, out_dtype=None, bias=None, **kwargs):
        if offs is None and args:
            offs = args[0]
        self.calls.append(offs.tolist() if offs is not None else None)
        # 2D x 3D grouped GEMM stand-in: mat_a [M,K], mat_b [G,K,N], offs [G]
        outputs = []
        start = 0
        for group, end in enumerate(offs.tolist()):
            outputs.append(mat_a[start:end] @ mat_b[group])
            start = end
        return torch.cat(outputs, dim=0)


def check_fused_dispatcher_still_uses_grouped_mm():
    inputs, weights = _fixture(requires_grad=False)
    stub = _CapturingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        actual = cuda_gmm.grouped_matmul_cuda(inputs, M_SPLIT, weights)
        matched = torch.allclose(actual, _reference_forward(inputs, weights))
        used = len(stub.calls) == 1 and stub.calls[0] == GROUP_ENDS
    finally:
        _restore(original)
    check(matched and used, "无 sink, vmm_safe=False 走 fused grouped_mm")


def main() -> int:
    check_forward_skips_grouped_mm()
    check_empty_group_is_zero()
    check_backward_matches_reference()
    check_sink_disables_fused_weight_grad()
    check_sink_without_vmm_safe_uses_fused_grouped_mm()
    check_fused_dispatcher_still_uses_grouped_mm()

    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
