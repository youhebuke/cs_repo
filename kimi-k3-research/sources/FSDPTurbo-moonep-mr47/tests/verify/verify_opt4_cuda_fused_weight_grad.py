#!/usr/bin/env python3
"""Verify OPT-4: CUDA computes the weight gradient with one fused GEMM per range.

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_opt4_cuda_fused_weight_grad.py

Expected output (exit code 0):

    [1/5] PASS  融合路径结果与逐 group 参考一致
    [2/5] PASS  BF16 输入直出 FP32, 未使用 staging
    [3/5] PASS  只对本地 row range 发起计算, 临时张量为 [E/R, out, in]
    [4/5] PASS  算子不支持时回退到逐 group, 结果一致且只探测一次
    [5/5] PASS  远端行在任何 GEMM 之前就被拒绝, buffer 保持干净

CPU 上用模拟的 grouped_mm 覆盖融合与回退两条路径。装有 CUDA 时
tests/unit/test_moonep_cuda_weight_grad.py 里另有一条真实设备用例。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsdp_turbo.ops.cuda import grouped_matmul as cuda_gmm  # noqa: E402
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink  # noqa: E402

# Groups 0 and 3 are empty; their rows belong to remote ranks.
GROUP_ENDS = [0, 4, 9, 9, 13]
LOCAL_RANGES = ((1, 3), (4, 5))
OUT_FEATURES, IN_FEATURES = 6, 4

_checks = []
_TOTAL = 5


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/{_TOTAL}] {'PASS' if passed else 'FAIL'}  {message}")


def _fixture(dtype=torch.float32):
    torch.manual_seed(0)
    grad_output = torch.randn(GROUP_ENDS[-1], OUT_FEATURES, dtype=dtype)
    inputs = torch.randn(GROUP_ENDS[-1], IN_FEATURES, dtype=dtype)
    sink = GradWeightSink(
        buffer=torch.zeros(len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES),
        row_ranges=LOCAL_RANGES,
    )
    return grad_output, inputs, sink


def _reference(grad_output, inputs):
    expected = torch.zeros(len(GROUP_ENDS), OUT_FEATURES, IN_FEATURES)
    start = 0
    for group, end in enumerate(GROUP_ENDS):
        if end != start:
            expected[group] = (
                grad_output[start:end].float().transpose(0, 1) @ inputs[start:end].float()
            )
        start = end
    return expected


class _RecordingGroupedMm:
    """CPU stand-in that records the shape of every fused call."""

    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def __call__(self, mat_a, mat_b, *, offs, out_dtype=None, bias=None):
        self.calls.append((tuple(mat_a.shape), tuple(mat_b.shape), offs.tolist()))
        if self.fail:
            raise RuntimeError("grouped_mm is unsupported on this device")
        outputs = []
        start = 0
        for end in offs.tolist():
            outputs.append(mat_a[:, start:end] @ mat_b[start:end])
            start = end
        result = torch.stack(outputs)
        return result if out_dtype is None else result.to(out_dtype)


def _with_grouped_mm(stub):
    original = cuda_gmm.F.grouped_mm
    cuda_gmm.F.grouped_mm = stub
    cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD = None
    return original


def _restore(original):
    cuda_gmm.F.grouped_mm = original
    cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD = None


def check_fused_matches_reference():
    grad_output, inputs, sink = _fixture()
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
        used_fused = cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD is True
    finally:
        _restore(original)
    check(
        used_fused and torch.allclose(sink.buffer, _reference(grad_output, inputs)),
        "融合路径结果与逐 group 参考一致",
    )


def check_bf16_inputs_emit_fp32_directly():
    grad_output, inputs, sink = _fixture(dtype=torch.bfloat16)
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
    finally:
        _restore(original)
    check(
        sink.buffer.dtype == torch.float32
        and sink.scratch is None
        and torch.allclose(
            sink.buffer, _reference(grad_output, inputs), rtol=2e-2, atol=2e-2
        ),
        "BF16 输入直出 FP32, 未使用 staging",
    )


def check_only_local_ranges_are_computed():
    grad_output, inputs, sink = _fixture()
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    try:
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
    finally:
        _restore(original)

    rows_per_range = [end - start for start, end in LOCAL_RANGES]
    computed_rows = [len(offsets) for _, _, offsets in stub.calls]
    # Tokens 0..9 back rows [1,3); tokens 9..13 back row [4,5).
    spans = [(mat_a[1], mat_b[0]) for mat_a, mat_b, _ in stub.calls]
    check(
        len(stub.calls) == len(LOCAL_RANGES)
        and computed_rows == rows_per_range
        and spans == [(9, 9), (4, 4)],
        f"只对本地 row range 发起计算, 临时张量为 [E/R, out, in]",
    )


def check_fallback_is_probed_once():
    grad_output, inputs, sink = _fixture()
    stub = _RecordingGroupedMm(fail=True)
    original = _with_grouped_mm(stub)
    try:
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
        fell_back = cuda_gmm._SUPPORTS_FUSED_WEIGHT_GRAD is False
    finally:
        _restore(original)
    check(
        fell_back
        and len(stub.calls) == 1
        and torch.allclose(sink.buffer, _reference(grad_output, inputs)),
        "算子不支持时回退到逐 group, 结果一致且只探测一次",
    )


def check_remote_rows_rejected_before_any_gemm():
    grad_output, inputs, sink = _fixture()
    sink.row_ranges = ((4, 5),)  # groups 1 and 2 now belong to a remote rank
    stub = _RecordingGroupedMm()
    original = _with_grouped_mm(stub)
    raised = ""
    try:
        cuda_gmm.write_weight_grads_cuda(sink, grad_output, inputs, GROUP_ENDS)
    except RuntimeError as error:
        raised = str(error)
    finally:
        _restore(original)
    check(
        "owned by a remote rank" in raised
        and not stub.calls
        and sink.buffer.abs().sum() == 0,
        "远端行在任何 GEMM 之前就被拒绝, buffer 保持干净",
    )


def main() -> int:
    check_fused_matches_reference()
    check_bf16_inputs_emit_fp32_directly()
    check_only_local_ranges_are_computed()
    check_fallback_is_probed_once()
    check_remote_rows_rejected_before_any_gemm()

    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
