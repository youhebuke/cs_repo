#!/usr/bin/env python3
"""Verify OPT-2: the grouped matmul writes weight gradients into the buffer.

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_opt2_grad_weight_sink.py

Expected output (exit code 0):

    [1/8] PASS  sink 结果与纯 autograd 参考逐位一致, weights 未进计算图
    [2/8] PASS  FP32 目标 + BF16 计算: 复用单块 [out,in] staging, 无 [E+B] 分配
    [3/8] PASS  空组保持原值未被清零 (对称内存下清零会覆盖 home rank 梯度)
    [4/8] PASS  非空组落在远端行区间时报错, buffer 保持干净
    [5/8] PASS  reduce_gradient 不再拷贝, 直接消费 main_grad 并返回独立张量
    [6/8] PASS  reduce 在两个 GMM 反向之后触发, 顺序固定为 down -> gate_up
    [7/8] PASS  端到端参数梯度与纯 autograd 参考逐位一致
    [8/8] PASS  NPU 算子: 不传 sink 时与基线逐位一致; 传 sink 时只对本地 2*(E/R) 行做 GMM
    ALL PASS
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (  # noqa: E402
    MoonEPSymmetricProjection,
    moonep_bind_weight_grads,
)
from fsdp_turbo.distributed.expert_parallel.moonep_dispatcher import (  # noqa: E402
    _grouped_matmul_with_static_tail,
)
from fsdp_turbo.ops.grad_weight_sink import GradWeightSink  # noqa: E402

NUM_EXPERTS = 4
EXPERTS_PER_RANK = 2
RANK = 1
TOTAL_ROWS = NUM_EXPERTS + EXPERTS_PER_RANK
LOCAL_START = RANK * EXPERTS_PER_RANK
LOCAL_END = LOCAL_START + EXPERTS_PER_RANK
LOCAL_RANGES = ((LOCAL_START, LOCAL_END), (NUM_EXPERTS, TOTAL_ROWS))

_checks = []
_TOTAL = 8


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/{_TOTAL}] {'PASS' if passed else 'FAIL'}  {message}")


def _sink(num_groups, out_features, in_features, row_ranges, dtype=torch.float32):
    return GradWeightSink(
        buffer=torch.zeros(num_groups, out_features, in_features, dtype=dtype),
        row_ranges=row_ranges,
    )


# --------------------------------------------------------------------------
# 1-4: the sink contract
# --------------------------------------------------------------------------
def check_matches_autograd_reference():
    offsets = torch.tensor([0, 4, 7], dtype=torch.int32)
    inputs = torch.randn(9, 8, requires_grad=True)
    weights = torch.randn(3, 16, 8)

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

    check(
        torch.equal(actual, reference)
        and torch.equal(inputs.grad, reference_inputs.grad)
        and torch.equal(sink.buffer, reference_weights.grad)
        and weights.grad is None,
        "sink 结果与纯 autograd 参考逐位一致, weights 未进计算图",
    )


def check_mixed_dtype_uses_one_staging_buffer():
    offsets = torch.tensor([2, 5], dtype=torch.int32)
    inputs = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(2, 6, 4, dtype=torch.bfloat16)

    sink = _sink(2, 6, 4, row_ranges=((0, 2),), dtype=torch.float32)
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    output.backward(torch.randn_like(output))

    check(
        sink.buffer.dtype == torch.float32
        and sink.buffer.abs().sum() > 0
        and sink.scratch is not None
        and tuple(sink.scratch.shape) == (6, 4),
        "FP32 目标 + BF16 计算: 复用单块 [out,in] staging, 无 [E+B] 分配",
    )


def check_empty_groups_are_left_alone():
    offsets = torch.tensor([0, 5], dtype=torch.int32)
    inputs = torch.randn(5, 4, requires_grad=True)
    weights = torch.randn(2, 6, 4)

    sink = _sink(2, 6, 4, row_ranges=((1, 2),))
    sink.buffer[0].fill_(7.0)
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    output.backward(torch.randn_like(output))

    check(
        torch.equal(sink.buffer[0], torch.full((6, 4), 7.0))
        and sink.buffer[1].abs().sum() > 0,
        "空组保持原值未被清零 (对称内存下清零会覆盖 home rank 梯度)",
    )


def check_remote_rows_are_rejected():
    offsets = torch.tensor([3, 6], dtype=torch.int32)
    inputs = torch.randn(6, 4, requires_grad=True)
    weights = torch.randn(2, 6, 4)

    sink = _sink(2, 6, 4, row_ranges=((1, 2),))
    output = _grouped_matmul_with_static_tail(
        inputs, offsets, weights, grad_weight_sink=sink
    )
    raised = ""
    try:
        output.backward(torch.randn_like(output))
    except RuntimeError as error:
        raised = str(error)

    check(
        "owned by a remote rank" in raised and sink.buffer[0].abs().sum() == 0,
        "非空组落在远端行区间时报错, buffer 保持干净",
    )


# --------------------------------------------------------------------------
# 5-7: the adapter side
# --------------------------------------------------------------------------
def _projection(events, name, out_features, in_features):
    projection = object.__new__(MoonEPSymmetricProjection)
    projection._grad_sink = None
    main_grad = torch.zeros(TOTAL_ROWS, out_features, in_features)
    projection.pool = SimpleNamespace(
        main_grad=main_grad,
        owner_grad_full=main_grad[:NUM_EXPERTS],
        reduce_buffers=main_grad[:NUM_EXPERTS].view(
            NUM_EXPERTS // EXPERTS_PER_RANK, EXPERTS_PER_RANK, out_features, in_features
        ),
        cuda_local_sink=False,
    )
    projection.runtime = SimpleNamespace(
        imports=SimpleNamespace(
            launch_inter_rank_sync=lambda ctx: events.append(f"sync:{name}"),
            launch_grad_reduce=lambda *a, **k: events.append(f"reduce:{name}"),
        ),
        num_experts=NUM_EXPERTS,
        rank=RANK,
        experts_per_rank=EXPERTS_PER_RANK,
        config=SimpleNamespace(num_sms=8),
    )
    return projection


_PLAN = SimpleNamespace(
    experts_to_copy=torch.full((2, EXPERTS_PER_RANK), -1, dtype=torch.int32)
)
_CONTEXT = {
    "meta_buf": torch.zeros(1, dtype=torch.int32),
    "meta_chunk_padded": 1,
    "BARRIER_OFF": 0,
    "grid_sync_bar": torch.zeros(1, dtype=torch.int32),
}


def _build_layer(events, seed=0, hidden=4, intermediate=6, tokens=6):
    torch.manual_seed(seed)
    gate_up = _projection(events, "gate_up", intermediate, hidden)
    down = _projection(events, "down", hidden, intermediate)
    gate_up_weight = torch.randn(TOTAL_ROWS, intermediate, hidden)
    down_weight = torch.randn(TOTAL_ROWS, hidden, intermediate)
    gate_up_param = torch.nn.Parameter(gate_up_weight[LOCAL_START:LOCAL_END].clone())
    down_param = torch.nn.Parameter(down_weight[LOCAL_START:LOCAL_END].clone())
    cu_seqlens = torch.tensor([0, 0, 3, 5, 5, tokens], dtype=torch.int32)
    dispatched = torch.randn(tokens, hidden, requires_grad=True)

    bound = moonep_bind_weight_grads(
        dispatched, gate_up_param, down_param, gate_up, down, _PLAN, _CONTEXT
    )
    fc1 = _grouped_matmul_with_static_tail(
        bound, cu_seqlens, gate_up_weight, grad_weight_sink=gate_up.grad_weight_sink()
    )
    output = _grouped_matmul_with_static_tail(
        torch.nn.functional.silu(fc1).contiguous(),
        cu_seqlens,
        down_weight,
        grad_weight_sink=down.grad_weight_sink(),
    )
    return SimpleNamespace(
        output=output,
        dispatched=dispatched,
        cu_seqlens=cu_seqlens,
        gate_up=gate_up,
        down=down,
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
        gate_up_param=gate_up_param,
        down_param=down_param,
    )


def check_reduce_gradient_no_longer_copies():
    events = []
    projection = _projection(events, "gate_up", 6, 4)
    projection.pool.main_grad.copy_(
        torch.arange(projection.pool.main_grad.numel(), dtype=torch.float32).reshape(
            projection.pool.main_grad.shape
        )
    )
    snapshot = projection.pool.main_grad.clone()

    local = projection.reduce_gradient(_PLAN, _CONTEXT, torch.bfloat16)

    independent = local.untyped_storage().nbytes() == (
        local.numel() * local.element_size()
    )
    check(
        torch.equal(projection.pool.main_grad, snapshot)
        and torch.equal(local, snapshot[LOCAL_START:LOCAL_END].to(torch.bfloat16))
        and local.dtype == torch.bfloat16
        and independent,
        "reduce_gradient 不再拷贝, 直接消费 main_grad 并返回独立张量",
    )


def check_reduce_runs_after_both_matmuls():
    events = []
    layer = _build_layer(events)
    written = {}
    original = MoonEPSymmetricProjection.reduce_gradient

    def recording(self, plan, comm_context, dtype):
        name = "gate_up" if self is layer.gate_up else "down"
        written[name] = self.pool.main_grad.abs().sum().item()
        return original(self, plan, comm_context, dtype)

    MoonEPSymmetricProjection.reduce_gradient = recording
    try:
        layer.output.sum().backward()
    finally:
        MoonEPSymmetricProjection.reduce_gradient = original

    check(
        written.get("down", 0) > 0
        and written.get("gate_up", 0) > 0
        and events == ["sync:down", "reduce:down", "sync:gate_up", "reduce:gate_up"],
        "reduce 在两个 GMM 反向之后触发, 顺序固定为 down -> gate_up",
    )


def check_end_to_end_gradients():
    layer = _build_layer([])
    layer.output.sum().backward()

    reference = _build_layer([], seed=0)
    gate_up_reference = reference.gate_up_weight.detach().clone().requires_grad_(True)
    down_reference = reference.down_weight.detach().clone().requires_grad_(True)
    dispatched_reference = reference.dispatched.detach().clone().requires_grad_(True)
    fc1 = _grouped_matmul_with_static_tail(
        dispatched_reference, reference.cu_seqlens, gate_up_reference
    )
    output = _grouped_matmul_with_static_tail(
        torch.nn.functional.silu(fc1).contiguous(),
        reference.cu_seqlens,
        down_reference,
    )
    output.sum().backward()

    matches = (
        torch.equal(layer.output, output)
        and torch.equal(layer.dispatched.grad, dispatched_reference.grad)
        and torch.equal(
            layer.gate_up_param.grad,
            gate_up_reference.grad[LOCAL_START:LOCAL_END],
        )
        and torch.equal(
            layer.down_param.grad, down_reference.grad[LOCAL_START:LOCAL_END]
        )
        and torch.equal(
            layer.gate_up.pool.main_grad[NUM_EXPERTS:],
            gate_up_reference.grad[NUM_EXPERTS:],
        )
    )
    check(matches, "端到端参数梯度与纯 autograd 参考逐位一致")


# --------------------------------------------------------------------------
# 8: the NPU operator, against a mocked torch_npu
# --------------------------------------------------------------------------
def _mock_torch_npu():
    wgrad_group_counts = []

    def npu_grouped_matmul(x_list, w_list, bias=None, group_list=None, split_item=None,
                           group_type=None, group_list_type=None, output_dtype=None):
        x = x_list[0]
        ends = torch.cumsum(group_list, 0).tolist()
        starts = [0] + ends[:-1]
        if split_item == 2 and group_type == 0:
            out = x.new_zeros(x.shape[0], w_list[0].shape[1])
            for group, (start, end) in enumerate(zip(starts, ends)):
                if end > start:
                    out[start:end] = x[start:end] @ w_list[group]
            return [out]
        if split_item == 3 and group_type == 2:
            wgrad_group_counts.append(int(group_list.numel()))
            grad = w_list[0]
            result = x.new_zeros(len(starts), x.shape[0], grad.shape[1])
            for group, (start, end) in enumerate(zip(starts, ends)):
                if end > start:
                    result[group] = x[:, start:end] @ grad[start:end]
            return [result if output_dtype is None else result.to(output_dtype)]
        raise RuntimeError("unexpected mock call")

    module = types.ModuleType("torch_npu")
    module.npu_grouped_matmul = npu_grouped_matmul
    module.wgrad_group_counts = wgrad_group_counts
    return module


def _load_npu_operator():
    sys.modules["torch_npu"] = _mock_torch_npu()
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "fsdp_turbo", "ops", "npu", "grouped_matmul.py",
    )
    spec = importlib.util.spec_from_file_location("verify_npu_gmm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_npu_operator():
    npu = _load_npu_operator()
    group_sizes = torch.tensor([0, 3, 2, 0, 1, 3], dtype=torch.int64)
    torch.manual_seed(0)
    inputs = torch.randn(9, 4)
    weights = torch.randn(TOTAL_ROWS, 6, 4)
    grad_output = torch.randn(9, 6)

    # Baseline: weights inside the autograd graph.
    base_inputs = inputs.clone().requires_grad_(True)
    base_weights = weights.clone().requires_grad_(True)
    base_out = npu.GroupedMatmul.apply(base_inputs, base_weights, None, group_sizes, 1, None)
    base_out.backward(grad_output)

    # With a sink: no weight gradient flows through autograd.
    sink_inputs = inputs.clone().requires_grad_(True)
    sink = _sink(TOTAL_ROWS, 6, 4, row_ranges=LOCAL_RANGES)
    sentinel = 9.0
    sink.buffer[0].fill_(sentinel)
    sys.modules["torch_npu"].wgrad_group_counts.clear()
    sink_out = npu.GroupedMatmul.apply(
        sink_inputs, weights.clone(), None, group_sizes, 1, sink
    )
    sink_out.backward(grad_output)

    local_rows_match = all(
        torch.allclose(sink.buffer[row], base_weights.grad[row], atol=1e-5)
        for start, end in LOCAL_RANGES
        for row in range(start, end)
    )
    wgrad_counts = sys.modules["torch_npu"].wgrad_group_counts
    check(
        torch.equal(base_out, sink_out)
        and torch.allclose(base_inputs.grad, sink_inputs.grad, atol=1e-6)
        and local_rows_match
        and torch.equal(sink.buffer[0], torch.full((6, 4), sentinel))
        and wgrad_counts == [2, 2],
        "NPU 算子: 不传 sink 时与基线逐位一致; 传 sink 时只对本地 2*(E/R) 行做 GMM",
    )


def main() -> int:
    torch.manual_seed(0)
    check_matches_autograd_reference()
    check_mixed_dtype_uses_one_staging_buffer()
    check_empty_groups_are_left_alone()
    check_remote_rows_are_rejected()
    check_reduce_gradient_no_longer_copies()
    check_reduce_runs_after_both_matmuls()
    check_end_to_end_gradients()
    check_npu_operator()

    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
