#!/usr/bin/env python3
"""Verify OPT-1: the gradient buffer is one [E+B] FP32 symmetric mapping.

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_opt1_main_grad_mapping.py

Expected output (exit code 0):

    [1/5] PASS  main_grad 形状 (E+B, out, in) = (10, 8, 4), dtype = torch.float32
    [2/5] PASS  rows [0,E) 由 4 个 rank 的 owner 分配拼成, 顺序 rank0..rank3
    [3/5] PASS  rows [E,E+B) 复用本 rank 的 reduce 分配 (与 reduce_buffers[1] 同源)
    [4/5] PASS  reduce_buffers 由 4 个 rank 的 reduce 分配拼成
    [5/5] PASS  owner_grad_full 是 main_grad[:E] 的视图, 未额外分配; close() 无残留 fd
    ALL PASS

OPT-1 只改布局，梯度数值行为不变，由 tests/unit/test_moonep_gradient_reduce.py
覆盖。把 buffer 真正接到算子上是 OPT-2 的事，见 verify_opt2_grad_weight_sink.py。
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fake_moonep import FakeSymmetricMemory  # noqa: E402

EP_SIZE = 4
EXPERTS_PER_RANK = 2
NUM_EXPERTS = EP_SIZE * EXPERTS_PER_RANK
RANK = 1
OUT_FEATURES, IN_FEATURES = 8, 4

_checks = []


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/5] {'PASS' if passed else 'FAIL'}  {message}")


def build_pool():
    from fsdp_turbo.distributed.expert_parallel import moonep_adapter

    memory = FakeSymmetricMemory()
    imports = SimpleNamespace(
        nvl_dist_alloc=memory.nvl_dist_alloc,
        nvl_release_mem_handle=memory.nvl_release_mem_handle,
        exchange_ipc_fds=memory.exchange_ipc_fds,
        nvl_dist_map=memory.nvl_dist_map,
        get_vmm_granularity=memory.get_vmm_granularity,
        nvl_multicast_supported=memory.nvl_multicast_supported,
        launch_inter_rank_sync=lambda comm_context: None,
        launch_grad_reduce=lambda *args, **kwargs: None,
    )
    runtime = SimpleNamespace(
        imports=imports,
        accelerator_type="npu",
        rank=RANK,
        size=EP_SIZE,
        num_experts=NUM_EXPERTS,
        experts_per_rank=EXPERTS_PER_RANK,
        group=None,
        config=SimpleNamespace(num_sms=8),
    )
    with mock.patch.object(moonep_adapter.dist, "barrier", lambda group=None: None):
        pool = moonep_adapter._ProjectionPool(
            runtime, "gate_up", (OUT_FEATURES, IN_FEATURES)
        )
    return moonep_adapter, memory, runtime, pool


def main() -> int:
    moonep_adapter, memory, runtime, pool = build_pool()

    check(
        tuple(pool.main_grad.shape) == (NUM_EXPERTS + EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES)
        and pool.main_grad.dtype == torch.float32,
        f"main_grad 形状 (E+B, out, in) = {tuple(pool.main_grad.shape)}, "
        f"dtype = {pool.main_grad.dtype}",
    )

    # allocations: 0 = prefetch (bf16), 1 = reduce (fp32), 2 = owner (fp32)
    reduce_chunk = memory.local_chunk_name(1)
    owner_chunk = memory.local_chunk_name(2)
    reduce_mapping, main_grad_mapping = memory.mappings

    expected_owner_rows = [
        f"peer{peer}:{owner_chunk}" if peer != RANK else owner_chunk
        for peer in range(EP_SIZE)
    ]
    check(
        main_grad_mapping.chunk_names[:EP_SIZE] == expected_owner_rows,
        f"rows [0,E) 由 {EP_SIZE} 个 rank 的 owner 分配拼成, "
        f"顺序 rank0..rank{EP_SIZE - 1}",
    )

    check(
        main_grad_mapping.chunk_names[EP_SIZE:] == [reduce_chunk]
        and main_grad_mapping.world_size == EP_SIZE + 1,
        f"rows [E,E+B) 复用本 rank 的 reduce 分配 (与 reduce_buffers[{RANK}] 同源)",
    )

    expected_reduce_rows = [
        f"peer{peer}:{reduce_chunk}" if peer != RANK else reduce_chunk
        for peer in range(EP_SIZE)
    ]
    check(
        reduce_mapping.chunk_names == expected_reduce_rows
        and reduce_mapping.world_size == EP_SIZE
        and tuple(pool.reduce_buffers.shape)
        == (EP_SIZE, EXPERTS_PER_RANK, OUT_FEATURES, IN_FEATURES),
        f"reduce_buffers 由 {EP_SIZE} 个 rank 的 reduce 分配拼成",
    )

    owner_is_view = (
        pool.owner_grad_full._base is pool.main_grad
        and pool.owner_grad_full.data_ptr() == pool.main_grad.data_ptr()
        and pool.owner_grad_full.shape[0] == NUM_EXPERTS
    )
    pool.close()
    check(
        owner_is_view and not memory.open_handles(),
        "owner_grad_full 是 main_grad[:E] 的视图, 未额外分配; close() 无残留 fd",
    )

    memory.close_all()
    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
