#!/usr/bin/env python3
"""Verify OPT-3: per-group weight views are cached and can be released.

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_opt3_weight_view_cache.py

Expected output (exit code 0):

    [1/5] PASS  同一权重张量重复取视图命中缓存, 只构建一次
    [2/5] PASS  同存储的新张量 (如 DTensor.to_local()) 复用同一条目
    [3/5] PASS  显式 release 后底层 storage 可被回收
    [4/5] PASS  未显式 release 时缓存有界, 不会无限增长
    [5/5] PASS  projection.close() 会在置空 full_weight 前释放视图

第 3 项是本优化的关键约束：视图通过 _base 强引用权重张量，而 PyTorch 的
GC 遍历不打破这条边，所以缓存必须显式释放，否则 MoonEP 的对称内存映射在
close() 之后仍然解不掉。第 4 项是漏调用时的兜底。
"""

import gc
import os
import sys
import weakref
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsdp_turbo.ops import weight_views  # noqa: E402

_checks = []
_TOTAL = 5


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/{_TOTAL}] {'PASS' if passed else 'FAIL'}  {message}")


def check_cache_hit():
    weight_views._CACHE.clear()
    weights = torch.randn(4, 8, 6)
    chunks, transposed = weight_views.grouped_weight_views(weights)
    again_chunks, again_transposed = weight_views.grouped_weight_views(weights)
    check(
        again_chunks is chunks
        and again_transposed is transposed
        and weight_views.cached_entry_count() == 1
        and len(chunks) == 4
        and tuple(chunks[0].shape) == (8, 6)
        and tuple(transposed[0].shape) == (6, 8)
        and torch.equal(chunks[2], weights[2]),
        "同一权重张量重复取视图命中缓存, 只构建一次",
    )


def check_same_storage_new_tensor():
    weight_views._CACHE.clear()
    weights = torch.randn(4, 8, 6)
    weight_views.grouped_weight_views(weights)
    for _ in range(8):
        weight_views.grouped_weight_views(weights.view_as(weights))
    check(
        weight_views.cached_entry_count() == 1,
        "同存储的新张量 (如 DTensor.to_local()) 复用同一条目",
    )


def check_release_frees_storage():
    weight_views._CACHE.clear()
    weights = torch.randn(8, 16, 16)
    storage = weakref.ref(weights.untyped_storage())
    weight_views.grouped_weight_views(weights)

    # Without the release the storage survives: views hold the base and the
    # cycle is not collectable.
    held = weights
    weights = None
    gc.collect()
    still_pinned = storage() is not None

    weight_views.release_weight_views(held)
    held = None
    gc.collect()
    check(
        still_pinned
        and storage() is None
        and weight_views.cached_entry_count() == 0,
        "显式 release 后底层 storage 可被回收",
    )


def check_cache_is_bounded():
    weight_views._CACHE.clear()
    original = weight_views._MAX_ENTRIES
    weight_views._MAX_ENTRIES = 4
    try:
        for _ in range(32):
            weight_views.grouped_weight_views(torch.randn(4, 8, 6))
        bounded = weight_views.cached_entry_count() == 4
    finally:
        weight_views._MAX_ENTRIES = original
        weight_views._CACHE.clear()
    check(bounded, "未显式 release 时缓存有界, 不会无限增长")


def check_projection_close_releases():
    from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (
        MoonEPSymmetricProjection,
    )

    weight_views._CACHE.clear()
    projection = object.__new__(MoonEPSymmetricProjection)
    projection.closed = False
    projection._grad_sink = None
    projection.full_weight = torch.randn(4, 8, 6)
    projection.expert_allocation = None
    weight_views.grouped_weight_views(projection.full_weight)

    before = weight_views.cached_entry_count()
    projection.close()
    check(
        before == 1 and weight_views.cached_entry_count() == 0,
        "projection.close() 会在置空 full_weight 前释放视图",
    )


def main() -> int:
    check_cache_hit()
    check_same_storage_new_tensor()
    check_release_frees_storage()
    check_cache_is_bounded()
    check_projection_close_releases()

    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
