"""Cached grouped-matmul weight views must hit across steps and be releasable."""

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.ops import weight_views


@pytest.fixture(autouse=True)
def _clear_cache():
    weight_views._CACHE.clear()
    yield
    weight_views._CACHE.clear()


def test_views_are_rebuilt_only_once_per_weight_tensor():
    weights = torch.randn(4, 8, 6)

    chunks, transposed = weight_views.grouped_weight_views(weights)
    again_chunks, again_transposed = weight_views.grouped_weight_views(weights)

    assert again_chunks is chunks
    assert again_transposed is transposed
    assert weight_views.cached_entry_count() == 1
    assert len(chunks) == 4
    assert chunks[0].shape == (8, 6)
    assert transposed[0].shape == (6, 8)
    torch.testing.assert_close(chunks[2], weights[2], rtol=0, atol=0)


def test_a_new_tensor_over_the_same_storage_reuses_the_entry():
    # ``DTensor.to_local()`` hands out a fresh Python tensor over the same
    # storage on every step; that must not add a cache entry.
    weights = torch.randn(4, 8, 6)
    weight_views.grouped_weight_views(weights)
    for _ in range(8):
        weight_views.grouped_weight_views(weights.view_as(weights))

    assert weight_views.cached_entry_count() == 1


def test_release_lets_the_weight_storage_be_freed():
    # MoonEP's full_weight is symmetric memory that MoonEPRuntime.close() must
    # unmap. Views hold it through ``_base`` and PyTorch's GC traversal does not
    # break that cycle, so the cache needs an explicit release.
    weights = torch.randn(8, 16, 16)
    storage = weakref.ref(weights.untyped_storage())
    weight_views.grouped_weight_views(weights)

    weight_views.release_weight_views(weights)
    weights = None
    gc.collect()

    assert storage() is None
    assert weight_views.cached_entry_count() == 0


def test_cache_is_bounded_even_without_release(monkeypatch):
    monkeypatch.setattr(weight_views, "_MAX_ENTRIES", 4)

    for _ in range(32):
        weight_views.grouped_weight_views(torch.randn(4, 8, 6))

    assert weight_views.cached_entry_count() == 4


def test_release_of_an_unknown_tensor_is_a_no_op():
    weight_views.release_weight_views(None)
    weight_views.release_weight_views(torch.randn(2, 4, 4))

    assert weight_views.cached_entry_count() == 0
