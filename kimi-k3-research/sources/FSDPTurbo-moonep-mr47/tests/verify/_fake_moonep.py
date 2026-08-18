"""A single-process stand-in for MoonEP's symmetric-memory primitives.

The verification scripts exercise ``_ProjectionPool`` and
``MoonEPSymmetricProjection`` without a multicast-capable accelerator. This fake
implements the handful of ``moonep`` entry points the adapter uses and records
which chunks each mapping was assembled from.

What it can check: the shape, dtype and *composition* of every mapping -- which
allocation backs which row range -- plus file-descriptor hygiene. Turning those
chunks into one address range is ``nvl_dist_map``'s job on real hardware, so the
scripts assert the composition the adapter asks for rather than re-implementing
virtual memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class Chunk:
    """One allocation, identified by a stable name for readable assertions."""

    name: str
    dtype: torch.dtype
    shape: tuple
    tensor: torch.Tensor


@dataclass
class Mapping:
    """One ``nvl_dist_map`` call."""

    chunk_names: List[str]
    dtype: torch.dtype
    chunk_shape: List[int]
    world_size: int
    local_rank: int
    tensor: torch.Tensor = field(repr=False)


class FakeSymmetricMemory:
    def __init__(self):
        self._chunks: Dict[tuple, Chunk] = {}
        self._alloc_count = 0
        self._live_handles: set[int] = set()
        self.allocations: List[Chunk] = []
        self.mappings: List[Mapping] = []

    # -- handles ----------------------------------------------------------
    def _handle_for(self, chunk: Chunk) -> int:
        # Real handles are file descriptors, so use pipe ends: the code under
        # test dups and closes them exactly as it would in production, and
        # duplicates of the same allocation share an inode.
        read_end, write_end = os.pipe()
        os.close(write_end)
        self._chunks[_identity(read_end)] = chunk
        self._live_handles.add(read_end)
        return read_end

    def _chunk_of(self, handle: int) -> Chunk:
        return self._chunks[_identity(handle)]

    # -- moonep API -------------------------------------------------------
    def nvl_dist_alloc(self, shape, dtype):
        self._alloc_count += 1
        chunk = Chunk(
            name=f"alloc{self._alloc_count}",
            dtype=dtype,
            shape=tuple(shape),
            tensor=torch.zeros(*shape, dtype=dtype),
        )
        self.allocations.append(chunk)
        return chunk.tensor, self._handle_for(chunk), self._handle_for(chunk)

    def nvl_release_mem_handle(self, handle: int) -> None:
        os.close(handle)
        self._live_handles.discard(handle)

    def exchange_ipc_fds(self, local_handle, ranks, rank, size, group):
        """Hand back one handle per rank; peers get their own simulated chunk."""
        local = self._chunk_of(local_handle)
        handles = {}
        for peer in ranks:
            if peer == rank:
                handles[peer] = os.dup(local_handle)
                self._live_handles.add(handles[peer])
                continue
            self._alloc_count += 1
            peer_chunk = Chunk(
                name=f"peer{peer}:{local.name}",
                dtype=local.dtype,
                shape=local.shape,
                tensor=torch.zeros(*local.shape, dtype=local.dtype),
            )
            handles[peer] = self._handle_for(peer_chunk)
        return handles

    def nvl_dist_map(self, chunk_shape, dtype, fds, local_rank, world_size):
        chunks = [self._chunk_of(handle) for handle in fds]
        tensor = torch.zeros(
            chunk_shape[0] * world_size, *chunk_shape[1:], dtype=dtype
        )
        self.mappings.append(
            Mapping(
                chunk_names=[chunk.name for chunk in chunks],
                dtype=dtype,
                chunk_shape=list(chunk_shape),
                world_size=world_size,
                local_rank=local_rank,
                tensor=tensor,
            )
        )
        return tensor

    def get_vmm_granularity(self):
        return 1

    def nvl_multicast_supported(self):
        return True

    def create_nvl_dist_tensor(self, chunk_shape, dtype, local_rank, world_size, group=None):
        allocation, export_handle, owned_handle = self.nvl_dist_alloc(chunk_shape, dtype)
        self.nvl_release_mem_handle(owned_handle)
        ranks = list(range(world_size))
        exchanged = self.exchange_ipc_fds(
            export_handle, ranks, local_rank, world_size, group
        )
        os.close(export_handle)
        handles = [exchanged[rank] for rank in ranks]
        try:
            return self.nvl_dist_map(
                chunk_shape, dtype, handles, local_rank, world_size
            )
        finally:
            for handle in handles:
                os.close(handle)

    # -- assertions -------------------------------------------------------
    def open_handles(self) -> List[int]:
        return sorted(handle for handle in self._live_handles if _is_open(handle))

    def close_all(self) -> None:
        for handle in list(self._live_handles):
            if _is_open(handle):
                os.close(handle)
        self._live_handles.clear()

    def local_chunk_name(self, allocation_index: int) -> str:
        return self.allocations[allocation_index].name


def _is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def _identity(fd: int) -> tuple:
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino)
