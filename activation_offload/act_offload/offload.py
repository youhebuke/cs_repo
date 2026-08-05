"""Activation CPU-offload manager.

A from-scratch reproduction of the *offload storage policy* from Kimi K3's
"unified activation manager": every tensor saved for the backward pass is routed
through a policy layer; large activations are asynchronously copied to pinned CPU
memory on a side stream during the forward pass (freeing GPU HBM before backward),
and copied back on demand during the backward pass. This lowers the peak activation
memory that dominates HBM usage for long-context / deep models.

Design goals (mirroring the report):
  * Decoupled from model code -> installed via ``torch.autograd.graph.saved_tensors_hooks``
    so any ``nn.Module`` region can be wrapped without edits.
  * Tensor-granularity policy -> a size threshold + a parameter filter decide what to offload.
  * Overlap -> D2H copies run on a dedicated CUDA stream; a bounded number of in-flight
    copies act as a double buffer so extra resident memory stays small.
  * Pinned buffer pool -> reused across steps to avoid allocation churn.

The mechanism is device-agnostic for *correctness*; the *memory saving* is realized on
CUDA (host<->device move). A CPU "emulation" mode is provided purely to unit-test the
autograd correctness of the pack/unpack round-trip on machines without a GPU.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Set, Tuple

import torch

_HANDLE_TAG = "__act_offload_cpu__"


class _PinnedPool:
    """Reuse pinned (page-locked) CPU byte buffers keyed by size to avoid churn."""

    def __init__(self, pin: bool = True):
        self._free: Dict[int, List[torch.Tensor]] = {}
        self._pin = pin

    def get(self, nbytes: int) -> torch.Tensor:
        bucket = self._free.get(nbytes)
        if bucket:
            return bucket.pop()
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=self._pin)

    def put(self, buf: torch.Tensor) -> None:
        self._free.setdefault(buf.numel(), []).append(buf)


class ActivationOffloadManager:
    """Packs large saved activations to pinned CPU memory and restores them on unpack.

    Args:
        min_bytes: only tensors at least this large are offloaded (skip tiny norms/biases).
        param_ptrs: ``data_ptr()`` set of model parameters, which are skipped (offloading
            persistent weights is wasteful and re-fetched every step).
        enabled: master switch; when False the hooks are pass-through.
        max_inflight: max number of not-yet-completed D2H copies kept resident at once
            (this is the "double buffer" that bounds extra memory while overlapping).
        emulate_cpu: for CPU-only testing -- treat CPU float tensors as offloadable and
            move them to a detached clone, exercising the exact pack/unpack path.
    """

    def __init__(
        self,
        min_bytes: int = 1 << 20,
        param_ptrs: Optional[Set[int]] = None,
        enabled: bool = True,
        max_inflight: int = 2,
        emulate_cpu: bool = False,
    ):
        self.min_bytes = int(min_bytes)
        self.param_ptrs = param_ptrs or set()
        self.enabled = enabled
        self.max_inflight = max(1, int(max_inflight))
        self.emulate_cpu = emulate_cpu

        self._cuda = torch.cuda.is_available()
        self._copy_stream = torch.cuda.Stream() if self._cuda else None
        self._pool = _PinnedPool(pin=self._cuda)
        # in-flight D2H copies: (event, source_tensor_ref) -- ref keeps GPU mem alive
        # until the copy completes, after which dropping it frees the HBM.
        self._pending: List[Tuple[torch.cuda.Event, torch.Tensor]] = []
        self.stats = {
            "offloaded_bytes": 0,
            "offloaded_count": 0,
            "kept_bytes": 0,
            "kept_count": 0,
        }

    # -- internal helpers -------------------------------------------------
    def _release_completed(self, force: bool = False) -> None:
        if not self._pending:
            return
        keep = []
        for ev, ref in self._pending:
            if force or ev.query():
                # drop `ref` -> frees the GPU activation storage
                continue
            keep.append((ev, ref))
        self._pending = keep

    def _should_offload(self, t: torch.Tensor, nbytes: int) -> bool:
        if nbytes < self.min_bytes:
            return False
        if t.data_ptr() in self.param_ptrs:
            return False
        return True

    # -- autograd hooks ---------------------------------------------------
    def pack(self, t: torch.Tensor):
        if not self.enabled or not isinstance(t, torch.Tensor):
            return t
        nbytes = t.element_size() * t.numel()

        # CPU emulation path (correctness testing only).
        if self.emulate_cpu and not t.is_cuda:
            if not t.is_floating_point() or not self._should_offload(t, nbytes):
                self.stats["kept_bytes"] += nbytes
                self.stats["kept_count"] += 1
                return t
            self.stats["offloaded_bytes"] += nbytes
            self.stats["offloaded_count"] += 1
            # a distinct detached tensor holding identical values
            return (_HANDLE_TAG, "emu", t.detach().clone())

        if not t.is_cuda or not self._should_offload(t, nbytes):
            self.stats["kept_bytes"] += nbytes
            self.stats["kept_count"] += 1
            return t

        cur = torch.cuda.current_stream()
        # bound in-flight copies (double buffering): reclaim finished, then wait if needed
        self._release_completed()
        while len(self._pending) >= self.max_inflight:
            self._pending[0][0].synchronize()
            self._release_completed()

        tc = t.contiguous()  # ensure a linear byte layout to reinterpret on restore
        buf = self._pool.get(nbytes)
        cpu_typed = buf[:nbytes].view(t.dtype).view(t.shape)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(cur)  # wait for producing kernels
            cpu_typed.copy_(tc, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
        # keep tc alive until the D2H copy completes, then it is freed
        self._pending.append((ev, tc))
        self.stats["offloaded_bytes"] += nbytes
        self.stats["offloaded_count"] += 1
        return (_HANDLE_TAG, "cuda", buf, tuple(t.shape), t.dtype, t.device, nbytes)

    def unpack(self, h):
        if not (isinstance(h, tuple) and len(h) >= 2 and h[0] == _HANDLE_TAG):
            return h
        kind = h[1]
        if kind == "emu":
            return h[2]
        # kind == "cuda"
        _, _, buf, shape, dtype, device, nbytes = h
        cur = torch.cuda.current_stream()
        cpu_typed = buf[:nbytes].view(dtype).view(shape)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(cur)
            gpu = torch.empty(shape, dtype=dtype, device=device)
            gpu.copy_(cpu_typed, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
        cur.wait_event(ev)  # compute stream waits for the H2D restore
        self._pool.put(buf)  # recycle the pinned buffer
        return gpu

    def finalize(self) -> None:
        """Flush any in-flight copies (call at the end of an offload region)."""
        if self._cuda and self._pending:
            for ev, _ in self._pending:
                ev.synchronize()
        self._pending.clear()

    # -- reporting --------------------------------------------------------
    def summary(self) -> str:
        mb = 1024 * 1024
        return (
            f"offloaded {self.stats['offloaded_count']} tensors "
            f"= {self.stats['offloaded_bytes'] / mb:.1f} MiB moved to CPU; "
            f"kept {self.stats['kept_count']} tensors "
            f"= {self.stats['kept_bytes'] / mb:.1f} MiB on device"
        )


@contextlib.contextmanager
def offload_activations(
    model: Optional[torch.nn.Module] = None,
    min_bytes: int = 1 << 20,
    enabled: bool = True,
    max_inflight: int = 2,
    manager: Optional[ActivationOffloadManager] = None,
    emulate_cpu: bool = False,
):
    """Context manager: within the block, large saved activations are offloaded to CPU.

    Example::

        mgr = None
        with offload_activations(model, min_bytes=1<<20) as mgr:
            loss = model(x).sum()
            loss.backward()
        print(mgr.summary())
    """
    param_ptrs = {p.data_ptr() for p in model.parameters()} if model is not None else set()
    mgr = manager or ActivationOffloadManager(
        min_bytes=min_bytes,
        param_ptrs=param_ptrs,
        enabled=enabled,
        max_inflight=max_inflight,
        emulate_cpu=emulate_cpu,
    )
    if not enabled:
        yield mgr
        return
    with torch.autograd.graph.saved_tensors_hooks(mgr.pack, mgr.unpack):
        try:
            yield mgr
        finally:
            mgr.finalize()
