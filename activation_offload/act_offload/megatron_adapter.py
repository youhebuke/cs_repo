"""Megatron-LM integration helpers for the activation offload manager.

Key fact that makes integration cheap: with ``torch.autograd.graph.saved_tensors_hooks``,
the ``pack`` hook fires while saving (during forward, so the context only needs to be
active in the forward pass), while the ``unpack`` hook is bound to the saved tensor and
fires automatically during backward -- even outside the context. So we only need to wrap
the *forward*; backward restores activations on its own.

Two integration points (pick one):

1. Whole-step wrap (simplest, recommended):
   Wrap the call to Megatron's ``forward_backward_func`` (or your ``train_step``'s
   forward) in ``offload_activations(model, min_bytes=...)``. One shared manager covers
   all micro-batches of the 1F1B schedule.

2. Per-layer wrap (finer control): call ``wrap_megatron_transformer_layers(model)`` once
   after building the model; it monkeypatches each ``TransformerLayer.forward`` to run
   under a shared offload manager. Call ``manager.finalize()`` after ``backward`` each step.

This module imports Megatron lazily so importing it never hard-fails without Megatron.
"""
from __future__ import annotations

import functools
from typing import Optional

import torch

from .offload import ActivationOffloadManager


def wrap_megatron_transformer_layers(
    model: torch.nn.Module,
    min_bytes: int = 1 << 20,
    max_inflight: int = 2,
    manager: Optional[ActivationOffloadManager] = None,
) -> ActivationOffloadManager:
    """Monkeypatch every Megatron ``TransformerLayer.forward`` to offload activations.

    Returns the shared :class:`ActivationOffloadManager` (call ``.finalize()`` after
    each step's backward, and read ``.summary()`` / ``.stats`` for accounting).
    """
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except Exception as e:  # pragma: no cover - only hit without Megatron installed
        raise RuntimeError(
            "Megatron-Core not importable; use option (1) whole-step wrap instead, "
            "or install megatron.core. Original error: %r" % (e,)
        )

    param_ptrs = {p.data_ptr() for p in model.parameters()}
    mgr = manager or ActivationOffloadManager(
        min_bytes=min_bytes, param_ptrs=param_ptrs, max_inflight=max_inflight
    )

    layers = [m for m in model.modules() if isinstance(m, TransformerLayer)]
    for layer in layers:
        if getattr(layer, "_act_offload_wrapped", False):
            continue
        orig_forward = layer.forward

        @functools.wraps(orig_forward)
        def wrapped(*args, __orig=orig_forward, **kwargs):
            # pack runs here (forward); unpack is bound to the saved tensors and
            # fires during backward automatically -- no need to hold the context then.
            with torch.autograd.graph.saved_tensors_hooks(mgr.pack, mgr.unpack):
                return __orig(*args, **kwargs)

        layer.forward = wrapped  # type: ignore[assignment]
        layer._act_offload_wrapped = True

    return mgr
