import logging
import time

import torch

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


class Monitor:
    """Basic training monitor that tracks loss, step time, and grad norm."""

    def __init__(self, logging_steps: int = 10):
        self.logging_steps = logging_steps
        self.loss_history = []
        self._step_start_time = None
        self._memory_backend = None

    def on_step_start(self):
        """Record the start time of a step. Call before train_step."""
        self._step_start_time = time.perf_counter()
        self._memory_backend = self._get_memory_backend()
        if self._memory_backend is not None:
            self._memory_backend.reset_peak_memory_stats()

    def log(self, step: int, loss: float, grad_norm: float = None, aux_loss: float = None):
        step_time = None
        if self._step_start_time is not None:
            step_time = time.perf_counter() - self._step_start_time
            self._step_start_time = None
        self.loss_history.append((step, loss, step_time, grad_norm))

        parts = [f"Step {step} | Loss: {loss:.6f}"]
        if grad_norm is not None:
            parts.append(f"Grad Norm: {grad_norm:.4f}")
        if aux_loss is not None:
            parts.append(f"Aux Loss: {aux_loss:.6f}")
        if step_time is not None:
            parts.append(f"Step Time: {step_time:.4f}s")
        memory_stats = self._get_memory_stats()
        if memory_stats is not None:
            allocated_gb, peak_gb = memory_stats
            parts.append(f"Memory: {allocated_gb:.4f} GB")
            parts.append(f"Peak Memory: {peak_gb:.4f} GB")

        print_rank(logger.info, f"  {' | '.join(parts)}")

    def summary(self):
        if not self.loss_history:
            return
        losses = [l for _, l, _, _ in self.loss_history]
        times = [t for _, _, t, _ in self.loss_history if t is not None]
        avg_loss = sum(losses) / len(losses)
        parts = [f"{len(self.loss_history)} steps", f"avg loss = {avg_loss:.6f}"]
        if times:
            avg_time = sum(times) / len(times)
            parts.append(f"avg step time = {avg_time:.4f}s")
        print_rank(logger.info, f"Training summary: {', '.join(parts)}")

    @staticmethod
    def _get_memory_backend():
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu
        if torch.cuda.is_available():
            return torch.cuda
        return None

    def _get_memory_stats(self):
        backend = self._memory_backend or self._get_memory_backend()
        if backend is None:
            return None
        backend.synchronize()
        return backend.memory_allocated() / 1024**3, backend.max_memory_allocated() / 1024**3
