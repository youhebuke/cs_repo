import os
import logging
from datetime import datetime

import torch

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

# torch.distributed.checkpoint (DCP) — available in PyTorch >= 2.2.
try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
    from torch.distributed.checkpoint.stateful import Stateful
    _HAS_DCP = True
except ImportError:
    _HAS_DCP = False


def dataset_fingerprint(data_cfg) -> str:
    """Build a string that uniquely identifies the training dataset.

    Two configs that produce the same fingerprint are considered to refer
    to the same dataset, making it safe to restore ``global_step`` from
    a checkpoint.
    """
    if data_cfg is None:
        return ""
    return f"{data_cfg.dataset_path}|{data_cfg.dataset_config}|{data_cfg.split}|{data_cfg.batch_size}"


# ------------------------------------------------------------------
# Stateful wrappers for DCP
# ------------------------------------------------------------------

if _HAS_DCP:

    class _AppState(Stateful):
        """Stateful wrapper for model + optimizer + scheduler + metadata.

        DCP automatically calls ``state_dict`` on save and
        ``load_state_dict`` on load, so all application state is
        handled through this single object.
        """

        def __init__(self, model, optimizer, scheduler, global_step=0,
                     data_cfg=None, dataloader=None, strict=True, consumed_samples=0):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.global_step = global_step
            self.data_cfg = data_cfg
            self.dataloader = dataloader
            self.strict = strict
            self.consumed_samples = consumed_samples

        def state_dict(self):
            result = {}

            # Model + optimizer via get_state_dict (handles FSDP FQNs
            # and sharded state dicts automatically).
            if self.model is not None:
                model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
                result["model"] = model_sd
                result["optim"] = optim_sd

            # Scheduler (plain state dict).
            if self.scheduler is not None:
                result["scheduler"] = self.scheduler.state_dict()

            # Metadata.
            result["global_step"] = self.global_step
            result["consumed_samples"] = self.consumed_samples
            result["dataset_fingerprint"] = dataset_fingerprint(self.data_cfg)

            if (self.dataloader is not None
                    and hasattr(self.dataloader, "sampler")
                    and hasattr(self.dataloader.sampler, "state_dict")):
                sampler_state = self.dataloader.sampler.state_dict()
                result["dataloader_state_dict"] = {
                    k: v for k, v in sampler_state.items()
                    if isinstance(v, (int, float, str, bool, list, dict, torch.Tensor))
                }

            return result

        def load_state_dict(self, state_dict):
            # Model + optimizer via set_state_dict (applies in-place,
            # correctly handling FSDP sharded parameters).
            if self.model is not None:
                set_state_dict(
                    self.model,
                    self.optimizer,
                    model_state_dict=state_dict.get("model", {}),
                    optim_state_dict=state_dict.get("optim", {}),
                    options=StateDictOptions(strict=self.strict)
                )

            # Scheduler.
            if self.scheduler is not None and state_dict.get("scheduler"):
                self.scheduler.load_state_dict(state_dict["scheduler"])

            # Metadata.
            if "global_step" in state_dict:
                self.global_step = int(state_dict["global_step"])
            self.consumed_samples = int(state_dict.get("consumed_samples", 0))
            self.dataset_fingerprint = state_dict.get("dataset_fingerprint", "")
            self.dataloader_state_dict = state_dict.get("dataloader_state_dict", None)


class Checkpointer:
    """Checkpoint manager using ``torch.distributed.checkpoint`` for
    distributed saves/loads.

    All application state (model, optimizer, scheduler, global_step,
    dataset fingerprint, dataloader state) is wrapped in a single
    ``Stateful`` object so DCP handles it as one unit.

    Falls back to ``torch.save``/``torch.load`` on rank 0 only when
    DCP is not available (e.g. PyTorch < 2.2).
    """

    def __init__(self, output_dir: str, save_optim: bool = True, load_optim: bool = True):
        self.output_dir = output_dir
        self.save_optim = save_optim
        self.load_optim = load_optim

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, model, optimizer, scheduler, global_step, data_cfg=None, dataloader=None,
             consumed_samples=0):
        """Save training checkpoint.

        When DCP is available, every rank participates in the save
        (each writes its own shard).  All state is wrapped in a
        ``Stateful`` object and saved in a single ``dcp.save`` call.

        When DCP is not available, falls back to rank-0-only
        ``torch.save``.
        """
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = os.path.join(self.output_dir, f"{current_time}-checkpoint-{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if _HAS_DCP:
            self._save_dcp(ckpt_dir, model, optimizer, scheduler, global_step, data_cfg, dataloader,
                           consumed_samples)
        else:
            self._save_torch(ckpt_dir, model, optimizer, scheduler, global_step, data_cfg, dataloader,
                             consumed_samples)

    def _save_dcp(self, ckpt_dir, model, optimizer, scheduler, global_step, data_cfg, dataloader,
                  consumed_samples=0):
        """Save using ``torch.distributed.checkpoint`` (all ranks)."""
        optim_to_save = optimizer if self.save_optim else None
        state_dict = {"app": _AppState(model, optim_to_save, scheduler, global_step, data_cfg, dataloader,
                                       consumed_samples=consumed_samples)}
        dcp.save(state_dict, checkpoint_id=ckpt_dir)
        print_rank(logger.info, f"DCP checkpoint saved to {ckpt_dir}")

    def _save_torch(self, ckpt_dir, model, optimizer, scheduler, global_step, data_cfg, dataloader,
                    consumed_samples=0):
        """Fallback save using ``torch.save`` (rank 0 only)."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        state = {"global_step": global_step}
        state["consumed_samples"] = consumed_samples
        state["dataset_fingerprint"] = dataset_fingerprint(data_cfg)

        if dataloader is not None and hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "state_dict"):
            state["dataloader_state_dict"] = dataloader.sampler.state_dict()

        if model is not None:
            state["model_state_dict"] = model.state_dict()
        if optimizer is not None and self.save_optim:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        ckpt_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save(state, ckpt_path)
        print_rank(logger.info, f"Checkpoint saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, checkpoint_path, model, optimizer, scheduler, strict: bool = True):
        """Load training checkpoint.

        When DCP is available, every rank participates in the load
        (each reads its own shard).  The ``_AppState`` wrapper's
        ``load_state_dict`` is called automatically by DCP, which
        applies model/optimizer/scheduler state via
        ``set_state_dict``.

        When DCP is not available, falls back to rank-0-only
        ``torch.load``.

        Args:
            checkpoint_path: Path to the checkpoint directory/file.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: LR scheduler to load state into.
            strict: If True, ``model.load_state_dict`` requires exact
                key match.  If False, missing keys are ignored (useful
                when loading a checkpoint trained with a different
                model architecture or partial checkpoint).

        Returns:
            A dict with saved metadata (``global_step``,
            ``dataset_fingerprint``, ``dataloader_state_dict``, etc.)
            so the caller can decide whether to restore ``global_step``.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if _HAS_DCP:
            return self._load_dcp(checkpoint_path, model, optimizer, scheduler, strict=strict)
        else:
            return self._load_torch(checkpoint_path, model, optimizer, scheduler, strict=strict)

    def _load_dcp(self, checkpoint_path, model, optimizer, scheduler, strict: bool = True):
        """Load using ``torch.distributed.checkpoint`` (all ranks)."""
        optim_to_load = optimizer if self.load_optim else None
        app = _AppState(model, optim_to_load, scheduler, strict=strict)
        state_dict = {"app": app}
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        print_rank(logger.info, f"DCP checkpoint loaded from {checkpoint_path}")
        return {
            "global_step": app.global_step,
            "consumed_samples": app.consumed_samples,
            "dataset_fingerprint": app.dataset_fingerprint,
            "dataloader_state_dict": app.dataloader_state_dict,
        }

    def _load_torch(self, checkpoint_path, model, optimizer, scheduler, strict: bool = True):
        """Fallback load using ``torch.load`` (rank 0 only)."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if model is not None and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in state and self.load_optim:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        print_rank(logger.info, f"Checkpoint loaded from {checkpoint_path}")

        return {
            "global_step": state.get("global_step", 0),
            "consumed_samples": state.get("consumed_samples", 0),
            "dataset_fingerprint": state.get("dataset_fingerprint", ""),
            "dataloader_state_dict": state.get("dataloader_state_dict", None),
        }