import os
import logging
from typing import Optional

import torch

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import set_log_level, print_rank
from fsdp_turbo.training.checkpoint import Checkpointer
from fsdp_turbo.training.monitor import Monitor

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer with common training lifecycle.

    Subclasses override builder methods to customize component creation.
    Default flow: init distributed -> build components -> train.

    Attributes:
        model: The training model (after parallelism wrapping).
        tokenizer: The tokenizer instance.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler.
        data_manager: The data loader / manager.
        config: The training configuration.
    """

    def __init__(self, config: Optional[FSDPTurboConfig] = None):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self._ckpt_manager = None
        self._monitor = None
        self._global_step = 0
        self._consumed_samples = 0
        self._accumulated_losses = {"loss": [], "aux_loss": []}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Build all components. Call before train()."""
        self._init_distributed()
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_scheduler()
        self.dataloader = self.build_dataloader()
        self._ckpt_manager = self.build_checkpointer()
        self._monitor = self.build_monitor()

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        print_rank(logger.info, "\n"+ str(self.config))

        self._prepare_training(resume_from_checkpoint)
        max_steps = self._resolve_max_steps()

        print_rank(logger.info, "***** Running training *****")
        print_rank(logger.info, f"  Max steps = {max_steps}")
        print_rank(logger.info, f"  Gradient accumulation steps = {self.config.run.gradient_accumulation_steps}")

        with self._create_profiler() as prof:
            # Compute start epoch and skip batches from consumed_samples.
            dataset_size = len(self.dataloader.dataset) if self.dataloader is not None else 0
            samples_per_epoch = dataset_size if dataset_size > 0 else 0
            if self._consumed_samples > 0 and samples_per_epoch > 0:
                start_epoch = self._consumed_samples // samples_per_epoch
                skip_batches = (self._consumed_samples % samples_per_epoch) // self.config.data.batch_size
                print_rank(logger.info, f"  Resuming from consumed_samples={self._consumed_samples}: "
                                        f"start_epoch={start_epoch}, skip_batches={skip_batches}")
            else:
                start_epoch = 0
                skip_batches = 0

            for epoch in range(self.config.run.num_train_epochs):
                print_rank(logger.info, f"  Epoch {epoch + 1}/{self.config.run.num_train_epochs}")
                # Set epoch on DistributedSampler for proper shuffling across epochs.
                if self.dataloader is not None and hasattr(self.dataloader, "sampler") \
                        and hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(epoch)

                # Skip epochs and batches we already consumed.
                if epoch < start_epoch:
                    continue
                for step, batch in enumerate(self.dataloader):
                    if epoch == start_epoch and step < skip_batches:
                        continue
                    self._monitor.on_step_start()
                    batch = self.prepare_batch(batch)
                    loss, aux_loss = self.train_step(batch)
                    self._accumulated_losses["loss"].append(loss.detach())
                    if aux_loss is not None:
                        self._accumulated_losses["aux_loss"].append(aux_loss.detach())

                    # Track how many samples have been processed.
                    if isinstance(batch, dict):
                        batch_size = next((v.shape[0] for v in batch.values()
                                          if isinstance(v, torch.Tensor)), 1)
                    elif isinstance(batch, (list, tuple)):
                        batch_size = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else 1
                    else:
                        batch_size = 1
                    self._consumed_samples += batch_size

                    if (step + 1) % self.config.run.gradient_accumulation_steps == 0:
                        synced = self.sync_loss(self._accumulated_losses)
                        loss = synced["loss"]
                        aux_loss = synced.get("aux_loss")
                        grad_norm = self._on_optimizer_step()
                        for v in self._accumulated_losses.values():
                            v.clear()
                        self._global_step += 1
                        self._on_log_step(loss, grad_norm, aux_loss)
                        self._on_checkpoint_step()
                        if prof is not None:
                            prof.step()

                    if 0 < max_steps <= self._global_step:
                        break
                if 0 < max_steps <= self._global_step:
                    break

        print_rank(logger.info, f"Training complete. Total steps: {self._global_step}")
        self._monitor.summary()

    # ------------------------------------------------------------------
    # Training loop hooks
    # ------------------------------------------------------------------

    def _prepare_training(self, resume_from_checkpoint: Optional[str] = None):
        """Resume from checkpoint and set model to train mode."""
        self._global_step = 0
        checkpoint = resume_from_checkpoint or self.config.checkpoint.resume_from_checkpoint
        if checkpoint and self._ckpt_manager is not None:
            self._load_checkpoint(checkpoint)
        self.model.train()

        if self.dataloader is None:
            raise RuntimeError("dataloader is None. Override build_dataloader() to provide a dataloader.")

    def _on_optimizer_step(self) -> float:
        """Clip gradients, step optimizer and scheduler, zero grads.

        Returns the gradient norm before clipping (for monitoring).
        Override to inject custom logic between clipping and stepping.
        """
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm

    def _on_log_step(self, loss: torch.Tensor, grad_norm: float, aux_loss: Optional[torch.Tensor] = None):
        """Log training metrics if the logging interval is reached."""
        if self._global_step % self.config.run.logging_steps != 0:
            return
        loss_val = loss.item()
        aux_loss_val = aux_loss.item() if aux_loss is not None else None
        if self._monitor is not None:
            self._monitor.log(self._global_step, loss_val, grad_norm, aux_loss=aux_loss_val)
        else:
            parts = [f"Step {self._global_step} | Loss: {loss_val:.6f}"]
            if aux_loss_val is not None:
                parts.append(f"Aux Loss: {aux_loss_val:.6f}")
            print_rank(logger.info, f"  {' | '.join(parts)}")

    def _on_checkpoint_step(self):
        """Save checkpoint if the save interval is reached."""
        if self._ckpt_manager is not None and self._global_step % self.config.checkpoint.save_steps == 0:
            self._save_checkpoint()

    def train_step(self, batch):
        """Execute a single training step: forward + backward."""
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else None
        return loss, aux_loss

    @classmethod
    def sync_loss(cls, accumulated_losses: dict) -> dict:
        """Synchronize accumulated losses across the CDP communication domain.

        For each key in *accumulated_losses*, sums the list of detached loss
        tensors locally, then performs an all-reduce average across ranks in
        the CDP (Context Data Parallel) group.  When CDP is not enabled
        (world_size == 1) the local sums are returned unchanged.

        The method is generic and supports any number of named losses (e.g.
        "loss", "aux_loss", or additional custom losses).

        Args:
            accumulated_losses: Dict mapping loss names to lists of detached
                loss tensors accumulated across gradient-accumulation
                micro-steps.

        Returns:
            Dict mapping the same loss names to scalar tensors with the
            CDP-averaged sums.  Keys whose list is empty map to
            None.
        """
        from fsdp_turbo.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()

        cp_enabled = parallel_state.is_cp_enable()
        if cp_enabled:
            cp_group = parallel_state.get_cp_group()
            cp_world_size = parallel_state.get_cp_group_size()

        result = {}
        for name, losses in accumulated_losses.items():
            if not losses:
                result[name] = None
                continue
            summed = torch.stack(losses).sum()
            if cp_enabled:
                synced = summed.clone()
                torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM, group=cp_group)
                synced.div_(cp_world_size)
                result[name] = synced
            else:
                result[name] = summed
        return result

    # ------------------------------------------------------------------
    # Distributed initialization
    # ------------------------------------------------------------------

    def _init_distributed(self):
        """Initialize distributed backend, process group, seed, and logging."""
        if torch.distributed.is_initialized():
            return

        from fsdp_turbo.utils.device import set_accelerator_compatible
        backend = set_accelerator_compatible()
        set_log_level("INFO")

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(local_rank)
        torch.accelerator.set_device(local_rank)

        self._set_seed()

    def _set_seed(self):
        """Set random seed. Override to customize seed source."""
        from fsdp_turbo.utils.random import set_seed
        set_seed(self.config.run.seed, set_deterministic=self.config.run.deterministic)

    # ------------------------------------------------------------------
    # Component builders - override in subclasses
    # ------------------------------------------------------------------

    def build_tokenizer(self):
        """Build tokenizer using TokenizerFactory."""
        if not self.config.model.tokenizer_name_or_path:
            print_rank(logger.warning, "tokenizer_name_or_path is empty, skipping tokenizer build.")
            return None
        from fsdp_turbo.training.factories import TokenizerFactory
        print_rank(logger.info, "> Building Tokenizer...")
        return TokenizerFactory.create(self.config.model.tokenizer_name_or_path)

    def build_model(self):
        """Build model using ModelFactory."""
        if not self.config.model.model_name_or_path:
            print_rank(logger.warning, "model_name_or_path is empty, skipping model build.")
            return None
        from fsdp_turbo.training.factories import ModelFactory
        print_rank(logger.info, "> Building Model...")
        return ModelFactory.create(
            self.config.model.model_name_or_path,
            torch_dtype=get_dtype(self.config.model.torch_dtype),
        )

    def build_optimizer(self):
        """Build optimizer using OptimizerFactory."""
        if self.model is None:
            print_rank(logger.warning, "Model is None, skipping optimizer build.")
            return None
        print_rank(logger.info, "> Building Optimizer...")
        from fsdp_turbo.training.factories import OptimizerFactory
        print(self.config.optimizer.lr, type(self.config.optimizer.lr))
        return OptimizerFactory.create(
            self.config.optimizer.optimizer_type,
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

    def build_scheduler(self):
        """Build LR scheduler using SchedulerFactory."""
        if self.optimizer is None:
            print_rank(logger.warning, "Optimizer is None, skipping scheduler build.")
            return None
        print_rank(logger.info, "> Building LR Scheduler...")
        from fsdp_turbo.training.factories import SchedulerFactory
        max_steps = self._resolve_max_steps()
        return SchedulerFactory.create(
            self.config.optimizer.lr_scheduler_type,
            self.optimizer,
            T_max=max_steps,
            eta_min=self.config.optimizer.min_lr,
        )

    def build_dataloader(self):
        """Build dataloader from HuggingFace datasets."""
        from fsdp_turbo.training.dataloader import build_dataloader
        return build_dataloader(self.config.data, tokenizer=self.tokenizer)

    def build_checkpointer(self):
        """Build checkpoint manager."""
        print_rank(logger.info, "> Building Checkpointer...")
        return Checkpointer(
            self.config.checkpoint.output_dir,
            save_optim=self.config.checkpoint.save_optim,
            load_optim=self.config.checkpoint.load_optim,
        )

    def build_monitor(self):
        """Build training monitor."""
        print_rank(logger.info, "> Building Monitor...")
        return Monitor(logging_steps=self.config.run.logging_steps)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_profiler(self):
        from fsdp_turbo.training.profiler import create_profiler
        return create_profiler(self.config.run.profile)

    def _resolve_max_steps(self):
        if self.config.run.max_steps > 0:
            return self.config.run.max_steps
        if self.dataloader is not None:
            try:
                num_batches = len(self.dataloader)
                return num_batches * self.config.run.num_train_epochs // self.config.run.gradient_accumulation_steps
            except TypeError:
                pass
        return 100000

    def prepare_batch(self, batch):
        """Move batch to device and optionally split along sequence dim for context parallelism."""
        batch = self._move_batch_to_device(batch)
        batch = self._split_batch_for_kv_allgather(batch)
        batch = self._split_batch_for_ulysses(batch)
        return batch

    def _move_batch_to_device(self, batch):
        device = torch.accelerator.current_accelerator()
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def _split_batch_for_ulysses(self, batch):
        """Split batch tensors along the sequence dimension for Ulysses context parallelism."""
        if self.config.distributed.ulysses_parallel_size <= 1:
            return batch
        from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_split_batch_data
        from fsdp_turbo.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()
        return ulysses_split_batch_data(batch, parallel_state.get_ulysses_group())

    def _split_batch_for_kv_allgather(self, batch):
        """Split batch tensors along the sequence dimension for KV all-gather context parallelism."""
        if self.config.distributed.kv_allgather_parallel_size <= 1:
            return batch
        from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather_split_batch_data
        from fsdp_turbo.distributed.parallel_state import get_parallel_state
        parallel_state = get_parallel_state()
        return kv_allgather_split_batch_data(batch, parallel_state.get_kvallgather_group())

    def _clip_grad_norm(self):
        """Clip gradient norm and return the total norm before clipping.

        Uses clip_grad and clip_grad_norm_type from OptimizerConfig.
        Delegates to the FSDP-aware clip_grads module.
        Returns the gradient norm before clipping for monitoring.
        """
        from fsdp_turbo.training.clip_grads import clip_grad_norm
        return clip_grad_norm(
            self.model,
            max_norm=self.config.optimizer.clip_grad,
            norm_type=self.config.optimizer.clip_grad_norm_type,
            group=self._get_grad_sync_group(),
        )

    def _compute_grad_norm(self):
        """Compute the total gradient norm without clipping."""
        from fsdp_turbo.training.clip_grads import compute_grad_norm
        return compute_grad_norm(
            list(self.model.parameters()),
            norm_type=self.config.optimizer.clip_grad_norm_type,
            group=self._get_grad_sync_group(),
        )

    def _get_grad_sync_group(self):
        """Return the process group for gradient norm all-reduce.

        Override this in subclasses to provide a specific group (e.g.
        the data-parallel group).  Returns ``None`` by default, which
        lets ``clip_grads`` infer the group from DTensor gradients
        automatically.
        """
        return None

    def _save_checkpoint(self):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "save"):
            self._ckpt_manager.save(
                self.model, self.optimizer, self.lr_scheduler, self._global_step,
                data_cfg=self.config.data, dataloader=self.dataloader,
                consumed_samples=self._consumed_samples,
            )

    def _load_checkpoint(self, checkpoint_path):
        if self._ckpt_manager is not None and hasattr(self._ckpt_manager, "load"):
            ckpt_info = self._ckpt_manager.load(
                checkpoint_path, self.model, self.optimizer, self.lr_scheduler,
                strict=self.config.checkpoint.strict,
            )
            # Restore global_step only if the dataset hasn't changed.
            saved_fp = ckpt_info.get("dataset_fingerprint", "")
            current_fp = None
            if self.config is not None and self.config.data is not None:
                from fsdp_turbo.training.checkpoint import dataset_fingerprint
                current_fp = dataset_fingerprint(self.config.data)

            if saved_fp and saved_fp == current_fp:
                self._global_step = ckpt_info.get("global_step", 0)
                self._consumed_samples = ckpt_info.get("consumed_samples", 0)
                print_rank(logger.info, f"Restored global_step to {self._global_step}, "
                                        f"consumed_samples to {self._consumed_samples} (dataset unchanged)")
            else:
                self._global_step = 0
                self._consumed_samples = 0
                print_rank(logger.warning, "Dataset changed since checkpoint; global_step and consumed_samples reset")

            # Restore dataloader sampler state if available.
            dl_state = ckpt_info.get("dataloader_state_dict")
            if dl_state is not None and self.dataloader is not None:
                if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "load_state_dict"):
                    self.dataloader.sampler.load_state_dict(dl_state)
