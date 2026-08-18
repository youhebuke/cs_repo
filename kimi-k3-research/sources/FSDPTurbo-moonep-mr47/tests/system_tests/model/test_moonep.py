"""Train a randomly initialized Qwen3 MoE model with FSDP2 and MoonEP.

This is an accelerator system-test workload, not a pytest test. Run it on one
multicast-capable node, for example:

    torchrun --nproc-per-node=8 tests/system_tests/model/test_moonep.py

MoonEP requires PyTorch 2.9+, ``moonep==0.0.1``, BF16 expert weights, and a
multicast-capable expert-parallel group.
"""

import glob
import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import (
    DistributedConfig,
    EPPlanConfig,
    FSDPPlanConfig,
    FSDPTurboConfig,
    MoonEPConfig,
)
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import print_rank, set_log_level
from fsdp_turbo.utils.random import set_seed
from fsdp_turbo.training.clip_grads import clip_grad_norm, compute_grad_norm


logger = logging.getLogger(__name__)


# Model and dataset configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-30B-A3B")
DATASET_PATH = os.environ.get("DATASET_PATH", "wikitext")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "wikitext-2-raw-v1")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
DATASET_PARQUET_PATH = os.environ.get("DATASET_PARQUET_PATH")

# Training configuration
SEED = int(os.environ.get("SEED", "42"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4196"))
LR = float(os.environ.get("LR", "1e-4"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))

# PyTorch profiler configuration. Each rank writes to its own subdirectory so
# that distributed traces do not overwrite one another.
ENABLE_PROFILER = os.environ.get("ENABLE_PROFILER", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PROFILE_DIR = os.environ.get("PROFILE_DIR", "./profile/moonep")
PROFILE_WAIT_STEPS = int(os.environ.get("PROFILE_WAIT_STEPS", "1"))
PROFILE_WARMUP_STEPS = int(os.environ.get("PROFILE_WARMUP_STEPS", "1"))
PROFILE_ACTIVE_STEPS = int(os.environ.get("PROFILE_ACTIVE_STEPS", "2"))

# The FSDP and EP meshes are independent views over the same torchrun world.
FULLY_SHARD_PARALLEL_SIZE = int(
    os.environ.get("FULLY_SHARD_PARALLEL_SIZE", "8")
)
EXPERT_PARALLEL_SIZE = int(os.environ.get("EXPERT_PARALLEL_SIZE", "8"))

# MoonEP tuning
MOONEP_NUM_SMS = int(os.environ.get("MOONEP_NUM_SMS", "32"))
MOONEP_TOKEN_PADDING = int(os.environ.get("MOONEP_TOKEN_PADDING", "128"))
MOONEP_ASYNC_FINISH = os.environ.get("MOONEP_ASYNC_FINISH", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MOONEP_ENABLE_PDL = os.environ.get("MOONEP_ENABLE_PDL", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Gradient monitoring. Set CLIP_GRAD>0 to clip; otherwise only log the norm.
CLIP_GRAD = float(os.environ.get("CLIP_GRAD", "0"))
GRAD_NORM_TYPE = float(os.environ.get("GRAD_NORM_TYPE", "2.0"))


def initialize_distributed() -> tuple[int, int, int]:
    """Initialize the local accelerator and its distributed process group."""
    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "This script must be launched with torchrun; missing environment "
            f"variables: {', '.join(missing)}"
        )
    if not hasattr(torch, "accelerator"):
        raise RuntimeError("MoonEP integration requires PyTorch 2.9 or newer.")

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = set_accelerator_compatible()
    accelerator_type = torch.accelerator.current_accelerator().type
    if accelerator_type not in {"cuda", "npu"}:
        raise RuntimeError(
            f"MoonEP requires a CUDA or NPU accelerator, detected {accelerator_type!r}."
        )
    torch.accelerator.set_device_index(local_rank)
    expected_backend = "nccl" if accelerator_type == "cuda" else "hccl"
    if backend != expected_backend:
        raise RuntimeError(
            f"MoonEP expected backend {expected_backend!r} for {accelerator_type}, "
            f"detected {backend!r}."
        )

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if EXPERT_PARALLEL_SIZE <= 1:
        raise ValueError("EXPERT_PARALLEL_SIZE must be greater than 1 for MoonEP.")
    if world_size % EXPERT_PARALLEL_SIZE:
        raise ValueError(
            f"WORLD_SIZE={world_size} is not divisible by "
            f"EXPERT_PARALLEL_SIZE={EXPERT_PARALLEL_SIZE}."
        )
    if world_size % FULLY_SHARD_PARALLEL_SIZE:
        raise ValueError(
            f"WORLD_SIZE={world_size} is not divisible by "
            f"FULLY_SHARD_PARALLEL_SIZE={FULLY_SHARD_PARALLEL_SIZE}."
        )

    set_seed(SEED)
    set_log_level(LOG_LEVEL)
    return local_rank, rank, world_size


def build_model_and_tokenizer(local_rank: int):
    """Build a randomly initialized BF16 model from config."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.torch_dtype = torch.bfloat16
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
    )
    model.to(torch.device(torch.accelerator.current_accelerator().type, local_rank))
    return model, tokenizer


def get_dataloader(tokenizer, rank: int, world_size: int) -> DataLoader:
    """Build a sharded, fixed-shape causal-language-modeling DataLoader."""
    if DATASET_PARQUET_PATH:
        files = sorted(glob.glob(os.path.join(DATASET_PARQUET_PATH, "*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"No parquet files were found in {DATASET_PARQUET_PATH!r}."
            )
        dataset = load_dataset("parquet", data_files=files, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_PATH, DATASET_CONFIG, split=DATASET_SPLIT)

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain a 'text' column, got {dataset.column_names}."
        )

    def tokenize_function(examples):
        encoded = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        encoded["labels"] = [
            [token if mask else -100 for token, mask in zip(input_ids, attention_mask)]
            for input_ids, attention_mask in zip(
                encoded["input_ids"], encoded["attention_mask"]
            )
        ]
        return encoded

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # Causal LM loss shifts labels by one position, so every sample needs at
    # least two non-padding tokens to contribute a finite loss.
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: sum(example["attention_mask"]) >= 2
    )
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=SEED,
        drop_last=True,
    )
    return DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
    )


def get_moonep_fsdp_config(top_k: int) -> FSDPTurboConfig:
    """Configure MoonEP for experts and FSDP2 for all non-expert weights."""
    return FSDPTurboConfig(
        distributed=DistributedConfig(
            fully_shard_parallel_size=FULLY_SHARD_PARALLEL_SIZE,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={
                    "model.layers.{*}": {},
                    "model.embed_tokens": {},
                    "lm_head": {},
                },
                param_dtype="bf16",
                reduce_dtype="fp32",
                output_dtype="bf16",
                hook_modules=["model.layers.{*}"],
                fsdp_implementation="native",
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=1,
            ep_plan=EPPlanConfig(
                apply_modules=["model.layers.{*}.mlp.experts"],
                dispatcher="moonep",
                moonep_config=MoonEPConfig(
                    num_sms=MOONEP_NUM_SMS,
                    token_padding=MOONEP_TOKEN_PADDING,
                    async_finish=MOONEP_ASYNC_FINISH,
                    enable_pdl=MOONEP_ENABLE_PDL,
                    tokens_per_rank=BATCH_SIZE * MAX_LENGTH,
                    top_k=top_k,
                ),
            ),
        ),
    )


def get_router_top_k(model: torch.nn.Module) -> int:
    """Read the model's static number of selected experts per token."""
    top_k = getattr(model.config, "num_experts_per_tok", None)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            "The model config must define a positive integer "
            f"num_experts_per_tok, got {top_k!r}."
        )
    return top_k


def create_profiler(rank: int):
    """Create a scheduled CUDA/CPU profiler or a no-op context manager."""
    if not ENABLE_PROFILER:
        return nullcontext()

    accelerator_type = torch.accelerator.current_accelerator().type
    if accelerator_type != "cuda":
        raise RuntimeError(
            "The system-test profiler currently supports only CUDA; disable "
            "ENABLE_PROFILER for NPU training."
        )

    if PROFILE_WAIT_STEPS < 0 or PROFILE_WARMUP_STEPS < 0:
        raise ValueError("Profiler wait and warmup steps must be non-negative.")
    if PROFILE_ACTIVE_STEPS <= 0:
        raise ValueError("PROFILE_ACTIVE_STEPS must be greater than zero.")

    required_steps = (
        PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS
    )
    if NUM_STEPS < required_steps:
        raise ValueError(
            f"NUM_STEPS={NUM_STEPS} is too small for the profiler schedule; "
            f"at least {required_steps} steps are required."
        )

    rank_profile_dir = os.path.join(PROFILE_DIR, f"rank{rank}")
    os.makedirs(rank_profile_dir, exist_ok=True)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=PROFILE_WAIT_STEPS,
            warmup=PROFILE_WARMUP_STEPS,
            active=PROFILE_ACTIVE_STEPS,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            rank_profile_dir
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def train(
    model: FSDPTurbo,
    dataloader: DataLoader,
    local_rank: int,
    rank: int,
) -> None:
    """Run the configured number of full forward/backward optimization steps."""
    model.train()
    optimizer = AdamW(model.model.parameters(), lr=LR)
    device = torch.device(torch.accelerator.current_accelerator().type, local_rank)

    if rank == 0:
        print_rank(
            logger.info,
            "Starting randomly initialized Qwen3 training with "
            f"MoonEP={EXPERT_PARALLEL_SIZE}, FSDP={FULLY_SHARD_PARALLEL_SIZE}, "
            f"batch_size={BATCH_SIZE}, sequence_length={MAX_LENGTH}, "
            f"steps={NUM_STEPS}",
        )
        if ENABLE_PROFILER:
            print_rank(
                logger.info,
                f"Profiler enabled: output={os.path.abspath(PROFILE_DIR)}, "
                f"wait={PROFILE_WAIT_STEPS}, warmup={PROFILE_WARMUP_STEPS}, "
                f"active={PROFILE_ACTIVE_STEPS}",
            )

    completed_steps = 0
    with create_profiler(rank) as prof:
        for step, batch in enumerate(dataloader):
            if step >= NUM_STEPS:
                break

            optimizer.zero_grad(set_to_none=True)
            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }

            dist.barrier()
            torch.accelerator.memory.reset_peak_memory_stats(device)
            started = time.perf_counter()

            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {loss.item()}"
                )
            loss.backward()
            if CLIP_GRAD > 0:
                grad_norm = clip_grad_norm(
                    model.model,
                    max_norm=CLIP_GRAD,
                    norm_type=GRAD_NORM_TYPE,
                )
            else:
                grad_norm = compute_grad_norm(
                    model.model.parameters(),
                    norm_type=GRAD_NORM_TYPE,
                )
            optimizer.step()

            torch.accelerator.synchronize(device)
            dist.barrier()
            elapsed = time.perf_counter() - started
            completed_steps += 1

            if prof is not None:
                prof.step()

            if rank == 0:
                print_rank(
                    logger.info,
                    f"step={step} loss={loss.item():.6f} grad_norm={grad_norm:.6f} "
                    f"time={elapsed:.4f}s "
                    f"memory={torch.accelerator.memory.memory_allocated(device) / 2**30:.2f}GiB "
                    f"peak_memory={torch.accelerator.memory.max_memory_allocated(device) / 2**30:.2f}GiB",
                )

    if completed_steps != NUM_STEPS:
        raise RuntimeError(
            f"DataLoader exhausted after {completed_steps} steps; expected {NUM_STEPS}."
        )
    if rank == 0:
        print_rank(logger.info, "MoonEP full-model training completed successfully.")


def main() -> None:
    """Set up, train, and reliably release distributed MoonEP resources."""
    model = None
    wrapped_model = None
    try:
        local_rank, rank, world_size = initialize_distributed()
        model, tokenizer = build_model_and_tokenizer(local_rank)
        top_k = get_router_top_k(model)
        config = get_moonep_fsdp_config(top_k)
        wrapped_model = FSDPTurbo(config, model)
        dataloader = get_dataloader(tokenizer, rank, world_size)
        train(wrapped_model, dataloader, local_rank, rank)
    finally:
        try:
            if wrapped_model is not None:
                wrapped_model.close()
            elif model is not None:
                # FSDPTurbo may have attached the runtime before its constructor
                # raised, in which case no wrapper object was returned to us.
                runtime = getattr(model, "_moonep_runtime", None)
                if runtime is not None:
                    runtime.close()
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
