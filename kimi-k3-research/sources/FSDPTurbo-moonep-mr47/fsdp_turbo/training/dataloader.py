# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Dataloader creation utilities.

Builds a ``DataLoader`` from a ``DataConfig``, optionally tokenizing
the dataset with a HuggingFace tokenizer and wrapping it with a
``DistributedSampler`` for multi-process training.
"""

import logging

import torch

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def build_dataloader(data_cfg, tokenizer=None):
    """Build a DataLoader from a HuggingFace dataset.

    Args:
        data_cfg: A ``DataConfig`` dataclass instance.
        tokenizer: An optional HuggingFace tokenizer. When provided,
            the dataset is tokenized and formatted for language modeling.

    Returns:
        A ``torch.utils.data.DataLoader``, or ``None`` if
        ``data_cfg.dataset_path`` is empty.
    """
    if data_cfg is None or not data_cfg.dataset_path:
        print_rank(logger.warning, "dataset_path is empty, skipping dataloader build.")
        return None

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "`datasets` library is required for build_dataloader. "
            "Install it with: pip install datasets"
        )

    print_rank(
        logger.info,
        f"> Building DataLoader for {data_cfg.dataset_path} {data_cfg.dataset_config} {data_cfg.split}",
    )
    dataset = load_dataset(data_cfg.dataset_path, data_cfg.dataset_config, split=data_cfg.split)

    if tokenizer is not None:
        def tokenize_function(examples):
            return tokenizer(
                examples[data_cfg.text_column],
                truncation=True,
                max_length=data_cfg.max_seq_length,
                padding="max_length",
            )

        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        def add_labels(examples):
            examples["labels"] = examples["input_ids"].clone()
            return examples

        dataset = dataset.map(add_labels)

    sampler = None
    if torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=data_cfg.shuffle)
        shuffle = False
    else:
        shuffle = data_cfg.shuffle

    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
