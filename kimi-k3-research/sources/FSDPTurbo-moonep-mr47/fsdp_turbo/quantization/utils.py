# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import re
from typing import Callable, Dict, Type
import torch
from torch import nn

from fsdp_turbo.quantization.mxfp8_config import QuantBaseConfig
from fsdp_turbo.utils.str_match import module_name_match

_QUANTIZE_CONFIG_HANDLER: Dict[
    Type[QuantBaseConfig],
    Callable[[torch.nn.Module, QuantBaseConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func

    return decorator


class TensorWithTranspose:
    def __init__(
        self,
        fp8_dtype: torch.dtype,
        data: torch.Tensor,
        scale: torch.Tensor,
        data_t: torch.Tensor,
        scale_t: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.fp8_dtype = fp8_dtype
        self.data = data
        self.scale = scale
        self.data_t = data_t
        self.scale_t = scale_t
        self.ori_dtype = dtype

    def get_by_trans(self, transpose=False):
        if transpose:
            return self.data_t, self.scale_t
        return self.data, self.scale


def module_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    def ignored_modules(fqn: str, config: QuantBaseConfig):
        for pattern in config.mxfp8_ignored_modules:
            if module_name_match(pattern, fqn):
                return True
        return False

    if not isinstance(mod, nn.Linear):
        return False

    ignored_modules_flag = ignored_modules(fqn, config)
    if ignored_modules_flag:
        return False

    for pattern in config.mxfp8_apply_modules:
        m = re.match(r"(.*?layers\.\d+)", fqn)
        if m is not None:
            prefix = m.group(1)
            return module_name_match(pattern, prefix)

    return False


def moe_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    """Filter function for MoE expert modules.

    Matches modules whose FQN contains "experts" and whose parent layer
    matches the configured apply_modules patterns.
    """
    if "experts" not in fqn.lower():
        return False

    m = re.match(r"(.*?layers\.\d+)", fqn)
    if m is None:
        return False
    prefix = m.group(1)

    for pattern in config.mxfp8_apply_modules:
        return module_name_match(pattern, prefix)

    return False


def view_as_n_dim(input_tensor, dim=2):
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1 :])
    return input_tensor
