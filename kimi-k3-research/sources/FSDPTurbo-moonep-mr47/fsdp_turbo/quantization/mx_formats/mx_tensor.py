# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from fsdp_turbo.quantization.utils import TensorWithTranspose, view_as_n_dim


class MXTensor:
    def __init__(self, config):
        self.config = config
        self.fp8_format_dtype = None

    def to_mxfp8(self, data_hf, key):
        if data_hf is None:
            return data_hf
        try:
            data_hf = data_hf.to_local()
        except (AttributeError, TypeError):
            pass
        if not data_hf.is_npu:
            data_hf = data_hf.npu()
        ori_dtype = data_hf.dtype

        if data_hf.dtype == torch.float32:
            data_hf = data_hf.to(torch.bfloat16)

        self.fp8_format_dtype = self.config.get_key_dtype(key)
        tensor_2d = view_as_n_dim(data_hf)

        col_data, col_scale, row_data, row_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            tensor_2d, dst_type=self.fp8_format_dtype
        )

        return TensorWithTranspose(self.fp8_format_dtype, col_data, col_scale, row_data, row_scale, ori_dtype)
