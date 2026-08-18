# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch


class _MissingTorchNpu:
    def __getattr__(self, name):
        raise ImportError("torch_npu is not available. MX quantization requires Ascend NPU with torch_npu installed.")


try:
    import torch_npu
except ImportError:
    torch_npu = _MissingTorchNpu()


def dynamic_mx_quant(tensor: torch.Tensor, axis: int = -1, dst_type=None) -> tuple:
    """Single-axis dynamic MX quantization."""
    return torch_npu.npu_dynamic_mx_quant(tensor, axis=axis, dst_type=dst_type)


def dynamic_mx_quant_with_dual_axis(tensor: torch.Tensor, dst_type=None) -> tuple:
    """Dual-axis dynamic MX quantization.

    Returns:
        (col_data, col_scale, row_data, row_scale)
    """
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(tensor, dst_type=dst_type)


def grouped_dynamic_mx_quant(
    tensor: torch.Tensor, group_list, round_mode: str = "rint", dst_type=None, blocksize: int = 32
) -> tuple:
    """Grouped dynamic MX quantization for MoE token groups."""
    return torch_npu.npu_grouped_dynamic_mx_quant(
        tensor,
        group_list,
        round_mode=round_mode,
        dst_type=dst_type,
        blocksize=blocksize,
    )


def weight_quant_gmm(weight: torch.Tensor, dst_type, new_shape: tuple) -> tuple:
    """Quantize weight for GMM, reshaping before quantization.

    The weight is reshaped to 3D [num_experts, in_dim, out_dim] before
    quantization so that dual-axis quantization produces correct per-expert
    scales.

    Returns:
        (weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd)
    """
    original_shape = weight.shape
    weight = weight.reshape(new_shape)
    weight_bwd, weight_scale_bwd, weight_fwd, weight_scale_fwd = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        weight, dst_type=dst_type
    )
    weight_fwd = weight_fwd.reshape(original_shape)
    weight_bwd = weight_bwd.reshape(original_shape)
    return weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply SiGLU activation."""
    return torch_npu.npu_swiglu(x, dim=dim)
