# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from fsdp_turbo.ops.registry import register_op


@torch._dynamo.allow_in_graph
class MxGroupedMatmul(torch.autograd.Function):
    """FP8 grouped matmul on NPU with pre-quantized inputs."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_scale_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        weight_scale_bwd: torch.Tensor,
        weight_2d_shape: tuple,
        weight_transposed: bool,
        grad_fp8_dtype,
        input_fp8_dtype,
        input_dtype,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        ctx.weight_2d_shape = weight_2d_shape
        ctx.weight_transposed = weight_transposed
        ctx.grad_fp8_dtype = grad_fp8_dtype
        ctx.input_fp8_dtype = input_fp8_dtype
        ctx.input_dtype = input_dtype
        ctx.group_list_type = group_list_type

        ctx.save_for_backward(x, group_list, weight_bwd, weight_scale_bwd)

        output = torch_npu.npu_grouped_matmul(
            [x_fp8],
            [weight_fwd],
            scale=[weight_scale_fwd],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=input_dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, group_list, weight_bwd, weight_scale_bwd = ctx.saved_tensors
        group_list_int32 = group_list.to(torch.int32)
        total_tokens = group_list[-1].item()

        grad_fp8, grad_scale = torch_npu.npu_dynamic_mx_quant(grad_output, axis=-1, dst_type=ctx.grad_fp8_dtype)

        dx = torch_npu.npu_grouped_matmul(
            [grad_fp8],
            [weight_bwd],
            scale=[weight_scale_bwd],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=grad_output.dtype,
            group_list_type=ctx.group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        if total_tokens == 0:
            dw = torch.zeros(ctx.weight_2d_shape, dtype=ctx.input_dtype, device=x.device)
        else:
            x_q, x_q_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                x,
                group_list_int32,
                round_mode="rint",
                dst_type=ctx.input_fp8_dtype,
                blocksize=32,
            )
            grad_q, grad_q_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                grad_output,
                group_list_int32,
                round_mode="rint",
                dst_type=ctx.grad_fp8_dtype,
                blocksize=32,
            )
            dw = torch_npu.npu_grouped_matmul(
                [x_q.t()],
                [grad_q],
                scale=[grad_q_scale],
                per_token_scale=[x_q_scale.transpose(0, 1)],
                group_list=group_list,
                group_type=2,
                output_dtype=ctx.input_dtype,
                group_list_type=ctx.group_list_type,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                split_item=3,
            )[0]

        if ctx.weight_transposed:
            dw = dw.transpose(1, 2).reshape(ctx.weight_2d_shape)
        else:
            dw = dw.reshape(ctx.weight_2d_shape)

        return (
            dx,
            dw,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@register_op('mx_grouped_matmul', 'npu')
def mx_grouped_matmul_npu(
    x,
    weight,
    x_fp8,
    x_scale,
    weight_fwd,
    weight_scale_fwd,
    weight_bwd,
    weight_scale_bwd,
    weight_2d_shape,
    weight_transposed,
    grad_fp8_dtype,
    input_fp8_dtype,
    input_dtype,
    group_list,
    group_list_type,
):
    return MxGroupedMatmul.apply(
        x,
        weight,
        x_fp8,
        x_scale,
        weight_fwd,
        weight_scale_fwd,
        weight_bwd,
        weight_scale_bwd,
        weight_2d_shape,
        weight_transposed,
        grad_fp8_dtype,
        input_fp8_dtype,
        input_dtype,
        group_list,
        group_list_type,
    )
