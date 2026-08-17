# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the prototype UX for converting a model to use mx weights
"""

from functools import partial
from typing import Optional
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None
from torch.distributed.tensor import DTensor

from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.core.pre_quant_weight import PreQuantWeight
from fsdp_turbo.quantization.mx_formats.mx_tensor import MXTensor
from fsdp_turbo.quantization.mxfp8_config import MXFP8LinearConfig
from fsdp_turbo.quantization.utils import register_quantize_module_handler


@torch._dynamo.allow_in_graph
class mx_mm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        mx_tensor: MXTensor,
    ):
        ctx.mx_tensor = mx_tensor
        # input @ weight_t = output
        ctx.input_orig_shape = input_hp.shape
        input_hp = input_hp.view(-1, input_hp.size(-1))

        # 低精量化
        input_mx = mx_tensor.to_mxfp8(input_hp, 'inputs')

        if isinstance(weight_hp, PostQuantWeight):
            ctx.is_post_quant = True
            ctx.weight_pq = weight_hp

            output = torch_npu.npu_quant_matmul(
                input_mx.data,
                weight_hp._weight_fwd.t(),
                weight_hp._scale_fwd.transpose(0, 1),
                pertoken_scale=input_mx.scale,
                output_dtype=input_hp.dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
        else:
            ctx.is_post_quant = False
            weight_mx = mx_tensor.to_mxfp8(weight_hp, 'weight')
            ctx.weight_mx = weight_mx

            output = torch_npu.npu_quant_matmul(
                input_mx.data,
                weight_mx.data.t(),
                weight_mx.scale.transpose(0, 1),
                pertoken_scale=input_mx.scale,
                output_dtype=input_hp.dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
            weight_mx.data.untyped_storage().resize_(0)
            weight_mx.scale.untyped_storage().resize_(0)

        ctx.input_mx = input_mx

        input_mx.data.untyped_storage().resize_(0)
        input_mx.scale.untyped_storage().resize_(0)

        if len(ctx.input_orig_shape) != 2:
            output = output.reshape(*ctx.input_orig_shape[:-1], output.shape[-1])
        if weight_hp.requires_grad:
            output.requires_grad = True
        return output

    @staticmethod
    def backward(ctx, grad_output_hp: torch.Tensor):
        input_mx = ctx.input_mx
        mx_tensor = ctx.mx_tensor
        ori_dtype = input_mx.ori_dtype
        grad_orig_shape = grad_output_hp.shape
        grad_output_hp = grad_output_hp.view(-1, grad_output_hp.size(-1))

        # 低精量化
        grads_mx = mx_tensor.to_mxfp8(grad_output_hp, 'grads')

        if ctx.is_post_quant:
            weight_pq = ctx.weight_pq
            dx = torch_npu.npu_quant_matmul(
                grads_mx.data,
                weight_pq._weight_bwd,
                weight_pq._scale_bwd,
                pertoken_scale=grads_mx.scale,
                output_dtype=ori_dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
        else:
            weight_mx = ctx.weight_mx
            dx = torch_npu.npu_quant_matmul(
                grads_mx.data,
                weight_mx.data_t,
                weight_mx.scale_t,
                pertoken_scale=grads_mx.scale,
                output_dtype=ori_dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )

        if len(grad_orig_shape) != 2:
            dx = dx.reshape(*grad_orig_shape[:-1], dx.shape[-1])

        grads_mx.data.untyped_storage().resize_(0)
        grads_mx.scale.untyped_storage().resize_(0)

        if ctx.is_post_quant:
            dw = torch_npu.npu_quant_matmul(
                grads_mx.data_t.t(),
                input_mx.data_t,
                input_mx.scale_t,
                pertoken_scale=grads_mx.scale_t.transpose(0, 1),
                output_dtype=ori_dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
        else:
            weight_mx = ctx.weight_mx
            dw = torch_npu.npu_quant_matmul(
                grads_mx.data_t.t(),
                input_mx.data_t,
                input_mx.scale_t,
                pertoken_scale=grads_mx.scale_t.transpose(0, 1),
                output_dtype=ori_dtype,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            )
            weight_mx.data_t.untyped_storage().resize_(0)
            weight_mx.scale_t.untyped_storage().resize_(0)

        grads_mx.data_t.untyped_storage().resize_(0)
        grads_mx.scale_t.untyped_storage().resize_(0)
        input_mx.data_t.untyped_storage().resize_(0)
        input_mx.scale_t.untyped_storage().resize_(0)

        return dx, dw, None


class MXLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._mx_tensor = None
        self.config = None
        self._tp_mesh = None
        self._tp_output_placements = None

    @classmethod
    @torch.no_grad()
    # pylint: disable=attribute-defined-outside-init
    def from_float(cls, mod, config: Optional[MXFP8LinearConfig] = None):
        if config is None:
            config = MXFP8LinearConfig()

        if getattr(config, 'enable_fsdp_low_precision_all_gather', False):
            with torch.device('meta'):
                new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
            new_mod.weight = mod.weight
            new_mod.bias = mod.bias
            new_mod.config = config

            weight_dtype = config.get_key_dtype('weight')
            quantizer_fn = partial(
                torch_npu.npu_dynamic_mx_quant_with_dual_axis,
                dst_type=weight_dtype,
            )

            w = new_mod.weight
            if isinstance(w, DTensor):
                local_w = w.to_local()
                pre_quant = PreQuantWeight(
                    local_w,
                    quantizer_fn,
                    config,
                    mod.weight.dtype,
                )
                new_mod.weight = torch.nn.Parameter(
                    DTensor.from_local(
                        pre_quant,
                        device_mesh=w.device_mesh,
                        placements=w.placements,
                    ),
                    requires_grad=w.requires_grad,
                )
            else:
                new_mod.weight = torch.nn.Parameter(
                    PreQuantWeight(
                        w,
                        quantizer_fn,
                        config,
                        mod.weight.dtype,
                    ),
                    requires_grad=w.requires_grad,
                )
            return new_mod

        mod.__class__ = MXLinear
        mod.config = config
        mod._mx_tensor = None
        mod._tp_mesh = None
        mod._tp_output_placements = None
        return mod

    @property
    def mx_tensor(self):
        if self._mx_tensor is None:
            self._mx_tensor = MXTensor(self.config)
        return self._mx_tensor

    def forward(self, x):
        w = self.weight
        try:
            x = x.to_local()
        except (AttributeError, TypeError):
            pass

        if not isinstance(w, PostQuantWeight):
            try:
                w = w.to_local()
            except (AttributeError, TypeError):
                pass

        y = mx_mm.apply(x, w, self.mx_tensor)
        if self.bias is not None:
            y = y + self.bias

        if self._tp_mesh is not None:
            y = DTensor.from_local(y, self._tp_mesh, self._tp_output_placements)
        return y


@register_quantize_module_handler(MXFP8LinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXFP8LinearConfig):
    return MXLinear.from_float(module, config=config)
