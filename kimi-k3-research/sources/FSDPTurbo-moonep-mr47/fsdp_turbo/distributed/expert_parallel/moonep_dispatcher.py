# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Fused-expert forward implementation backed by MoonEP."""

import logging
import os
import types

import torch

from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (
    MoonEPRuntime,
    moonep_bind_weight_grads,
    moonep_dispatch,
    moonep_weighted_combine,
)
from fsdp_turbo.distributed.expert_parallel.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)
from fsdp_turbo.ops.moe import grouped_matmul

logger = logging.getLogger(__name__)

_REQUIRED_EXPERT_ATTRIBUTES = (
    "gate_up_proj",
    "down_proj",
    "act_fn",
    "hidden_dim",
)


def _debug_sync(label: str) -> None:
    """Optional device-wide barrier used to pinpoint a native SIGSEGV.

    Set ``MOONEP_DEBUG_SYNC=1`` to log after dispatch, prefetch, each grouped
    matmul, and combine. The last printed label is the stage that crashed.
    """
    flag = os.environ.get("MOONEP_DEBUG_SYNC", "").lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    torch.accelerator.synchronize()
    logger.info("MoonEP debug sync after %s", label)


def _grouped_matmul_with_static_tail(
    inputs: torch.Tensor,
    cumulative_group_ends: torch.Tensor,
    weights: torch.Tensor,
    grad_weight_sink=None,
    snapshot=False,
) -> torch.Tensor:
    """Run the common grouped matmul over MoonEP's static communication buffer.

    ``snapshot=True`` clones ``inputs`` because MoonEP reuses the dispatch
    buffer across layers. Down-projection activations already have private
    storage; cloning them again was a full ``[NvS, H']`` copy per layer.
    """
    if cumulative_group_ends.numel() != weights.shape[0]:
        raise RuntimeError(
            f"MoonEP provided {cumulative_group_ends.numel()} groups but the "
            f"projection contains {weights.shape[0]} weight groups."
        )

    # Keep the planner's exclusive ends on CUDA/CPU. Do not rewrite the last
    # end to NvS and do not multiply the whole dispatch buffer to zero the
    # static tail: that clone is [NvS, H] per GMM (~64-170 MiB at Qwen3-30B
    # S=4196 K=8) and is saved until backward, so it inflates peak_memory.
    # CUDA grouped_mm only reads [:cu_seqlens[-1]]; the CUDA op zeros unused
    # output rows. A .cpu() here still deadlocks FSDP prefetch all-gather.
    if snapshot:
        inputs = inputs.clone()
    cumulative_group_ends = cumulative_group_ends.to(dtype=torch.int32).contiguous()
    if inputs.device.type == "npu":
        # npu_grouped_matmul requires the group list to cover every row of
        # the static buffer. Zero the tail first so folding it into the last
        # expert cannot pollute that expert's wgrad.
        cumulative_group_ends = cumulative_group_ends.clone()
        live = (
            torch.arange(inputs.shape[0], device=inputs.device)
            < cumulative_group_ends[-1]
        )
        inputs = inputs * live.unsqueeze(-1).to(inputs.dtype)
        cumulative_group_ends[-1] = inputs.shape[0]
    group_starts = torch.cat(
        [torch.zeros_like(cumulative_group_ends[:1]), cumulative_group_ends[:-1]]
    )
    group_sizes = cumulative_group_ends - group_starts
    return grouped_matmul(
        inputs,
        group_sizes,
        weights,
        use_eager=inputs.device.type == "cpu",
        grad_weight_sink=grad_weight_sink,
    )


def _validate_expert_module(module: torch.nn.Module) -> None:
    missing = [name for name in _REQUIRED_EXPERT_ATTRIBUTES if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "MoonEP currently supports fused expert modules with gate_up_proj, "
            f"down_proj, act_fn and hidden_dim; missing {missing}."
        )
    for name in ("gate_up_proj", "down_proj"):
        value = getattr(module, name)
        if not isinstance(value, torch.nn.Parameter):
            raise RuntimeError(
                f"MoonEP requires {name} to be an unquantized nn.Parameter, "
                f"got {type(value).__name__}."
            )
    gate_up = module.gate_up_proj
    down = module.down_proj
    expected_experts = int(module.num_global_experts)
    if gate_up.ndim != 3 or down.ndim != 3:
        raise RuntimeError("MoonEP fused expert weights must both be rank-3 tensors.")
    if gate_up.shape[0] != expected_experts or down.shape[0] != expected_experts:
        raise RuntimeError(
            "MoonEP projection expert dimensions must match num_global_experts."
        )
    if (
        gate_up.shape[-1] != module.hidden_dim
        or down.shape[-2] != module.hidden_dim
        or gate_up.shape[-2] != 2 * down.shape[-1]
    ):
        raise RuntimeError(
            "MoonEP expects gate_up_proj=[E,2*Hp,H] and down_proj=[E,H,Hp]; "
            f"got {tuple(gate_up.shape)} and {tuple(down.shape)} with H={module.hidden_dim}."
        )


def parallelize_moonep_module(module, runtime: MoonEPRuntime, ep_mesh, fixed_router=False):
    _validate_expert_module(module)
    gate_up = runtime.distribute_projection("gate_up", module.gate_up_proj, ep_mesh)
    down = runtime.distribute_projection("down", module.down_proj, ep_mesh)

    module.register_parameter("gate_up_proj", gate_up.parameter)
    module.register_parameter("down_proj", down.parameter)
    # Resource holders are deliberately not modules/buffers and therefore do
    # not alter state_dict or optimizer parameter discovery.
    object.__setattr__(module, "_moonep_runtime", runtime)
    object.__setattr__(module, "_moonep_gate_up", gate_up)
    object.__setattr__(module, "_moonep_down", down)
    module.forward = types.MethodType(
        get_moonep_experts_forward_fn(runtime, fixed_router=fixed_router), module
    )


def get_moonep_experts_forward_fn(runtime: MoonEPRuntime, fixed_router=False):
    def prefetch_before_backward(output, projection, call):
        if output.requires_grad:
            def hook(grad):
                projection.prefetch(call.plan)
                return grad

            output.register_hook(hook)
        return output

    def moonep_experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ):
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim).contiguous()
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                f"MoonEP expert input must be BF16, got {hidden_states.dtype}."
            )

        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(
                top_k_index, top_k_weights, self.num_global_experts
            )
        if top_k_index.ndim == 1:
            top_k_index = top_k_index.unsqueeze(-1)
            top_k_weights = top_k_weights.unsqueeze(-1)

        tokens_per_rank = hidden_states.shape[0]
        top_k = top_k_index.shape[-1]
        if (
            top_k_index.shape[0] != tokens_per_rank
            or top_k_weights.shape != top_k_index.shape
        ):
            raise RuntimeError(
                "MoonEP expects hidden [S,H], expert indices [S,K], and route "
                f"weights [S,K]; got {tuple(hidden_states.shape)}, "
                f"{tuple(top_k_index.shape)}, {tuple(top_k_weights.shape)}."
            )

        top_k_index = top_k_index.to(dtype=torch.int32).contiguous()
        top_k_weights = top_k_weights.float().contiguous()
        torch._assert_async((top_k_index >= 0).all(), "MoonEP expert ids must be non-negative")
        torch._assert_async(
            (top_k_index < self.num_global_experts).all(),
            "MoonEP expert ids must be smaller than num_global_experts",
        )
        tokens_per_expert = torch.bincount(
            top_k_index.reshape(-1).long(), minlength=self.num_global_experts
        ).to(dtype=torch.int32).contiguous()

        call = runtime.new_call(tokens_per_rank, self.hidden_dim, top_k)
        dispatched, dispatched_route_weights = moonep_dispatch(
            call,
            hidden_states,
            top_k_weights,
            top_k_index,
            tokens_per_expert,
        )
        _debug_sync("dispatch")
        call.prefetch(self._moonep_gate_up, self._moonep_down)
        _debug_sync("prefetch")

        # Binding the parameters here, upstream of both grouped matmuls, makes
        # the cross-rank gradient reduction the last node of this layer's
        # backward pass, once both projections have written their gradients
        # into the [E+B] buffers.
        dispatched = moonep_bind_weight_grads(
            dispatched,
            self.gate_up_proj,
            self.down_proj,
            self._moonep_gate_up,
            self._moonep_down,
            call.plan,
            call.comm_context,
        )
        with torch.profiler.record_function("moonep.gmm.gate_up"):
            fc1 = _grouped_matmul_with_static_tail(
                dispatched,
                call.cu_seqlens,
                self._moonep_gate_up.full_weight,
                grad_weight_sink=self._moonep_gate_up.grad_weight_sink(),
                snapshot=True,
            )
        _debug_sync("gmm.gate_up")
        fc1 = prefetch_before_backward(fc1, self._moonep_gate_up, call)
        gate, up = fc1.chunk(2, dim=-1)
        act_limit = self.limit if hasattr(self, "limit") else None
        if act_limit is not None:
            gate = gate.clamp(max=act_limit)
            up = up.clamp(min=-act_limit, max=act_limit)
        activated = self.act_fn(gate) * up
        with torch.profiler.record_function("moonep.gmm.down"):
            expert_output = _grouped_matmul_with_static_tail(
                activated.contiguous(),
                call.cu_seqlens,
                self._moonep_down.full_weight,
                grad_weight_sink=self._moonep_down.grad_weight_sink(),
            )
        _debug_sync("gmm.down")
        expert_output = prefetch_before_backward(expert_output, self._moonep_down, call)
        output = moonep_weighted_combine(
            call, expert_output, dispatched_route_weights
        )
        _debug_sync("combine")
        return output.view(*hidden_shape)

    return moonep_experts_forward
