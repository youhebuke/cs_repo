# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from torch.distributed.tensor import DTensor
from fsdp_turbo.distributed.dist_ops import gather_along_first_dim_expert_parallel
from fsdp_turbo.ops.moe import permute, unpermute, grouped_matmul
from fsdp_turbo.distributed.expert_parallel.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)

_DOMINO_NUM_SLICES = 2


def experts_computation(hidden_states, split_list, gate_up_weights, down_weights, act_fn, act_limit, fused):
    gate, up = grouped_matmul(hidden_states, split_list, gate_up_weights, use_eager=not fused).chunk(2, dim=-1)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    act = act_fn(gate) * up
    hidden_states = grouped_matmul(act, split_list, down_weights, use_eager=not fused)
    return hidden_states


def get_domino_experts_forward_fn(ep_group, fused=True, fixed_router=False):
    """
    Generate a forward function for expert parallel processing with DOMINO optimization.

    Args:
        ep_group: Expert parallel process group
        fused: Whether to use fused grouped matmul operations

    """

    comm_streams = {}

    def domino_experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                               top_k_weights: torch.Tensor):
        """
        Forward pass for expert parallel processing with DOMINO optimization.

        Args:
            hidden_states: Input tensor of shape [batch_size, sequence_length, hidden_dim]
            top_k_index: Top-k expert indices.
            top_k_weights: Top-k expert weights.

        Returns:
            Output tensor with the same shape as hidden_states
        """
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        hidden_states_list, top_k_index_list, top_k_weights_list = slice_hidden_states_for_domino(
            hidden_states, top_k_index, top_k_weights)
        gate_up_proj = (self.gate_up_proj.to_local()
                        if isinstance(self.gate_up_proj, DTensor)
                        else self.gate_up_proj)
        down_proj = (self.down_proj.to_local()
                     if isinstance(self.down_proj, DTensor)
                     else self.down_proj)
        weights = (gate_up_proj, down_proj)
        act_fn = self.act_fn
        num_global_experts = self.num_global_experts
        expert_ids_per_ep_rank = self.expert_ids_per_ep_rank
        act_limit = self.limit if hasattr(self, 'limit') else None
        device_index = torch.accelerator.current_device_index()
        comm_stream = comm_streams.get(device_index)
        if comm_stream is None:
            comm_stream = torch.accelerator.Stream()
            comm_streams[device_index] = comm_stream

        hidden_states_list = dispatch_mlp_combine(ep_group, fused, hidden_states_list, top_k_index_list,
                                                  top_k_weights_list, weights, act_fn, act_limit,
                                                  num_global_experts, expert_ids_per_ep_rank, comm_stream)
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states.view(*hidden_states_shape)

    return domino_experts_forward


def dispatch_mlp_combine(ep_group, fused,
                         hidden_states_list: list[torch.Tensor],
                         top_k_index_list: list[torch.Tensor],
                         top_k_weights_list: list[torch.Tensor],
                         weights,
                         act_fn,
                         act_limit,
                         num_global_experts,
                         expert_ids_per_ep_rank,
                         comm_stream=None):
    if not (
        len(hidden_states_list) == len(top_k_index_list) == len(top_k_weights_list) == _DOMINO_NUM_SLICES
    ):
        raise AssertionError(f"DOMINO dispatcher expects exactly {_DOMINO_NUM_SLICES} slices.")

    preprocess_ctx = dispatch_preprocess(ep_group, top_k_index_list, num_global_experts, expert_ids_per_ep_rank)
    gate_up_weights, down_weights = weights
    if comm_stream is None:
        comm_stream = torch.accelerator.Stream()
    hidden_dim = hidden_states_list[0].shape[-1]

    dispatch_fwd_events = [None] * _DOMINO_NUM_SLICES
    dispatch_bwd_events = [None] * _DOMINO_NUM_SLICES
    combine_fwd_events = [None] * _DOMINO_NUM_SLICES
    combine_bwd_events = [None] * _DOMINO_NUM_SLICES

    unpermute_indices_list = [[None, None] for _ in range(_DOMINO_NUM_SLICES)]
    permute_indices_list = [None] * _DOMINO_NUM_SLICES
    split_sizes_list = [None] * _DOMINO_NUM_SLICES

    p1_buf = [None] * _DOMINO_NUM_SLICES
    dispatch_buf = [None] * _DOMINO_NUM_SLICES
    up2_buf = [None] * _DOMINO_NUM_SLICES
    combine_buf = [None] * _DOMINO_NUM_SLICES
    output_list = [None] * _DOMINO_NUM_SLICES
    dispatch_producer_events = [None] * _DOMINO_NUM_SLICES

    def prepare_dispatch_input(i):
        permute_indices_list[i], split_sizes_list[i] = prepare_dispatch_slice(preprocess_ctx, i)
        hs = hidden_states_list[i].reshape(-1, hidden_states_list[i].shape[-1])
        p1_out, up_idx1 = permute(hs, top_k_index_list[i], use_eager=not fused)
        unpermute_indices_list[i][0] = up_idx1
        if p1_out.grad_fn is not None:
            p1_out.grad_fn.register_prehook(
                wait_event_backward_pre_hook(dispatch_bwd_events, i))
        p1_buf[i] = p1_out

        dispatch_producer_events[i] = torch.accelerator.current_stream().record_event()

    def launch_dispatch_slice(i):
        inp_split, out_splits = split_sizes_list[i]
        dispatch_buf[i] = domino_all_to_all(
            ep_group, p1_buf[i], out_splits, inp_split,
            comm_stream=comm_stream, producer_event=dispatch_producer_events[i],
            forward_event_list=dispatch_fwd_events,
            backward_event_list=dispatch_bwd_events, index=i)

    def wait_dispatch_and_permute_local(i):
        torch.accelerator.current_stream().wait_event(dispatch_fwd_events[i])

        num_tokens, global_idx = permute_indices_list[i]
        inp_split, out_splits = split_sizes_list[i]
        is_empty = out_splits.sum() == 0

        if is_empty:
            up2_buf[i] = empty_expert_output(dispatch_buf[i], hidden_dim, combine_bwd_events, i)
            return inp_split, out_splits, None, None, None

        p2_out, up_idx2 = permute(dispatch_buf[i], global_idx, use_eager=not fused)
        unpermute_indices_list[i][1] = up_idx2
        return inp_split, out_splits, num_tokens, p2_out, up_idx2

    def finish_local_expert_compute(i, num_tokens, p2_out, up_idx2):
        if p2_out is None:
            return

        mlp_out = experts_computation(
            p2_out, num_tokens, gate_up_weights, down_weights, act_fn, act_limit, fused)

        up2_out = unpermute(mlp_out, up_idx2, use_eager=not fused)
        if up2_out.grad_fn is not None:
            up2_out.grad_fn.register_prehook(
                wait_event_backward_pre_hook(combine_bwd_events, i))
        up2_buf[i] = up2_out

    def launch_combine_slice(i, inp_split, out_splits):
        combine_producer_event = torch.accelerator.current_stream().record_event()
        combine_buf[i] = domino_all_to_all(
            ep_group, up2_buf[i], inp_split, out_splits,
            comm_stream=comm_stream, producer_event=combine_producer_event,
            forward_event_list=combine_fwd_events,
            backward_event_list=combine_bwd_events, index=i)

    prepare_dispatch_input(0)
    launch_dispatch_slice(0)
    prepare_dispatch_input(1)

    for i in range(_DOMINO_NUM_SLICES):
        inp_split, out_splits, num_tokens, p2_out, up_idx2 = wait_dispatch_and_permute_local(i)
        if i == 0:
            launch_dispatch_slice(1)
        finish_local_expert_compute(i, num_tokens, p2_out, up_idx2)
        launch_combine_slice(i, inp_split, out_splits)

    for i in range(_DOMINO_NUM_SLICES):
        torch.accelerator.current_stream().wait_event(combine_fwd_events[i])
        up_idx1, _ = unpermute_indices_list[i]
        output_list[i] = unpermute(combine_buf[i], up_idx1, top_k_weights_list[i], use_eager=not fused)

    return output_list


def dispatch_preprocess(ep_group, top_k_index_list, num_global_experts, expert_ids_per_ep_rank):
    """
    Preprocess step to calculate permutation indices and communication split sizes.

    Args:
        ep_group: Expert parallel process group
        top_k_index_list: List of top-k index tensors
        num_global_experts: Total number of experts across all processes
        expert_ids_per_ep_rank: Expert IDs assigned to current process rank

    """
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts
    # [S, E] --> [EP * S, E], where S is the fixed number of DOMINO slices.
    local_tokens_per_experts = torch.stack([
        torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
        for top_k_index in top_k_index_list
    ], dim=0)
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(local_tokens_per_experts, ep_group)
    # [EP * S, E] --> [EP, S, E]
    num_global_tokens_per_expert = num_global_tokens_per_expert.reshape(
        ep_size, _DOMINO_NUM_SLICES, num_global_experts)

    return {
        "ep_size": ep_size,
        "num_local_experts": num_local_experts,
        "local_experts_start_id": local_experts_start_id,
        "local_experts_end_id": local_experts_end_id,
        "expert_ids_per_ep_rank": expert_ids_per_ep_rank,
        "local_tokens_per_experts": local_tokens_per_experts,
        "global_tokens_per_expert": num_global_tokens_per_expert,
        "prepared_slices": [None] * _DOMINO_NUM_SLICES,
    }


def prepare_dispatch_slice(preprocess_ctx, list_index):
    prepared_slice = preprocess_ctx["prepared_slices"][list_index]
    if prepared_slice is not None:
        return prepared_slice

    ep_size = preprocess_ctx["ep_size"]
    num_local_experts = preprocess_ctx["num_local_experts"]
    local_experts_start_id = preprocess_ctx["local_experts_start_id"]
    local_experts_end_id = preprocess_ctx["local_experts_end_id"]
    expert_ids_per_ep_rank = preprocess_ctx["expert_ids_per_ep_rank"]
    local_tokens_per_experts = preprocess_ctx["local_tokens_per_experts"]
    num_global_tokens_per_expert = preprocess_ctx["global_tokens_per_expert"]

    num_local_tokens_per_expert = local_tokens_per_experts[list_index]
    # [EP, 2, E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, list_index, local_experts_start_id: local_experts_end_id]

    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    # [E] --> [EP, local_E] --> [EP]
    input_split = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1).to(
        torch.device("cpu")).numpy()
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()

    prepared_slice = (num_tokens_per_local_expert, global_indices), (input_split, output_splits)
    preprocess_ctx["prepared_slices"][list_index] = prepared_slice
    return prepared_slice


def domino_all_to_all(group, inputs, output_split_sizes=None, input_split_sizes=None, comm_stream=None,
                      producer_event=None, forward_event_list=None, backward_event_list=None, index=0):
    """
    Wrapper for autograd function implementing DOMINO optimized AllToAll communication.

    Args:
        group: Process group for communication
        inputs: Input tensor to distribute
        output_split_sizes: Sizes for output splitting (for alltoall-v)
        input_split_sizes: Sizes for input splitting (for alltoall-v)
        comm_stream: CUDA stream for communication
        forward_event: Event to synchronize forward pass
        backward_event: Event to synchronize backward pass

    """
    return Domino_AllToAll.apply(group, inputs, output_split_sizes, input_split_sizes, comm_stream, producer_event,
                                 forward_event_list, backward_event_list, index)


def empty_expert_output(dispatch_buf: torch.Tensor, hidden_dim: int, backward_event_list: list, index: int):
    output = dispatch_buf.reshape(-1, hidden_dim)[:0]
    if output.grad_fn is not None:
        output.grad_fn.register_prehook(wait_event_backward_pre_hook(backward_event_list, index))
    return output


class Domino_AllToAll(torch.autograd.Function):
    """
    Custom autograd function implementing DOMINO optimized AllToAll communication
    with overlapping computation and communication.
    """

    @staticmethod
    def forward(ctx, group, inputs, output_split_sizes, input_split_sizes,
                comm_stream: torch.cuda.Stream = None,
                producer_event: torch.cuda.Event = None,
                forward_event_list: torch.cuda.Event = None,
                backward_event_list: torch.cuda.Event = None,
                index: int = 0):
        """
        Forward pass implementing AllToAll communication.

        Args:
            ctx: Context object to save variables for backward pass
            group: Process group for communication
            inputs: Input tensor to distribute
            output_split_sizes: Sizes for output splitting
            input_split_sizes: Sizes for input splitting
            comm_stream: CUDA stream for communication
            forward_event: Event to synchronize forward pass
            backward_event: Event to record for backward synchronization

        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.forward_event_list = forward_event_list
        ctx.backward_event_list = backward_event_list
        ctx.comm_stream = comm_stream
        ctx.index = index
        default_stream = torch.accelerator.current_stream()
        with torch.accelerator.stream(comm_stream):
            if producer_event is None:
                comm_stream.wait_stream(default_stream)
            else:
                comm_stream.wait_event(producer_event)
            output = _all_to_all_single(group, inputs, output_split_sizes, input_split_sizes)
            forward_event_list[index] = torch.accelerator.current_stream().record_event()
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        comm_stream = ctx.comm_stream
        index = ctx.index
        default_stream = torch.accelerator.current_stream()
        with torch.accelerator.stream(comm_stream):
            comm_stream.wait_stream(default_stream)
            res = _all_to_all_single(ctx.group, grad_output[0], ctx.input_split_sizes, ctx.output_split_sizes)
            ctx.backward_event_list[index] = torch.accelerator.current_stream().record_event()
        return None, res, None, None, None, None, None, None, None


def _all_to_all_single(group, inputs, output_split_sizes=None, input_split_sizes=None):
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return inputs

    inputs = inputs.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(inputs)
    else:
        output = inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]),
                                  dtype=inputs.dtype, device=torch.accelerator.current_device_index())
    torch.distributed.all_to_all_single(output, inputs, output_split_sizes=output_split_sizes,
                                        input_split_sizes=input_split_sizes, group=group)
    return output


def wait_event_backward_pre_hook(backward_event_list: list, index: int):
    """
    Create a backward pre-hook that waits for an event recorded by an async
    backward all-to-all.

    Args:
        backward_event_list: List populated by Domino_AllToAll.backward.
        index: Event index for the current DOMINO slice.

    Returns:
        Hook function that can be registered with autograd functions
    """

    def _backward_pre_hook(*_):
        backward_event = backward_event_list[index]
        if backward_event is not None:
            torch.accelerator.current_stream().wait_event(backward_event)

    return _backward_pre_hook


def slice_hidden_states_for_domino(
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Slice hidden states and related tensors into two chunks for DOMINO processing.

    Args:
        hidden_states: Input tensor to be sliced
        top_k_index: Top-k indices tensor
        top_k_weights: Top-k weights tensor

    Returns:
        Tuple of (hidden_states_list, top_k_index_list, top_k_weights_list)
    """

    batch_size = hidden_states.size(0)
    split_idx = (batch_size + 1) // 2

    hidden_states_list = [hidden_states[:split_idx], hidden_states[split_idx:]]
    top_k_index_list = [top_k_index[:split_idx], top_k_index[split_idx:]]
    top_k_weights_list = [top_k_weights[:split_idx], top_k_weights[split_idx:]]

    return hidden_states_list, top_k_index_list, top_k_weights_list
