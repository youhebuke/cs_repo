# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


@register_op('all2all_grouped_matmul', 'cpu')
def all2all_grouped_matmul_cpu(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None
):
    """
    CPU implementation of all2all_grouped_matmul.

    This performs all-to-all communication followed by grouped matrix multiplication.
    For CPU, we simulate the all-to-all operation using standard distributed operations.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    world_size = torch.distributed.get_world_size(group)

    if torch.is_tensor(send_counts):
        send_counts = send_counts.tolist()
    if torch.is_tensor(recv_counts):
        recv_counts = recv_counts.tolist()

    # send_counts/recv_counts are per-expert (length == world_size * num_experts),
    # but all_to_all_single requires per-rank (length == world_size), so sum the
    # expert counts belonging to each rank.
    num_cols = len(recv_counts) // world_size
    send_counts_per_rank = [sum(send_counts[i * num_cols : (i + 1) * num_cols]) for i in range(world_size)]
    recv_counts_per_rank = [sum(recv_counts[i * num_cols : (i + 1) * num_cols]) for i in range(world_size)]

    # Perform all-to-all communication
    send_tensor = inputs
    total_recv = sum(recv_counts_per_rank)
    recv_tensor = torch.empty(
        (total_recv,) + tuple(send_tensor.shape[1:]), dtype=send_tensor.dtype, device=send_tensor.device
    )
    torch.distributed.all_to_all_single(
        recv_tensor, send_tensor, recv_counts_per_rank, send_counts_per_rank, group=group
    )

    # Compute per-expert group_sizes for grouped matmul splitting
    group_list_chunk = [recv_counts[i * num_cols : (i + 1) * num_cols] for i in range(world_size)]
    group_sizes = [sum(col) for col in zip(*group_list_chunk)]

    # Split received tensor into groups and perform matrix multiplication
    output_chunks = []
    start_idx = 0
    for i, size in enumerate(group_sizes):
        end_idx = start_idx + size
        chunk = recv_tensor[start_idx:end_idx]
        # Perform matrix multiplication with corresponding weight
        output_chunk = torch.matmul(chunk, weights[i].t())
        output_chunks.append(output_chunk)
        start_idx = end_idx

    # Concatenate results
    output = torch.cat(output_chunks, dim=0)

    # Handle shared expert if provided
    shared_output = None
    if shared_inputs is not None and shared_weight is not None:
        shared_output = torch.matmul(shared_inputs, shared_weight.t())

    if shared_inputs is not None:
        return output, shared_output
    return output


@register_op('grouped_matmul_all2all', 'cpu')
def grouped_matmul_all2all_cpu(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None
):
    """
    CPU implementation of grouped_matmul_all2all.

    This performs grouped matrix multiplication followed by all-to-all communication.
    For CPU, we simulate the operation using standard distributed operations.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    world_size = torch.distributed.get_world_size(group)

    if torch.is_tensor(send_counts):
        send_counts = send_counts.tolist()
    if torch.is_tensor(recv_counts):
        recv_counts = recv_counts.tolist()

    # send_counts/recv_counts are per-expert (length == world_size * num_experts),
    # but all_to_all_single requires per-rank (length == world_size), so sum the
    # expert counts belonging to each rank.
    num_cols = len(send_counts) // world_size

    # Compute per-expert group_sizes for grouped matmul splitting
    group_list_chunk = [send_counts[i * num_cols : (i + 1) * num_cols] for i in range(world_size)]
    group_sizes = [sum(col) for col in zip(*group_list_chunk)]

    # Sum per-rank for all_to_all_single
    send_counts_per_rank = [sum(chunk) for chunk in group_list_chunk]
    recv_counts_per_rank = [sum(recv_counts[i * num_cols : (i + 1) * num_cols]) for i in range(world_size)]

    # Perform grouped matrix multiplication first
    output_chunks = []
    start_idx = 0
    for i, size in enumerate(group_sizes):
        end_idx = start_idx + size
        chunk = inputs[start_idx:end_idx]
        # Perform matrix multiplication with corresponding weight
        output_chunk = torch.matmul(chunk, weights[i].t())
        output_chunks.append(output_chunk)
        start_idx = end_idx

    # Concatenate results
    gmm_output = torch.cat(output_chunks, dim=0)

    # Perform all-to-all communication
    send_tensor = gmm_output
    total_recv = sum(recv_counts_per_rank)
    recv_tensor = torch.empty(
        (total_recv,) + tuple(send_tensor.shape[1:]), dtype=send_tensor.dtype, device=send_tensor.device
    )

    # Perform all-to-all
    torch.distributed.all_to_all_single(
        recv_tensor, send_tensor, recv_counts_per_rank, send_counts_per_rank, group=group
    )

    output = recv_tensor

    # Handle shared expert if provided
    shared_output = None
    if shared_inputs is not None and shared_weight is not None:
        shared_output = torch.matmul(shared_inputs, shared_weight.t())

    if shared_inputs is not None:
        return output, shared_output
    return output
