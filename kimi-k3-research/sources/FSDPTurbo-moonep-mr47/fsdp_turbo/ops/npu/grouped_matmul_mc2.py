# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


class AllToAllGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)

        if torch.is_tensor(send_counts):
            send_counts = send_counts.tolist()
        if torch.is_tensor(recv_counts):
            recv_counts = recv_counts.tolist()

        output, shared_output, permute_output = torch_npu.npu_alltoallv_gmm(
            inputs,
            weights,
            hcomm,
            ep_world_size,
            send_counts,
            recv_counts,
            mm_x=shared_inputs,
            mm_weight=shared_weight,
            trans_gmm_weight=True,
            trans_mm_weight=shared_weight is not None,
            permute_out_flag=True,
        )

        ctx.save_for_backward(weights, shared_inputs, shared_weight, permute_output)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        weights, shared_inputs, shared_weight, permute_output = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts

        num_cols = len(recv_counts) // ep_world_size
        group_list_chunk = [recv_counts[i * num_cols : (i + 1) * num_cols] for i in range(ep_world_size)]
        group_list = [sum(col) for col in zip(*group_list_chunk)]
        group_list = torch.tensor(group_list, dtype=torch.int64, device=weights.device)

        inputs_grad, shared_inputs_grad = torch_npu.npu_gmm_alltoallv(
            output_grad,
            weights,
            hcomm,
            ep_world_size,
            recv_counts,
            send_counts,
            mm_x=shared_output_grad,
            mm_weight=shared_weight,
        )

        weights_grad = torch_npu.npu_grouped_matmul(
            [output_grad.T],
            [permute_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        shared_weight_grad = None if shared_inputs is None else torch.matmul(shared_inputs.T, shared_output_grad)
        return inputs_grad, weights_grad, None, None, None, shared_inputs_grad, shared_weight_grad


class GroupedMatmulAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)

        if torch.is_tensor(send_counts):
            send_counts = send_counts.tolist()
        if torch.is_tensor(recv_counts):
            recv_counts = recv_counts.tolist()

        output, shared_output = torch_npu.npu_gmm_alltoallv(
            inputs,
            weights,
            hcomm,
            ep_world_size,
            send_counts,
            recv_counts,
            mm_x=shared_inputs,
            mm_weight=shared_weight,
            trans_gmm_weight=True,
            trans_mm_weight=shared_weight is not None,
        )

        ctx.save_for_backward(inputs, weights, shared_inputs, shared_weight)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        inputs, weights, shared_inputs, shared_weight = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts

        num_cols = len(send_counts) // ep_world_size
        group_list_chunk = [send_counts[i * num_cols : (i + 1) * num_cols] for i in range(ep_world_size)]
        group_list = [sum(col) for col in zip(*group_list_chunk)]
        group_list = torch.tensor(group_list, dtype=torch.int64, device=weights.device)

        inputs_grad, shared_inputs_grad, permute_grad = torch_npu.npu_alltoallv_gmm(
            output_grad,
            weights,
            hcomm,
            ep_world_size,
            recv_counts,
            send_counts,
            mm_x=shared_output_grad,
            mm_weight=shared_weight,
            permute_out_flag=True,
        )

        weights_grad = torch_npu.npu_grouped_matmul(
            [permute_grad.T], [inputs], bias=None, group_list=group_list, split_item=3, group_type=2, group_list_type=1
        )[0]
        shared_weight_grad = None if shared_inputs is None else torch.matmul(shared_inputs.T, shared_output_grad)
        return inputs_grad, weights_grad, None, None, None, shared_inputs_grad, shared_weight_grad


@register_op('all2all_grouped_matmul', 'npu')
def all2all_grouped_matmul_npu(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None
):
    output = AllToAllGroupedMatmul.apply(inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]


@register_op('grouped_matmul_all2all', 'npu')
def grouped_matmul_all2all_npu(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None
):
    output = GroupedMatmulAllToAll.apply(inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]
