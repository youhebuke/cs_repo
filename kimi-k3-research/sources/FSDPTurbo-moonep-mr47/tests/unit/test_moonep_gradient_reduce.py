from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (
    MoonEPSymmetricProjection,
)


def test_gradient_reduce_synchronizes_before_remote_reads_and_releases_full_storage():
    calls = []
    num_experts, experts_per_rank, rank = 4, 2, 1
    full_grad = torch.arange(
        (num_experts + experts_per_rank) * 2 * 2, dtype=torch.bfloat16
    ).reshape(num_experts + experts_per_rank, 2, 2)
    reduce_buffers = torch.zeros(
        2, experts_per_rank, 2, 2, dtype=torch.float32
    )
    owner_grad_full = torch.zeros(num_experts, 2, 2, dtype=torch.float32)
    owner_grad_buffers = owner_grad_full.view(2, experts_per_rank, 2, 2)

    def launch_inter_rank_sync(comm_context):
        assert comm_context is context
        calls.append("sync")

    def launch_grad_reduce(remote_expert_grads, remote_reduce_buffers, *args, **kwargs):
        assert calls == ["sync"]
        assert remote_expert_grads is owner_grad_full
        assert remote_reduce_buffers is reduce_buffers
        calls.append("reduce")
        # Simulate a contribution received for this rank's owned experts.
        start = rank * experts_per_rank
        remote_expert_grads[start:start + experts_per_rank].add_(100)

    imports = SimpleNamespace(
        launch_inter_rank_sync=launch_inter_rank_sync,
        launch_grad_reduce=launch_grad_reduce,
    )
    runtime = SimpleNamespace(
        imports=imports,
        num_experts=num_experts,
        rank=rank,
        experts_per_rank=experts_per_rank,
        config=SimpleNamespace(num_sms=8),
    )
    projection = object.__new__(MoonEPSymmetricProjection)
    projection.runtime = runtime
    projection.pool = SimpleNamespace(
        reduce_buffers=reduce_buffers,
        owner_grad_full=owner_grad_full,
        owner_grad_buffers=owner_grad_buffers,
    )
    plan = SimpleNamespace(
        experts_to_copy=torch.full((2, experts_per_rank), -1, dtype=torch.int32)
    )
    context = {
        "meta_buf": torch.zeros(1, dtype=torch.int32),
        "meta_chunk_padded": 1,
        "BARRIER_OFF": 0,
        "grid_sync_bar": torch.zeros(1, dtype=torch.int32),
    }

    local_grad = projection.reduce_gradient(full_grad, plan, context)

    assert calls == ["sync", "reduce"]
    torch.testing.assert_close(
        reduce_buffers[rank],
        full_grad[num_experts:num_experts + experts_per_rank].float(),
        rtol=0,
        atol=0,
    )
    owned_start = rank * experts_per_rank
    owned_end = owned_start + experts_per_rank
    expected = full_grad[owned_start:owned_end].float() + 100
    torch.testing.assert_close(owner_grad_buffers[rank], expected, rtol=0, atol=0)
    torch.testing.assert_close(local_grad, expected.to(torch.bfloat16), rtol=0, atol=0)
    assert local_grad.dtype == torch.bfloat16
    assert local_grad.untyped_storage().nbytes() == (
        local_grad.numel() * local_grad.element_size()
    )
