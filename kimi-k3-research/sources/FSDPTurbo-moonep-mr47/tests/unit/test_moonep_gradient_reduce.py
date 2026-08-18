from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (
    MoonEPSymmetricProjection,
)


def _projection(num_experts=4, experts_per_rank=2, rank=1):
    """Build a projection whose pool mimics the [E+B] FP32 VMM mapping."""
    main_grad = torch.zeros(
        num_experts + experts_per_rank, 2, 2, dtype=torch.float32
    )
    owner_grad_full = main_grad[:num_experts]
    # Rows [E, E+B) alias this rank's reduce-buffer slots, exactly as the VMM
    # mapping does on device.
    reduce_buffers = torch.zeros(
        num_experts // experts_per_rank, experts_per_rank, 2, 2, dtype=torch.float32
    )

    projection = object.__new__(MoonEPSymmetricProjection)
    projection._grad_sink = None
    projection.pool = SimpleNamespace(
        main_grad=main_grad,
        owner_grad_full=owner_grad_full,
        owner_grad_buffers=owner_grad_full.view(
            num_experts // experts_per_rank, experts_per_rank, 2, 2
        ),
        reduce_buffers=reduce_buffers,
    )
    return projection, main_grad, owner_grad_full, reduce_buffers


def test_gradient_reduce_synchronizes_before_remote_reads_and_casts_locally():
    calls = []
    num_experts, experts_per_rank, rank = 4, 2, 1
    projection, main_grad, owner_grad_full, reduce_buffers = _projection(
        num_experts, experts_per_rank, rank
    )
    main_grad.copy_(
        torch.arange(main_grad.numel(), dtype=torch.float32).reshape(main_grad.shape)
    )
    before_reduce = main_grad.clone()

    def launch_inter_rank_sync(comm_context):
        assert comm_context is context
        calls.append("sync")

    def launch_grad_reduce(remote_expert_grads, remote_reduce_buffers, *args, **kwargs):
        assert calls == ["sync"]
        assert remote_expert_grads is owner_grad_full
        assert remote_reduce_buffers is reduce_buffers
        calls.append("reduce")
        start = rank * experts_per_rank
        remote_expert_grads[start:start + experts_per_rank].add_(100)

    projection.runtime = SimpleNamespace(
        imports=SimpleNamespace(
            launch_inter_rank_sync=launch_inter_rank_sync,
            launch_grad_reduce=launch_grad_reduce,
        ),
        num_experts=num_experts,
        rank=rank,
        experts_per_rank=experts_per_rank,
        config=SimpleNamespace(num_sms=8),
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

    local_grad = projection.reduce_gradient(plan, context, torch.bfloat16)

    assert calls == ["sync", "reduce"]
    owned_start = rank * experts_per_rank
    owned_end = owned_start + experts_per_rank
    expected = before_reduce[owned_start:owned_end] + 100
    torch.testing.assert_close(main_grad[owned_start:owned_end], expected, rtol=0, atol=0)
    torch.testing.assert_close(local_grad, expected.to(torch.bfloat16), rtol=0, atol=0)
    assert local_grad.dtype == torch.bfloat16
    # Autograd must own an independent, local-sized tensor because the pool is
    # reused by every layer with the same projection shape.
    assert local_grad.untyped_storage().nbytes() == (
        local_grad.numel() * local_grad.element_size()
    )
    # The remaining rows still hold the values the grouped matmul wrote; nothing
    # copies them into a second buffer any more.
    torch.testing.assert_close(
        main_grad[num_experts:], before_reduce[num_experts:], rtol=0, atol=0
    )


def test_grad_weight_sink_covers_local_experts_and_prefetch_slots_only():
    num_experts, experts_per_rank, rank = 4, 2, 1
    projection, main_grad, _, _ = _projection(num_experts, experts_per_rank, rank)
    projection.runtime = SimpleNamespace(
        num_experts=num_experts, rank=rank, experts_per_rank=experts_per_rank
    )

    sink = projection.grad_weight_sink()

    assert sink.buffer is main_grad
    assert sink.row_ranges == ((2, 4), (4, 6))
    assert [index for index in range(6) if sink.owns(index)] == [2, 3, 4, 5]
    # The sink is cached so its staging buffer survives across steps.
    assert projection.grad_weight_sink() is sink
