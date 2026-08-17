"""The weight-gradient bridge must reduce after both expert matmuls."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep_adapter import (
    MoonEPSymmetricProjection,
    moonep_bind_weight_grads,
)
from fsdp_turbo.distributed.expert_parallel.moonep_dispatcher import (
    _grouped_matmul_with_static_tail,
)

NUM_EXPERTS = 4
EXPERTS_PER_RANK = 2
RANK = 1
TOTAL_ROWS = NUM_EXPERTS + EXPERTS_PER_RANK
LOCAL_START = RANK * EXPERTS_PER_RANK
LOCAL_END = LOCAL_START + EXPERTS_PER_RANK

PLAN = SimpleNamespace(
    experts_to_copy=torch.full((2, EXPERTS_PER_RANK), -1, dtype=torch.int32)
)
COMM_CONTEXT = {
    "meta_buf": torch.zeros(1, dtype=torch.int32),
    "meta_chunk_padded": 1,
    "BARRIER_OFF": 0,
    "grid_sync_bar": torch.zeros(1, dtype=torch.int32),
}


def _projection(events, name, out_features, in_features):
    projection = object.__new__(MoonEPSymmetricProjection)
    projection._grad_sink = None
    main_grad = torch.zeros(TOTAL_ROWS, out_features, in_features, dtype=torch.float32)
    projection.pool = SimpleNamespace(
        main_grad=main_grad,
        owner_grad_full=main_grad[:NUM_EXPERTS],
        reduce_buffers=main_grad[:NUM_EXPERTS].view(
            NUM_EXPERTS // EXPERTS_PER_RANK, EXPERTS_PER_RANK, out_features, in_features
        ),
    )
    projection.runtime = SimpleNamespace(
        imports=SimpleNamespace(
            launch_inter_rank_sync=lambda comm_context: events.append(f"sync:{name}"),
            launch_grad_reduce=lambda *args, **kwargs: events.append(f"reduce:{name}"),
        ),
        num_experts=NUM_EXPERTS,
        rank=RANK,
        experts_per_rank=EXPERTS_PER_RANK,
        config=SimpleNamespace(num_sms=8),
    )
    return projection


def _build_layer(events, hidden_dim=4, intermediate=6, tokens=6, seed=0):
    torch.manual_seed(seed)
    gate_up = _projection(events, "gate_up", intermediate, hidden_dim)
    down = _projection(events, "down", hidden_dim, intermediate)

    gate_up_weight = torch.randn(TOTAL_ROWS, intermediate, hidden_dim)
    down_weight = torch.randn(TOTAL_ROWS, hidden_dim, intermediate)
    gate_up_param = torch.nn.Parameter(gate_up_weight[LOCAL_START:LOCAL_END].clone())
    down_param = torch.nn.Parameter(down_weight[LOCAL_START:LOCAL_END].clone())

    # Every token lands on a local expert or a prefetch slot, matching the
    # planner's guarantee that remote rows stay empty.
    cu_seqlens = torch.tensor([0, 0, 3, 5, 5, tokens], dtype=torch.int32)
    dispatched = torch.randn(tokens, hidden_dim, requires_grad=True)

    bound = moonep_bind_weight_grads(
        dispatched, gate_up_param, down_param, gate_up, down, PLAN, COMM_CONTEXT
    )
    fc1 = _grouped_matmul_with_static_tail(
        bound, cu_seqlens, gate_up_weight, grad_weight_sink=gate_up.grad_weight_sink()
    )
    activated = torch.nn.functional.silu(fc1)
    output = _grouped_matmul_with_static_tail(
        activated.contiguous(),
        cu_seqlens,
        down_weight,
        grad_weight_sink=down.grad_weight_sink(),
    )
    return SimpleNamespace(
        output=output,
        dispatched=dispatched,
        cu_seqlens=cu_seqlens,
        gate_up=gate_up,
        down=down,
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
        gate_up_param=gate_up_param,
        down_param=down_param,
    )


def test_reduce_runs_after_both_projections_wrote_their_gradients():
    events = []
    layer = _build_layer(events)

    written = {}
    original = MoonEPSymmetricProjection.reduce_gradient

    def recording_reduce_gradient(self, plan, comm_context, dtype):
        name = "gate_up" if self is layer.gate_up else "down"
        written[name] = self.pool.main_grad.abs().sum().item()
        return original(self, plan, comm_context, dtype)

    MoonEPSymmetricProjection.reduce_gradient = recording_reduce_gradient
    try:
        layer.output.sum().backward()
    finally:
        MoonEPSymmetricProjection.reduce_gradient = original

    # Both sinks already hold gradients when the first reduction starts.
    assert written["down"] > 0
    assert written["gate_up"] > 0
    # The reduction order is fixed so every rank enters MoonEP's barriers alike.
    assert [event for event in events if event.startswith("reduce")] == [
        "reduce:down",
        "reduce:gate_up",
    ]
    assert events == ["sync:down", "reduce:down", "sync:gate_up", "reduce:gate_up"]


def test_bridge_forwards_activations_and_returns_parameter_gradients():
    layer = _build_layer([])
    layer.output.sum().backward()

    assert layer.dispatched.grad is not None
    for param, projection in (
        (layer.gate_up_param, layer.gate_up),
        (layer.down_param, layer.down),
    ):
        assert param.grad is not None
        assert param.grad.dtype == param.dtype
        torch.testing.assert_close(
            param.grad,
            projection.pool.main_grad[LOCAL_START:LOCAL_END].to(param.dtype),
            rtol=0,
            atol=0,
        )


def test_expert_gradients_match_a_plain_autograd_reference():
    layer = _build_layer([])
    layer.output.sum().backward()

    # Same computation with the weights inside the autograd graph.
    reference = _build_layer([], seed=0)
    gate_up_reference = reference.gate_up_weight.detach().clone().requires_grad_(True)
    down_reference = reference.down_weight.detach().clone().requires_grad_(True)
    dispatched_reference = reference.dispatched.detach().clone().requires_grad_(True)
    fc1 = _grouped_matmul_with_static_tail(
        dispatched_reference, reference.cu_seqlens, gate_up_reference
    )
    output = _grouped_matmul_with_static_tail(
        torch.nn.functional.silu(fc1).contiguous(),
        reference.cu_seqlens,
        down_reference,
    )
    output.sum().backward()

    torch.testing.assert_close(layer.output, output)
    torch.testing.assert_close(layer.dispatched.grad, dispatched_reference.grad)
    for projection, expected in (
        (layer.gate_up, gate_up_reference.grad),
        (layer.down, down_reference.grad),
    ):
        # Only rows this rank owns are written; the rest belong to remote ranks.
        torch.testing.assert_close(
            projection.pool.main_grad[LOCAL_START:LOCAL_END],
            expected[LOCAL_START:LOCAL_END],
        )
        torch.testing.assert_close(
            projection.pool.main_grad[NUM_EXPERTS:], expected[NUM_EXPERTS:]
        )
