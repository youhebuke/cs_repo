import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "_dynamo"):
    pytest.skip("the installed torch package is not a supported PyTorch build", allow_module_level=True)

from fsdp_turbo.distributed.expert_parallel.moonep_dispatcher import (
    _grouped_matmul_with_static_tail,
)


def _reference(inputs, offsets, weights):
    # Live groups stop at the planner's last exclusive end. Rows past that
    # belong to MoonEP's static tail and must be zero, not folded into the
    # last expert (folding clones the whole NvS buffer).
    outputs = inputs.new_zeros(inputs.shape[0], weights.shape[1])
    start = 0
    for group, end in enumerate(offsets.tolist()):
        outputs[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
        start = end
    return outputs


@pytest.mark.parametrize("requires_input_grad", [False, True])
def test_grouped_matmul_forward_backward_with_empty_group_and_static_tail(
    requires_input_grad,
):
    offsets = torch.tensor([3, 3, 7], dtype=torch.int32)
    inputs = torch.randn(
        9,
        8,
        dtype=torch.float32,
        requires_grad=requires_input_grad,
    )
    weights = torch.randn(
        3, 16, 8, dtype=torch.float32, requires_grad=True
    )
    inputs_ref = inputs.detach().clone().requires_grad_(requires_input_grad)
    weights_ref = weights.detach().clone().requires_grad_(True)

    actual = _grouped_matmul_with_static_tail(inputs, offsets, weights)
    expected = _reference(inputs_ref, offsets, weights_ref)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(actual, expected)
    if requires_input_grad:
        torch.testing.assert_close(inputs.grad, inputs_ref.grad)
    else:
        assert inputs.grad is None
    torch.testing.assert_close(weights.grad, weights_ref.grad)


def test_static_tail_garbage_does_not_enter_the_last_group():
    offsets = torch.tensor([3, 7], dtype=torch.int32)
    inputs = torch.zeros(9, 4)
    inputs[:7] = 1.0
    inputs[7:] = 100.0
    weights = torch.ones(2, 8, 4)

    actual = _grouped_matmul_with_static_tail(inputs, offsets, weights)

    torch.testing.assert_close(actual[7:], torch.zeros(2, 8), rtol=0, atol=0)
    torch.testing.assert_close(actual[:7], _reference(inputs, offsets, weights)[:7])


def test_static_tail_garbage_does_not_change_last_group_wgrad():
    offsets = torch.tensor([3, 7], dtype=torch.int32)
    dirty = torch.zeros(9, 4)
    dirty[:7] = 1.0
    dirty[7:] = 100.0
    clean = dirty.clone()
    clean[7:] = 0.0
    weights_dirty = torch.randn(2, 8, 4, requires_grad=True)
    weights_clean = weights_dirty.detach().clone().requires_grad_(True)

    _grouped_matmul_with_static_tail(dirty, offsets, weights_dirty).sum().backward()
    _grouped_matmul_with_static_tail(clean, offsets, weights_clean).sum().backward()

    torch.testing.assert_close(weights_dirty.grad, weights_clean.grad)


def test_snapshot_clones_only_when_requested(monkeypatch):
    from fsdp_turbo.distributed.expert_parallel import moonep_dispatcher

    offsets = torch.tensor([3, 7], dtype=torch.int32)
    inputs = torch.randn(9, 8, requires_grad=True)
    weights = torch.randn(2, 16, 8)
    seen = []
    original = moonep_dispatcher.grouped_matmul

    def spy(captured, *args, **kwargs):
        seen.append(captured.data_ptr())
        return original(captured, *args, **kwargs)

    monkeypatch.setattr(moonep_dispatcher, "grouped_matmul", spy)

    _grouped_matmul_with_static_tail(inputs, offsets, weights, snapshot=False)
    _grouped_matmul_with_static_tail(inputs, offsets, weights, snapshot=True)

    assert seen[0] == inputs.data_ptr()
    assert seen[1] != inputs.data_ptr()


def test_grouped_matmul_rejects_group_count_mismatch():
    with pytest.raises(RuntimeError, match="provided 2 groups"):
        _grouped_matmul_with_static_tail(
            torch.randn(4, 8),
            torch.tensor([2, 4], dtype=torch.int32),
            torch.randn(3, 16, 8),
        )


def test_grouped_matmul_on_available_accelerator():
    if not hasattr(torch, "accelerator"):
        pytest.skip("torch.accelerator is unavailable")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type not in {"cuda", "npu"}:
        pytest.skip("CUDA or NPU accelerator is unavailable")

    offsets = torch.tensor([3, 3, 7], dtype=torch.int32, device=accelerator)
    inputs = torch.randn(
        9,
        128,
        dtype=torch.bfloat16,
        device=accelerator,
        requires_grad=True,
    )
    weights = torch.randn(
        3,
        256,
        128,
        dtype=torch.bfloat16,
        device=accelerator,
        requires_grad=True,
    )
    actual = _grouped_matmul_with_static_tail(inputs, offsets, weights)

    assert actual.shape == (9, 256)
    actual.float().sum().backward()
    assert inputs.grad is not None
    assert weights.grad is not None


def test_cuda_grouped_matmul_accepts_noncontiguous_output_gradient():
    if not hasattr(torch, "accelerator"):
        pytest.skip("torch.accelerator is unavailable")
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "cuda":
        pytest.skip("CUDA accelerator is unavailable")

    offsets = torch.tensor([3, 3, 7], dtype=torch.int32, device=accelerator)
    inputs = torch.randn(
        9,
        128,
        dtype=torch.bfloat16,
        device=accelerator,
        requires_grad=True,
    )
    weights = torch.randn(
        3,
        256,
        128,
        dtype=torch.bfloat16,
        device=accelerator,
        requires_grad=True,
    )
    output = _grouped_matmul_with_static_tail(inputs, offsets, weights)
    grad_storage = torch.randn(
        output.shape[1],
        output.shape[0],
        dtype=output.dtype,
        device=output.device,
    )
    grad_output = grad_storage.transpose(0, 1)
    assert not grad_output.is_contiguous()

    output.backward(grad_output)

    assert inputs.grad is not None
    assert weights.grad is not None
