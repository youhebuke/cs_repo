import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "_dynamo"):
    pytest.skip("the installed torch package is not a supported PyTorch build", allow_module_level=True)

from fsdp_turbo.distributed.expert_parallel.moonep_dispatcher import (
    _grouped_matmul_with_static_tail,
)


def _reference(inputs, offsets, weights):
    active_rows = torch.arange(inputs.shape[0]) < offsets[-1]
    inputs = inputs * active_rows.unsqueeze(-1).to(inputs.dtype)
    offsets = offsets.clone()
    offsets[-1] = inputs.shape[0]
    outputs = []
    start = 0
    for group, end in enumerate(offsets.tolist()):
        outputs.append(inputs[start:end] @ weights[group].transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


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
