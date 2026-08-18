import socket
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel import moonep_adapter
from fsdp_turbo.fsdp_turbo_config import MoonEPConfig


def _patch_runtime_environment(monkeypatch, accelerator_type):
    accelerator = SimpleNamespace(
        current_accelerator=lambda: SimpleNamespace(type=accelerator_type),
        current_device_index=lambda: 0,
    )
    monkeypatch.setattr(torch, "accelerator", accelerator)
    monkeypatch.setattr(moonep_adapter.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moonep_adapter.dist, "get_world_size", lambda group=None: 2)

    def gather_hosts(hosts, local_host, group=None):
        assert local_host == socket.gethostname()
        hosts[:] = [local_host, local_host]

    monkeypatch.setattr(moonep_adapter.dist, "all_gather_object", gather_hosts)
    imports = SimpleNamespace(nvl_multicast_supported=lambda: True)
    monkeypatch.setattr(moonep_adapter, "_load_moonep", lambda: imports)
    return imports


def _projection_tensor(accelerator_type):
    return SimpleNamespace(
        device=SimpleNamespace(type=accelerator_type),
        dtype=torch.bfloat16,
        ndim=3,
        shape=(4, 128, 128),
        is_contiguous=lambda: True,
        element_size=lambda: 2,
    )


@pytest.mark.parametrize("accelerator_type", ["cuda", "npu"])
def test_projection_validation_uses_current_accelerator_type(accelerator_type):
    moonep_adapter._validate_projection_shape(
        "gate_up",
        _projection_tensor(accelerator_type),
        ep_size=2,
        granularity=65536,
        accelerator_type=accelerator_type,
    )


def test_projection_validation_rejects_mismatched_accelerator():
    with pytest.raises(RuntimeError, match="current npu accelerator"):
        moonep_adapter._validate_projection_shape(
            "gate_up",
            _projection_tensor("cuda"),
            ep_size=2,
            granularity=65536,
            accelerator_type="npu",
        )


@pytest.mark.parametrize("accelerator_type", ["cuda", "npu"])
def test_runtime_uses_the_same_moonep_api_for_accelerators(
    monkeypatch, accelerator_type
):
    imports = _patch_runtime_environment(monkeypatch, accelerator_type)
    runtime = moonep_adapter.MoonEPRuntime(
        group=object(),
        ep_mesh=None,
        num_experts=4,
        config=MoonEPConfig(),
    )

    assert runtime.imports is imports
    assert runtime.accelerator_type == accelerator_type


def test_runtime_rejects_unsupported_accelerator(monkeypatch):
    _patch_runtime_environment(monkeypatch, "cpu")
    with pytest.raises(RuntimeError, match="CUDA or NPU"):
        moonep_adapter.MoonEPRuntime(
            group=object(),
            ep_mesh=None,
            num_experts=4,
            config=MoonEPConfig(),
        )


def test_runtime_locks_shape_without_mutating_config(monkeypatch):
    _patch_runtime_environment(monkeypatch, "cuda")
    config = MoonEPConfig()
    runtime = moonep_adapter.MoonEPRuntime(
        group=object(), ep_mesh=None, num_experts=4, config=config
    )
    buffer = object()
    runtime._buffers[(512, 128, 4)] = buffer

    call = runtime.new_call(tokens_per_rank=512, hidden_dim=128, top_k=4)

    assert call.buffer is buffer
    assert config.tokens_per_rank is None
    assert config.top_k is None
    with pytest.raises(RuntimeError, match="static token shape"):
        runtime.new_call(tokens_per_rank=256, hidden_dim=128, top_k=4)
    with pytest.raises(RuntimeError, match="static top-k"):
        runtime.new_call(tokens_per_rank=512, hidden_dim=128, top_k=2)


def test_dispatch_async_finish_is_forced_off_on_cuda():
    cuda_runtime = SimpleNamespace(
        accelerator_type="cuda",
        config=SimpleNamespace(async_finish=True),
    )
    npu_runtime = SimpleNamespace(
        accelerator_type="npu",
        config=SimpleNamespace(async_finish=True),
    )
    assert moonep_adapter._dispatch_async_finish(cuda_runtime) is False
    assert moonep_adapter._dispatch_async_finish(npu_runtime) is True


def test_cuda_dispatch_and_combine_use_dispatch_async_helper():
    import inspect

    dispatch_src = inspect.getsource(moonep_adapter._MoonEPDispatch)
    combine_src = inspect.getsource(moonep_adapter._MoonEPWeightedCombine)
    assert "_dispatch_async_finish" in dispatch_src
    assert "_dispatch_async_finish" in combine_src
    assert "async_finish=config.async_finish" not in dispatch_src
    assert "async_finish=config.async_finish" not in combine_src


def test_prefetch_makes_experts_contiguous():
    import inspect

    src = inspect.getsource(moonep_adapter.MoonEPSymmetricProjection.prefetch)
    assert "experts.contiguous()" in src
