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


def test_ensure_comm_buffer_is_noop_without_static_shape(monkeypatch):
    imports = _patch_runtime_environment(monkeypatch, "cuda")
    runtime = moonep_adapter.MoonEPRuntime(
        group=object(), ep_mesh=None, num_experts=4, config=MoonEPConfig()
    )

    def should_not_create(**kwargs):
        raise AssertionError("Buffer must not be created without S and K")

    runtime.imports.Buffer = should_not_create
    runtime.ensure_comm_buffer(128)
    assert runtime._buffers == {}


def test_ensure_comm_buffer_creates_once_when_shape_is_known(monkeypatch):
    imports = _patch_runtime_environment(monkeypatch, "cuda")
    config = MoonEPConfig(tokens_per_rank=512, top_k=8, num_sms=32)
    runtime = moonep_adapter.MoonEPRuntime(
        group=object(), ep_mesh=None, num_experts=4, config=config
    )
    created = []

    def fake_buffer(**kwargs):
        created.append(kwargs)
        return object()

    runtime.imports.Buffer = fake_buffer
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None, raising=False)

    runtime.ensure_comm_buffer(2048)
    runtime.ensure_comm_buffer(2048)

    assert len(created) == 1
    assert created[0]["S"] == 512
    assert created[0]["H"] == 2048
    assert created[0]["K"] == 8
    assert created[0]["E"] == 4
    call = runtime.new_call(tokens_per_rank=512, hidden_dim=2048, top_k=8)
    assert call.buffer is runtime._buffers[(512, 2048, 8)]
    assert len(created) == 1
