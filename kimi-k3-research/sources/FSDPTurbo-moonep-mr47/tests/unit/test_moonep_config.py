import textwrap

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.fsdp_turbo_config import (
    DistributedConfig,
    EPPlanConfig,
    FSDPTurboConfig,
    MoonEPConfig,
    QuantizationConfig,
    QuantizeConfig,
    load_config_from_yaml,
)


def _distributed_config(**kwargs):
    values = {
        "expert_parallel_size": 2,
        "expert_fully_shard_parallel_size": 1,
        "ep_plan": EPPlanConfig(
            apply_modules=["model.layers.{*}.mlp.experts"], dispatcher="moonep"
        ),
    }
    values.update(kwargs)
    return DistributedConfig(**values)


def test_moonep_defaults_are_materialized():
    config = FSDPTurboConfig(distributed=_distributed_config())
    moonep_config = config.distributed.ep_plan.moonep_config
    assert moonep_config == MoonEPConfig()
    assert config.distributed.ep_plan.apply_efsdp_modules == []


@pytest.mark.parametrize(
    "moon_config, message",
    [
        (MoonEPConfig(num_sms=0), "num_sms"),
        (MoonEPConfig(token_padding=0), "token_padding"),
        (MoonEPConfig(tokens_per_rank=0), "tokens_per_rank"),
        (MoonEPConfig(top_k=0), "top_k"),
    ],
)
def test_moonep_rejects_invalid_tuning(moon_config, message):
    plan = EPPlanConfig(
        apply_modules=["experts"], dispatcher="moonep", moonep_config=moon_config
    )
    with pytest.raises(ValueError, match=message):
        FSDPTurboConfig(distributed=_distributed_config(ep_plan=plan))


def test_moonep_rejects_expert_fsdp():
    with pytest.raises(ValueError, match="expert FSDP"):
        FSDPTurboConfig(
            distributed=_distributed_config(expert_fully_shard_parallel_size=2)
        )


def test_moonep_requires_parallel_group():
    with pytest.raises(ValueError, match="expert_parallel_size"):
        FSDPTurboConfig(
            distributed=DistributedConfig(
                expert_parallel_size=1,
                ep_plan=EPPlanConfig(apply_modules=["experts"], dispatcher="moonep"),
            )
        )


def test_moonep_rejects_moe_quantization():
    with pytest.raises(ValueError, match="MoE quantization"):
        FSDPTurboConfig(
            distributed=_distributed_config(),
            quantization=QuantizationConfig(
                quantization_plan=QuantizeConfig(
                    quant_recipe="mxfp8", converters=["quantize.moe.mx"]
                )
            ),
        )


def test_yaml_loads_nested_moonep_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            distributed:
              expert_parallel_size: 2
              expert_fully_shard_parallel_size: 1
              ep_plan:
                apply_modules: ["model.layers.{*}.mlp.experts"]
                dispatcher: moonep
                moonep_config:
                  num_sms: 24
                  token_padding: 64
                  tokens_per_rank: 512
                  top_k: 4
            """
        ),
        encoding="utf-8",
    )
    config = load_config_from_yaml(path)
    moonep_config = config.distributed.ep_plan.moonep_config
    assert isinstance(moonep_config, MoonEPConfig)
    assert (moonep_config.num_sms, moonep_config.token_padding) == (24, 64)
    assert (moonep_config.tokens_per_rank, moonep_config.top_k) == (512, 4)
