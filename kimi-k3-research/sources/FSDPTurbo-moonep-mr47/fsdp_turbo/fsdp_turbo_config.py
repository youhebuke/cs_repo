# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Callable, Literal, Union, Optional
from pathlib import Path

import torch
import yaml

from fsdp_turbo.utils.dtype import get_dtype

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    torch_dtype: torch.dtype = torch.bfloat16


@dataclass
class OptimizerConfig:
    optimizer_type: str = "AdamW"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr: float = 1e-4
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    min_lr: float = 0.0
    clip_grad: float = 1.0
    clip_grad_norm_type: float = 2.0


@dataclass
class ProfileConfig:
    enabled: bool = False
    wait_steps: int = 0
    warmup_steps: int = 0
    active_steps: int = 1
    repeat: int = 1
    skip_first: int = 0
    output_dir: str = "./profile"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False


@dataclass
class DataConfig:
    dataset_path: str = ""
    dataset_config: Optional[str] = None
    text_column: str = "text"
    split: str = "train"
    batch_size: int = 1
    max_seq_length: int = 4096
    num_workers: int = 0
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class TrainRunConfig:
    seed: int = 42
    deterministic: bool = False
    max_steps: int = -1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    profile: ProfileConfig = field(default_factory=ProfileConfig)


@dataclass
class CheckpointConfig:
    output_dir: str = "./output"
    resume_from_checkpoint: Optional[str] = None
    save_steps: int = 500
    strict: bool = True
    save_optim: bool = True
    load_optim: bool = True


@dataclass
class FSDPPlanConfig:
    ignored_modules: List[str] = field(default_factory=list)
    apply_modules: Dict[str, Any] = None

    # mp_policy settings
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    cast_forward_inputs: bool = True

    # offload_policy settings
    cpu_offload: bool = False
    pin_memory: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    # fsdp2 hook manager
    hook_modules: Optional[List[str]] = None

    # FSDP implementation strategy
    # 'custom': Use FSDPTurbo custom FSDP implementation
    # 'native': Use PyTorch native FSDP implementation (default)
    fsdp_implementation: Literal['custom', 'native'] = 'native'

    # Gradient divide factor for HSDP. Applied at reduce-scatter time so that
    # the subsequent all-reduce over the dp (replicate) dim yields the correct
    # mean across both dp and fsdp ranks. When unset (None) it is computed in
    # the config validation from the launcher-provided WORLD_SIZE (dp =
    # world_size // (fsdp * tp)). Users may override it explicitly.
    gradient_divide_factor: Optional[float] = None


@dataclass
class TPPlanConfig:
    colwise_parallel: List[str] = None
    rowwise_parallel: List[str] = None
    sequence_parallel: List[str] = None


@dataclass
class CPPlanConfig:
    ulysses_core_attention_function: Union[str, List[str]] = None
    ulysses_attention_type: Union[Literal["default", "with_compressor", "GDN"], Callable] = "default"
    ulysses_attn_output_transposed: bool = False
    kv_allgather_core_attention_function: Union[str, List[str]] = None
    kv_allgather_attention_type: Union[Literal["default", "with_compressor"], Callable] = "default"


@dataclass
class MoonEPConfig:
    """MoonEP communication and static-shape tuning options.

    ``tokens_per_rank`` and ``top_k`` may be omitted. In that case they are
    inferred on the first expert forward and locked for the lifetime of the
    runtime. MoonEP training always uses one prefetch slot per local expert.
    """

    num_sms: int = 32
    token_padding: int = 128
    comm_stream_priority: int = -1
    # CUDA forces both False at runtime: the dispatch epilogue is a cooperative
    # kernel and PDL is unsafe on MoonEP's side comm stream (Hopper SIGSEGV).
    enable_pdl: bool = True
    async_finish: bool = True
    tokens_per_rank: Optional[int] = None
    top_k: Optional[int] = None


@dataclass
class EPPlanConfig:
    apply_modules: List[str] = None
    dispatcher: Union[Literal["eager", "fused", "mc2", "domino", "moonep"], Callable] = None
    fixed_router: bool = False
    apply_efsdp_modules: List[str] = None
    moonep_config: Optional[MoonEPConfig] = None
    # Gradient divide factor for expert HSDP, analogous to
    # ``FSDPPlanConfig.gradient_divide_factor``. When unset (None) it is
    # computed in the config validation from the launcher-provided WORLD_SIZE
    # (edp = world_size // (efsdp * ep)).
    gradient_divide_factor: Optional[float] = None


@dataclass
class QuantizeConfig:
    quant_format: Optional[str] = None
    quant_recipe: Optional[str] = None
    block_size: int = 32
    quant_apply_modules: List[str] = None
    quant_ignored_modules: List[str] = None
    converters: List[str] = None
    enable_fsdp_low_precision_all_gather: bool = True
    fsdp_low_precision_all_gather_mode: str = "on-demand"
    quant_gmm: bool = False
    gemm_gradient_accumulation_fusion: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)  # for future extensibility


@dataclass
class DistributedConfig:
    data_parallel_size: int = 1

    fully_shard_parallel_size: int = 1
    fsdp_plan: FSDPPlanConfig = None

    tensor_parallel_size: int = 1
    tp_plan: TPPlanConfig = None

    kv_allgather_parallel_size: int = 1
    ulysses_parallel_size: int = 1
    cp_plan: CPPlanConfig = None

    expert_parallel_size: int = 1
    expert_fully_shard_parallel_size: int = 1
    expert_data_parallel_size: int = 1
    ep_plan: EPPlanConfig = None


@dataclass
class ChunkBatchPlanConfig:
    """Configuration for chunked micro-batch execution.

    The feature wraps selected module forwards and executes them with smaller
    slices along the configured batch dimension. This is useful when a large
    module's activation or temporary workspace peak is proportional to batch
    size.

    Example:
        ``chunk_mbs=1`` with an input tensor shaped ``[4, 2048, hidden]`` runs
        the selected forward four times with tensors shaped ``[1, 2048, hidden]``
        and then concatenates the outputs back to ``[4, 2048, hidden]``.

        For Qwen-style decoder layers, a typical plan is:
            apply_modules=["model.layers.{*}"]
            batch_dim=0
            chunk_arg_indexs=[0]
            chunk_kwarg_names=["hidden_states", "attention_mask"]
    """

    chunk_mbs: int = 1
    # Module-name patterns to patch, for example "model.layers.{*}".
    apply_modules: List[str] = None
    # Tensor dimension that represents batch. HF decoder layers usually use 0.
    batch_dim: int = 0
    # Positional forward arguments that should be sliced by batch.
    chunk_arg_indexs: Optional[List[int]] = None
    # Keyword forward arguments that should be sliced by batch.
    chunk_kwarg_names: Optional[List[str]] = None


@dataclass
class MemoryConfig:
    """Memory-saving feature switches.

    ``recompute`` reduces saved activations by re-running module forwards during
    backward. ``chunk_batch`` reduces per-module forward/backward peaks by
    splitting the batch dimension into smaller micro batches.
    """

    recompute: bool = False
    recompute_plan: List[str] = None
    chunk_batch: bool = False
    chunk_batch_plan: Optional[ChunkBatchPlanConfig] = None


@dataclass
class QuantizationConfig:
    quantization_plan: Optional[QuantizeConfig] = None


@dataclass
class FSDPTurboConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: TrainRunConfig = field(default_factory=TrainRunConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    module_patches: List[Dict[str, Any]] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def __post_init__(self):
        self.validate_tp_config()
        self.validate_cp_config()
        self.validate_module_patches()
        self.validate_ep_config()
        self.validate_recompute_config()
        self.validate_chunk_batch_config()
        self.validate_quantization_config()
        self.validate_fsdp_config()

    def validate_fsdp_config(self):
        '''fully shard plan
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                fsdp_plan=FSDPPlanConfig(
                    'ignored_modules':['*mlp.experts*'],
                    'apply_modules': {
                        'model.layers.*': {reshard_after_forward=None, shard_placement_fn=None}
                    }
                )
            )
        )
        '''
        self.distributed.fsdp_plan = (
            FSDPPlanConfig() if self.distributed.fsdp_plan is None else self.distributed.fsdp_plan
        )
        if self.distributed.fsdp_plan.gradient_divide_factor is None:
            # Derive the real dp size from the launcher-provided world size:
            # dp = world_size // (fsdp * tp). torchrun sets WORLD_SIZE
            # automatically; fall back to 1 for non-distributed runs.
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            dp = world_size // (self.distributed.fully_shard_parallel_size * self.distributed.tensor_parallel_size)
            self.distributed.fsdp_plan.gradient_divide_factor = self.distributed.fully_shard_parallel_size * dp
        if self.distributed.fully_shard_parallel_size > 1:
            if self.distributed.expert_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.ep_plan.apply_modules)
            if self.distributed.tensor_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.tp_plan.colwise_parallel)
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.tp_plan.rowwise_parallel)
            self.distributed.fsdp_plan.ignored_modules = list(
                set(self.distributed.fsdp_plan.ignored_modules)
            )  # remove duplicates

    def validate_tp_config(self):
        '''tensor parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                tp_plan=TPPlanConfig(
                    colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                    rowwise_parallel=['*.o_proj']
                )
            )
        )
        '''
        self.distributed.tp_plan = TPPlanConfig() if self.distributed.tp_plan is None else self.distributed.tp_plan
        self.distributed.tp_plan.colwise_parallel = (
            [] if self.distributed.tp_plan.colwise_parallel is None else self.distributed.tp_plan.colwise_parallel
        )
        self.distributed.tp_plan.rowwise_parallel = (
            [] if self.distributed.tp_plan.rowwise_parallel is None else self.distributed.tp_plan.rowwise_parallel
        )
        self.distributed.tp_plan.sequence_parallel = (
            [] if self.distributed.tp_plan.sequence_parallel is None else self.distributed.tp_plan.sequence_parallel
        )

    def validate_cp_config(self):
        '''context parallelize plan (Ulysses)

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                kv_allgather_parallel_size=2,
                ulysses_parallel_size=2,
                cp_plan=CPPlanConfig(
                    ulysses_core_attention_function='transformers.models.deepseek_v4.modeling_deepseek_v4.eager_attention_forward'
                )
            )
        )
        '''
        self.distributed.cp_plan = CPPlanConfig() if self.distributed.cp_plan is None else self.distributed.cp_plan
        if self.distributed.cp_plan.ulysses_core_attention_function is None:
            self.distributed.cp_plan.ulysses_core_attention_function = []
        elif isinstance(self.distributed.cp_plan.ulysses_core_attention_function, str):
            self.distributed.cp_plan.ulysses_core_attention_function = [
                self.distributed.cp_plan.ulysses_core_attention_function
            ]
        if self.distributed.cp_plan.kv_allgather_core_attention_function is None:
            self.distributed.cp_plan.kv_allgather_core_attention_function = []
        elif isinstance(self.distributed.cp_plan.kv_allgather_core_attention_function, str):
            self.distributed.cp_plan.kv_allgather_core_attention_function = [
                self.distributed.cp_plan.kv_allgather_core_attention_function
            ]

    def validate_ep_config(self):
        '''expert parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ep_plan=EPPlanConfig(
                    apply_modules: ['*mlp.experts*'],
                    dispatcher: 'eager', 'fused', 'mc2'
                )
            )
        )
        '''
        self.distributed.ep_plan = (
            EPPlanConfig(apply_modules=[], dispatcher='eager')
            if self.distributed.ep_plan is None
            else self.distributed.ep_plan
        )
        if self.distributed.ep_plan.apply_modules is None:
            self.distributed.ep_plan.apply_modules = []

        if self.distributed.ep_plan.gradient_divide_factor is None:
            # Derive the real edp size from the launcher-provided world size:
            # edp = world_size // (efsdp * ep). torchrun sets WORLD_SIZE
            # automatically; fall back to 1 for non-distributed runs.
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            edp = world_size // (
                self.distributed.expert_fully_shard_parallel_size * self.distributed.expert_parallel_size
            )
            self.distributed.ep_plan.gradient_divide_factor = self.distributed.expert_fully_shard_parallel_size * edp

        if self.distributed.ep_plan.dispatcher == "moonep":
            if self.distributed.expert_parallel_size <= 1:
                raise ValueError(
                    "MoonEP requires expert_parallel_size > 1; use the model's "
                    "local expert path when EP is disabled."
                )
            if self.distributed.expert_fully_shard_parallel_size != 1:
                raise ValueError(
                    "MoonEP does not support expert FSDP; set "
                    "expert_fully_shard_parallel_size=1 or use another dispatcher."
                )
            moonep_config = self.distributed.ep_plan.moonep_config
            if moonep_config is None:
                moonep_config = MoonEPConfig()
                self.distributed.ep_plan.moonep_config = moonep_config
            if moonep_config.num_sms <= 0:
                raise ValueError("MoonEP num_sms must be positive.")
            if moonep_config.token_padding <= 0:
                raise ValueError("MoonEP token_padding must be positive.")
            if (
                moonep_config.tokens_per_rank is not None
                and moonep_config.tokens_per_rank <= 0
            ):
                raise ValueError("MoonEP tokens_per_rank must be positive when set.")
            if moonep_config.top_k is not None and moonep_config.top_k <= 0:
                raise ValueError("MoonEP top_k must be positive when set.")
        if self.distributed.ep_plan.apply_efsdp_modules is None:
            self.distributed.ep_plan.apply_efsdp_modules = []
            if self.distributed.ep_plan.dispatcher != "moonep":
                for ep_module in self.distributed.ep_plan.apply_modules:
                    if ep_module.endswith('.experts'):
                        self.distributed.ep_plan.apply_efsdp_modules.append(ep_module.removesuffix('.experts'))

    def validate_module_patches(self):
        """Validate the namespace-neutral module patch configuration.

        Each entry describes exactly one dotted ``target`` and one callable
        ``replacement``. Callable objects are accepted for direct Python API
        use; YAML normally supplies dotted replacement paths.
        """
        if self.module_patches is None:
            self.module_patches = []
        elif isinstance(self.module_patches, dict):
            self.module_patches = [self.module_patches]
        elif not isinstance(self.module_patches, list):
            raise ValueError("module_patches must be a mapping or a list of mappings.")
        for patch in self.module_patches:
            if not isinstance(patch, dict):
                raise ValueError("module_patches entries must be mappings.")
            target = patch.get("target")
            replacement = patch.get("replacement")
            if not isinstance(target, str) or not target:
                raise ValueError("module_patches entries must set target.")
            if replacement is None:
                raise ValueError("module_patches entries must set replacement.")
            target_parts = str(target).split(".")
            if len(target_parts) < 2 or any(not part for part in target_parts):
                raise ValueError("module_patches target must use a dotted module.member or module.Class.member path.")
            if isinstance(replacement, str):
                replacement_parts = replacement.split(".")
                if ":" in replacement:
                    raise ValueError("module_patches replacement must use a dotted callable path, not a colon path.")
                if len(replacement_parts) < 2 or any(not part for part in replacement_parts):
                    raise ValueError("module_patches replacement must use a dotted callable path.")
            elif not callable(replacement):
                raise ValueError("module_patches replacement must be callable or a dotted callable path.")

    def validate_recompute_config(self):
        self.memory.recompute_plan = [] if self.memory.recompute_plan is None else self.memory.recompute_plan

    def validate_chunk_batch_config(self):
        """Normalize and validate chunk-batch configuration.

        The selected modules and at least one sliced input are required. Without
        ``apply_modules`` no module would be patched; without arg/kwarg selectors
        the wrapper cannot infer batch size or know which tensors must be sliced.
        """
        if not self.memory.chunk_batch:
            return
        self.memory.chunk_batch_plan = (
            ChunkBatchPlanConfig() if self.memory.chunk_batch_plan is None else self.memory.chunk_batch_plan
        )
        if self.memory.chunk_batch_plan.chunk_mbs <= 0:
            raise ValueError("chunk_mbs must be positive.")
        self.memory.chunk_batch_plan.apply_modules = (
            [] if self.memory.chunk_batch_plan.apply_modules is None else self.memory.chunk_batch_plan.apply_modules
        )
        self.memory.chunk_batch_plan.chunk_arg_indexs = (
            []
            if self.memory.chunk_batch_plan.chunk_arg_indexs is None
            else self.memory.chunk_batch_plan.chunk_arg_indexs
        )
        self.memory.chunk_batch_plan.chunk_kwarg_names = (
            []
            if self.memory.chunk_batch_plan.chunk_kwarg_names is None
            else self.memory.chunk_batch_plan.chunk_kwarg_names
        )
        if not self.memory.chunk_batch_plan.apply_modules:
            raise ValueError("chunk_batch_plan.apply_modules must not be empty when chunk_batch is enabled.")
        if not self.memory.chunk_batch_plan.chunk_arg_indexs and not self.memory.chunk_batch_plan.chunk_kwarg_names:
            raise ValueError(
                "chunk_batch_plan must specify chunk_arg_indexs or chunk_kwarg_names when chunk_batch is enabled."
            )

    def validate_quantization_config(self):
        self.quantization.quantization_plan = (
            QuantizeConfig() if self.quantization.quantization_plan is None else self.quantization.quantization_plan
        )
        ep_plan = self.distributed.ep_plan
        quant_plan = self.quantization.quantization_plan
        if ep_plan.dispatcher == "moonep" and quant_plan.quant_recipe:
            converters = quant_plan.converters or []
            if "quantize.moe.mx" in converters or quant_plan.quant_gmm:
                raise ValueError(
                    "MoonEP requires BF16 expert weights and is incompatible with MoE quantization. "
                    "Remove quantize.moe.mx/quant_gmm or use another dispatcher."
                )

    def __str__(self):
        import dataclasses as dc

        def _format_value(v):
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return "[]"
                if len(v) <= 3:
                    return str(v)
                return f"[{v[0]}, {v[1]}, ... ] (len={len(v)})"
            if isinstance(v, dict):
                if len(v) == 0:
                    return "{}"
                return str(v)
            if isinstance(v, torch.dtype):
                return str(v)
            return repr(v)

        # First pass: collect all leaf fields to determine max name width.
        def _collect_leaves(obj, prefix=""):
            leaves = []
            for f in dc.fields(obj):
                val = getattr(obj, f.name)
                display = f.name if not prefix else f"{prefix}.{f.name}"
                if val is None:
                    leaves.append((display, "None"))
                elif dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                    leaves.extend(_collect_leaves(val, display))
                else:
                    leaves.append((display, _format_value(val)))
            return leaves

        sep = "=" * 60
        # Auto-discover sections from dataclass fields that are themselves dataclasses.
        sections = []
        for f in dc.fields(self):
            val = getattr(self, f.name)
            if dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                sections.append((f.name.capitalize(), val))

        # Collect all leaves across sections to find max name width.
        all_leaves = []
        for section_name, section_obj in sections:
            all_leaves.extend(_collect_leaves(section_obj))
        max_name_len = max(len(name) for name, _ in all_leaves) if all_leaves else 20

        # Second pass: render with aligned values.
        def _format_dataclass(obj, indent=2, prefix=""):
            lines = []
            for f in dc.fields(obj):
                val = getattr(obj, f.name)
                display = f.name if not prefix else f"{prefix}.{f.name}"
                if val is None:
                    val_str = "None"
                elif dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                    lines.extend(_format_dataclass(val, indent, display))
                    continue
                else:
                    val_str = _format_value(val)
                # Pad name + dashes so that value starts at value_col.
                padded_name = " " * indent + display
                total_name_width = indent + max_name_len
                dash_count = max(1, total_name_width - len(padded_name) + 2)
                lines.append(f"{padded_name} {'-' * dash_count} {val_str}")
            return lines

        lines = [sep, f'{"FSDPTurboConfig":^60}', sep]
        for section_name, section_obj in sections:
            lines.append(f"  [{section_name}]")
            lines.extend(_format_dataclass(section_obj, indent=4))
            lines.append("")
        if self.module_patches:
            lines.append("  [Module_patches]")
            lines.append(f"    module_patches ---- {self.module_patches!r}")
            lines.append("")
        lines.append(sep)
        return "\n".join(lines)


def _coerce_value(value, field_type):
    if not isinstance(value, str):
        return value

    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = [a for a in field_type.__args__ if a is not type(None)]  # pylint: disable=unidiomatic-typecheck
        if args:
            field_type = args[0]

    if field_type is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning("Cannot convert '%s' to float, keeping as str", value)
            return value

    if field_type is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning("Cannot convert '%s' to int, keeping as str", value)
            return value

    if field_type is torch.dtype:
        try:
            return get_dtype(value)
        except (ValueError, TypeError) as e:
            logger.warning("Cannot convert '%s' to torch.dtype: %s, keeping as str", value, e)
            return value

    return value


def _dict_to_dataclass(cls, data):
    if data is None:
        return None
    if not isinstance(data, dict):
        return data
    import dataclasses as dc

    field_types = {f.name: f.type for f in dc.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            logger.warning("Ignoring unknown config key '%s' for %s", key, cls.__name__)
            continue
        ft = field_types[key]
        origin = getattr(ft, "__origin__", None)
        optional_dataclass = None
        if origin is Union:
            for arg in ft.__args__:
                if arg is not type(None) and dc.is_dataclass(arg):  # pylint: disable=unidiomatic-typecheck
                    optional_dataclass = arg
                    break
        if dc.is_dataclass(ft) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        elif optional_dataclass is not None and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(optional_dataclass, value)
        else:
            kwargs[key] = _coerce_value(value, ft)
    return cls(**kwargs)


def load_config_from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Extract top-level sub-configs (promoted from old TrainConfig)
    model_raw = raw.pop("model", None)
    optimizer_raw = raw.pop("optimizer", None)
    data_raw = raw.pop("data", None)
    run_raw = raw.pop("run", None)
    checkpoint_raw = raw.pop("checkpoint", None)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw) if model_raw is not None else ModelConfig()
    optimizer_cfg = (
        _dict_to_dataclass(OptimizerConfig, optimizer_raw) if optimizer_raw is not None else OptimizerConfig()
    )
    data_cfg = _dict_to_dataclass(DataConfig, data_raw) if data_raw is not None else DataConfig()
    run_cfg = _dict_to_dataclass(TrainRunConfig, run_raw) if run_raw is not None else TrainRunConfig()
    checkpoint_cfg = (
        _dict_to_dataclass(CheckpointConfig, checkpoint_raw) if checkpoint_raw is not None else CheckpointConfig()
    )

    # Extract distributed sub-config
    distributed_raw = raw.pop("distributed", None)
    if distributed_raw is not None:
        # Extract plan configs from distributed dict
        fsdp_plan_raw = distributed_raw.pop("fsdp_plan", None)
        tp_plan_raw = distributed_raw.pop("tp_plan", None)
        ep_plan_raw = distributed_raw.pop("ep_plan", None)
        cp_plan_raw = distributed_raw.pop("cp_plan", None)

        fsdp_plan = _dict_to_dataclass(FSDPPlanConfig, fsdp_plan_raw) if fsdp_plan_raw is not None else None
        tp_plan = _dict_to_dataclass(TPPlanConfig, tp_plan_raw) if tp_plan_raw is not None else None
        ep_plan = _dict_to_dataclass(EPPlanConfig, ep_plan_raw) if ep_plan_raw is not None else None
        cp_plan = _dict_to_dataclass(CPPlanConfig, cp_plan_raw) if cp_plan_raw is not None else None

        distributed_valid_fields = {f.name for f in fields(DistributedConfig)}
        extra_keys = set(distributed_raw.keys()) - distributed_valid_fields
        if extra_keys:
            logger.warning("Ignoring unknown distributed config keys: %s", extra_keys)
        filtered_distributed_raw = {k: v for k, v in distributed_raw.items() if k in distributed_valid_fields}

        distributed_cfg = DistributedConfig(
            fsdp_plan=fsdp_plan, tp_plan=tp_plan, ep_plan=ep_plan, cp_plan=cp_plan, **filtered_distributed_raw
        )
    else:
        distributed_cfg = DistributedConfig()

    # Extract memory sub-config
    memory_raw = raw.pop("memory", None)
    if memory_raw is not None:
        memory_cfg = _dict_to_dataclass(MemoryConfig, memory_raw)
    else:
        memory_cfg = MemoryConfig()

    # Extract quantization sub-config
    quantization_raw = raw.pop("quantization", None)
    if quantization_raw is not None:
        quantization_plan_raw = quantization_raw.pop("quantization_plan", None)
        quantization_plan = (
            _dict_to_dataclass(QuantizeConfig, quantization_plan_raw) if quantization_plan_raw is not None else None
        )
        quantization_cfg = QuantizationConfig(quantization_plan=quantization_plan)
    else:
        quantization_cfg = QuantizationConfig()

    valid_fields = {f.name for f in fields(FSDPTurboConfig)}
    extra_keys = set(raw.keys()) - valid_fields
    if extra_keys:
        logger.warning("Ignoring unknown top-level config keys: %s", extra_keys)
    filtered_raw = {k: v for k, v in raw.items() if k in valid_fields}

    return FSDPTurboConfig(
        model=model_cfg,
        optimizer=optimizer_cfg,
        data=data_cfg,
        run=run_cfg,
        checkpoint=checkpoint_cfg,
        distributed=distributed_cfg,
        memory=memory_cfg,
        quantization=quantization_cfg,
        **filtered_raw,
    )
