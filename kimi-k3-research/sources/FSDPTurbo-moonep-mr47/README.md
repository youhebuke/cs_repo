# FSDPTurbo

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**兼具高性能与易用性的分布式训练加速库**

[English](README_EN.md) | [简体中文](#概述)

</div>

## 概述

FSDPTurbo 是一个兼具高性能与易用性的分布式训练加速库，基于高内聚低耦合的架构设计理念，打造一套与模型结构解耦的通用加速方案，用于训练大规模深度学习模型。它提供了业界领先的并行策略、先进的内存优化技术和量化支持，能够高效训练用户的大模型。

***

## 主要特性

### 🚀 分布式训练并行

- **全分片数据并行（FSDP2）**：实现 PyTorch 的 FSDP2 API，支持混合精度训练、前向/反向预取和选择性模块分片
- **张量并行（TP）**：支持列并行、行并行和序列并行，集成 DTensor
- **专家并行（EP）**：专为混合专家（MoE）模型设计，支持多种调度器类型（'eager'、'fused'、'mc2'）
- **上下文并行**：Ulysses 风格的上下文并行，支持长序列训练

### 💾 内存优化

- **梯度检查点**：选择性激活重计算以减少内存占用
- **分块损失计算**：将损失计算分块处理，适用于大词表模型
- **激活交换**：异步将激活值卸载到 CPU 内存
- **优化器状态交换**：将优化器状态卸载到 CPU 以减少 GPU 内存使用

### 🔢 量化

- **MXFP8 量化**：支持 E4M3、E5M2 和 HIF8（华为 hifloat8）格式
- **分块量化**：可配置块大小，实现细粒度控制
- **模型转换框架**：轻松在模型中部署量化

### 🎯 灵活配置

- **基于模式的模块选择**：使用通配符模式选择模块（如 `*.q_proj`、`*.k_proj`）
- **层次化并行**：组合多种并行策略（DP × FSDP × TP × EP）
- **模型无关**：适用于各种 Transformer 架构

***

## 安装

### 环境要求

- Python >= 3.11
- PyTorch >= 2.9.0, < 2.12.0
- [torch\_npu](https://gitee.com/ascend/pytorch) >= 2.9.0, < 2.12.0

### 从源码安装

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

***

## 快速开始

### 基础用法

```python
import torch
from fsdp_turbo.fsdp_turbo_config import (
    FSDPTurboConfig, FSDPPlanConfig, TPPlanConfig
)
from fsdp_turbo.fsdp_turbo import FSDPTurbo

# 创建模型
model = create_your_model()

# 配置并行策略
config = FSDPTurboConfig(
    # FSDP 配置
    fully_shard_parallel_size=8,
    fsdp_plan=FSDPPlanConfig(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        num_to_forward_prefetch=1,
        num_to_backward_prefetch=1,
    ),

    # 张量并行
    tensor_parallel_size=4,
    tp_plan=TPPlanConfig(
        colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
        rowwise_parallel=['*.o_proj']
    ),

    # 内存优化
    recompute=True,
    recompute_plan=['model.layers.*'],
)

# 应用并行引擎到模型
model = FSDPTurbo(config, model)

# 训练循环
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 高级用法：专家并行

```python
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, QuantizeConfig

config = FSDPTurboConfig(
    # MoE 模型的专家并行
    expert_parallel_size=2,
    ep_plan=EPPlanConfig(
        apply_modules=['*mlp.experts*'],
        dispatcher='fused'  # 或 'eager'、'mc2'
    ),

    # 量化
    quantization_plan=QuantizeConfig(
        quant_format='E4M3',  # 或 'E5M2'、'HIF8'
        block_size=32
    ),

    # 其他配置...
)
```

### MoonEP（GPU/NPU）

MoonEP 是面向固定 shape BF16 训练的可选专家并行后端。FSDPTurbo 通过
`torch.accelerator` 使用 GPU/NPU 上同名且功能等价的 MoonEP 接口。当前同级 MoonEP
仓库提供 GPU 实现；NPU 实现可使用相同的 `moonep==0.0.1` 包接口接入。安装后显式选择 dispatcher：

```bash
pip install -e ../MoonEP
```

```python
from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

config.distributed.ep_plan = EPPlanConfig(
    apply_modules=["model.layers.{*}.mlp.experts"],
    dispatcher="moonep",
    moonep_config=MoonEPConfig(
        tokens_per_rank=batch_size * sequence_length,
        top_k=8,
        num_sms=32,
        token_padding=128,
    ),
)
config.distributed.expert_fully_shard_parallel_size = 1
```

专家模块需提供融合的 `gate_up_proj`、`down_proj`、`act_fn` 和 `hidden_dim`。
EP 组必须位于同一支持 MoonEP multicast 的加速器互联域中；当前 GPU 实现要求
NVLink/NVSwitch。动态 token shape、专家 FSDP 和量化 MoE 权重会明确报错而不会静默回退。
销毁分布式进程组前需调用 `model.close()`。

***

## 架构

```
FSDPTurbo
├── distributed/           # 分布式训练模块
│   ├── fully_shard_parallel/   # FSDP2 实现
│   ├── tensor_parallel/        # 张量并行
│   ├── expert_parallel/        # 专家并行（MoE）
│   └── parallel_state.py       # 全局并行状态管理
├── memory/                # 内存优化技术
│   ├── recompute/              # 梯度检查点
│   ├── chunk_loss/             # 分块损失计算
│   ├── swap_activation/        # 激活值卸载
│   └── swap_optimizer/         # 优化器状态卸载
├── quantization/          # 量化支持
│   ├── mx_formats/             # MX 格式实现
│   ├── converter/              # 模型转换框架
│   └── mxfp8_config.py         # FP8 配置
└── utils/                 # 工具函数
```

***

# 免责声明

## 致 FSDPTurbo 使用者

1. FSDPTurbo 提供的所有内容仅供您用于非商业目的。
2. 对于 FSDPTurbo 测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 FSDPTurbo 过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。
4. FSDPTurbo 功能依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，FSDPTurbo 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## 致数据所有者

如果您不希望您的模型或数据集在 FSDPTurbo 中被提及，或希望更新 FSDPTurbo 中有关的描述，请在提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对 FSDPTurbo 的理解和贡献。

## License声明

FSDPTurbo 中涉及的文件，如文件目录下存在 License 的，以该 License 为准。如文件目录下不存在 License 的，以Apache 2.0 许可证许可，对应许可证文本可查阅 FSDPTurbo 根目录。

***

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

***

## 支持

- **问题反馈**：[Issues](https://gitcode.com/Ascend/FSDPTurbo/issues)
- **文档**：即将推出

***
