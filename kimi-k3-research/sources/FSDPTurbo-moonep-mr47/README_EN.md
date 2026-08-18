# FSDPTurbo

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A High-Performance and Easy-to-Use Distributed Training Acceleration Library**

[English](#overview) | [简体中文](README.md)

</div>

## Overview

FSDPTurbo is a high-performance and easy-to-use distributed training acceleration library. Based on the architectural design philosophy of high cohesion and low coupling, it provides a general acceleration solution decoupled from model structures for training large-scale deep learning models. It offers state-of-the-art parallelism strategies, advanced memory optimization techniques, and quantization support to efficiently train your large models.

***

## Key Features

### 🚀 Distributed Training

- **FSDP2**: Fully Sharded Data Parallel with enhanced performance
- **Tensor Parallelism**: Efficient model parallelism for large models
- **Expert Parallelism**: Optimized MoE (Mixture of Experts) training
- **Hybrid Parallelism**: Flexible combination of different parallel strategies

### 💾 Memory Optimization

- **Gradient Checkpointing**: Reduce memory footprint by recomputing activations
- **Chunked Loss Computation**: Memory-efficient loss calculation
- **Activation Offloading**: Offload activations to CPU/NVMe
- **Optimizer State Offloading**: Offload optimizer states to CPU/NVMe

### 🔢 Quantization Support

- **MX Formats**: Support for E4M3, E5M2, and HIF8 formats
- **FP8 Training**: Native FP8 training support
- **Mixed Precision**: Flexible mixed precision training strategies

### 🛠️ Easy to Use

- **Simple API**: Minimal code changes required
- **Model Agnostic**: Works with various model architectures
- **Flexible Configuration**: Customizable parallel and memory strategies

***

## Installation

### Dependencies

- Python >= 3.11
- PyTorch >= 2.9.0, < 2.12.0
- [torch\_npu](https://gitee.com/ascend/pytorch) >= 2.9.0, < 2.12.0

### From Source

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

***

## Quick Start

### Basic Usage

```python
import torch
from fsdpturbo import FSDPTurboConfig, FSDPTurboEngine

# Define your model
model = create_your_model()

# Configure FSDPTurbo
config = FSDPTurboConfig(
    # FSDP configuration
    fully_shard_parallel_size=8,
    fsdp_plan=FSDPPlanConfig(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        num_to_forward_prefetch=1,
        num_to_backward_prefetch=1,
    ),

    # tensor parallel
    tensor_parallel_size=4,
    tp_plan=TPPlanConfig(
        colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
        rowwise_parallel=['*.o_proj']
    ),

    # memory optimization
    recompute=True,
    recompute_plan=['model.layers.*'],
)

# Initialize the engine
model = FSDPTurbo(config, model)

# Training loop
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Advanced Configuration

```python
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, QuantizeConfig

config = FSDPTurboConfig(
    # Expert parallel of MoE model
    expert_parallel_size=2,
    ep_plan=EPPlanConfig(
        apply_modules=['*mlp.experts*'],
        dispatcher='fused'  # or 'eager'、'mc2'
    ),

    # Quantization
    quantization_plan=QuantizeConfig(
        quant_format='E4M3',  # or 'E5M2'、'HIF8'
        block_size=32
    ),

    # Other configurations...
)
```

### MoonEP (GPU/NPU)

MoonEP is an optional expert-parallel backend for static-shape BF16 training.
FSDPTurbo uses the same functionally equivalent MoonEP API on GPU and NPU via
`torch.accelerator`. The sibling MoonEP repository currently provides the GPU
implementation; an NPU implementation can use the same `moonep==0.0.1` package
interface. Install it and select the dispatcher explicitly:

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

The selected expert module must expose fused `gate_up_proj`, `down_proj`,
`act_fn`, and `hidden_dim` attributes. The EP group must be inside one
MoonEP multicast-capable accelerator domain; the current GPU implementation
requires NVLink/NVSwitch. Dynamic token shapes, expert FSDP, and quantized MoE
weights are rejected instead of silently falling back. Call `model.close()`
before destroying the distributed process group.

***

## Architecture

```
FSDPTurbo
├── distributed/           # Distributed training modules
│   ├── fully_shard_parallel/   # FSDP2 implementation
│   ├── tensor_parallel/        # Tensor parallelism
│   ├── expert_parallel/        # Expert parallelism (MoE)
│   └── parallel_state.py       # Global parallel state management
├── memory/                # Memory optimization techniques
│   ├── recompute/              # Gradient checkpointing
│   ├── chunk_loss/             # Chunked loss computation
│   ├── swap_activation/        # Activation offloading
│   └── swap_optimizer/         # Optimizer state offloading
├── quantization/          # Quantization support
│   ├── mx_formats/             # MX format implementations
│   ├── converter/              # Model converter framework
│   └── mxfp8_config.py         # FP8 configuration
└── utils/                 # Utility functions
```

***

# Disclaimer

## To FSDPTurbo Users

1. All content provided by FSDPTurbo is for your non-commercial use only.
2. For the models and datasets involved in FSDPTurbo test cases and example files, the platform is only used for functional testing. Huawei does not provide any model weights or datasets. If you use this data for training, please pay special attention to comply with the corresponding model and dataset licenses. If you encounter infringement disputes due to using these models and datasets, Huawei assumes no responsibility.
3. If you discover any issues (including but not limited to functional issues, compliance issues) while using FSDPTurbo, please submit an issue on Gitee, and we will review and resolve it promptly.
4. The third-party open-source software that FSDPTurbo's functionality depends on is provided and maintained by third-party communities. Fixes for issues caused by third-party open-source software depend on the contributions and feedback of relevant communities. You should understand that the FSDPTurbo repository does not guarantee fixes for issues in the third-party open-source software itself, nor does it guarantee testing and correction of all vulnerabilities and errors in third-party open-source software.

## To Data Owners

If you do not wish for your model or dataset to be mentioned in FSDPTurbo, or wish to update the description related to FSDPTurbo, please submit an issue, and we will delete or update your relevant description according to your issue requirements. We sincerely thank you for your understanding and contribution to FSDPTurbo.

## License Statement

For files involved in FSDPTurbo, if a License exists in the file directory, that License shall prevail. If no License exists in the file directory, it is licensed under the Apache 2.0 License, and the corresponding license text can be found in the FSDPTurbo root directory.

***

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

***

## Support

- **Issues**: [Issues](https://gitcode.com/Ascend/FSDPTurbo/issues)
- **Documentation**: Coming soon

***
