# MoonEP 与 FSDPTurbo MR#47 适配分析

> 分析日期：2026-08-17  
> 上游仓库：
> - MoonEP：https://github.com/MoonshotAI/MoonEP
> - FSDPTurbo MR#47：https://gitcode.com/Ascend/FSDPTurbo/pull/47/diffs

本地镜像路径：

- `sources/MoonEP/` — MoonEP 官方仓库快照
- `sources/FSDPTurbo-moonep-mr47/` — FSDPTurbo MR#47 分支快照（`feature/moonep-integration`）

---

## 1. MoonEP 是什么

MoonEP 是 Moonshot AI 为 **MoE Expert Parallelism（EP）** 设计的通信库，核心目标是：

1. **完美负载均衡**：通过在线 planning + 动态冗余专家（prefetch），让每个 EP rank 始终处理固定 `S × K` 个 token，路由再偏也不拖慢最慢 rank。
2. **零拷贝 + 静态 shape**：dispatch 直接把 token 写到远端 rank 上按 expert 分组后的最终位置；返回 buffer view 给 Group GEMM，避免 permute in/out 和 host sync。
3. **训练闭环**：prefetch 冗余专家权重 → Group GEMM → combine；backward 时 combine 梯度 + `reduce_grad` 把冗余专家梯度归并回 home rank。

### 1.1 关键 API 契约

MoonEP 与训练框架的接口非常「硬」：

| 组件 | 形状 / 约束 |
|---|---|
| 权重 buffer | 每个 projection 一个连续 `[E+B, H, H']` BF16 tensor；`B = E/R`（训练） |
| dispatch 输出 | `hidden_nvsh [NvS, H]`、`cu_seqlens [E+B]`、`plan` |
| Group GEMM | 按 `cu_seqlens` 做 grouped matmul |
| prefetch | 把远端专家权重拷到本地 prefetch slot |
| reduce_grad | 冗余专家梯度跨 rank 归并 |

底层依赖 **NVLink symmetric memory / VMM multicast**（GPU 上），要求 EP group 在同一节点内。

### 1.2 MoonEP 代码结构

```
MoonEP/
├── moonep/
│   ├── api.py / buffer.py      # Buffer 高层 API
│   ├── planning.py             # 在线 planning kernel
│   ├── dispatch.py / combine.py
│   ├── prefetch.py / grad_reduce.py
│   └── _C (CUDA extension)     # VMM、NVL shared buffer
├── tests/                      # dispatch/combine/e2e 单测
└── benchmarks/                 # vs DeepEP 对比
```

MoonEP **本身不包含** FSDP、模型定义或 Group GEMM 实现——它只负责 EP 通信 + 权重/梯度 buffer 管理。

---

## 2. FSDPTurbo MR#47 做了什么

FSDPTurbo 是华为 Ascend 基于 PyTorch 的分布式训练加速库（FSDP2 + TP + EP + CP + 量化等）。MR#47（分支 `feature/moonep-integration`，13 个专属 commit）在现有 EP dispatcher 体系上 **新增 `dispatcher="moonep"`**，把 MoonEP 接入 FSDPTurbo 的 MoE 训练路径。

### 2.1 改动概览（相对 main）

| 类别 | 主要文件 | 说明 |
|---|---|---|
| **核心适配** | `fsdp_turbo/distributed/expert_parallel/moonep_adapter.py` (~700 行) | VMM 权重映射、prefetch/grad_reduce、autograd Function |
| **MoE forward** | `moonep_dispatcher.py` (~210 行) | dispatch → GMM gate_up → act → GMM down → combine |
| **配置** | `fsdp_turbo_config.py` | 新增 `MoonEPConfig`；校验 EP>1、禁 expert FSDP、禁 MoE 量化 |
| **EP 入口** | `expert_parallel.py` | `dispatcher=="moonep"` 时创建 `MoonEPRuntime` 并 patch expert module |
| **依赖** | `pyproject.toml` / `setup.py` | 可选 extra：`fsdp-turbo[moonep]` → `moonep==0.0.1`, `torch>=2.9` |
| **测试** | `tests/unit/test_moonep_*.py`, `tests/system_tests/model/test_moonep.py` | 配置校验、GMM tail、8卡 Qwen3 MoE 训练 |
| **文档** | `README.md` / `README_EN.md` | MoonEP 使用说明 |

MR 基于较早的 main（尚未包含后续 Ulysses CP、Qwen3.5 等 commit），diff 约 **+2810 / -4150** 行。

### 2.2 架构：FSDPTurbo 如何「包」住 MoonEP

```
FSDPTurbo Expert Module (fused gate_up_proj + down_proj)
        │
        ▼
parallelize_moonep_module()
  ├─ MoonEPRuntime：全局 Buffer + VMM pool + comm stream
  ├─ MoonEPSymmetricProjection ×2：把 nn.Parameter 映射到 [E+B] VMM
  └─ 替换 forward → moonep_experts_forward()
        │
        ▼
Forward 流水线：
  1. moonep_dispatch()      ← MoonEP Buffer.dispatch
  2. prefetch(gate_up, down)← launch_prefetch
  3. grouped_matmul (fc1)   ← FSDPTurbo ops.moe
  4. act_fn(gate) * up
  5. grouped_matmul (down)
  6. moonep_weighted_combine() ← MoonEP Buffer.combine
        │
        ▼
Backward（autograd Function）：
  - dispatch backward → combine grad
  - combine backward → re-dispatch grad
  - _MoonEPWeightsBridge → launch_grad_reduce（gate_up + down 同一节点，防 barrier 死锁）
```

### 2.3 适配层关键设计

#### （1）延迟 import + 版本锁定

`moonep_adapter.py` 只在选择 `dispatcher="moonep"` 时才 import `moonep`，并要求 **`moonep==0.0.1`**，避免 VMM/kernel 接口漂移。

#### （2）VMM 对称权重映射

`MoonEPSymmetricProjection` 把每个 rank 的 local expert 权重通过 `nvl_dist_alloc` + IPC fd exchange 映射成全局 `[E+B, H, H']` view，同时保留 `DTensor(Shard(0))` 作为 `nn.Parameter`，让 FSDPTurbo 其它逻辑仍能识别参数归属。

#### （3）静态 token shape 锁定

`MoonEPRuntime.new_call()` 在首次 forward 记录 `tokens_per_rank` 和 `top_k`，之后必须一致——这与 MoonEP「固定 S×K buffer」的设计一致。可通过 `MoonEPConfig(tokens_per_rank=..., top_k=...)` 预声明。

#### （4）Grouped Matmul 静态 tail 处理

MoonEP 通信 buffer 尾部可能有 padding token。`_grouped_matmul_with_static_tail()` 把 tail 归到最后一个 group，避免 host 读 scalar 或动态 shape。

#### （5）双卡梯度归并顺序

`_MoonEPWeightsBridge` 把 gate_up 和 down 的 grad_reduce **放在同一个 autograd Function** 里，保证各 rank 以相同顺序进入 MoonEP 内部 barrier，避免死锁。

#### （6）硬约束（配置层拒绝，不 silent fallback）

- EP size > 1，且 **同一 hostname**（intra-node）
- **不支持** expert FSDP（`expert_fully_shard_parallel_size` 必须为 1）
- **不支持** MoE 权重量化（必须 BF16 `nn.Parameter`）
- 需要 **multicast-capable** 互联（GPU：NVLink/NVSwitch）
- 模型需 fused expert：`gate_up_proj [E,2Hp,H]`、`down_proj [E,H,Hp]`

### 2.4 配置示例

```python
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, MoonEPConfig

config.distributed.ep_plan = EPPlanConfig(
    apply_modules=["model.layers.{*}.mlp.experts"],
    dispatcher="moonep",
    moonep_config=MoonEPConfig(
        tokens_per_rank=batch_size * seq_len,
        top_k=8,
        num_sms=32,
        token_padding=128,
    ),
)
config.distributed.expert_fully_shard_parallel_size = 1
```

安装：`pip install -e ../MoonEP` 或 `pip install -e .[moonep]`。

---

## 3. 两份代码的关系对照

| 维度 | MoonEP（官方） | FSDPTurbo MR#47（适配） |
|---|---|---|
| **职责** | EP 通信原语 + buffer 管理 | 把 MoonEP 嵌入 FSDP2/TP/训练器 |
| **接口形态** | Python `Buffer` + CUDA ext | `dispatcher="moonep"` + autograd 桥 |
| **权重** | 用户提供 `[E+B,H,H']` contiguous BF16 | 从 `gate_up_proj/down_proj` 自动建 VMM |
| **计算** | 不含 GEMM | 调用 FSDPTurbo `grouped_matmul` |
| **并行** | 仅 EP | EP + FSDP2（非 expert 参数）+ 可选 TP |
| **设备** | NVIDIA GPU（PPU 待审） | GPU/NPU 统一 API（`torch.accelerator`）；MoonEP 包当前主要是 GPU 实现 |
| **约束** | S×K 固定、intra-node、B=E/R | 继承 MoonEP 全部约束 + 禁 expert FSDP/量化 |

**一句话**：MoonEP 解决「EP 通信与负载均衡」；FSDPTurbo MR#47 解决「如何在现有分布式训练框架里安全、可 autograd 地使用 MoonEP」。

---

## 4. Kimi K3 训练 Infra 语境

根据 Kimi K3 技术报告，MoonEP 是预训练 Infra 的关键组件之一，用于：

- 消除 MoE routing 不平衡导致的 straggler 和 OOM（静态 activation shape）
- 相比 DeepEP，在高 maxvio 场景下通信与 e2e 迭代时间更稳定

FSDPTurbo 的 MR#47 把同一套 MoonEP 能力带到 **Ascend NPU + FSDPTurbo 生态**，便于在华为栈上复现类似 EP 策略（README 声明 NPU 可走相同 `moonep==0.0.1` 接口，GPU 实现已在同级仓库提供）。

---

## 5. MR#47 commit 历史（专属部分）

```
4471a2f add moonep
8222a34 bugfix
37f5ab4 bugfix for oom
62b3d6c bugfix for bound
8f4576c / 92aeacf / 58c059e revise test_moonep
040a6f9 add test maxvio
6b507e3 delete md and bench
efdc270 Remove MoonEP maxvio diagnostics
d5e6b62 refactor code for both cuda and npu
063f1a3 revise gmm
```

演进路径：先打通基本 dispatch/combine → 修 OOM/bound → 补系统测试 → 清理 benchmark → CUDA/NPU 统一 refactor → GMM 尾部处理修订。

---

## 6. 使用与验证建议

1. **环境**：PyTorch ≥ 2.9、BF16、8×GPU 同节点 NVLink、`pip install moonep==0.0.1`
2. **冒烟**：`tests/unit/test_moonep_*.py`
3. **集成**：`torchrun --nproc-per-node=8 tests/system_tests/model/test_moonep.py`
4. **与 MoonEP 官方 benchmark 对照**：`sources/MoonEP/benchmarks/bench_vs_deepep.py`

---

## 7. 参考链接

- MoonEP GitHub：https://github.com/MoonshotAI/MoonEP
- FSDPTurbo GitCode：https://gitcode.com/Ascend/FSDPTurbo
- MR#47 diff：https://gitcode.com/Ascend/FSDPTurbo/pull/47/diffs
- Kimi K3 技术分享材料：[`README.md`](README.md)
