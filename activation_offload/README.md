# 激活 CPU Offload 管理器（复现 Kimi K3「统一激活管理器」的 offload 存储策略）

一个从零实现、可上库、**单卡即可实测显存收益**的激活显存优化模块，对应 Kimi K3 技术报告
§5.2「统一激活管理器」里的 **offload 存储策略**：把前向为反向保存的大激活异步搬到 **pinned CPU**，
在反向到达前再取回，从而**降低训练峰值 HBM**。可与 Megatron-LM 的 `TransformerLayer` 对接。

## 为什么选这个（三选一的结论）

在「基于 Megatron 复现 K3 内存优化能力」的三个候选里：

| 能力 | 在 Megatron 现状 | 是否需要真代码 | 单卡可实测收益 |
|---|---|---|---|
| ZeRO-2 梯度 sharding & offload | 分布式优化器 + `--optimizer-cpu-offload` **已内建** | 否（配置即得） | 是，但非代码产出 |
| PP rank 间激活均衡（借邻卡 HBM） | 无（仅 CPU delta 近似为配置项） | 是，但需**多卡流水**才能验证 | 否（本环境无法验证） |
| **激活 offload（本项目）** | 有生产版，但**核心机制可独立复现** | **是** | **是（`max_memory_allocated` 直接读）** |

因此选 **激活 offload**：需要真实代码（autograd hook + pinned 缓冲池 + 拷贝流 overlap + 策略），
且单张 GPU 就能量化峰值 HBM 下降，便于产出与上库。

## 原理

- **解耦模型代码**：通过 `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` 安装，
  任意 `nn.Module` 区域无需改动即可启用。
- **张量粒度策略**：`min_bytes` 阈值（跳过小张量）+ 参数指针过滤（不 offload 权重）决定哪些激活外移。
- **overlap + 双缓冲**：D2H 拷贝在**独立 CUDA stream** 上进行；`max_inflight` 限制在途拷贝数，
  既与计算重叠、又把额外常驻显存控制在很小范围。
- **pinned 缓冲池**：按字节数复用页锁定 CPU 缓冲，跨 step 免反复分配。
- **关键实现点**：`pack` 在前向保存时触发（需上下文在前向活跃）；`unpack` 绑定到被保存的张量、
  在**反向时自动触发**（无需反向仍处于上下文中）——这让 Megatron 集成只需包住**前向**。

数学上无损：`unpack` 返回的张量与 `pack` 收到的张量数值完全一致，故梯度精确不变（见测试：`max |grad diff| = 0`）。

## 快速开始

```python
from act_offload import offload_activations

with offload_activations(model, min_bytes=1<<20) as mgr:   # 只 offload ≥1MiB 的激活
    loss = model(x).sum()
    loss.backward()
print(mgr.summary())   # 打印本步 offload 了多少 MiB
```

## 运行

```bash
pip install -r requirements.txt   # CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu

# 数值正确性（CPU 或 GPU 均可；CPU 走 emulate 验证 pack/unpack 无损）
python tests/test_correctness.py

# 显存基准（GPU 上读 max_memory_allocated；CPU 上打印字节记账）
python benchmark/bench_memory.py --layers 24 --dmodel 2048 --seq 4096 --batch 1
```

## 实测结果

**正确性（本环境 CPU）**：前向输出与**全部参数梯度逐位一致**——
`max |grad diff| = 0.000e+00`；本步 offload 49 个张量 / 17.3 MiB。

**收益记账（本环境 CPU，12 层 / d=1024 / seq=2048 / batch=1）**：
`offloaded 133 tensors = 1640.0 MiB moved to CPU; kept 156 tensors = 578.1 MiB on device`
——即在 GPU 上这 **1.64 GiB** 激活会在前向期被移出 HBM（约占该配置 saved 激活的 74%），
对应等量的峰值 HBM 下降。

**在 GPU 上**运行 `bench_memory.py` 会直接打印：
```
baseline : peak HBM = ... MiB   step = ... ms
offload  : peak HBM = ... MiB   step = ... ms
--> saved XXX MiB (YY%)  time overhead +Z%
```
（收益随层数、seq、batch 增大而增大；overlap 让时间开销可控。）

> 本环境无 GPU，故给出的是 CPU 正确性 + 字节记账 + 可直接在 GPU 复跑的基准脚本；
> 峰值 HBM 的直接数字请在任意一张 CUDA 卡上跑 `bench_memory.py` 获得。

## 与 Megatron-LM 对接

见 `act_offload/megatron_adapter.py`。两种方式（因 unpack 自动在反向触发，**只需包前向**）：

1. **整步包裹（推荐、最简）**：把 `forward_backward_func`（或你 `train_step` 的前向）包进
   `offload_activations(model, min_bytes=...)`，一个共享 manager 覆盖 1F1B 的所有 micro-batch。
2. **逐层包裹（更细）**：
   ```python
   from act_offload.megatron_adapter import wrap_megatron_transformer_layers
   mgr = wrap_megatron_transformer_layers(model, min_bytes=1<<20)
   # ... 每步 backward 后：mgr.finalize()；mgr.summary() 看记账
   ```
   它会 monkeypatch 每个 `TransformerLayer.forward` 在共享 manager 下运行。

与 Megatron 内建 fine-grained activation offloading 的关系：Megatron 生产版绑定 TransformerEngine
算子与固定模块边界；本模块是**框架无关**的 saved-tensor 级实现，可独立验证、便于教学/移植，
也可作为对 K3「按张量选择存储策略」思想的最小复现。

## 局限与后续

- 反向的 H2D 目前为**按需取回**（在拷贝流上做，不阻塞其它流）；可进一步做**反向预取**
  （记录反向访问顺序，提前 H2D）以进一步隐藏延迟。
- 参数过滤用 `data_ptr()` 集合；视图/非连续张量已用 `contiguous()` 兜底（多一次拷贝）。
- 与 `activation recomputation`/FP8 量化可组合（K3 的「可插拔存储策略」全集），此处先落地 offload 一策。
