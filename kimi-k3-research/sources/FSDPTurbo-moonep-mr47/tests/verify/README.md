# OPT-1 / OPT-2：NPU MoE 梯度工作区显存

基于原始 MoonEP MR#47。目标是 **NPU 上降低 MoE 梯度工作区峰值**，不是 FSDP 全量参数峰值，也不是打 step time。

GPU 只做功能验证（H20 已跑通）。NPU 上再验功能并对比显存。

对应评审：

1. **OPT-1** `main_grad` 做成一块 `[E+B]` FP32 VMM，反向不再另申请全尺寸 grad
2. **OPT-2** grouped matmul 反向写入该 buffer，不再「算子返回全量再 `copy_`」

从本 FSDPTurbo 根目录：

```bash
PYTHONPATH=. python3 tests/verify/verify_opt1_main_grad_mapping.py
PYTHONPATH=. python3 tests/verify/verify_opt2_grad_weight_sink.py
```

## OPT-1 `[E+B]` FP32 对称映射

`_ProjectionPool.main_grad` 是一块连续 `[E+B, out, in]` FP32：

- `[0, E)`：各 rank 参数梯度（owner 块拼成）
- `[E, E+B)`：本 rank reduce 槽（与 `reduce_buffers[rank]` 同一物理页）
- `owner_grad_full = main_grad[:E]`，不再另分配

物理增量主要是 VA。对称内存若不走 caching allocator，`memory_allocated` 可能几乎不动。

## OPT-2 GMM 反向写入 `main_grad`

`GradWeightSink` 把 grouped_matmul 反向接到这块 buffer，权重退出 autograd 图，`reduce_gradient` 不再 `copy_`。

| | NPU（收益目标） | GPU（功能） |
|---|---|---|
| 前向 GMM | `npu_grouped_matmul`，group list 覆盖整段静态 buffer | PT 2.9 `torch._grouped_mm`，`offs[-1] == NvS` |
| wgrad | 算子 `output_dtype=FP32` 仍返回 `[E+B]`，只 `copy_` 本地 `2*(E/R)` 行 | 逐 expert `torch.mm(..., out=main_grad[g])` |
| 省掉的张量 | 框架 `.grad`；少一次 BF16→FP32 staging / `stack` | `empty_like([E+B])` |
| 不能做的 | CANN 不能安全 `out=` 进对称内存（会写远端行） | 前向不要 `.cpu()` / 逐 expert `torch.mm`（和 FSDP prefetch 死锁） |

空组跳过而非清零：对称内存上清零会覆盖 home rank 梯度。

## NPU 上机（功能 + 显存）

`dispatcher="moonep"`，不要 `"fused"`。`PYTHONPATH` 必须指向本树。

NPU 保持原始 `MOONEP_ASYNC_FINISH=1`（默认）。adapter 只在 CUDA 上强制 `async_finish=False`。

对比同一 `test_moonep.py` 在原始 MR#47 与本分支上的：

- `torch.npu.memory_allocated`
- `torch.npu.max_memory_allocated`

逐步日志里的 `memory=` / `peak_memory=` 主要是 FSDP 全量峰值，OPT-1/OPT-2 不一定把它打下来。看 MoE 梯度工作区：少一份 `[E+B]` FP32 `.grad` / 全量 `copy_`。

Qwen3-30B-A3B EP=4 量级：gate_up `[160,1536,2048]` FP32 ≈ 1.88 GiB，down `[160,2048,768]` FP32 ≈ 0.94 GiB（若完全物化）。

## GPU 功能（已验证，不要当显存优化）

H20 / PT 2.9 能跑通需要的最小差异，NPU 不依赖这些：

- CUDA `Buffer.dispatch` / `combine` 走计算流（`async_finish=False`）
- `Event.wait` 改为 `Stream.wait_event`
- `torch._grouped_mm`，前向 offsets 留在 GPU，`offs[-1] == NvS`
- 启动脚本导出 `MOONEP_ASYNC_FINISH=0`

```bash
export MOONEP_ASYNC_FINISH=0
export FSDP_TURBO_ROOT=/path/to/this/FSDPTurbo
export PYTHONPATH=$FSDP_TURBO_ROOT:$PYTHONPATH
bash tests/system_tests/model/run_test_qwen3_moonep.sh
```
