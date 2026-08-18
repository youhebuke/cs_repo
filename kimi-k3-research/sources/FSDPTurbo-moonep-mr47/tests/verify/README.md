# 两个显存优化（基于原始 MR#47）

对应评审：

1. **OPT-1** `main_grad` 未做 `[E+B]` 的 VMM 映射，反向会重新申请 grad tensor
2. **OPT-2** npu gmm 算子封装，避免返回值后再写入 buffer

两者主目标都是 **MoE 梯度工作区峰值**，不是 FSDP 全量参数峰值，也不是把 step time 打下去。GPU 只做功能验证。

从仓库根目录：

```bash
PYTHONPATH=. python3 tests/verify/verify_opt1_main_grad_mapping.py
PYTHONPATH=. python3 tests/verify/verify_opt2_grad_weight_sink.py
```

## OPT-1 `[E+B]` FP32 对称映射

`_ProjectionPool.main_grad` 是一块连续 `[E+B, out, in]` FP32：

- `[0, E)`：各 rank 参数梯度（owner 块拼成）
- `[E, E+B)`：本 rank reduce 槽（与 `reduce_buffers[rank]` 同一物理页）
- `owner_grad_full = main_grad[:E]`，不再另分配

物理增量是 VA，不是再买一份远端 HBM。GPU 上 `nvl_dist_map(..., allocator=nullptr)` 通常 **不进** `memory_allocated`。

## OPT-2 GMM 反向写入 `main_grad`

`GradWeightSink` 把 grouped_matmul 反向接到这块 buffer，权重退出 autograd 图，`reduce_gradient` 不再 `copy_`。

| | GPU | NPU |
|---|---|---|
| 前向 GMM | `torch._grouped_mm`（PT 2.9 无 `F.grouped_mm`） | `npu_grouped_matmul` |
| wgrad | 逐 expert `torch.mm(..., out=main_grad[g])` | 算子 `output_dtype=FP32` 返回 `[E+B]`，只 `copy_` 本地 2*(E/R) 行 |
| 省掉的 allocator 张量 | `empty_like([E+B])`，约 0.5–1 GiB | autograd `.grad`；算子返回值仍在（CANN 不能安全 `out=` 进对称内存） |
| FSDP `peak_memory` | 基本不动 | 基本不动 |

空组跳过而非清零：对称内存上清零会覆盖 home rank 梯度。

## GPU 上机（功能，不是显存）

实验结论：

- `dispatcher="moonep"`，不要 `"fused"`
- 必须 `export MOONEP_ASYNC_FINISH=0`（`async_finish=1` 在 H20 CuTe 编译路径 SIGSEGV）
- 前向 offsets 留在 GPU。`.cpu()` / 逐 expert 前向会和 FSDP prefetch 死锁，NCCL timeout 600s
- 静态 buffer **必须** `offs[-1] == NvS`（mask + 折进最后一组）。截断 `cu_seqlens` 会 `unspecified launch failure` + `cross_rank_barrier` 100s
- `PYTHONPATH` 必须指向本树；probe 两行应是本 checkout，不是另一份 pip/editable 安装

```bash
export MOONEP_ASYNC_FINISH=0
export FSDP_TURBO_ROOT=/path/to/this/FSDPTurbo
export PYTHONPATH=$FSDP_TURBO_ROOT:$PYTHONPATH
bash tests/system_tests/model/run_test_qwen3_moonep.sh
```

启动应打印：

```
[probe] adapter .../FSDPTurbo/.../moonep_adapter.py
[probe] cuda_gmm .../FSDPTurbo/.../ops/cuda/grouped_matmul.py
```

## NPU 显存怎么看

对比原始 MR#47 的 `torch.npu.memory_allocated` / `max_memory_allocated`：

- OPT-1：布局；若 NPU 对称内存不走 caching allocator，数字可能几乎不动
- OPT-2：少一份框架 `.grad` 和 `torch.stack` 全尺寸拷；`output_dtype` 少一份 BF16→FP32 临时。算子仍会物化 `[E+B]` 返回值再拷本地行
