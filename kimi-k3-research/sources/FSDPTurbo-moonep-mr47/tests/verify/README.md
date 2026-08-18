# 两处 NPU 显存改动（基于原始 MR#47）

评审项：

1. `main_grad` 未做 `[E+B]` 的 VMM 映射，反向会重新申请 grad tensor
2. NPU GMM 算子封装，避免返回值后再写入 buffer

目标是 **NPU 上 MoE 梯度工作区峰值**。GPU 只用来验证同一套训练图能跑通，没有额外 GPU 显存优化。

```bash
PYTHONPATH=. python3 tests/verify/verify_opt1_main_grad_mapping.py
PYTHONPATH=. python3 tests/verify/verify_opt2_grad_weight_sink.py
```

## 1. `main_grad` 做成 `[E+B]` VMM

`_ProjectionPool.main_grad` 是一块连续 `[E+B, out, in]` FP32：

- `[0, E)`：各 rank 参数梯度
- `[E, E+B)`：本 rank reduce 槽（与 `reduce_buffers[rank]` 同一物理页）

反向不再另申请一块全尺寸 grad 再拷进去。

## 2. NPU GMM 封装：只算本地行并写入 buffer

`npu_grouped_matmul` 不能安全 `out=` 进整块对称内存（会写远端行）。封装层在反向里：

- 只对本地 owner + prefetch/reduce 槽做 wgrad，返回 `2*(E/R)` 行而不是 `[E+B]`
- `copy_` 进 `main_grad` 的对应切片
- 权重退出 autograd，不再生成第二份 `.grad`
- 能走 `output_dtype=FP32` 时少一次 BF16 staging

## NPU 上机

`dispatcher="moonep"`。`PYTHONPATH` 指向本树。`MOONEP_ASYNC_FINISH` 保持默认 1。

对比原始 MR#47 的 `torch.npu.memory_allocated` / `max_memory_allocated`。逐步日志的 `peak_memory=` 主要是 FSDP 全量峰值，这两处改的是 MoE 梯度工作区。

## GPU 功能（不要当显存优化）

H20 / PT 2.9 没有 `F.grouped_mm`，CUDA 路径回退 `torch._grouped_mm`。启动时导出 `MOONEP_ASYNC_FINISH=0`。

```bash
export MOONEP_ASYNC_FINISH=0
export FSDP_TURBO_ROOT=/path/to/this/FSDPTurbo
export PYTHONPATH=$FSDP_TURBO_ROOT:$PYTHONPATH
bash tests/system_tests/model/run_test_qwen3_moonep.sh
```
