# 逐优化点验证脚本

每个优化点一个独立脚本，可在 CPU 上直接跑，无需 NPU/GPU 与 `moonep` 包。
全部从仓库根目录执行：

```bash
cd <FSDPTurbo repo root>
PYTHONPATH=. python3 tests/verify/verify_opt1_main_grad_mapping.py
PYTHONPATH=. python3 tests/verify/verify_opt2_grad_weight_sink.py
PYTHONPATH=. python3 tests/verify/verify_opt3_weight_view_cache.py
PYTHONPATH=. python3 tests/verify/verify_opt4_cuda_fused_weight_grad.py   # 仅 GPU 分支
PYTHONPATH=. python3 tests/verify/verify_cuda_vmm_safe_grouped_matmul.py  # 仅 GPU 分支
PYTHONPATH=. python3 tests/verify/verify_cuda_prefetch_event_wait.py      # 仅 GPU 分支
```

每个脚本逐项打印 `PASS`/`FAIL`，全通过时最后一行是 `ALL PASS` 且退出码为 0；
任一项失败则打印 `FAILED N check(s)` 并以 1 退出。

`_fake_moonep.py` 是共用的单进程替身，实现 adapter 用到的那几个
symmetric-memory 入口，用真实文件描述符做 handle，因此 dup/close 的行为与
线上一致，能查出描述符泄漏。

---

## OPT-1 `[E+B]` FP32 对称映射

对应评审意见「main_grad 未做 `[E+B]` 的 VMM 映射」。

```
[1/5] PASS  main_grad 形状 (E+B, out, in) = (10, 8, 4), dtype = torch.float32
[2/5] PASS  rows [0,E) 由 4 个 rank 的 owner 分配拼成, 顺序 rank0..rank3
[3/5] PASS  rows [E,E+B) 复用本 rank 的 reduce 分配 (与 reduce_buffers[1] 同源)
[4/5] PASS  reduce_buffers 由 4 个 rank 的 reduce 分配拼成
[5/5] PASS  owner_grad_full 是 main_grad[:E] 的视图, 未额外分配; close() 无残留 fd
ALL PASS
```

配置：EP=4、每 rank 2 个专家（E=8、B=2、E+B=10）、rank=1、`[out,in]=[8,4]`。

脚本校验的是**映射的构成**——哪块分配支撑哪段行区间、`world_size` 传得对不对、
fd 有没有漏。把这些 chunk 拼成一段连续地址是 `nvl_dist_map` 在真实硬件上的职责，
脚本不重新实现虚拟内存。

OPT-1 只改布局，梯度数值行为不变，由 `tests/unit/test_moonep_gradient_reduce.py` 覆盖。

## OPT-2 gmm 反向直写梯度缓冲

对应评审意见「npu gmm 算子封装，避免返回值后再写入 buffer」。

```
[1/8] PASS  sink 结果与纯 autograd 参考逐位一致, weights 未进计算图
[2/8] PASS  FP32 目标 + BF16 计算: 复用单块 [out,in] staging, 无 [E+B] 分配
[3/8] PASS  空组保持原值未被清零 (对称内存下清零会覆盖 home rank 梯度)
[4/8] PASS  非空组落在远端行区间时报错, buffer 保持干净
[5/8] PASS  reduce_gradient 不再拷贝, 直接消费 main_grad 并返回独立张量
[6/8] PASS  reduce 在两个 GMM 反向之后触发, 顺序固定为 down -> gate_up
[7/8] PASS  端到端参数梯度与纯 autograd 参考逐位一致
[8/8] PASS  NPU 算子: 不传 sink 时与基线逐位一致; 传 sink 时只写本地行
ALL PASS
```

第 1、7 项是数值正确性的主证据：与「权重留在 autograd 图里」的普通实现
`torch.equal` 逐位相等。第 3、4 项守的是对称内存特有的风险。第 8 项用
mock 的 `torch_npu` 对拍，确保不传 sink 时与基线完全等价（可回退）。

## OPT-3 权重视图缓存

```
[1/5] PASS  同一权重张量重复取视图命中缓存, 只构建一次
[2/5] PASS  同存储的新张量 (如 DTensor.to_local()) 复用同一条目
[3/5] PASS  显式 release 后底层 storage 可被回收
[4/5] PASS  未显式 release 时缓存有界, 不会无限增长
[5/5] PASS  projection.close() 会在置空 full_weight 前释放视图
```

第 3 项是本优化的关键约束：脚本先确认**不 release 时 storage 确实被钉住**
（`gc.collect()` 也收不掉），再确认 release 之后可回收。这条决定了缓存必须
显式释放，否则 MoonEP 的 `explicitly_destroy` 退出路径会解不掉对称内存映射。

## OPT-4 CUDA 融合权重梯度（仅 GPU 分支）

```
[1/5] PASS  融合路径结果与逐 group 参考一致
[2/5] PASS  BF16 输入直出 FP32, 未使用 staging
[3/5] PASS  只对本地 row range 发起计算, 临时张量为 [E/R, out, in]
[4/5] PASS  算子不支持时回退到逐 group, 结果一致且只探测一次
[5/5] PASS  远端行在任何 GEMM 之前就被拒绝, buffer 保持干净
```

CPU 上用模拟的 `grouped_mm` 覆盖融合与回退两条路径；装有 CUDA 时
`tests/unit/test_moonep_cuda_weight_grad.py` 里还有一条真实设备用例。

MoonEP 训练的 sink 反向走 OPT-2 逐 group `torch.mm`（`use_fused=False`），
不走 OPT-4：dispatch buffer 是 VMM 映射。OPT-4 仍可供普通 CUDA 内存探测。

## CUDA fused grouped_mm（PyTorch 2.9）与 vmm_safe 逃生舱

PyTorch 2.9 没有 `F.grouped_mm`，但有 `torch._grouped_mm` 和
`aten._grouped_mm`。MoonEP 训练前向必须走这条 fused 路径，offsets 留在 GPU。
此前 `vmm_safe` 默认打开，前向 `_group_ends()` 里 `.cpu()`，会在 FSDP
prefetch all-gather 进行时把某个 rank 卡在 D2H，其余 rank 进入 NCCL，
最终 `mesh_fsdp` ALLGATHER vs `mesh_ep` ALLREDUCE 对不上，NCCL timeout 600s。

`vmm_safe=True` 仍保留为逐 expert `torch.mm` 逃生舱，训练不要开。

```
[1/6] PASS  vmm_safe 前向不调用 grouped_mm, 结果与逐 expert 参考一致
[2/6] PASS  空组被跳过, 输出对应行为 0
[3/6] PASS  dgrad / wgrad 与参考一致, 仍不调用 grouped_mm
[4/6] PASS  sink + vmm_safe 只写本地行, 且关闭融合 grouped_mm
[5/6] PASS  传入 sink 且未设 vmm_safe 时走 fused grouped_mm
[6/6] PASS  无 sink, vmm_safe=False 走 fused grouped_mm
ALL PASS
```

`async_finish` 的 FSDPTurbo 包装 GPU/NPU 是同一份 Python；开源 MoonEP 0.0.1
只有 CUDA Buffer（`torch.cuda.Stream` + CuTe），NPU 跑不到这条 kernel。
H20 上 `async_finish=1` 的 SIGSEGV 是 adapter 里
`Event.wait(torch.accelerator.current_stream())`，不是 GPU 单独关 flag，
也不是已证实的 cooperative-epilogue-on-side-stream。PDL 可保持默认开。
修复后 CUDA/NPU 都按配置传递 `async_finish` / `enable_pdl`。
训练请继续 `MOONEP_ASYNC_FINISH=0`，直到确认 `async_finish=1` 可跑。

上机若仍 SIGSEGV，在脚本里打开 `MOONEP_DEBUG_SYNC=1`：最后一条
`MoonEP debug sync after <stage>` 就是崩溃点。若从未打印 `dispatch`，
崩溃在 MoonEP 自己的 dispatch kernel，而不是 grouped matmul。

## CUDA prefetch Event.wait SIGSEGV（GPU 正确性修复）

对应现象：`PYTHONFAULTHANDLER=1` 后堆栈停在

```
torch/cuda/streams.py:203  Event.wait
moonep_adapter.py          prefetch
```

根因是 `done.wait(torch.accelerator.current_stream())`。CUDA 的
`Event.wait` 是 `cudaStreamWaitEvent`，不能吃 device-agnostic 的
accelerator Stream，PyTorch 2.9 上会直接 SIGSEGV。应改成
`cuda.Stream.wait_event(event)`，与 Domino 一致。

```
[1/4] PASS  CUDA 路径调用 stream.wait_event, 不调用 event.wait
[2/4] PASS  event 为 None 时是空操作
[3/4] PASS  传入没有 wait_event 的 stream 时回退到 torch.cuda.current_stream()
[4/4] PASS  adapter 源码不再把 accelerator stream 传给 Event.wait
ALL PASS
```

---

## 上机验证

单元与验证脚本只能覆盖数值与契约，实际收益需要在真机上量：

```bash
bash tests/system_tests/model/run_test_qwen3_moonep.sh
```

判据：10 步跑完、每步 loss 有限、正常退出码 0；与优化前对比 `peak_memory`
应下降，`time` 不应变差。
