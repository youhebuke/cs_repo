# MoonEP × FSDPTurbo：反向梯度缓冲优化

针对 [Ascend/FSDPTurbo MR#47](https://gitcode.com/Ascend/FSDPTurbo/pull/47) 评审意见中的两个优化点：

> 1. `main_grad` 未做 `[E+B]` 的 VMM 映射，反向会重新申请 grad tensor
> 2. NPU gmm 算子封装，避免返回值后再写入 buffer

---

## 1. 优化前的数据流

每层、每个 projection（`gate_up` / `down`）在反向阶段：

```
grouped_matmul.backward
  └─ torch.empty_like(weights)              ← ① 分配 [E+B, out, in] BF16 全尺寸张量
     （NPU 额外 torch.stack([w.T ...])）     ← ② 再分配一份 + 全量拷贝
        ↓ 作为 autograd 梯度传给 bridge
_MoonEPWeightsBridge.backward
  └─ reduce_gradient(full_grad, ...)
       ├─ owner_grad_buffers[rank].copy_(full_grad[start:end])          ← ③ BF16→FP32 拷贝
       ├─ reduce_buffers[rank].copy_(full_grad[E:E+B])                  ← ④ BF16→FP32 拷贝
       ├─ launch_inter_rank_sync + launch_grad_reduce
       └─ owner_grad_buffers[rank].to(bf16)                             ← ⑤ 局部尺寸拷贝
```

问题在于：MoonEP 的契约（见其 README「Gradient buffers」一节）本来就要求
**每个 projection 一块连续的 `[E+B, H, H']` FP32 梯度缓冲**，其中

- 行 `[0, E)` 是各 rank 参数梯度（对称内存映射）
- 行 `[E, E+B)` 由 reduce buffer 支撑

MR#47 里只有权重做了 `[E+B]` VMM 映射（`full_weight`），梯度侧退化成
「临时全尺寸张量 + 两次切片拷贝进池」，于是每次反向都要重新申请 ①（NPU 还多一份 ②）。

---

## 2. 优化后的数据流

```
grouped_matmul.backward(grad_weight_sink=[E+B] FP32 VMM buffer)
  └─ 直接把每个 group 的权重梯度写进 buffer 对应行            ← 无 ①②③④
_MoonEPWeightGradBridge.backward
  └─ reduce_gradient(plan, ctx, dtype)
       ├─ launch_inter_rank_sync + launch_grad_reduce
       └─ main_grad[start:end].to(param_dtype)                ← 仅保留 ⑤
```

### 2.1 `[E+B]` FP32 VMM 映射（优化点 1）

`_ProjectionPool` 现在按 MoonEP 的布局构建梯度缓冲：

| 行区间 | 物理来源 |
|---|---|
| `[0, E)` | 每个 rank 一块 `[E/R, out, in]` FP32 分配，经 IPC fd 交换后映射成连续 `[E]` |
| `[E, E+B)` | 本 rank reduce buffer 分片（与 `reduce_buffers[rank]` 同一块物理内存） |

关键点：这块缓冲仍然**按 (role, shape) 在所有层之间共享**，因此显存占用与优化前
完全一致——不会退化成「每层一份 FP32 全量梯度」。共享是安全的，因为某一层的
bridge 反向紧跟在该层两个 GMM 反向之后、且早于上一层的 GMM 反向。

`reduce_buffers` 与 `main_grad[E:]` 现在指向同一块物理内存，正好对应 MoonEP README 的
「rows `[E, E+B)` are backed by the reduce buffer」。

### 2.2 权重梯度 sink（优化点 2）

新增 `fsdp_turbo/ops/grad_weight_sink.py`：

```python
@dataclass
class GradWeightSink:
    buffer: torch.Tensor                     # [E+B, out, in]，通常是 FP32 VMM 映射
    row_ranges: Tuple[Tuple[int, int], ...]  # 本 rank 物理拥有的行区间
```

`grouped_matmul(..., grad_weight_sink=sink)` 时，反向不再返回权重梯度，而是原地写入。

**NPU**（`fsdp_turbo/ops/npu/grouped_matmul.py`）：

- 用 `output_dtype=torch.float32` 让 gmm 直接产出 FP32，省掉一次 cast（首次调用探测，
  旧版 CANN/torch_npu 不支持时自动回退）
- 用 `.transpose(1, 2)` 视图替代 `torch.stack([w.T for w in ...])`，去掉一次全量拷贝
  （这条对未使用 sink 的旧路径同样生效）
- 只把 `row_ranges` 覆盖的 `2×(E/R)` 行写进 buffer，而不是整个 `E+B` 行

**CUDA / CPU**：逐 group `torch.mm(..., out=buffer[g])`；dtype 不一致时复用一块
`[out, in]` staging，而不是分配整个 `[E+B, out, in]`。

### 2.3 一个必须处理的正确性风险

优化前的 `grad_full` 是**进程本地临时张量**，把空 group 的行清零无害。
优化后 buffer 行 `[0, E)` 是**对称内存**——某个 rank 往非本地行写零，会直接
破坏 home rank 的梯度。

因此 `write_group_grads` 的语义是：

- **跳过**空 group（不清零）
- 非空 group 若落在 `row_ranges` 之外，直接 `RuntimeError`

依据是 MoonEP 的训练契约（`B = E/R`，planner 保证「every expert the group GEMM
touches is local」）。真出现越界说明 planning 与 EP 配置不一致，宁可报错也不要静默写坏。

### 2.4 bridge 位置调整

权重不再进 autograd 图（`full_weight` 直接传给 GMM），所以原来「挂在权重上」的
bridge 失去了触发点。新的 `_MoonEPWeightGradBridge` 改挂在 **`dispatched` 激活**上：

```
forward :  dispatch → bridge(identity) → GMM(gate_up) → act → GMM(down) → combine
backward:  combine → GMM(down)↓写 buffer → act → GMM(gate_up)↓写 buffer → bridge → reduce
```

由于 bridge 在前向位于两个 GMM 之**前**，反向就自然位于两者之**后**，
满足「两块 buffer 都写完才做 reduce」。它依然是单个多输出节点，
保证各 rank 以相同顺序（先 `down` 后 `gate_up`）进入 MoonEP 内部 barrier。

---

## 3. 收益

设 `R` = EP size，`B = E/R`，单个 projection 权重梯度尺寸 `W = (E+B)·out·in`。

| 项 | 优化前 | 优化后 |
|---|---|---|
| 反向全尺寸临时分配 | 1×`W` BF16（NPU 2×） | 0（NPU 仍由算子内部产出 1 份，见下） |
| 全量拷贝 | NPU `torch.stack` 1× | 0 |
| 切片拷贝进池 | 2×`B·out·in` | 0 |
| 参数梯度 cast | 1×`B·out·in` | 1×`B·out·in`（autograd 需要） |
| 写入对称内存的行数 | `E+B`（含远端行） | `2B`（仅本地行） |

NPU 侧的说明：`torch_npu.npu_grouped_matmul` 目前没有 `out=` 语义，算子仍会分配自己的
输出。真正做到「零返回值」需要算子层支持写入指定 buffer，或按 host 侧 group 边界
只对本地 group 发起计算——后者会引入 device→host 同步，与 MoonEP 免同步的设计相冲突，
因此这一版保留算子输出，只消除框架侧的 stack / 拷贝。CUDA 侧因为是逐 group `torch.mm`，
可以做到完全不分配全尺寸张量。

---

## 4. 验证

```bash
cd sources/FSDPTurbo-moonep-mr47
PYTHONPATH=. python3 -m pytest tests/unit -q
```

新增/更新的用例：

| 文件 | 覆盖内容 |
|---|---|
| `tests/unit/test_moonep_grad_weight_sink.py` | sink 结果与 autograd 参考一致；FP32 目标 + BF16 计算；越界行报错；空 group 不被清零；shape 校验 |
| `tests/unit/test_moonep_weight_grad_bridge.py` | reduce 在两个 GMM 反向之后触发；固定 `down → gate_up` 顺序；参数梯度与纯 autograd 参考逐元素相等 |
| `tests/unit/test_moonep_gradient_reduce.py` | `reduce_gradient` 不再拷贝、返回独立局部张量 |

上机验证（需 NPU/GPU 实机）：

```bash
bash tests/system_tests/model/run_test_qwen3_moonep.sh
```

判据与优化前一致：10 步跑完、loss 有限、正常退出；并对比 `peak_memory` 应下降。
