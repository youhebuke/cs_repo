# 两版实现对比评审：MoonEP 反向梯度缓冲优化

对比对象：

- **A 版**（用户上传，`optimized_inline.zip`）：改动 `moonep_adapter.py`、`moonep_dispatcher.py`、`ops/moe.py`、`ops/npu/grouped_matmul.py`
- **B 版**（本仓库）：分支 `cursor/moonep-npu-main-grad-vmm-380a` / `cursor/moonep-gpu-main-grad-vmm-380a`

结论先说：**两版都不是完全正确的**。A 版的优化点 1 没有真正落地（用的是本地普通张量，不是 VMM 映射），且视图缓存有资源泄漏；B 版漏掉了 A 版里一个真正有价值的优化（权重视图缓存）。最终版本 = B 版的 VMM 映射 + A 版的视图缓存（修掉泄漏）。

---

## 1. 优化点 1：`main_grad` 的 `[E+B]` VMM 映射

### A 版做法

```python
# _ProjectionPool.__init__
self.main_grad_full = torch.empty(
    runtime.num_experts + experts_per_rank, out_features, in_features,
    dtype=torch.bfloat16,
    device=torch.accelerator.current_accelerator(),
)
```

一块**进程本地的普通 BF16 张量**，不是 VMM 映射。GMM 反向经 `grad_weight_out=` 直写它，然后 `reduce_gradient` 里原来的两次 `copy_` **原样保留**。

A 版自己在注释里说明了为什么不做 VMM：

> 若进一步做成与 `owner_grad_full` 同构的 `[E]` 全局 VMM 映射……需要 GMM 支持组掩码或把远端行映射到本地哑页 —— 否则 GMM 反向对空组的 dense 零写会变成跨卡写流量

**这个顾虑本身是对的**：`npu_grouped_matmul` 会对所有 G 组稠密输出（空组写零），如果目标直接是对称内存，空组的零写就会打到远端 rank 的显存上，既是跨卡流量，更会**覆盖 home rank 的梯度**。

**但结论下早了**：不需要让算子直接写 VMM 缓冲。算子照常写它自己的输出，框架侧只把**本 rank 拥有的行**拷进 VMM 缓冲即可。这样 VMM 映射成立，同时零跨卡写。

### B 版做法

`_ProjectionPool` 按 MoonEP README 的布局构建：

| 行区间 | 物理来源 |
|---|---|
| `[0, E)` | 每 rank 一块 `[E/R, out, in]` FP32 分配，IPC fd 交换后映射成连续 `[E]` |
| `[E, E+B)` | 本 rank reduce buffer 分片（与 `reduce_buffers[rank]` 同一块物理内存） |

于是 `reduce_gradient` 里两次 `copy_` **可以彻底删掉**——数据本来就已经在归约缓冲里了。

### 差异量化

设 `R` = EP size，`B = E/R`，`G = E+B`。单 projection 单层单步：

| 项 | 基线 | A 版 | B 版 |
|---|---|---|---|
| autograd 全尺寸分配 | `G` 行 BF16 | 0 | 0 |
| 算子自身输出 | `G` 行 | `G` 行 | `G` 行 |
| `torch.stack` 全量拷贝 | `G` 行 | 0 | 0 |
| 转置拷贝进 main_grad | — | **`G` 行** | 0 |
| 拷进 FP32 归约池 | `2B` 行 | **`2B` 行** | **0** |
| 写进对称内存的行数 | — | — | `2B` 行 |
| **合计拷贝行数** | `G + 2B` | `G + 2B` | **`2B`** |

R=8 时 `G = 9B`，A 版拷 11B 行、B 版拷 2B 行，**相差 5.5 倍**。A 版消掉的是 autograd 分配和 `torch.stack`，但自己又引入了一次 `G` 行的转置拷贝，总拷贝量与基线持平。

### 判定

| | 优化点 1 |
|---|---|
| A 版 | ❌ 未实现 VMM 映射；消除了重复分配，但拷贝量未减 |
| B 版 | ✅ 实现了 `[E+B]` FP32 VMM 映射，拷贝量降到 `2B` 行 |

---

## 2. 优化点 2：算子封装

### 2.1 `grad_weight_out` 直写（两版都对）

A 版：

```python
grad_weight_out.copy_(grad_weight.transpose(1, 2))
grad_weight_result = grad_weight_out
```

B 版：

```python
for start, end in sink.row_ranges:
    sink.buffer[start:end].copy_(grad_weight[start:end])
```

两版都用 `.transpose(1, 2)` 视图替掉了基线的 `torch.stack([w.T for w in grad_weight])`（一次全尺寸分配 + 拷贝）。**这一条 A 版做得完全正确**，回退路径的 `.transpose(1,2).contiguous()` 与基线逐位等价，A 版自带的 `test_gmm_equiv.py` 也验证了（我在本地跑通，4 项全 PASS）。

差别只在目标：A 版拷全部 `G` 行到本地缓冲，B 版拷 `2B` 行到 VMM 缓冲。

B 版另外用 `output_dtype=torch.float32` 让算子直出 FP32（带能力探测回退），省掉一次 BF16→FP32 转换。

### 2.2 `out=` 前向输出缓冲（A 版独有，但是无效优化）

```python
fwd_output = torch_npu.npu_grouped_matmul(...)   # 算子仍然分配
if out is not None:
    out.copy_(fwd_output)                        # 又加了一次全量拷贝
    ctx.mark_dirty(out)
    fwd_output = out
```

A 版注释声称「当前已把 per-step 的 `[M, N]` 分配消掉」——**这句不成立**：算子的输出 `fwd_output` 照样分配了，`out=` 只是在后面**多加了一次 `[M, N]` 拷贝**。A 版自己的 TODO 也承认要等算子支持 `out=` 形参才能真正零拷贝。

A 版把它默认关闭（`MOONEP_GMM_STATIC_OUT=0`），理由分析得很准确：跨层复用同一块前向输出缓冲会覆盖 autograd 已保存的激活（`gate/up` 被 SiLU 反向保存、`expert_output` 被 `_MoonEPWeightedCombine.backward` 保存），只有在 MoE 层被 recompute 完全覆盖或纯推理时才安全。

所以这段是**默认不生效的死代码，且启用后是净负收益**。B 版没有实现它，这是对的。

### 2.3 权重视图缓存（A 版独有，是真正有价值的优化）

```python
_VIEW_CACHE: dict = {}
def _cached_weight_views(weights):
    key = (weights.data_ptr(), tuple(weights.shape), tuple(weights.stride()))
    ...
```

基线在 forward 和 backward 里各重建一次 `[w[0] for w in weights.chunk(G, dim=0)]`，共 `2G` 次 Python 级视图构建。G 取 K3 满配的 1008、48 层，每步就是十万量级的 host 操作，直接堵在 kernel 下发前面。**这是 B 版原本漏掉的一个实打实的优化点，已采纳。**

但 A 版的实现有两个缺陷，我实测复现了：

**缺陷 1：缓存无界增长。** key 用 `data_ptr()`，永不淘汰。A 版自己的测试就暴露了这点——`PASS 4` 打印 `视图缓存 entries 3→4`，条目明明增加了，断言文案却写「无重复构建」：

```
第1个权重张量 -> _VIEW_CACHE entries = 1
...
第5个权重张量 -> _VIEW_CACHE entries = 5
```

**缺陷 2：强引用钉住 VMM 映射，`close()` 失效。**

```
close() 之后底层 storage 是否已释放: False
_VIEW_CACHE 仍持有 entries = 1
清空缓存后才释放: True
```

`MoonEPSymmetricProjection.close()` 里 `self.full_weight = None` 本意是释放对称内存映射，但缓存里的视图通过 `_base` 强引用着它，映射释放不掉。MoonEP 用 `explicitly_destroy=True`，`model.close()` 必须在 `destroy_process_group()` 之前真正解映射，否则退出路径可能挂死。

**修复过程中的一个坑**：我先试了 `weakref.WeakKeyDictionary`，不行——value 里的视图通过 `_base` 强引用 key，环打不破。又试了把缓存挂到张量自己的属性上靠 GC 收环，实测也不行：

```
chunk _base is w: True
collect#1 freed=20 tensor_alive=True storage_alive=True
collect#2 freed=0 tensor_alive=True storage_alive=True
```

PyTorch 的 tensor GC 遍历不报告 `_base` 这条边，环收不掉。**结论：只要缓存了视图，就必然钉住权重张量，只能显式释放。**

最终方案（`fsdp_turbo/ops/weight_views.py`）：

- 独立模块，`grouped_weight_views()` / `release_weight_views()`
- `MoonEPSymmetricProjection.close()` 在置空 `full_weight` **之前**调用 `release_weight_views()`
- 有界 FIFO（`_MAX_ENTRIES=256`）兜底，防止漏调用时无限增长；在张量存活期间淘汰是安全的（下次重建），而不是等张量释放后才淘汰（那样会有地址复用导致的悬垂视图）
- key 加上 dtype 和 device，避免同地址不同解释

---

## 3. 其他差异

### 3.1 bridge 位置

- **A 版**：保持基线，bridge 挂在权重上。正确——`_MoonEPWeightsBridge` 是单节点双输出，autograd 会等两个输出的梯度都到齐才执行 backward，所以 reduce 天然在两个 GMM 反向之后。
- **B 版**：bridge 改挂在 `dispatched` 激活上。**这不是风格选择而是必需**：B 版让 GMM 完全不返回权重梯度（FP32 缓冲与 BF16 权重 dtype 不符，autograd 会校验失败），权重因此退出计算图，挂在权重上的 bridge 就没有触发点了。改挂上游后，前向在两个 GMM 之前 ⇒ 反向在两者之后，语义等价。

A 版的位置**改动更小**，在它自己的设计下是正确的。

### 3.2 `reduce_gradient` 的 dtype 转换

A 版把 `reduce_gradient` 改成返回池内 FP32 视图，由 bridge 统一做 `.to(dtype, copy=True)`，注释说「消除原先的双重转换」。

**这个收益不存在**：基线里 `reduce_gradient` 返回 `.to(full_grad.dtype)` 已经是 BF16，bridge 里的 `.to(ctx.gate_up_dtype)` 是同 dtype 的 **no-op**，本来就只有一次转换。改动本身无害（异 dtype 时更稳），但不是优化。

### 3.3 空组处理

- **A 版**：目标是本地缓冲，空组写零无害，不需要处理。
- **B 版**：目标是对称内存，**必须**跳过空组而不是清零，否则会覆盖 home rank 的梯度；非空组落在本地行区间之外直接 `RuntimeError`。

这正是 A 版顾虑的那个风险，B 版通过「只拷本地行」而非「让算子直写」解决了。

---

## 4. 汇总

| 维度 | A 版 | B 版（最终） |
|---|---|---|
| OPT-1 `[E+B]` VMM 映射 | ❌ 本地普通张量 | ✅ FP32 对称内存映射 |
| OPT-1 消除 autograd 全尺寸分配 | ✅ | ✅ |
| OPT-1 消除归约池两次 `copy_` | ❌ 保留 | ✅ |
| 单步拷贝行数 | `G + 2B` | `2B` |
| OPT-2 去除 `torch.stack` | ✅ | ✅ |
| OPT-2 算子直出 FP32 | ❌ | ✅（带探测回退） |
| OPT-2 `out=` 前向缓冲 | ⚠️ 无效且默认关闭 | 未实现（有意） |
| 权重视图缓存 | ✅ 想法正确 ❌ 实现泄漏 | ✅ 已采纳并修复 |
| 空组/远端行防护 | 不适用 | ✅ 跳过 + 越界报错 |
| GPU 适配 | ❌ 未涉及 | ✅ 独立分支 |
| 等价性测试 | ✅ mock 算子对拍基线 | ✅ 34 条单测，含与纯 autograd 参考逐位比对 |

**A 版值得保留的**：`grad_weight_out` 直写、`transpose` 替代 `torch.stack`、权重视图缓存的思路、`out=` 的风险分析、mock 对拍测试方法。

**A 版需要修正的**：`main_grad` 改成真正的 `[E+B]` VMM 映射并删掉归约池的两次 `copy_`；视图缓存加显式释放与有界淘汰；删掉或明确标注 `out=` 路径当前无收益。

---

## 5. GPU 与 NPU 的实现差异

`[E+B]` VMM 映射、`GradWeightSink` 契约、bridge 位置、空组防护全部共用，差异只在算子层。

| | NPU | GPU |
|---|---|---|
| 权重梯度算子 | `npu_grouped_matmul(split_item=3, group_type=2)` | `F.grouped_mm(offs=..., out_dtype=...)` |
| 直出 FP32 | `output_dtype=torch.float32`（探测回退） | `out_dtype=torch.float32` |
| 写入粒度 | 算子输出全部 `G` 行 → 拷本地 `2B` 行 | 按本地 row range 分两次算，只出 `2B` 行 |
| 回退路径 | 原 dtype 输出 + 转换拷贝 | 逐 group `torch.mm(out=buffer[g])`，零全尺寸分配 |
| 视图缓存 | 需要（算子吃 per-group list） | 不需要（`grouped_mm` 吃整块 3D 张量） |

GPU 侧有一条 NPU 没有的硬约束：`torch.mm` 不允许 out 张量 dtype 与输入不同

```
RuntimeError: Expected out tensor to have dtype c10::BFloat16, but got float instead
```

所以 BF16 激活没法用 `torch.mm(out=fp32_buffer)` 直写，必须走 `F.grouped_mm(out_dtype=...)`，或退回「小块 staging + cast」。

---

## 6. 复现

```bash
cd sources/FSDPTurbo-moonep-mr47
PYTHONPATH=. python3 -m pytest tests/unit -q     # 34 passed, 2 skipped
```

A 版缺陷的复现脚本见本文第 2.3 节的输出，等价性测试可用 A 版自带的：

```bash
FSDPTURBO_GMM_BASELINE=<MR47 原始 grouped_matmul.py> python3 test_gmm_equiv.py
```
