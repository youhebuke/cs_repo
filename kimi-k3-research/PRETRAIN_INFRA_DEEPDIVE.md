# Kimi K3 · 3T 级预训练 Infra 深挖分析

> 本文对应技术报告 §5.2「Infra for 3T-class Pre-Training」，是 `DESIGN_NOTES.md` §4 的**加深版**：不复述结论，而是逐条拆解**方案细节 + 背后原理 + 数学/定量推导 + 权衡与对比 + 组件间耦合**。
>
> 读者默认熟悉 PP / TP / EP / DP / ZeRO / all-to-all / 1F1B 等概念。

---

## 0. 先立论：为什么 2.8T 预训练首先是"系统问题"

一个直觉：104B 激活参数、1M 上下文、原生多模态，单看**算法**都不新；难点在于**把它们同时喂饱数千卡且不 OOM、不空转**。K3 的 infra 哲学可以概括成一句话：

> **每一个算法选择都要先回答"它对通信量、显存账本、kernel 形状意味着什么"。**

三个数量级事实定义了问题边界：
- **参数显存**：2.78T 参数即使 BF16 也要 ~5.5 TB，MXFP4 专家权重是压显存的前提；但训练期梯度/优化器状态/激活才是真正的峰值来源。
- **激活显存**：93 层 × 1M token（长序列阶段）× hidden 7168，激活是 TB 级，必须靠"量化 + offload + 重计算 + 重平衡"组合拳而非单一手段。
- **专家通信**：896 选 16 的 all-to-all dispatch/combine，是 EP 的主带宽消耗；负载不均会让"最热 rank"决定整轮迭代时间。

因此 §5.2 的三大痛点——**EP 负载不均、显存超预算、ViT 计算方差暴露在关键路径**——本质是"**计算/通信/显存三者都被推到边界**"的直接后果。下面逐一深挖。

---

## 9.2.1 并行组合：为什么恰好是这一套

**组合**：`PP（interleaved 1F1B + VPP）+ EP + ZeRO-1 DP + Pipeline ZeRO-2 梯度分片 + CP(KCP)`。

### (a) 每一维的职责与"为什么必须有它"

| 维度 | 切什么 | 为什么在 K3 里不可或缺 |
|---|---|---|
| **PP（+VPP）** | 93 层按 stage 切 | 层数多、单层大；PP 把权重/优化器分摊到多机；VPP 压 pipeline bubble |
| **EP** | 896 专家按 rank 切 | 专家权重占参数大头，DP 复制会爆显存；EP 让每 rank 只存 `E/R` 个专家 |
| **ZeRO-1 DP** | 优化器状态分片 | 数据并行扩批量；ZeRO-1 只分片 optimizer state，通信最省 |
| **Pipeline ZeRO-2** | 梯度分片（+CPU） | 在 PP 之上再把**梯度**按 DP 分片、下沉 CPU，压 GPU 峰值 |
| **CP（KCP）** | 序列维切分 | 1M 上下文单卡放不下激活/状态；KDA 的递归态要用 KCP 专门处理（见 §4.1 KCP） |

**关键约束**：这几维不是自由叠加，而是相互耦合的——
- EP 的 all-to-all 与 PP 的 P2P、DP 的 reduce 争抢同一张网卡带宽 → 必须**用计算把通信盖住**（overlap 是全篇主线）。
- CP 把序列切开后，KDA 的**跨段递归**不能像 vanilla 线性注意力那样"各段独立求和"，所以要专门的 KCP（转移矩阵分解 + 前缀扫描）。
- ZeRO-2 把梯度下沉 CPU，是为了给**外部激活 offload / KV pool** 腾 HBM——显存是全局零和博弈。

### (b) interleaved 1F1B + VPP 的原理与收益（定量）

- 朴素 GPipe 的 bubble 占比 ≈ `(p−1)/m`（p=stage 数，m=micro-batch 数）。1F1B（one-forward-one-backward）不改 bubble 比例但**大幅降低激活峰值**：稳态下每 stage 只需驻留 ~`p` 份在飞 micro-batch 的激活，而非全部 m 份。
- **VPP（virtual pipeline）**：把每个物理 stage 再拆成 `v` 个非连续"虚拟块"，bubble 占比降到 ≈ `(p−1)/(m·v)`，代价是**通信次数 ×v**。K3 层多、机器多（p 大），VPP 的 bubble 收益值得。
- **副作用（后面要治）**：1F1B 暖机期各 stage 驻留激活**不均**（stage 0 在暖机时压着最多的在飞 micro-batch）→ 见 §9.2.4 的「PP rank 间激活再平衡」。

### (c) Fig.11 调度解读：一切都在"叠瓦"

报告 Fig.11 的时间线把四条流并置：**计算流**（DataLoader → ViT fwd → Attn → SE1 → MLP → SE2 → WGrad …）、**EP 通信**（EP-D 派发 / EP-C 合并 / EP-DR 派发重算）、**NCCL 通信**（gather param、reduce grad = `reduce_scatter + onload + add + offload`）、**本地/远程激活 offload**。可读出三条设计意图：
1. **shared expert 拆成 SE1/SE2 两段**并派到独立 stream，用来填 all-to-all（EP-D/EP-C）的通信空档。
2. **reduce grad 是复合操作**（reduce_scatter 到 GPU double buffer → onload CPU 分片 → add → offload 回 CPU），正是 Pipeline ZeRO-2 + CPU 梯度的落地。
3. **EP-DR（dispatch 重算）**出现在反向，对应 memory-efficient MoE 的"反向重算 dispatch 并与 group-GEMM 反向 overlap"。

> 一句话：**这套并行组合的本质不是"切得多"，而是"切完之后每一条通信都能被某段计算盖住"**。

---

## 9.2.2 三大痛点的本质（为什么是这三个）

| 痛点 | 直接症状 | 根因 |
|---|---|---|
| **EP 负载不均** | 热 rank 决定迭代时间；动态形状碎片化直至 OOM；每层 host↔device 同步取真实形状 | token 路由是**数据相关**的，专家负载天然长尾 |
| **显存超预算** | 激活/梯度/优化器状态之和 > HBM | 2.8T 参数 + 1M 序列 + 多模态，三者叠加 |
| **ViT 计算方差** | 大图/长视频让 encoder 计算量剧烈波动，落在关键路径拖慢整条流水 | 视觉样本尺寸方差大，且 encoder 与 backbone 串行 |

这三条分别由 **MoonEP（§9.2.3）**、**显存组合拳（§9.2.4）**、**多模态 encoder 优化（§9.2.5）** 接管。

---

## 9.2.3 MoonEP：把 EP 负载均衡从"缓解"做成"消灭"

### (a) 常规 EP 到底痛在哪（三连击）

设 `S` = 每 rank 输入 token 数、`K` = top-k、`E` = 专家数、`R` = EP size。
1. **热 rank 决定迭代时间**：EP 是同步屏障，最慢的 rank 拖住所有 rank。负载 `T_e/T̄ − 1`（maxvio）越大越慢。
2. **动态形状 → 显存碎片 → OOM**：每步每层路由到各专家的 token 数都不同，routed 激活 shape 逐步/逐层变化，分配器反复申请释放不同大小的块 → 碎片累积，高不均时直接 OOM。
3. **每层 host-device 同步**：grouped GEMM 需要 `cu_seqlens`（每专家 token 数）才能确定 kernel 形状；这些计数是**设备上算出来的数据相关量**，host 必须先把它读回来（device→host 同步/停顿）才能按正确形状发射 GEMM——**每层一次**，把 CPU 卡在关键路径上。

> 注意：这三点**同源**——都是"token→专家的映射是动态且不均"的后果。所以治本之道是**让映射的结果对系统而言变成静态且均衡**。

### (b) 核心机制：动态冗余专家实现"完美均衡"

**目标**：让每个 rank 恰好收到 `S×K` 个 token（所有 rank 计算量完全相同）。
**手段**：**冗余专家（redundant expert）**——把一个热专家临时复制到别的 rank 上，让它的 token 可以被拆分到多个副本，从而把长尾"摊平"。

- **前向**：从当前 micro-batch/层的 router 输出**在线规划**冗余专家放置，并**预取（prefetch）**其权重到本地预取槽；然后所有 rank 做形状一致的 grouped GEMM。
- **反向**：冗余专家产生的梯度先 stage 到**本地 reduce buffer**，计算完成后再 `reduce` 回其 **home rank**（权重真正的属主 rank）的梯度 buffer。README 的布局：权重 `[E+B, H, H']`（`[0,E)` 是各 rank 本地专家、`[E,E+B)` 是预取槽），梯度镜像同布局，reduce buffer `[R,B,H,H']` 让每 rank 通过 NVLink 远程读回属于自己专家的分片并累加。

### (c) 理论保证：`E/R` 冗余专家上界（重述证明 + 解读）

这是 MoonEP 相对 ECHO/UltraEP 的**根本差异**——它不是"预设一个够用的冗余数"，而是**证明了一个恒成立的上界**。

记 `m_r(P)` = 计划 `P` 下 rank `r` 的冗余专家数，`M(I) = min_P max_r m_r(P)`（在最优计划下的最坏 rank 冗余数）。

**定理 1（上界）**：对任意路由输出 `I`，`M(I) ≤ E/R`。

**证明的关键引理 + 构造**：存在一个计划使 (i) 每 rank 恰收 `S×K`；(ii) 每 rank 的远程 token 只来自**一个**其它 rank。
- 初始：每 rank 只持本地 token，按是否达到 `S×K` 分成"欠载/过载"。
- 迭代：取一个欠载 rank 和一个过载 rank，从过载迁 token **恰好填满**欠载 rank 到 `S×K`；被填满的 rank 从此不再变。
- 终止：每次填满**一个**欠载 rank → 至多 `R−1` 次；每 rank 至多被填**一次** → 其远程 token 来自**单一** rank。
- 收尾：单一源 rank 只持有 `E/R` 个本地专家 → rank `r` 的远程 token 至多涉及 `E/R` 个专家 → `m_r ≤ E/R`。∎

**定理 2（基本紧）**：构造 `I*`：rank 0 的专家收 0 token，其余 `R−1` 个 rank 均分所有 token。则 rank 0 必须收 `S×K` 个**全远程** token，这些 token 至少涉及 `⌈E(R−1)/R²⌉` 个不同专家 → `M(I*) ≥ ⌈E(R−1)/R²⌉ ≈ E/R`。∎

**工程含义**：每 rank **预留 `E/R` 个冗余槽**（训练时 `B = E/R`），则在线规划**永远有可行解**，训练**永不因不均而中断**。
- 对比 **ECHO / UltraEP**：预设固定冗余数或 per-rank token cap。当路由极端倾斜、无可行方案落在 cap 内时，**训练被迫停止**；cap 还要人工调参，且往往**残留不均**。
- 对比 **DeepEP**：不做冗余专家，靠通信硬扛不均，延迟由最热 rank 决定。

### (d) 在线规划：ILP 做"标尺"，GPU kernel 做"生产"

- 每步精确求最优放置是 ILP，训练时在线跑 ILP 不可行。
- 做法：**离线用 ILP 对代表性用例求精确最优做参照**，据此设计一个**近最优的 GPU planning kernel**——开销可忽略、始终满足 `E/R` 上界。这是"用离线最优标定在线启发式"的经典范式。

### (e) Zero-copy：为什么 MoonEP 只要 `S×K` 而 DeepEP 要 `S×K×R`

- **原理**：规划 kernel **预计算每个 token 的最终目的地**（远端某 rank 上按专家分组后的确切槽位），dispatch 时 token **直接写到该最终位置**，通信 buffer 的**视图**直接交给 grouped GEMM 计算——**没有 comm-buffer → user-buffer 的拷贝**（这一步拷贝在常规实现里是 epilogue 的主开销）。
- **缓冲区大小推理**：
  - DeepEP 要保证零拷贝，必须为**最坏倾斜**预留空间——极端情况下某 rank 可能收到所有 `R` 个 rank 各自的 `S×K` token → 接收 buffer 需 `S×K×R`。
  - MoonEP 因为**完美均衡**，每 rank 恒收 `S×K` → buffer **固定 `S×K`**，与倾斜无关。
- 这解释了 README 基准里"通信时间随 imbalance 几乎平坦"：**buffer 不随倾斜膨胀、通信路径无拷贝**。

### (f) 静态形状：一次均衡换掉两类隐藏开销

完美均衡 ⇒ 每 rank 每层的 MoE 计算形状**静态已知**（固定 `E+B` 组、每组 token 数已知）：
1. **消除每层 MoE 的 host 同步**：不再需要把 `cu_seqlens` 读回 host 来决定 kernel 形状 → CPU 不再卡在关键路径。
2. **降 kernel 启动开销**：静态形状可提前编排、复用 launch 配置。

### (g) rank 内偏斜：均衡"总量"之后还要均衡"makespan"

即使各 rank 总 token 数相等，**rank 内每个专家的 token 数仍偏斜**。固定顺序、workload-无关的调度会把这种偏斜变成 SM worker 间不均的 makespan。
- 解法：**workload-aware GEMM 调度器**——发射前根据当前 token 分布调参、执行中固定；参数由**分析型硬件代价模型 + 离线 autotune 标定的系数**选出。
- **shared 专家 GEMM 派到独立 stream**，与其它 kernel overlap（呼应 Fig.11 的 SE1/SE2）。

### (h) 基准与对比总表

| 维度 | DeepEP | ECHO / UltraEP | **MoonEP** |
|---|---|---|---|
| 负载均衡 | 无（硬扛） | 固定冗余/ cap，可能无解 | **完美均衡，规划恒可行** |
| 零拷贝 buffer | 最坏 `S×K×R` | — | **固定 `S×K`** |
| 形状 | 动态（每层 host 同步） | 动态 | **静态（免同步）** |
| 高不均行为 | 通信恶化、碎片 → OOM | **训练可能中断** + 手调 | **迭代时间持平、不 OOM** |

> 一句话总结 MoonEP 的思想：**先花很小的代价（≤E/R 个冗余专家 + 一个规划 kernel）把"数据相关的不均"变成"系统层面的完全确定与均衡"，随后零拷贝、静态形状、免同步全都是这一步的免费红利。**

---

## 9.2.4 显存工程：把 2.8T 塞进预算的一整套组合拳

显存是**全局零和**：省下来的每一 GB 都用去支撑更大的批量 / 更长的序列 / 更少的重计算。K3 不靠单一手段，而是一套**可组合、可在张量粒度调度**的策略框架。

### (a) 统一激活管理器：把"存储策略"从模型代码里剥离

- **抽象**：每个"为反向保存的张量"挂一个**可插拔存储后端**；**重计算 / FP8 块量化 / offload / 远程 offload** 都只是"存储策略"，可在**张量粒度**自由组合、用**注解**声明、与模型代码**解耦**。
- **实现要点**：
  - **重计算按函数粒度** → 支持**跨层重计算**（不局限单层）。
  - **单一内存池、主计算流上分配** → 避免多 stream 分配器的碎片与 host-bound 开销（这点常被忽视却致命：多 stream 各自持有 allocator 缓存会互相割裂显存）。
  - **层粒度预取并与计算 overlap** → offload 的搬运不进关键路径。
- **K3 实际配置**：多数激活 **FP8 块量化 + offload/远程 offload**；逐元素算子（便宜、重算划算）**配重计算**。
- **原理**：这就是把经典的"激活显存 ↔ 重计算算力 ↔ offload 带宽"三角，做成**每个张量各自最优**的策略搜索空间，而不是全网一刀切。

### (b) Memory-efficient MoE：用"梯度数学变换"省掉前向输出的保存

- **痛点**：native 实现里，permuted probs 的梯度**依赖前向输出 `output`** → 必须把 `output` 存到反向。
- **解法（借 SonicMoE）**：把该梯度**数学变换**成只依赖**中间激活 `act_output`** 与**上游梯度 `doutput`** 的形式 → **消除对 `output` 的反向依赖**，代价是一点额外逐元素计算。
- **再加一层**：group GEMM 前向**只存 dispatch 的输入**；反向**重算 dispatch** 恢复输入。重算引入的通信被**与 group-GEMM 反向计算 overlap**（Fig.11 的 EP-DR）。
- **原理**：重计算只有在"重算能被现有计算盖住"时才真正免费——这里 dispatch 重算的 all-to-all 恰好塞进 group-GEMM 反向的计算窗口。**存储省掉、代价近零。**

### (c) Block AttnRes 的显存伴生优化：让深度维注意力"回到标准残差的足迹"

- **潜在问题**：AttnRes 需要"所有前层输出"活着 → 朴素实现 `O(Ld)` 激活。
- **三招压回标准残差足迹**：
  1. **块表示在边界层生成一次、后续层共享、常驻 GPU**（只有 `N≈8` 个块表示是额外状态）。
  2. **AttnRes 计算整体 checkpoint** → 每层为反向保存的激活与**标准残差架构完全相同**。
  3. **PP 用 cache-based 通信**：只**增量传新生成的块**、micro-batch 用完即释 → 达到**理论显存下界**。
- **原理**：把 `O(Ld)` 的"保活所有层输出"压成 `O(Nd)` 的"保活 N 个块表示 + 用重算换取其余"，且跨 stage 只传增量。

### (d) PP rank 间激活再平衡：治 1F1B 的"暖机不均"

- **现象**：interleaved 1F1B 暖机导致各 stage 驻留激活**递减**（越靠后的 PP rank 常驻激活越少；靠前的 rank 在暖机时压着最多在飞 micro-batch）。
- **解法**：用 **Mooncake Transfer Engine** 把激活**远程 offload 到别的 PP rank 的内存**，把峰值占用**拉平**，避免最紧的 rank OOM。
- **原理**：显存瓶颈是"单 rank 峰值"而非"总量"；把峰值从热 rank 挪到冷 rank，等效于提升可用批量/序列长度。

### (e) Pipeline ZeRO-2 + CPU 梯度

- 梯度按 DP rank **分片**；分片梯度存 **CPU 内存**降 GPU 峰值，**double grad buffer 留在 GPU**。
- 流程：DP 间 `reduce_scatter` 进 GPU double buffer → onload CPU 分片 → add → offload 回 CPU（即 Fig.11 的 `reduce grad` 复合操作）。
- **原理**：梯度只在优化器更新时才需要"完整"，其余时间可下沉 CPU；double buffer 保证 reduce 与 offload 能流水。

### (f) P2P Muon 正交化：为什么"点对点拉分片"远优于"全量 all-gather"

- **约束**：Newton–Schulz 正交化迭代 `X ← aX + b·X(XᵀX) + …` 是**整矩阵多项式**，必须要**完整参数矩阵**，无法在行/列分片上独立完成。但分布式优化器已把参数按 DP 分片。
- **朴素做法**：全量 all-gather 完整参数 buffer → **显存**（每 rank 多一份全参 buffer）+ **通信**（all-gather 全参）双爆炸，成为大规模主瓶颈。
- **K3 做法**：**每 rank 只 P2P 拉回"自己负责更新的那些参数"的分片**（向各 owner rank 点对点通信），重建自己要更新的完整矩阵即可 → **消除全参 buffer**，通信量与显存都降；再按 **model-chunk buffer 粒度**把通信与计算流水隐藏。
- **原理**：**只有"要更新某参数的 rank"才需要它的完整矩阵**，所以 gather 应是"属主→更新者"的**定向**传输，而非广播式 all-gather。定向后每 rank 的临时 buffer 只需其**负责的那部分**（`O(P/DP)`）而非全参（`O(P)`）。

### (g) 显存账本速查（这套组合拳各自省在哪）

| 手段 | 省的是 | 代价 |
|---|---|---|
| FP8 块量化激活 | 激活字节（~½） | 量化/反量化算力 |
| offload / 远程 offload | HBM ↔ 换 CPU/远端 HBM | PCIe/NVLink 带宽（被 overlap 盖住） |
| 跨层重计算 | 激活保存 | 前向重算算力 |
| Memory-efficient MoE | 前向 `output` + dispatch 输入 | 少量逐元素 + 可 overlap 的重算 |
| Block AttnRes checkpoint | `O(Ld) → O(Nd)` | AttnRes 重算 |
| PP 激活再平衡 | **峰值** rank 的 HBM | 跨 rank 传输 |
| Pipeline ZeRO-2 + CPU 梯度 | GPU 梯度显存 | CPU 内存 + offload 带宽 |
| P2P Muon | 全参 buffer + all-gather 通信 | P2P 通信（流水隐藏） |

---

## 9.2.5 多模态 encoder：把 ViT 的"计算方差"从关键路径上抹掉

### (a) 方差从哪来、为什么危险
大图 / 长视频让 ViT 的 patch 数剧烈波动 → **encoder 计算量高方差**；而 encoder 与 backbone **串行**，方差直接变成 PP 关键路径上的 straggler，拖慢整条流水。

### (b) Dynamic CP：大样本沿 patch 维切 + 组内均衡
- 单张大图**沿 patch 维切到多设备**，注意力用 **gather-KV**（跨 CP rank 收集 K/V）计算。
- 每个 CP 组再分成若干 **sub-CP 组**，把**多张大图负载均衡**分配到各 sub-group → **通信占比不随规模增长**（否则大图越多、gather-KV 通信越吃比例）。

### (c) Bubble filling：把 ViT 计算塞进 PP 气泡（承 K2.5 DEP，思路同 Optimus）
- K2.5 的 **Decoupled Encoder Process (DEP)**：把 ViT 与文本训练拆成独立 stage，并把视觉的前向/反向**均衡到各 PP stage**。
- K3 进一步**分解 ViT 计算**：观察到 interleaved 1F1B 下，**首批 PP micro-batch 的文本前向全排在最开头、末批的文本反向只在最末尾结束** → 中间有大量气泡。
  - **首批 micro-batch 的 ViT 前向同步先跑**，其余前向**调度进气泡**；反向同理。
  - 结果：**大部分 ViT 计算被气泡吸收**，视觉 encoder 的**有效开销基本消除**。
- **原理**：PP 天然存在 bubble；ViT 是"可搬动、无跨 micro-batch 依赖"的计算，正好用来填 bubble——**用一个本来就浪费的时间窗，装下一个本来在关键路径上的负载**。

---

## 9.2.6 组件间的耦合：谁解决谁、谁增强谁

infra 各模块不是孤立的，存在明确的"解决/增强"关系：

```
痛点①EP不均 ──► MoonEP(完美均衡)
                   │ 带来静态形状 ──► 免 host 同步、降 launch 开销
                   │ 带来零拷贝  ──► 通信不随倾斜膨胀
                   └ 带来无碎片  ──► 缓解痛点②(显存)

痛点②显存   ──► 统一激活管理(量化/offload/重算)
                + Memory-efficient MoE(免存输出)
                + Block AttnRes checkpoint(回到标准残差足迹)
                + PP激活再平衡(削峰)
                + Pipeline ZeRO-2 + CPU梯度(下沉梯度)
                + P2P Muon(消除全参buffer)
                   └ 腾出的 HBM ──► 支撑更长序列 / 外部KV pool(RL)

痛点③ViT方差 ──► Dynamic CP(切大图+组内均衡)
                + Bubble filling(ViT塞进PP气泡)

贯穿全篇     ──► overlap：每条通信(all-to-all/P2P/reduce/offload)都被某段计算盖住
```

两个"协同放大"的例子值得单独强调：
1. **MoonEP 的静态形状 × 统一内存池**：动态形状是碎片之源；MoonEP 消灭动态形状后，单内存池几乎不再碎片化——**均衡直接改善了显存**，而不仅是吞吐。
2. **KDA 的"下界化衰减"（算法侧）× FlashKDA（系统侧）**：给门加下界让对角块能上 Tensor Core，FlashKDA 才能把块内计算与跨块状态传播充分 overlap——**算法改动是系统优化的前提**。

---

## 9.2.7 权衡、风险与可迁移的经验

### (a) 主要权衡
- **复杂度换效率**：MoonEP/KCP/统一激活管理/P2P Muon 都显著增加系统复杂度；收益是"3T/1M 训得动且不空转"。这套复杂度只有在**规模足够大**时才回本。
- **CPU/远端内存与带宽成为新资源**：offload、CPU 梯度、外部 KV pool 把压力从 HBM 转到 CPU DRAM 与互联带宽——**带宽预算取代显存预算**成为新的一等约束。
- **冗余专家的额外算力/权重流量**：MoonEP 的均衡不是免费的，冗余专家要预取权重、反向要额外 reduce；但相对"热 rank 拖慢 + OOM 停训"，这点开销可忽略。

### (b) 风险点
- **规划 kernel 的近最优性**：在线 GPU planning 是近似解，极端分布下与 ILP 最优的差距需靠离线标定兜住。
- **offload/远程 offload 依赖互联**：Mooncake Transfer Engine / NVLink 带宽若不足，overlap 会失败、搬运漏进关键路径。
- **静态形状的前提**：一旦某处破坏"每 rank 恰好 S×K"（如实现 bug），静态形状假设失效会连带 host 同步回归。

### (c) 可迁移给我们自己项目的经验
1. **均衡优先于扩容**：先把负载/形状做"确定 + 均衡"，很多下游优化（零拷贝、静态形状、免同步、无碎片）会自动到手。
2. **重计算只有能 overlap 才算免费**：设计重算时，先确认它能塞进哪段现有计算/通信窗口。
3. **通信要"定向"而非"广播"**：P2P Muon 的教训——想清楚"谁真正需要这份数据"，把 all-gather 降级为点对点。
4. **把高方差、可搬动的计算塞进 bubble**：ViT bubble filling 的思路可推广到任何"与主流水无强依赖"的旁路计算。
5. **算法-系统协同**：改激活函数（SiTU-GLU 有界）、改门（KDA 下界）、改均衡（QB 分位数）时，同步评估其对 kernel/通信/显存的连锁影响——**最好的系统优化往往始于一个小的算法改动**。

---

### 附：符号表
`S`=每 rank 输入 token 数 · `K`=每 token top-k · `E`=路由专家数 · `R`=EP size · `H`=hidden · `H'`=专家 FFN 中间维 · `p`=PP stage 数 · `m`=micro-batch 数 · `v`=VPP 虚拟块数 · `L`=层数 · `N`=AttnRes block 数 · `P`=完整参数量。
