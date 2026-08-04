# Kimi K3 技术分享 · 思路梳理文档（落盘版）

> 本文件用于系统性梳理 Kimi K3 的设计动机、数学原理、核心模块演进、以及预训练 / RL / Infra 的方案与原理，作为 PPT 的思路底稿。
> **PPT 才是最终交付物**；本文档偏"全"，PPT 偏"讲清 why"。文中标注 **[PPT 重点]** 的部分对应 1 小时分享的主线，标注 **[略讲]** 的可在 PPT 中压缩。
>
> 目标读者：有大模型训练 infra / 算法背景的同事。基础结构（MoE、attention、GLU、RMSNorm 等）默认已知。

---

## 0. 资料来源与阅读说明

| 来源 | 内容 | 用途 |
|---|---|---|
| `k3_tech_report.pdf`（MoonshotAI/Kimi-K3） | K3 官方技术报告（正文 + 附录 B–F 含全部数学推导） | 主线依据 |
| `Attention-Residuals`（arXiv 2603.15031） | 深度维注意力残差论文 + 伪代码 | §2.2 依据 |
| `FlashKDA` | KDA 的 CUTLASS 高性能 kernel | §4 KDA 协同设计 |
| `MoonEP` | 完美负载均衡 EP 通信库 + 上界证明 | §4 MoE 训练 |
| `Kimi Linear`（arXiv 2510.26692） | KDA 前身，3:1 KDA:MLA 混合 | 演进脉络 |
| DeepSeek-V3（2412.19437）/ Qwen3（2505.09388） | 对比基线（真实公开架构） | §2.6 对比 |

> 说明：报告中出现的 Claude Fable 5 / GPT-5.6 Sol / DeepSeek-V4 / GLM-5.2 / Kimi K2.5 等为报告自身的时间线（2026-07）设定，评测数字以报告为准，不做外部考证；**架构与 infra 的技术内容是分享重点**。

---

## 1. K3 的定位与总体设计哲学 [PPT 重点]

### 1.1 一句话定位
Kimi K3 = **2.8T 总参 / 104B 激活**、原生多模态、1M 上下文的开源 MoE 模型；相比 Kimi K2 在**整体 scaling 效率上提升约 2.5×**。

### 1.2 为什么要做 3T 级？（背景动机）
- LLM 的 scaling 有两条轴：**(轴一) 预训练规模**（更大模型 + 更多数据）与 **(轴二) test-time compute**（RL + 推理 effort + 长程交互）。
- 近两年开源社区在**轴二**上进展很快（推理 / agentic RL），但在**轴一**上停滞——大量模型停留在 1T 级左右。若都在相似规模的底座上堆 RL，开源会**收敛**而与闭源前沿的差距**扩大**。
- K3 的选择：**两轴同时推到前沿**——底座推到史无前例的 3T 级，同时把 RL / reasoning effort / 长程交互推到 1M 上下文。

### 1.3 贯穿全文的主线：沿三个维度"扩展信息流"
这是理解 K3 结构设计的**总纲**——所有结构创新都在回答"信息如何更高效地流动"：

| 维度 | 机制 | 解决什么 |
|---|---|---|
| **序列维（token mixing）** | Hybrid Attention：3×KDA + 1×Gated MLA | 长序列高效混合 + 保留全局高容量注意力 |
| **深度维（layer mixing）** | Attention Residuals (AttnRes) | 每层可**选择性**检索所有前层表示，突破串行残差瓶颈 |
| **宽度维（channel mixing）** | Stable LatentMoE：896 选 16 | 稀疏扩展通道混合容量 |

> 记忆钩子：**KDA 管"跨 token"，AttnRes 管"跨 layer"，LatentMoE 管"跨 channel/expert"，Per-Head Muon 管"怎么训得稳"。**

### 1.4 关键规格（对比 K2）

| 项 | Kimi K2 | Kimi K3 | Δ |
|---|---|---|---|
| 总参 / 激活 | 1.04T / 32.6B | **2.78T / 104B** | +167% / +220% |
| 层数 | 61 | **93**（1 dense） | +52% |
| 注意力 | 全 MLA | **69 KDA + 24 Gated MLA**（3:1，末尾补 1 MLA） | 混合 |
| 路由专家 / 激活 | 384 / 8 | **896 / 16**（+2 shared） | +133% / +100% |
| Latent MoE 维 | – | **3584 (0.5× hidden)** | 新增 |
| 每专家 FFN 维 | 2048 | 3072 | +50% |
| 激活函数 | SwiGLU | **SiTU-GLU** | 新 |
| 训练上下文 | 128K | **1M** | 8× |
| 位置编码 | RoPE | **NoPE**（KDA 隐式提供位置） | 新 |
| ViT | – | **MoonViT-V2, 401M, 27 层**（from scratch） | 新增 |
| 优化器 | Muon | **Per-Head Muon** | 精化 |
| 量化 | – | **MXFP4 权重 / MXFP8 激活（QAT）** | 新 |

---

## 2. 模型结构：设计动机与数学原理

### 2.1 Hybrid Attention（序列维）[PPT 重点]

**总体**：每个 block = 3 层 KDA + 1 层 Gated MLA（3:1），全网重复；backbone 末尾再补一层 Gated MLA，保证最后一层一定是全局注意力。思路继承自 Kimi Linear：**用线性注意力（KDA）承担绝大多数 token mixing，用少量全局注意力（MLA）补齐长程精确检索**——线性注意力最大的弱点就是长程精确回忆，3:1 是"效率 vs. 召回"的折中。

#### 2.1.1 Kimi Delta Attention (KDA)

**它是什么**：带**通道级遗忘门**的 delta-rule 线性注意力（gated DeltaNet 的精化）。状态 `S_t ∈ R^{d_k×d_v}` 是**固定大小**的递归记忆（不像 softmax 注意力的 KV cache 随长度增长）。

核心递归（单头）：

```
S_t = ( I − β_t k_t k_t^T ) Diag(α_t) S_{t-1} + β_t k_t v_t^T
õ_t = S_t^T q_t
```

- `Diag(α_t)`：**通道级**（每个特征维独立）的一步保留因子 α_t ∈ (0,1)^{d_k} —— 这是 KDA 相对 Gated DeltaNet / Mamba-2（**头级**标量门）的关键升级：更细粒度地调控有限状态记忆的衰减与位置感知（借鉴 GLA 的 per-channel gate，但结合了 delta rule）。
- `(I − β_t k_t k_t^T)`：delta rule 的"先擦除再写入"——先按 key 方向删除旧的关联，再写入新的 `k_t v_t^T`，`β_t` 控制写入强度。
- 参数化：q/k/v 用 `ShortConv + Swish`，q/k 再做 `L2Norm`；`β=Sigmoid(W_β x)`；衰减 logit `z_t` 用**低秩**投影 + head bias 得到。

**演进点 1 —— 下界化衰减（K3 相对 Kimi Linear 的改动）[PPT 重点]**
- 问题：chunkwise 形式中，要用**累计衰减的倒数** `1/Γ` 对 key 重标定；`Γ` 是一堆 (0,1) 因子连乘，倒数会**无界增长**，在有限精度（BF16）下溢/上溢。Kimi Linear 的做法是在 log 空间算相对衰减 + 把 chunk 再切成 16-token 小 tile，其中**对角 tile 只能用逐位置对 (position-pair) 计算**，成为 intra-chunk 主要瓶颈（不能上 Tensor Core）。
- K3 的改法：把 log-decay 的映射从"负 Softplus（无下界）"改成**带下界的缩放 sigmoid**：

```
g_t = g_min · Sigmoid(e^{A_h} z_t) ∈ (g_min, 0)      # g_min = −5 固定, A_h 可学
α_t = exp(g_t) ∈ (e^{−5}, 1) ≈ (6.7e−3, 1)
```

- 数学后果：每个保留因子 > e^{−5}，则一个 16-token tile 的累计 log-decay ∈ (−80, 0)，倒数重标定因子 < e^{80}，**落在 BF16 动态范围内**。于是**对角 tile 也能用稠密 Tensor Core 矩阵乘**，彻底消掉 position-pair 对角路径 → 训练/prefill kernel 大幅提速。
- **一句话 why**：一个"给门加下界"的小改动，把数值稳定性问题转成了**硬件友好性**收益（对角块从标量循环变成 GEMM）。这是"算法-系统协同设计"的典型缩影。

**演进点 2 —— 全秩输出门**
- Kimi Linear 用**低秩**输出门；K3 改成**输入相关的全秩**投影：`y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]`。
- why：让每个 token 更充分地调制从递归记忆读出的通道（表达力↑），并与 K3 里 MLA 的新门参数化保持一致。

#### 2.1.2 Gated MLA + NoPE
- **MLA**（源自 DeepSeek-V2）：把每个 token 的 KV 压缩成低维 latent `c_t = W_c x_t` 缓存，用时再上投影重建 K/V → **KV cache 大幅缩小**同时保留全局注意力。K3 在**周期性全局层**保留 MLA。
- **改动 1：NoPE（No Position Encoding）**——所有 MLA 层不加任何显式位置编码。理由：**位置信息由中间的 KDA 层通过递归门控/衰减隐式提供**；MLA 只负责"无约束的全局内容交互"。副产品：**扩上下文时不用改任何位置编码参数**（不用重调 RoPE base、不用 YaRN 插值）→ 直接外推到 1M。
- **改动 2：全秩输出门**：`y_t = W_o [ Sigmoid(W_g x_t) ⊙ õ_t ]`，与 KDA 一致。
- **数值工程**：为纠正 flash-attention 的有偏 rounding，训练时把注意力输出保持 **FP32**；这会翻倍片上占用，于是重设计 kernel 让 FP32 输出与 KV staging buffer 重叠（而非与 query tile 重叠），腾出 shared memory 做更深的 KV 流水。

### 2.2 Attention Residuals（深度维）[PPT 重点]

#### 2.2.1 动机：标准残差是"深度维的 RNN"
- 展开 PreNorm 残差 `h_l = h_1 + Σ_{i<l} f_i(h_i)`：**每层都用"固定单位权重"把所有前层输出加起来**。这就像**时间维的 RNN**：把所有历史压进一个状态。
- 三个后果：① **无选择性访问**（attention 层和 MLP 层拿到同一个聚合态）；② **不可逆损失**（早层信息被埋没，深层无法选择性找回，实测可裁掉相当比例的层）；③ **输出膨胀**（PreNorm 下 hidden 幅度随深度 O(L) 增长，稀释每层贡献、破坏稳定性）。
- **类比**：序列维当年用 attention 取代 RNN 递归；深度维也应如此。

#### 2.2.2 Full AttnRes：深度维 softmax 注意力
每层 l 用一个**可学的伪 query** `w_l ∈ R^d`，对"embedding + 所有前层输出"做 softmax 注意力：

```
k_i = v_i = { h_1 (i=0);  f_i(h_i) (1≤i≤l−1) }
φ(q,k) = exp( q^T · RMSNorm(k) )          # RMSNorm 防大幅度层主导权重
α_{i→l} = φ(w_l, k_i) / Σ_j φ(w_l, k_j)
h_l = Σ_{i=0}^{l−1} α_{i→l} · v_i
```

- **理论视角**：标准残差 / 各种加权递归变体 = **深度维线性注意力**；AttnRes = **深度维 softmax 注意力**。完成了序列维当年"linear→softmax"的同款跃迁。
- 深度 L<100，`O(L²d)` 算力可忽略；代价是 `O(Ld)` 显存（保活所有层输出）+ PP 下的跨 stage 通信。

#### 2.2.3 Block AttnRes：把开销从 O(Ld) 降到 O(Nd)
- 把 L 层分成 N 个 block；**block 内用标准残差求和**得到 block 表示 `b_n`，**block 间才做全注意力**（只对 N 个 block 级表示）。`b_0 = h_1`（embedding 恒为一个源）。
- N=1 退化为标准残差，N=L 恢复 Full AttnRes；实测 **N≈8** 就能拿回大部分收益。
- **K3 配置**：93 层切成 8 个 block（每块 12 层），加上 embedding 层共 9 个源 + 一个不满的末 block。
- 收益（论文级）：Block AttnRes 用同算力能匹配"多 1.25× 算力"的 baseline；hidden 幅度随深度**有界**、梯度在层间**更均匀**（缓解 PreNorm dilution）。下游全面提升，GPQA-Diamond +7.5、HumanEval +3.1。
- **推理/训练友好**：`w_l` 与前向解耦 → block 间注意力可并行；block 内部分和用 **online softmax** 合并，保证与 Full 严格等价；块结构还**界定了推理时的状态大小**。

### 2.3 Stable LatentMoE（宽度维）[PPT 重点]

#### 2.3.1 LatentMoE：分离"模型宽度"与"路由专家宽度"
- 常规 MoE：每个被选专家都吃**完整 d 维** token → 通信量与专家权重流量随激活多重度线性增长，扩到 896 选 16 会爆。
- LatentMoE 的关键：**shared 专家走全宽路径**（处理共性变换），**routed 专家在压缩 latent 空间（宽度 ℓ=3584=0.5×hidden）里算**：

```
z = W↓ x ∈ R^ℓ                          # 下投影到 latent
u = Σ_{i∈Topk} p_i · E_i^routed(z)       # 在 latent 空间路由+专家
y = Σ_j E_j^shared(x) + W↑ · RMSNorm(u)  # 上投影回 d 维, 与 shared 相加
```

- K3：`shared=2`，`routed=896 选 16`，稀疏度 ≈ 56。组织沿用 DeepSeekMoE 的 shared+routed。

#### 2.3.2 极端稀疏放大了两个失效模式，Stable LatentMoE 用三招治
极端稀疏（896 专家）把 vanilla 设计的两个问题放大：**① routed 分支近似 4 连乘矩阵（W↓ · gated-FFN · W↑），病态 → 内部激活爆炸**；**② 近 10³ 个专家的负载均衡超出了现有 aux-loss-free bias 更新能稳住的范围**。三招：

1. **Normalized LatentMoE**：在专家聚合 `u` 与上投影 `W↑` 之间插 **RMSNorm**。`u` 的尺度随所选专家/权重变化很大，归一化降低 routed 分支对尺度变化的敏感度，再与全宽 shared 分支相加。除稳训练外，也**稳定改善验证 loss 与下游**。

2. **SiTU-GLU（Sigmoid Tanh Unit GLU）**：治激活爆炸。
   - 背景：SwiGLU 的两个乘子（`x·σ(x)` 和 `x`）**都无界**，同时出现大坐标会产生激活离群点、低精度下溢出风险高；原始 GLU 的 sigmoid 门虽有界但丢了 Swish 的"正区近线性"特性。
   - SiTU-GLU：对 Swish 的线性因子和 up 分支**各自套一个 softcap `β·tanh(·/β)`**：

     ```
     SiTU-GLU(x) = β1·tanh(W_g x / β1) ⊙ Sigmoid(W_g x) ⊙ β2·tanh(W_u x / β2)
     ```

   - 超参：`β1=4`（门分支）、`β2=25`（up 分支）。
   - 数学性质（附录 B）：① 近原点 `β·tanh(z/β)=z+O(z³/β²)` → **一阶匹配 SwiGLU**（保留局部行为）；② `β1,β2→∞` 时逐点恢复 SwiGLU；③ **输出有界** `‖SiTU-GLU(x)‖_∞ ≤ β1·β2 = 100`。相比硬 clamp，平滑 cap 在非饱和区**保留非零梯度** → 训练行为更好。

3. **Quantile Balancing (QB)**：治大规模负载均衡。见下。

#### 2.3.3 Quantile Balancing (QB)：从"符号步长"到"一步到位的分位数"[PPT 重点]
- 出发点：K3 用 **aux-loss-free** 路由（DeepSeek-V3 同思路）——给 router 分数加**专家偏置 `b_j`** 影响 Top-k 选择，但 `b_j` **不进** softmax 权重 `p_{i,j}`（因此不改混合权重、不干扰 router 的梯度优化）。
- DeepSeek-V3 的更新是**固定步长符号**：`b_j ← b_j + γ·sign(负载误差)`；γ 大则震荡、γ 小则收敛慢。896 专家下更难稳。
- **QB 的洞见**：把负载均衡看成**最优平衡指派**问题（最大化分数、每专家恰好收 `q=mk/n` 个 token），做**线性松驰 + 对偶**，可证专家侧对偶变量 `β_j` 的闭式解正好是**每专家 margin 分布的 (1−k/n) 分位数**（附录 C）：

  ```
  b̂_j^{(t+1)} = − quantile_{1−k/n}( s_{:,j} − α^{(t)} )   # 一次前向即可解
  b^{(t+1)} = b̂ − mean(b̂)                                   # 去公共偏移, 不改 Top-k
  ```

  用 Top-(k+1) 路由取每个 token 的第 (k+1) 名作为 cutoff `α_i`，避免单独算 token 侧分位数。更新滞后一步生效（因果，batch 不会用自己算出的 bias）。
- **与符号法的关系**：该对偶目标的（次）梯度恰是"目标负载 − 实际负载"；SignSGD 一步就退化成 DeepSeek 的符号更新——**QB 直接跳到该对偶目标的精确坐标最小值**。这解释了为何 **QB 无需 learning-rate 类超参、且在近 10³ 专家下几步就均衡**。
- **工程落地：直方图估计（附录 D）[略讲但重要]**：全局 batch 的 margin 有数百万且分散在各 rank/累积步，精确分位数不可行。改为**每专家维护一个直方图**（对 `r_{i,j}=α_i−s_{i,j}` 分箱），一次 `all-reduce` 求和 bin 计数即可从池化计数恢复分位数。counts 可加 → 与 token 如何分片无关，得到的是**全局 batch** 的分位数；通信量仅每层每步 `n×B` 个整数（B≈1000，误差 ~几×10⁻³）。可再对分位数做 EMA 降噪。

### 2.4 Native Vision（原生多模态）[略讲]
- 文本/图像/视频**同一 backbone、同一 context、同一 next-token 目标**，无事后模态对齐阶段。这是"vision-in-the-loop"（写代码→看截图→改）的架构基础。
- **MoonViT-V2（401M, 27 层 ViT）从零训练**（next-token prediction），**不再用 SigLIP 对比预训练初始化**。核心理由是**稳定性**：SigLIP 初始化的 MoonViT-3D 联合训练时梯度范数持续偏高且频繁尖峰；from-scratch 的 V2 全程稳定。且性能与 SigLIP-init 打平 → 说明大规模多模态 LLM **不必**依赖对比预训练做初始化。
- 细节：RMSNorm、去掉所有 bias；图/视频参数全共享（帧内空间 + 帧间时间分解注意力 + 时间池化）；投影前 2×2 pixel-shuffle 下采样把 visual token 减 4×。

### 2.5 Per-Head Muon [PPT 重点（优化器）]
- K3 沿用 **Muon**（对矩阵参数做动量 + Newton–Schulz 正交化的优化器）。
- **改动**：对注意力投影（Q/K/V），不再对整个投影矩阵做正交化，而是**沿 head 维切块、逐 head 分别正交化**。
- **why**：整矩阵正交化把所有 head 当成耦合块——梯度/动量尺度大的 head 会主导共享更新方向，小尺度 head 更新归一化不足。逐 head 正交化**均衡各 head 的更新尺度** → 各 head 学习动态更均衡、大规模训练更稳；顺带因为对高瘦 per-head 块做 Newton–Schulz 更便宜，**优化器开销略降**。

### 2.6 与 DeepSeek-V3 / Qwen3 的对比与优劣 [PPT 重点]

真实公开基线规格：
- **DeepSeek-V3**：671B/37B，61 层，全 **MLA**（128 头，KV 压缩维 512），DeepSeekMoE（**256 routed + 1 shared，选 8**，FFN 2048），**aux-loss-free 符号偏置**均衡，**MTP=1**，RoPE，FP8 训练 + DualPipe。
- **Qwen3-MoE（235B-A22B）**：**GQA + QK-Norm**，**128 experts 选 8、无 shared**，**global-batch aux-loss** 均衡，RoPE + SwiGLU + RMSNorm。

对比表：

| 维度 | DeepSeek-V3 | Qwen3-MoE | **Kimi K3** | K3 的取舍 / 优劣 |
|---|---|---|---|---|
| 注意力 | 全 MLA | GQA + QK-Norm | **3×KDA + 1×MLA 混合** | ✅ 1M 上下文 KV/算力显著更省；⚠️ 线性注意力长程精确召回靠 1/4 MLA 兜底、kernel/系统复杂度高 |
| 位置编码 | RoPE | RoPE | **NoPE**（KDA 隐式） | ✅ 扩长上下文零改动、无需 YaRN；⚠️ 强依赖混合结构中 KDA 提供位置 |
| 深度维聚合 | 标准残差 | 标准残差 | **Attention Residuals** | ✅ 缓解 PreNorm dilution、选择性跨层检索；⚠️ 新增 block 状态与 PP 通信/缓存工程 |
| MoE 专家 | 256 选 8 (+1) | 128 选 8 (无 shared) | **896 选 16 (+2)** | ✅ 稀疏度~56、专家特化空间大、scaling 效率高；⚠️ 均衡/激活爆炸/通信极端 → 需 Stable LatentMoE + MoonEP |
| 专家宽度 | 全宽 | 全宽 | **Latent 压缩 (0.5×)** | ✅ 让"896 选 16"通信/显存可负担；⚠️ 4 连乘病态 → 需 RMSNorm |
| 激活函数 | SwiGLU | SwiGLU | **SiTU-GLU（有界）** | ✅ 低精度更稳、利于 MXFP4/8 QAT；⚠️ 引入 β1,β2 超参 |
| 负载均衡 | aux-loss-free 符号步长 | global-batch aux-loss | **Quantile Balancing** | ✅ 无 lr 超参、几步收敛、不干扰 router 梯度；对 ~10³ 专家更稳 |
| 优化器 | AdamW | AdamW | **Per-Head Muon** | ✅ head 间更均衡、稳；⚠️ 需分布式正交化（P2P gather）工程 |
| 多 token 预测 | MTP=1 | – | **MTP=1 → EAGLE-3 draft** | 复用 MTP 层做投机解码 draft |
| 量化 | FP8 训练 | – | **MXFP4 权重 / MXFP8 激活 QAT** | ✅ 部署显存/成本；训练即量化、消除 train-infer 失配 |

**一句话总结优劣**：K3 是**"把 DeepSeek 系（MLA+shared/routed MoE+aux-loss-free）作为地基，叠加线性注意力（KDA）、深度维注意力（AttnRes）、latent 稀疏（LatentMoE）三层新机制，把规模与上下文同时推到 3T/1M"**——代价是**结构与系统复杂度显著上升**，因此才必须配套一整套 infra（FlashKDA / KCP / MoonEP / 显存管理）。Qwen3 走的是更"保守稳健"的稠密-注意力 + 标准 MoE 路线，工程简单但 scaling/长上下文上限更受限。

---

## 3. 预训练

### 3.1 数据 [略讲]
- 四大文本域：Web / Code / Math / Knowledge + 大规模视觉语料（caption、图文交织、OCR、感知、视频、visual coding）。
- 规则启发式 + 分类器质量打分 + 去重；小模型 ablation 定采样率。
- **Rephrasing**（沿用 K2）：对 knowledge/math 用风格与视角多样化 prompt 改写、chunk-wise 自回归生成、并对源文档做**保真校验**。
- **视觉**：坐标监督同时给绝对与归一化 [0,1] 两种格式（精确 + 分辨率鲁棒）；大规模 **programmatic 多模态数据**——代码片段配其渲染结果（SVG / 3D / 网页 / 游戏 / CAD）。

### 3.2 Scaling Law [PPT 重点]
- 结构/数据/训练一起改后，最优训练 regime 也变了 → 专门重调 **batch size、lr、TPP（tokens-per-parameter）、模型 shape**。
- 结论：整体 **scaling 效率提升约 2.5×**（held-out OOD 验证）。
- **调度选择**：一致偏好 **cosine decay 优于 WSD**。关键方法论：两种调度的**最优超参差别很大**，用同一套超参比较不公平 → **各自独立做 scaling-law 搜索**，在各自最优下 cosine 的最终 loss 更低。（对同事的启示：比较训练技巧时必须"各自调到最优再比"。）

### 3.3 训练配方 [PPT 重点]
- 原生多模态：语言与视觉**从训练一开始就联合优化**，视觉/文本 token 交织在单一 next-token 目标里。
- 优化器：**Per-Head Muon** + K2 的 **weight-clipping**；MoE 用 **QB** 均衡；**cosine lr + 1% 线性 warmup**；**weight decay=0.1** 全程。
- 上下文课程起步：预训练从 **8K → 64K**。

### 3.4 长上下文扩展（到 1M）[PPT 重点]
- **位置**：NoPE → **直接外推 1M，无需任何位置编码改动**（不 rescale RoPE、不 YaRN）。
- **数据**：长文/长视频含大量低质（近重复、二进制块、截断、无效日志）→ 专门清洗（精确+模糊去重、视频帧感知哈希、结构校验）。天然长文稀缺 → **上采样**；**长度本身不等于长程能力** → 合成"必须跨全 1M 上下文才能解"的任务（排列拼接多模态文档/子任务），把注意力"训练在目标尺度"，防退化成局部模式。
- **渐进式四阶段课程**：预训练期 **8K→64K**，cooldown 期 **256K→1M**。把昂贵的长序列计算**集中在训练预算的一小部分**，经济且渐进适应。

### 3.5 部署感知量化（QAT）[PPT 重点]
- MoE 专家权重（占参数显存大头）→ **MXFP4**，激活 **MXFP8**；非专家部分（注意力投影、latent MoE 投影、shared 专家、router）保持高精度。
- **从 SFT 阶段起全程 QAT**（覆盖 SFT + RL）→ 模型适应量化精度损失；**RL 时 rollout 与训练用同一量化方案 → 消除 train-infer 失配**。

---

## 4. 预训练 Infrastructure [PPT 最重点]

> K3 罕见地在**同一个模型**里同时叠了三种系统级挑战：**混合 KDA 注意力**、**3T 级稀疏多模态训练/推理**、**百万 token agentic 负载**。Infra 是全生命周期协同设计。

### 4.1 KDA 的算法-系统协同设计
KDA 用固定大小递归状态 `S` 换掉增长的 KV cache：**好处**是状态小、易传输/复用；**坏处**是串行更新与 GPU 偏好的宽并行相冲突，且在不同执行阶段是不同瓶颈 → 每个阶段一个专用 kernel。

- **FlashKDA（训练 & prefill）**：chunkwise 形式"块内并行、块间串行"，朴素实现会在串行状态传播时让 SM 空转。FlashKDA（CUTLASS）**把块内计算与跨块状态传播 overlap**，拆成 token-parallel 阶段 + head-parallel 递归，各自独立调度调优；显著超过 Triton 参考实现，并作为 flash-linear-attention 的自动派发后端。（配合 §2.1.1 的"下界化衰减"让对角块也能上 Tensor Core。）
- **intra-device Context Parallelism（长上下文 prefill）**：纯 TP 只切 head 不缩短递归——超长序列 prefill 时每 rank 只有几个 head，SM 大量空转。洞见：**每段的状态转移可独立于入态先算、之后精确合成**。SM 级 CP planner 把序列切到单 rank 的各 SM 上并行算段转移再合并，**完全 intra-device、无跨设备通信**。
- **KDA Context Parallelism（KCP，跨设备）[PPT 重点]**：
  - 线性注意力跨设备 CP 只需传**固定大小递归状态**（softmax 要传随长度增长的 KV 块）。但 vanilla 线性注意力的"各 rank 从 S=0 算本地态再求和"对 KDA **不成立**——因为 KDA 是 `S_t = M_t S_{t-1} + β_t k_t v_t^T`，**token 相关的转移矩阵 `M_t` 要作用在入态上**，段的效果依赖入态。
  - KCP 把每段效果分解成**两个本地可算量**：① 累计转移 `M^{T←1}`（作用于入态的部分）+ ② 从 0 生成的本地态 `S̃`。这两个量都能在拿到前序 rank 状态之前用本地 token 算完。
  - 这些 rank 级更新**满足结合律** → 用**前缀扫描（prefix scan）**恢复各 rank 入态；每 rank 只需一次**固定大小 all-gather** 同步递归状态，**线性算力扩展**。（基于 DeltaNet CP，KDA 实现见 FLA PR #691。）

### 4.2 MoonEP：完美负载均衡的专家并行 [PPT 最重点]

**问题**：常规 EP（如 DeepEP）token 负载在 rank 间不均 → ① 计算不均拖慢吞吐；② routed 激活的**动态 shape** 造成显存碎片、且每层要 host-device 同步取真实 shape → 流水停顿。

**MoonEP 的四个支柱**：
1. **完美均衡 + 有界冗余专家**：要求每 rank 恰好收 `S×K` token（S=每 rank token 数、K=top-k）。核心问题是"要多少冗余专家才能保证总能均衡"。**证明（附录 E）：每 rank 至多 `E/R` 个冗余专家就一定存在均衡方案，且此界基本紧**（构造性证明：反复"填满一个欠载 rank"，每次填满一个且不再变，≤R−1 次填满，每 rank 至多被一个 rank 填 → 远程 token 来自单一 rank 的 ≤E/R 个本地专家）。→ 每 rank 预留 `E/R` 冗余槽即保证**规划总有可行解、训练永不中断**。对比 ECHO/UltraEP 预设冗余数或 per-rank cap，无可行解时被迫停训、还要手调 cap 且残留不均。
2. **在线规划**：每步精确 ILP 太贵 → 离线 ILP 求代表性最优做参照，设计**GPU 规划 kernel**（近最优、开销可忽略、始终满足 E/R 上界），从当前 micro-batch/层的 router 输出规划冗余专家并**在专家计算前预取**；反向把其梯度先 stage 到本地 reduce buffer，算完再 reduce 回 home rank。
3. **Zero-copy 通信**：融合 permute/unpermute——规划 kernel 预算每个 token 的目的地，token **直接发到远程 rank 的专家分组位置**，通信 buffer 的**视图直接交给计算**，消除中间拷贝。完美均衡下 buffer 只需固定 `S×K`（DeepEP 最坏要 `S×K×R`）。
4. **静态 shape + sync-free + Expert-GEMM 调度**：完美均衡 → 每层 MoE 计算 shape **静态已知** → **消除每层 MoE 的 host 同步**、降 kernel 启动开销。即使聚合均衡，rank 内**每专家 token 数仍偏斜** → 用 **workload-aware 调度器**（分析型代价模型 + 离线 autotune 标定系数）在发射前按当前分布调参并固定；shared 专家 GEMM 派到单独 stream 与其他 kernel overlap。
- 效果（README）：通信时间随不均**几乎不变**（DeepEP 随最热 rank 恶化）；端到端迭代时间在各不均水平**持平且不 OOM**（DeepEP 高不均时 OOM）。

### 4.3 显存高效训练 [PPT 重点]
- **统一激活管理器**：每个为反向保存的 tensor 绑定一个**可插拔存储后端**；重计算 / 量化 / offload / remote-offload 都只是"存储策略"，可在 tensor 粒度自由组合、用轻量注解声明、与模型代码解耦。重计算按函数粒度（支持跨层重计算）。全部 GPU 显存在主计算流上单一内存池分配（避免多流碎片），激活按层粒度预取并与计算 overlap。K3 里多数激活用 **block-wise FP8 量化 + offload/remote-offload**，逐元素算子配重计算。
- **Memory-efficient MoE（借鉴 SonicMoE）**：把 permuted probs 的梯度**数学变换**成只依赖中间激活 `act_output` 与上游梯度 `doutput` 的形式，消除反向对前向 `output` 的依赖（代价是一点逐元素计算）；group GEMM 前向只存 dispatch 的输入，反向**重算 dispatch** 恢复输入，且该重算通信可与 group-GEMM 反向 overlap。
- **Memory-efficient AttnRes**：基于 Block AttnRes——block 表示在边界层生成一次、后续层共享、常驻 GPU；AttnRes 计算全部 checkpointing 包裹 → **每层为反向保存的激活与标准残差架构相同**。PP 用 **cache-based pipeline 通信**：只增量传新生成的 block、micro-batch 结束即释放，达到显存下界。
- **跨 PP rank 均衡激活**：interleaved 1F1B 下激活分布不均（越靠后 rank 常驻激活越少）→ 用 **Mooncake Transfer Engine** 把激活 remote-offload 到其他 PP rank 内存，均衡各 PP rank 显存、防 OOM。
- **Pipeline ZeRO-2 梯度分片 + offload**：梯度按 DP rank 分片，分片梯度存 **CPU 内存**降 GPU 峰值，double grad buffer 留在 GPU；DP reduce 进 double buffer 后累加到 CPU 分片。
- **P2P-based Muon 正交化 [PPT 重点]**：分布式优化器把参数均匀分片到 DP rank，但 Newton–Schulz 需要**完整参数矩阵** → 需先 gather。朴素做法全参 all-gather（显存 + 通信双瓶颈）。K3 改为**每 rank 只 P2P 取回自己负责参数的分片**（向 owner rank 点对点通信），消掉全参 buffer、降显存与通信量；再按 model-chunk buffer 粒度把通信与计算流水，隐藏通信。

### 4.4 多模态编码器优化 [略讲]
- **Dynamic CP**：大图沿 patch 维切到多设备、gather-KV 算注意力；每个 CP 组再分若干 sub-CP 组、把多张大图负载均衡分配 → 通信占比不随规模增长。
- **Bubble filling**：延续 K2.5 的 Decoupled Encoder Process，进一步分解 ViT 计算——首批 micro-batch 的 ViT 前向同步先算，其余塞进**流水气泡**，反向同理 → ViT 计算几乎全被气泡吸收，视觉编码器的有效开销基本消除。

### 4.5 并行组合总览 [PPT 重点]
PP（+虚拟阶段 VP）+ EP + ZeRO-1 DP + Pipeline ZeRO-2 梯度分片 + CP（KCP）。MoE 层 shared 专家跨 EP rank 复制；dispatch/combine 的 all-to-all 与计算 overlap。三个核心难题被逐一解决：token 负载不均→MoonEP；显存超预算→统一激活管理/ZeRO-2/remote-offload；ViT 可变计算暴露在关键路径→bubble filling。

---

## 5. 后训练 / RL（简化）[略讲，PPT 压缩]

### 5.1 三阶段范式
**SFT（冷启动）→ 按域×effort 做 RL 得到 9 个专家 → MOPD 多教师蒸馏合并成单模型。**

- **SFT**：用前代 Kimi 域专家合成轨迹 + 多阶段校验 + 人在环标注；统一用 **XTML chat 模板**（见 §5.5）序列化。从 SFT 起全程 QAT。
- **RL**：不为单任务训专用模型，而是三大域各训一个专家、每个 effort 一版：**(i) general、(ii) general agents、(iii) coding agents**，× {low, high, max} = **9 个专家**。scaling RL FLOPs → tool-call 步数与整体能力一致提升。
  - **算法**：扩展**部分 rollout（partial rollout）**——采样 N prompts × K completions，当 λ 比例轨迹完成就暂停生成、让策略优化先走（避免 straggler 长尾）；暂停的入队、下轮优先恢复（靠 sandbox）。长轨迹跨多迭代 → **数据 staleness**，靠**逐 token 正则**把更新限制在局部邻域，容忍极端 off-policy、稳住训练。
  - **Reasoning-effort RL**：每题给初始 token 预算 `b_0(x)`，超 `τ·b_0` 给 −1 奖励；按 τ 从大到小的 stage-wise 课程得到 max/high/low effort 专家。
  - **Agentic GRM**（不可验证的一般任务）：锦标赛式二元比较；judge 走"读产物→生成 rubric→逐项打分→记 scorepad"强制协议；用 verbosity 预算控制防"越长越好"的 reward hacking。
- **MOPD（多教师 on-policy 蒸馏）**：按域 d 与 effort e 用对应教师，逐 token 奖励 = `clip( sg(log π_teacher/π_student), −Rmax, Rmax )`（稠密信号，天然融入 RL 框架、支持 partial rollout）。

### 5.2 RL 环境与任务合成 [略讲]
- **统一 white-box 环境**：把 agent harness 抽象成可配置可组合模块（工具接口/系统 prompt/上下文管理/skills/memories/subagents），可实例化 Kimi Code / Claude Code / Codex / OpenClaw / Hermes 等 → 防对单一 harness 过拟合、促跨 scaffold 泛化。
- **知识图谱引导任务合成**：agent 递归扩展的自演化 DAG 知识图；采样节点组关键词→检索真实材料→合成各类任务，控制"粒度 + 覆盖"。
- 其他可验证环境：agentic search、专业知识工作（投行/数据/法律）、多步视觉推理（Python sandbox 里 crop/zoom/算）、**kernel 优化任务**（含防 reward-hacking 检测：CUDA graph replay / 输入缓存 / 降精度）、personal assistant（Gmail/Notion/Slack mock，跨多日、上千工具调用）、**Autonomous Execution Tasks (AET)**（只给目标+验证接口、无参考轨迹）、web development。

### 5.3 部署感知后训练
- **Draft model / EAGLE-3**：把预训练的 MTP 层微调成 EAGLE-3 draft（target 冻结、只训 draft 层 + feature-fusion 投影）；输入融合第 1/4/最后 AttnRes block 的低/中/高层特征，投影矩阵初始化为 `[0 0 I]`（初始等价高层特征，再逐渐学入低/中层）。
- **LK loss**：直接优化投机解码的**接受率**（`−log Σ min(p,q)`）而非 KL 代理；draft 训练也在 QAT 配置下（MoE 权重 MXFP4）。

### 5.5 XTML chat 模板 [略讲]
- 目标：可扩展（新能力靠向后兼容的 message 格式而非改模板）、低 alignment tax、解码友好。
- XTML = 类 XML 但用三个保留 special token `[open]/[sep]/[close]` + `[end_of_msg]`，消除元素边界的 tokenization 歧义、利于约束解码。
- channels：`think / response / tools`（灵感自 Harmony）；只保留 thinking 模式；reasoning-effort 作为**自然语言 option message** 注入（解耦接口与语法、低训练成本）。

---

## 6. 推理 & 在线服务 Infra（补充）[略讲]

- **KDA-aware prefix cache**：混合架构里 MLA KV（随长度增长、按 token 分页）与 KDA 递归态（固定大小、每请求一份）生命周期/大小完全不同，但一个缓存前缀**只有两者能在同一边界一起恢复**才可复用。
  - **统一分页布局**：把 KDA 态打进与 MLA KV 同一分页块池、统一页字节大小 → 分配/引用计数/驱逐一套实现；head 连续存放、每 head 自成最小跨节点传输单元；prefill/decode 分离且 TP 度不同的时候在传输路径上重排、GPU 侧零 reshuffle。
  - **细粒度前缀复用**：解耦"哈希粒度"与"物理块粒度"——前缀哈希在细 hash block（如 512 token）上跑，KDA checkpoint 只在 MLA hash 端点的稀疏子集保存（对话轮边界）。可在 512-token 边界命中、深入到大物理块内部（示例：2800 token 命中在 B=2560），从 B 恢复不重算 [0,B)。
  - **并发一致性**：命中块跨所有 cache 组 pin 后再分配；块内拷贝在前向前落地、未落地的块排除匹配；checkpoint 要么在每个 KDA 组都可命中、要么都不可命中（原子失效）。
- **专用 kernel**：① KDA 解码——MTP 投机解码里状态就地更新、拒绝的 draft 无法回滚 → **只缓存投影输入、片上重建被接受 token 的状态**（同期 ReplaySSM 独立提出），一个 fused kernel 覆盖 shortconv/norm/gating/递归/输出 norm；② Block AttnRes——两阶段（inter-block 批量读一次 + intra-block online-softmax 合并）、prefill 用 SP、decoding 用 side stream overlap + 融合进 TP all-reduce；③ Stable LatentMoE——latent 下投影与 router 融合成一个 GEMM、权重分片 + all-gather 融进 epilogue、小 batch 用 token-centric 的 WarpDecode kernel。
- **Fleet 级调度**：① **cache-aware affinity**——按前缀缓存所在集群路由（一致性哈希 pin 主+备集群，故障时备集群重 prefill、re-prefill 分散到多集群）；② **budget-based admission**——按请求类分资源预算，防长上下文突发拖垮短请求的 TTFT。

---

## 7. 评测与结论 [略讲]

- 整体：紧随最强闭源（Claude Fable 5 / GPT-5.6 Sol），稳定领先其他开源/部分闭源。SOTA/领先项：ProgramBench、SWE-Marathon、BrowseComp、DeepSearchQA、MCPMark、OmniDocBench、WebDev Arena（**首个登顶该榜的开源模型**）等。
- 短板：research 级推理（HLE-Full、CritPt）、部分知识工作 Elo、hardened target 的端到端 exploit。
- **成本效率**：多个 coding/agentic 榜上处于或接近成本-效率前沿（如 BrowseComp 最佳分且约为 GPT-5.6 Sol 一半成本）。
- Case study：GPU kernel 优化、从零写 Triton 类编译器 MiniTriton、48 小时自主设计推理芯片原型 nano-kpu、复现天体物理 I–Love–Q 关系、视频剪辑等。

---

## 8. 演进脉络与对我们的启示 [PPT 收尾]

**脉络**：Kimi K2（全 MLA、384 选 8、Muon、SwiGLU、128K）→ Kimi K2.5（视觉 agentic、DEP 编码器）→ **Kimi Linear**（提出 KDA、3:1 KDA:MLA、验证线性注意力可超全注意力）→ **Attention Residuals**（深度维注意力，48B 验证）→ **Kimi K3**（把 KDA + AttnRes + LatentMoE + Per-Head Muon + QAT + 全套 infra 集成到 2.8T/1M）。

**方法论启示（给同事）**：
1. **算法-系统协同设计**是主线：给 KDA 门"加下界"这种纯数学小改，直接换来"对角块能上 Tensor Core"的系统收益；MoonEP 的"完美均衡"直接换来"静态 shape + 零拷贝 + 免 host 同步"。**改算法时先想它对 kernel/通信/显存意味着什么。**
2. **把成熟维度的思想迁移到新维度**：attention 从序列维迁到深度维（AttnRes）；CP 从 softmax 迁到线性注意力（KCP，用前缀扫描 + 转移矩阵分解）。
3. **有界化 / 归一化是极端稀疏与低精度下的稳定器**：SiTU-GLU 有界、LatentMoE 里加 RMSNorm、KDA 门下界、per-head 正交化——都是为了在 2.8T/896 专家/MXFP4 下稳住优化。
4. **比较训练技巧要"各自调到最优再比"**（cosine vs WSD 的教训）。
5. **负载均衡从"启发式步长"升级为"有理论最优解的分位数"**（QB），并用直方图做全局可扩展估计。

---

### 附：本次分享可回答的高频问题（备问）
- *为什么 3:1 而不是全 KDA？* 线性注意力长程精确召回弱，1/4 全局 MLA 兜底；末层强制全局。
- *NoPE 会不会外推差？* 位置由 KDA 递归门控隐式提供，报告显示可直接外推 1M；MLA 层本就负责无约束全局内容交互。
- *896 选 16 会不会训不动/不均衡？* Stable LatentMoE（Latent 压缩 + RMSNorm + SiTU-GLU）治激活爆炸，QB + 直方图治均衡，MoonEP 治系统级不均。
- *AttnRes 推理会不会很贵？* Block(N=8) + online softmax + 专用 kernel，论文实测推理延迟开销 <2%。
- *和 DeepSeek 到底差在哪？* 见 §2.6：K3 = DeepSeek 地基 + KDA + AttnRes + LatentMoE + Per-Head Muon + QAT，规模与上下文都更激进，系统复杂度更高。
