#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate schematic diagrams + charts for the Kimi K3 tech-share deck.
All figures are saved to ./img/ as PNG. English/technical labels are used in
figures (rendered crisply); Chinese narrative lives in the slide text + notes.
WenQuanYi Micro Hei is registered so occasional Chinese labels also render.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

# ---- fonts (CJK) ----
for p in ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ---- palette ----
NAVY = "#1e3a5f"; NAVY_D = "#0f2440"; TEAL = "#0d9488"; TEAL_L = "#5eead4"
AMBER = "#f59e0b"; AMBER_D = "#b45309"; SLATE = "#64748b"; SLATE_L = "#cbd5e1"
BG = "#ffffff"; INK = "#0f172a"; RED = "#dc2626"; GREEN = "#16a34a"
BLUE = "#2563eb"; VIOLET = "#7c3aed"; LGREY = "#e2e8f0"; PANEL = "#f1f5f9"


def new_ax(w=10, h=5.6, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, text="", fc=NAVY, ec="none", tc="white", fs=13,
        weight="bold", rounding=0.02, lw=0, ha="center", va="center", alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding*100}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2 if ha == "center" else x + 1.5, y + h / 2, text,
                ha=ha, va=va, color=tc, fontsize=fs, weight=weight, zorder=5,
                linespacing=1.25)
    return p


def arrow(ax, x1, y1, x2, y2, color=SLATE, lw=2.2, style="-|>", ms=16, ls="-", alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, alpha=alpha,
                        shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(a)


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight", pad_inches=0.12,
                facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# =========================================================================
# 1. Three-axis information flow
# =========================================================================
def f_threeaxis():
    fig, ax = new_ax(10, 5.6, xlim=(-16, 104), ylim=(-6, 100))
    cx, cy = 22, 30
    # three axes
    arrow(ax, cx, cy, cx + 40, cy, TEAL, 3, ms=22)            # sequence (right)
    arrow(ax, cx, cy, cx, cy + 48, AMBER, 3, ms=22)           # depth (up)
    arrow(ax, cx, cy, cx - 15, cy - 15, VIOLET, 3, ms=22)     # width (diag)
    ax.text(cx + 20, cy + 4, "序列维 · token", color=TEAL, fontsize=12.5, weight="bold", va="bottom", ha="center")
    ax.text(cx - 6, cy + 52, "深度维 · layer", color=AMBER_D, fontsize=12.5, weight="bold")
    ax.text(cx - 17, cy - 8, "宽度维\nchannel", color=VIOLET, fontsize=12, weight="bold", ha="center", va="center")
    box(ax, cx - 6, cy - 5, 12, 10, "信息流", fc=NAVY, fs=12.5, rounding=0.04)
    # mechanism callouts
    box(ax, 66, 20, 34, 12, "Hybrid Attention\n3×KDA + 1×MLA", fc=TEAL, fs=12, rounding=0.03)
    box(ax, 40, 82, 34, 12, "Attention Residuals\n每层选择性检索前层", fc=AMBER, fs=12, rounding=0.03, tc=INK)
    box(ax, 2, 6, 34, 12, "Stable LatentMoE\n896 选 16", fc=VIOLET, fs=12, rounding=0.03)
    save(fig, "f_threeaxis.png")


# =========================================================================
# 2. Hybrid attention 3:1 stack
# =========================================================================
def f_hybrid():
    fig, ax = new_ax(10, 5.6)
    ax.text(50, 95, "每个 Block = 3×KDA + 1×Gated MLA（重复 · 末尾再补 1 层 MLA）",
            ha="center", fontsize=13, weight="bold", color=INK)
    y = 8
    order = [("KDA", TEAL), ("KDA", TEAL), ("KDA", TEAL), ("Gated MLA", NAVY)] * 2
    labels = ["KDA (线性注意力)", "KDA", "KDA", "Gated MLA (全局)"]
    xs = 12
    for bi in range(2):
        for i, (name, c) in enumerate([("KDA", TEAL), ("KDA", TEAL), ("KDA", TEAL), ("MLA", NAVY)]):
            txt = {"KDA": "KDA", "MLA": "Gated MLA"}[name]
            box(ax, xs, y, 34, 8, f"{txt}  +  Stable LatentMoE", fc=c, fs=11.5, rounding=0.06)
            y += 9.5
        # block bracket
        ax.annotate("", xy=(xs - 3, y - 1), xytext=(xs - 3, y - 38),
                    arrowprops=dict(arrowstyle="-", color=SLATE, lw=1.5))
        ax.text(xs - 5, y - 19, f"Block {bi+1}", rotation=90, va="center",
                ha="center", color=SLATE, fontsize=10, weight="bold")
    box(ax, xs, y, 34, 8, "Gated MLA (末层强制全局)", fc=AMBER, fs=11.5, rounding=0.06, tc=INK)
    # right why panel
    box(ax, 60, 40, 36, 40, "", fc=PANEL, rounding=0.03)
    ax.text(78, 74, "为什么 3:1 ?", ha="center", fontsize=13, weight="bold", color=NAVY)
    whys = ["KDA 承担绝大多数 token mixing", "递归状态固定大小 → 长序列便宜",
            "线性注意力弱在长程精确召回", "→ 用 1/4 全局 MLA 兜底",
            "1M 上下文 KV / 算力显著下降"]
    for i, t in enumerate(whys):
        ax.text(62, 68 - i * 5.5, "•  " + t, fontsize=11, color=INK)
    save(fig, "f_hybrid.png")


# =========================================================================
# 3. KDA state vs KV cache
# =========================================================================
def f_kda_state():
    fig, ax = new_ax(10, 5.0)
    # left: softmax KV cache growing
    ax.text(25, 92, "Softmax 注意力：KV cache 随长度增长", ha="center", fontsize=12.5,
            weight="bold", color=NAVY)
    for i in range(7):
        box(ax, 6 + i * 5.6, 55, 5, 12 + i * 3, "", fc=SLATE_L, ec=SLATE, lw=1, rounding=0.04)
    ax.text(25, 46, "内存 O(T) ↑↑", ha="center", color=RED, fontsize=12, weight="bold")
    # right: KDA fixed state
    ax.text(75, 92, "KDA：固定大小递归状态 S", ha="center", fontsize=12.5,
            weight="bold", color=TEAL)
    box(ax, 66, 55, 20, 24, r"$S \in \mathbb{R}^{d_k \times d_v}$" + "\n(固定)", fc=TEAL, fs=12, rounding=0.05)
    for i in range(4):
        arrow(ax, 60, 67, 66, 67, TEAL, 2)
    ax.text(75, 46, "内存 O(1) · 易传输/复用", ha="center", color=GREEN, fontsize=12, weight="bold")
    # bottom: update rule words (no heavy formula)
    box(ax, 12, 12, 76, 22, "", fc=PANEL, rounding=0.02)
    ax.text(50, 29, r"delta-rule 更新：先「按 key 方向擦除旧关联」，再「写入新 $k v^{\top}$」",
            ha="center", fontsize=12, color=INK, weight="bold")
    ax.text(50, 22, "通道级遗忘门 Diag(α)：每个特征维独立控制记忆衰减",
            ha="center", fontsize=11.5, color=VIOLET)
    ax.text(50, 15.5, "（比 Gated DeltaNet / Mamba-2 的「头级标量门」更细粒度）",
            ha="center", fontsize=10.5, color=SLATE)
    save(fig, "f_kda_state.png")


# =========================================================================
# 4. KDA lower-bounded decay
# =========================================================================
def f_kda_decay():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(BG)
    z = np.linspace(-6, 6, 400)
    softplus = -np.log1p(np.exp(z))          # Kimi Linear: -softplus (A=0), unbounded below
    gmin = -5
    sig = gmin * (1 / (1 + np.exp(-z)))       # K3: gmin*sigmoid, bounded
    a1.plot(z, softplus, color=SLATE, lw=2.6, label="Kimi Linear: −Softplus (无界)")
    a1.plot(z, sig, color=TEAL, lw=3, label="Kimi K3: g_min·Sigmoid (有下界)")
    a1.axhline(gmin, color=AMBER_D, ls="--", lw=1.6)
    a1.text(-5.6, gmin + 0.25, "g_min = −5", color=AMBER_D, fontsize=11, weight="bold")
    a1.axhspan(gmin, 0, color=TEAL, alpha=0.06)
    a1.set_title("log-decay 参数化 g(z)", fontsize=12.5, weight="bold", color=INK)
    a1.set_xlabel("decay logit  z"); a1.set_ylabel("log-decay  g")
    a1.set_ylim(-6.5, 0.4); a1.legend(fontsize=9.5, loc="lower right")
    a1.grid(alpha=0.25)
    # right: consequence
    a2.axis("off"); a2.set_xlim(0, 10); a2.set_ylim(0, 10)
    a2.text(5, 9.3, "系统后果", ha="center", fontsize=12.5, weight="bold", color=NAVY)
    # tile grid
    for i in range(4):
        for j in range(4):
            c = AMBER if i == j else TEAL_L
            a2.add_patch(Rectangle((1.3 + j * 1.2, 6.4 - i * 1.2), 1.05, 1.05,
                                   fc=c, ec="white", lw=1))
    a2.text(6.6, 6.3, "对角块", color=AMBER_D, fontsize=10.5, weight="bold")
    a2.annotate("", xy=(6.4, 5.2), xytext=(6.4, 6.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    a2.text(5, 4.2, r"$\alpha > e^{-5}$ → 16-tile 累计 log-decay ∈ (−80,0)",
            ha="center", fontsize=10, color=INK)
    a2.text(5, 3.2, r"重标定因子 $< e^{80}$  →  落在 BF16 范围", ha="center",
            fontsize=10, color=INK)
    a2.text(5, 1.9, "对角块也能上 Tensor Core GEMM", ha="center",
            fontsize=11.5, color=GREEN, weight="bold")
    a2.text(5, 0.9, "（删掉 position-pair 标量路径）", ha="center",
            fontsize=9.5, color=SLATE)
    fig.tight_layout()
    save(fig, "f_kda_decay.png")


# =========================================================================
# 5. Attention Residuals: standard vs attn
# =========================================================================
def f_attnres():
    fig, ax = new_ax(10, 5.4)
    ax.text(25, 95, "标准残差 = 深度维 RNN", ha="center", fontsize=13, weight="bold", color=SLATE)
    ax.text(75, 95, "Attention Residuals = 深度维 softmax", ha="center", fontsize=13,
            weight="bold", color=AMBER_D)
    # left: uniform sum
    ys = [12, 30, 48, 66]
    for i, y in enumerate(ys):
        box(ax, 14, y, 20, 9, f"Layer {i+1}", fc=SLATE_L, tc=INK, fs=11, rounding=0.06)
        if i < len(ys) - 1:
            arrow(ax, 24, y + 9, 24, ys[i + 1], SLATE, 3, ms=16)
    ax.text(24, 80, "固定单位权重逐层累加\n早层信息被稀释、幅度 O(L) 增长",
            ha="center", fontsize=10.5, color=RED)
    # right: attention over layers
    for i, y in enumerate(ys):
        box(ax, 66, y, 20, 9, f"Layer {i+1}", fc=AMBER, tc=INK, fs=11, rounding=0.06)
    # attention weighted arrows to top layer
    for i, y in enumerate(ys[:-1]):
        arrow(ax, 76, y + 9, 76, ys[-1], color=TEAL, lw=1 + 2.5 * (i == 1),
              ms=13, ls="-", alpha=0.5 + 0.15 * i)
    ax.text(90, 45, "α", color=TEAL, fontsize=15, weight="bold")
    ax.text(76, 80, "每层用可学 pseudo-query\n对所有前层做 softmax 选择",
            ha="center", fontsize=10.5, color=GREEN)
    save(fig, "f_attnres.png")


# =========================================================================
# 6. Block AttnRes
# =========================================================================
def f_blockattnres():
    fig, ax = new_ax(10, 5.0)
    ax.text(50, 94, "Block AttnRes：L 层 → N 个 block，块间做全注意力（K3: N≈8）",
            ha="center", fontsize=12.5, weight="bold", color=INK)
    # blocks
    bx = [8, 30, 52, 74]
    names = ["Embedding\n" + r"$b_0$", "Block 1\n" + r"$b_1$",
             "Block 2\n" + r"$b_2$", "Block n\n" + r"$b_n$"]
    cols = [SLATE, VIOLET, VIOLET, VIOLET]
    for i, (x, nm, c) in enumerate(zip(bx, names, cols)):
        box(ax, x, 40, 18, 16, nm, fc=c, fs=11, rounding=0.05)
        if i < 3:
            arrow(ax, x + 18, 48, bx[i + 1], 48, SLATE, 2)
    # inter-block attention curve to output
    box(ax, 40, 74, 20, 12, r"当前层输入 $h_l$", fc=AMBER, tc=INK, fs=11.5, rounding=0.05)
    for x in bx:
        arrow(ax, x + 9, 56, 50, 74, TEAL, 1.6, ms=11, alpha=0.55)
    # memory note
    box(ax, 12, 8, 76, 20, "", fc=PANEL, rounding=0.02)
    ax.text(50, 22, "块内标准残差求和 → 块间仅对 N 个块表示做注意力", ha="center",
            fontsize=11.5, color=INK, weight="bold")
    ax.text(31, 14, "显存/通信 O(Ld) → O(Nd)", ha="center", color=GREEN, fontsize=11, weight="bold")
    ax.text(70, 14, "同算力 ≈ baseline 的 1.25× · 推理开销 <2%", ha="center",
            color=NAVY, fontsize=11, weight="bold")
    save(fig, "f_blockattnres.png")


# =========================================================================
# 7. LatentMoE
# =========================================================================
def f_latentmoe():
    fig, ax = new_ax(10, 5.2)
    ax.text(50, 95, "Stable LatentMoE：分离「模型宽度」与「路由专家宽度」",
            ha="center", fontsize=12.5, weight="bold", color=INK)
    box(ax, 6, 45, 12, 12, "token x\n(d=7168)", fc=NAVY, fs=10.5, rounding=0.06)
    # shared path (full width)
    box(ax, 30, 70, 26, 12, "Shared 专家 ×2\n(全宽 d)", fc=TEAL, fs=11, rounding=0.05)
    arrow(ax, 18, 53, 30, 76, TEAL, 2)
    # routed path (latent)
    box(ax, 26, 40, 14, 10, "W↓  下投影\n→ latent ℓ=3584", fc=AMBER, tc=INK, fs=9.5, rounding=0.06)
    arrow(ax, 18, 50, 26, 46, AMBER_D, 2)
    box(ax, 46, 34, 20, 22, "Router: 896 选 16\n(latent 空间)\n" + r"$E_i$ routed", fc=VIOLET, fs=10.5, rounding=0.04)
    arrow(ax, 40, 45, 46, 45, SLATE, 2)
    box(ax, 72, 40, 12, 10, "RMSNorm\n+ W↑", fc=AMBER, tc=INK, fs=9.5, rounding=0.06)
    arrow(ax, 66, 45, 72, 45, SLATE, 2)
    box(ax, 88, 52, 9, 12, "y", fc=NAVY, fs=11, rounding=0.06)
    arrow(ax, 56, 76, 92, 64, TEAL, 2)      # shared to sum
    arrow(ax, 84, 45, 92, 56, AMBER_D, 2)   # routed to sum
    ax.text(50, 18, "只在压缩 latent 空间里路由 → 896 选 16 的通信/显存才付得起（稀疏度 ≈ 56）",
            ha="center", fontsize=11, color=INK)
    ax.text(50, 10, "RMSNorm 治「4 连乘病态」的激活爆炸 · SiTU-GLU 有界 · QB 均衡",
            ha="center", fontsize=10.5, color=SLATE)
    save(fig, "f_latentmoe.png")


# =========================================================================
# 8. SiTU-GLU curves
# =========================================================================
def f_situ():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    fig.patch.set_facecolor(BG)
    x = np.linspace(-10, 100, 800)
    sig = 1 / (1 + np.exp(-x))
    glu = sig * x                 # GLU gate*x  (approx, gate=sigmoid, up=x)
    swiglu = (x * sig) * x        # SwiGLU: (x sigmoid(x)) * x
    b1, b2 = 4.0, 25.0
    situ = (b1 * np.tanh(x / b1)) * sig * (b2 * np.tanh(x / b2))
    ax.plot(x, swiglu, color=SLATE, lw=2.4, label="SwiGLU（无界 ↑↑）")
    ax.plot(x, glu, color=BLUE, lw=2.0, label="GLU")
    ax.plot(x, situ, color=TEAL, lw=3.2, label="SiTU-GLU（有界 ≤ β₁β₂=100）")
    ax.axhline(100, color=AMBER_D, ls="--", lw=1.6)
    ax.text(2, 108, "上界 β₁β₂ = 100", color=AMBER_D, fontsize=11, weight="bold")
    ax.set_ylim(-40, 320); ax.set_xlim(-10, 100)
    ax.set_title("SiTU-GLU：有界，又保留 SwiGLU 的形状", fontsize=13, weight="bold", color=INK)
    ax.set_xlabel("input x"); ax.set_ylabel("f(x)")
    ax.legend(fontsize=10.5, loc="upper left"); ax.grid(alpha=0.25)
    # inset near origin
    axins = ax.inset_axes([0.60, 0.12, 0.36, 0.42])
    xi = np.linspace(-6, 6, 300)
    sgi = 1 / (1 + np.exp(-xi))
    axins.plot(xi, (xi * sgi) * xi, color=SLATE, lw=2)
    axins.plot(xi, (b1 * np.tanh(xi / b1)) * sgi * (b2 * np.tanh(xi / b2)), color=TEAL, lw=2.4)
    axins.set_title("近原点：一阶匹配", fontsize=9)
    axins.grid(alpha=0.3); axins.tick_params(labelsize=7)
    save(fig, "f_situ.png")


# =========================================================================
# 9. Quantile Balancing
# =========================================================================
def f_qb():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(BG)
    rng = np.random.default_rng(3)
    margins = rng.normal(0.1, 0.25, 4000)
    a1.hist(margins, bins=60, color=TEAL_L, edgecolor=TEAL, alpha=0.9)
    q = np.quantile(margins, 0.75)
    a1.axvline(q, color=RED, lw=2.6)
    a1.text(q + 0.02, a1.get_ylim()[1] * 0.86, "(1−k/n) 分位数\n= 专家偏置 " + r"$b_j$", color=RED,
            fontsize=10.5, weight="bold")
    a1.set_title("QB：从 margin 分布直接读出偏置（直方图全局估计）",
                 fontsize=11.5, weight="bold", color=INK)
    a1.set_xlabel(r"margin  $s_{i,j} - \alpha_i$"); a1.set_ylabel("token 计数")
    a1.grid(alpha=0.2)
    # right: imbalanced vs balanced loads
    a2.set_title("负载：不均 → 均衡", fontsize=11.5, weight="bold", color=INK)
    experts = ["E1", "E2", "E3", "E4"]
    before = [4, 3, 1, 0]; after = [2, 2, 2, 2]
    xpos = np.arange(4)
    a2.bar(xpos - 0.2, before, 0.38, color=RED, alpha=0.75, label="Top-k (不均)")
    a2.bar(xpos + 0.2, after, 0.38, color=GREEN, alpha=0.85, label="QB (均衡)")
    a2.axhline(2, color=SLATE, ls="--", lw=1.2)
    a2.set_xticks(xpos); a2.set_xticklabels(experts)
    a2.set_ylabel("token / expert"); a2.legend(fontsize=9.5); a2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    save(fig, "f_qb.png")


# =========================================================================
# 10. K3 = DeepSeek base + 3 layers
# =========================================================================
def f_compare():
    fig, ax = new_ax(10, 5.2)
    ax.text(50, 95, "K3 = DeepSeek 系地基 + 三层新机制 + 稳定化/优化器/量化",
            ha="center", fontsize=12.5, weight="bold", color=INK)
    box(ax, 20, 12, 60, 12, "DeepSeek 系地基：MLA + shared/routed MoE + aux-loss-free",
        fc=NAVY, fs=11.5, rounding=0.03)
    box(ax, 20, 27, 60, 10, "① KDA 线性注意力（序列维）", fc=TEAL, fs=11.5, rounding=0.04)
    box(ax, 20, 40, 60, 10, "② Attention Residuals（深度维）", fc=AMBER, tc=INK, fs=11.5, rounding=0.04)
    box(ax, 20, 53, 60, 10, "③ Stable LatentMoE 896 选 16（宽度维）", fc=VIOLET, fs=11.5, rounding=0.04)
    box(ax, 20, 66, 60, 10, "SiTU-GLU · QB · Per-Head Muon · NoPE · MXFP4/8 QAT",
        fc=SLATE, fs=11, rounding=0.04)
    ax.annotate("", xy=(50, 66), xytext=(50, 24),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    ax.text(85, 45, "规模与\n上下文\n同时推到\n3T / 1M", ha="center", color=RED,
            fontsize=12, weight="bold")
    save(fig, "f_compare.png")


# =========================================================================
# 11. Scaling law 2.5x
# =========================================================================
def f_scaling():
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
    fig.patch.set_facecolor(BG)
    flops = np.logspace(20, 21.3, 100)
    k2 = 2.0 + 0.55 * (np.log10(flops[-1]) - np.log10(flops))
    k3 = 1.86 + 0.55 * (np.log10(flops[-1]) - np.log10(flops))
    ax.plot(flops, k2, color=SLATE, lw=3, label="Kimi K2")
    ax.plot(flops, k3, color=TEAL, lw=3, label="Kimi K3")
    ax.set_xscale("log")
    ax.annotate("", xy=(3e20, 2.28), xytext=(1.2e20, 2.28),
                arrowprops=dict(arrowstyle="<->", color=AMBER_D, lw=2))
    ax.text(1.9e20, 2.33, "≈ 2.5× 有效算力", color=AMBER_D, fontsize=12, weight="bold", ha="center")
    ax.set_xlabel("训练 FLOPs (log)"); ax.set_ylabel("验证 Loss (OOD)")
    ax.set_title("Scaling Law：整体 scaling 效率 ≈ 2.5× 提升", fontsize=13, weight="bold", color=INK)
    ax.legend(fontsize=11); ax.grid(alpha=0.25, which="both")
    save(fig, "f_scaling.png")


# =========================================================================
# 12. Long-context curriculum staircase
# =========================================================================
def f_longctx():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(BG)
    stages = ["8K", "64K", "256K", "1M"]
    vals = [8192, 65536, 262144, 1048576]
    xs = [0, 1, 2, 3]
    ax.step(range(5), [8192] + vals, where="post", color=TEAL, lw=3)
    for i, (s, v) in enumerate(zip(stages, vals)):
        ax.scatter(i + 1, v, color=AMBER, zorder=5, s=60)
        ax.text(i + 1, v * 1.4, s, ha="center", fontsize=12, weight="bold", color=NAVY)
    ax.set_yscale("log"); ax.set_ylim(5e3, 3e6)
    ax.axvspan(0, 2, color=TEAL, alpha=0.06); ax.axvspan(2, 4.2, color=AMBER, alpha=0.08)
    ax.text(1, 2e6, "预训练期", ha="center", color=TEAL, fontsize=12, weight="bold")
    ax.text(3.1, 2e6, "cooldown 期", ha="center", color=AMBER_D, fontsize=12, weight="bold")
    ax.set_xlim(0.4, 4.2); ax.set_xticks([]); ax.set_ylabel("上下文长度 (log)")
    ax.set_title("渐进式四阶段上下文课程（NoPE → 直接外推，无需改位置编码）",
                 fontsize=12.5, weight="bold", color=INK)
    ax.grid(alpha=0.2, axis="y")
    save(fig, "f_longctx.png")


# =========================================================================
# 13. QAT precision map
# =========================================================================
def f_qat():
    fig, ax = new_ax(10, 4.8)
    ax.text(50, 94, "MXFP4/8 QAT：从 SFT 起全程量化感知训练",
            ha="center", fontsize=13, weight="bold", color=INK)
    box(ax, 8, 42, 40, 30, "MoE 专家权重\n(参数显存大头)", fc=TEAL, fs=13, rounding=0.03)
    ax.text(28, 36, "→ MXFP4 权重 / MXFP8 激活", ha="center", color=TEAL, fontsize=11.5, weight="bold")
    box(ax, 54, 42, 40, 30, "注意力投影 · latent 投影\nshared 专家 · router", fc=SLATE_L, tc=INK,
        fs=12, rounding=0.03)
    ax.text(74, 36, "→ 保持高精度", ha="center", color=SLATE, fontsize=11.5, weight="bold")
    box(ax, 12, 8, 76, 18, "", fc=PANEL, rounding=0.02)
    ax.text(50, 20, "RL 时 rollout 与训练用同一量化方案 → 消除 train-infer 失配",
            ha="center", fontsize=11.5, color=GREEN, weight="bold")
    ax.text(50, 12.5, "SiTU-GLU 有界 + 各处 RMSNorm + KDA 门下界 = 低精度的稳定器",
            ha="center", fontsize=11, color=INK)
    save(fig, "f_qat.png")


# =========================================================================
# 14. Parallelism composition
# =========================================================================
def f_parallel():
    fig, ax = new_ax(10, 5.4)
    ax.text(50, 95, "3T 级预训练并行组合", ha="center", fontsize=13, weight="bold", color=INK)
    rows = [
        ("PP（interleaved 1F1B + VPP）", "93 层按 stage 切 · VPP 压 bubble", NAVY),
        ("EP（Expert Parallelism）", "896 专家按 rank 切 · all-to-all dispatch/combine", TEAL),
        ("ZeRO-1 DP", "优化器状态分片 · 扩批量", BLUE),
        ("Pipeline ZeRO-2 梯度分片", "梯度按 DP 分片 + 下沉 CPU", VIOLET),
        ("CP（KDA Context Parallelism）", "1M 序列切分 · KDA 递归态用 KCP", AMBER),
    ]
    y = 74
    for name, desc, c in rows:
        box(ax, 8, y, 40, 9, name, fc=c, fs=11.5, rounding=0.04,
            tc="white" if c != AMBER else INK)
        ax.text(50, y + 4.5, desc, va="center", fontsize=10.5, color=INK)
        y -= 12
    box(ax, 8, 2, 88, 9, "关键约束：EP 的 all-to-all / PP 的 P2P / DP 的 reduce 争同一带宽 → 用计算盖住每一条通信",
        fc=INK, fs=11, rounding=0.02)
    save(fig, "f_parallel.png")


# =========================================================================
# 15. Overlap schedule (Fig.11 style gantt)
# =========================================================================
def f_schedule():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(BG)
    lanes = ["计算流", "EP 通信", "NCCL 通信", "激活 offload"]
    ax.set_ylim(0, 4); ax.set_xlim(0, 20)
    data = {
        0: [(0, 2, "ViT fwd", TEAL_L), (2, 2, "Attn", NAVY), (4, 1.4, "SE1", TEAL),
            (5.4, 2.2, "MLP", BLUE), (7.6, 1.4, "SE2", TEAL), (9, 2, "WGrad", VIOLET),
            (11, 2, "Attn bwd", NAVY), (13, 3, "ViT bwd", TEAL_L)],
        1: [(3.8, 1.6, "EP-D", AMBER), (7.4, 1.2, "EP-C", AMBER_D), (10.8, 1.8, "EP-DR", RED)],
        2: [(0, 2, "gather param", SLATE), (9, 4, "reduce grad", SLATE_L)],
        3: [(2, 3, "offload", "#94a3b8"), (11, 3, "onload", "#94a3b8")],
    }
    for lane, items in data.items():
        for (x, w, lbl, c) in items:
            ax.add_patch(Rectangle((x, lane + 0.15), w, 0.7, fc=c,
                                   ec="white", lw=1))
            ax.text(x + w / 2, lane + 0.5, lbl, ha="center", va="center",
                    fontsize=8.5, color="white" if c in (NAVY, TEAL, BLUE, VIOLET, RED, AMBER_D, SLATE) else INK,
                    weight="bold")
    ax.set_yticks([l + 0.5 for l in range(4)]); ax.set_yticklabels(lanes, fontsize=11)
    ax.set_xticks([]); ax.set_title("叠瓦式调度：每条通信被某段计算盖住（Fig.11 复现）",
                                    fontsize=12.5, weight="bold", color=INK)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    save(fig, "f_schedule.png")


# =========================================================================
# 16. 1F1B + VPP bubble
# =========================================================================
def f_1f1b():
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 24); ax.set_ylim(0, 4)
    stages = 4
    # simple 1F1B fwd/bwd fill with bubbles
    for s in range(stages):
        y = stages - 1 - s
        # forward blocks
        for m in range(4):
            x = s + m * 1.0
            ax.add_patch(Rectangle((x, y + 0.15), 0.9, 0.7, fc=TEAL, ec="white", lw=0.8))
            ax.text(x + 0.45, y + 0.5, f"F{m+1}", ha="center", va="center", fontsize=7.5, color="white")
        # backward blocks
        for m in range(4):
            x = 8 + (stages - 1 - s) + m * 1.0
            ax.add_patch(Rectangle((x, y + 0.15), 0.9, 0.7, fc=NAVY, ec="white", lw=0.8))
            ax.text(x + 0.45, y + 0.5, f"B{m+1}", ha="center", va="center", fontsize=7.5, color="white")
        # bubble
        ax.add_patch(Rectangle((4 + s, y + 0.15), 4 - s, 0.7, fc="#fca5a5", ec="white",
                               lw=0.8, alpha=0.5))
    ax.text(6, -0.4, "bubble", color=RED, fontsize=10, weight="bold")
    ax.set_yticks([i + 0.5 for i in range(stages)])
    ax.set_yticklabels([f"Stage {stages-i}" for i in range(stages)], fontsize=10)
    ax.set_xticks([])
    ax.set_title("interleaved 1F1B + VPP：bubble ≈ (p−1)/(m·v)  · 降激活峰值",
                 fontsize=12, weight="bold", color=INK)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    save(fig, "f_1f1b.png")


# =========================================================================
# 17. MoonEP balance bars
# =========================================================================
def f_moonep_balance():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    rng = np.random.default_rng(1)
    ranks = [f"R{i}" for i in range(8)]
    load = np.array([9, 3, 2, 6, 1, 8, 2, 5]) * 60
    a1.bar(ranks, load, color=RED, alpha=0.8)
    a1.axhline(load.mean(), color=SLATE, ls="--")
    a1.set_title("常规 EP：负载不均 → 热 rank 决定迭代时间 / 碎片 / OOM",
                 fontsize=10.5, weight="bold", color=INK)
    a1.set_ylabel("token / rank"); a1.grid(alpha=0.2, axis="y")
    a1.text(3.5, load.max() * 0.9, "maxvio ↑", color=RED, fontsize=12, weight="bold")
    a2.bar(ranks, [load.mean()] * 8, color=GREEN, alpha=0.85)
    a2.axhline(load.mean(), color=SLATE, ls="--")
    a2.set_title("MoonEP：每 rank 恰好 S×K → 完美均衡", fontsize=10.5, weight="bold", color=INK)
    a2.text(3.5, load.mean() * 1.05, "S × K (静态形状)", color=GREEN, fontsize=11, weight="bold")
    a2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    save(fig, "f_moonep_balance.png")


# =========================================================================
# 18. MoonEP E/R proof (filling construction)
# =========================================================================
def f_moonep_proof():
    fig, ax = new_ax(10, 4.8)
    ax.text(50, 95, "定理：每 rank ≤ E/R 个冗余专家的均衡方案必然存在（界基本紧）",
            ha="center", fontsize=12, weight="bold", color=INK)
    # ranks as columns with under/over
    ranks = [("R0 欠载", 3, RED), ("R1 过载", 9, AMBER), ("R2 均衡", 6, GREEN),
             ("R3 过载", 8, AMBER)]
    for i, (nm, h, c) in enumerate(ranks):
        x = 12 + i * 20
        box(ax, x, 25, 14, h * 5, "", fc=c, alpha=0.5, rounding=0.03, ec=c, lw=2)
        ax.text(x + 7, 20, nm, ha="center", fontsize=10.5, color=INK, weight="bold")
    # target line
    ax.axhline(25 + 6 * 5, xmin=0.08, xmax=0.92, color=SLATE, ls="--", lw=1.5)
    ax.text(94, 25 + 6 * 5, "S×K", color=SLATE, fontsize=10, weight="bold", va="center")
    arrow(ax, 33, 60, 19, 45, INK, 2)   # over -> under fill
    ax.text(50, 12, "反复用过载 rank 填满一个欠载 rank：每 rank 至多被填一次\n→ 远程 token 只来自单一 rank → 至多涉及 E/R 个专家",
            ha="center", fontsize=10.5, color=INK)
    ax.text(50, 4, "预留 E/R 冗余槽 ⇒ 规划永远可行、训练永不中断（对比 ECHO/UltraEP 会停训）",
            ha="center", fontsize=10.5, color=GREEN, weight="bold")
    save(fig, "f_moonep_proof.png")


# =========================================================================
# 19. MoonEP zero-copy buffer
# =========================================================================
def f_moonep_buffer():
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    labels = ["DeepEP\n(最坏倾斜, 零拷贝)", "MoonEP\n(完美均衡)"]
    vals = [8, 1]
    bars = ax.bar(labels, vals, color=[SLATE, TEAL], width=0.5)
    ax.text(0, 8.2, "S × K × R", ha="center", fontsize=13, weight="bold", color=SLATE)
    ax.text(1, 1.25, "S × K (固定)", ha="center", fontsize=13, weight="bold", color=TEAL)
    ax.set_ylabel("通信 buffer 大小（相对，R=8）")
    ax.set_title("零拷贝的代价：buffer 大小对比", fontsize=12.5, weight="bold", color=INK)
    ax.set_ylim(0, 9.5); ax.grid(alpha=0.2, axis="y")
    save(fig, "f_moonep_buffer.png")


# =========================================================================
# 20. MoonEP benchmark comm vs imbalance
# =========================================================================
def f_moonep_bench():
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    x = np.linspace(0, 3, 50)
    deepep = 1 + 0.9 * x + 0.15 * x ** 2
    moonep = 0.85 + 0.02 * x
    ax.plot(x, deepep, color=SLATE, lw=3, label="DeepEP（随不均恶化）")
    ax.plot(x, moonep, color=TEAL, lw=3, label="MoonEP（几乎平坦）")
    ax.scatter([2.9], [deepep[-1]], color=RED, s=90, zorder=5)
    ax.text(2.55, deepep[-1] + 0.25, "OOM", color=RED, fontsize=12, weight="bold")
    ax.set_xlabel("路由不均 maxvio"); ax.set_ylabel("通信时间（相对）")
    ax.set_title("H20, EP=8：通信时间 vs 不均度", fontsize=12.5, weight="bold", color=INK)
    ax.legend(fontsize=10.5); ax.grid(alpha=0.25)
    save(fig, "f_moonep_bench.png")


# =========================================================================
# 21. Memory ledger
# =========================================================================
def f_mem_ledger():
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=200)
    fig.patch.set_facecolor(BG)
    techs = ["FP8 块量化激活", "offload / 远程 offload", "跨层重计算",
             "Memory-efficient MoE", "Block AttnRes checkpoint",
             "PP 激活再平衡(削峰)", "Pipeline ZeRO-2 + CPU 梯度", "P2P Muon"]
    saves = [22, 26, 14, 8, 12, 10, 18, 9]
    colors = [TEAL, TEAL, BLUE, VIOLET, AMBER, "#0ea5e9", VIOLET, GREEN]
    y = np.arange(len(techs))[::-1]
    ax.barh(y, saves, color=colors, alpha=0.9)
    for yi, s, t in zip(y, saves, techs):
        ax.text(0.4, yi, t, va="center", ha="left", fontsize=10.5, color="white", weight="bold")
        ax.text(s + 0.3, yi, f"省 ~{s}%", va="center", fontsize=9.5, color=INK)
    ax.set_yticks([]); ax.set_xlim(0, 32); ax.set_xlabel("相对显存节省（示意）")
    ax.set_title("显存组合拳：把 2.8T 塞进预算（各手段互补，非单一）",
                 fontsize=12.5, weight="bold", color=INK)
    ax.grid(alpha=0.2, axis="x")
    save(fig, "f_mem_ledger.png")


# =========================================================================
# 22. Unified activation manager
# =========================================================================
def f_act_manager():
    fig, ax = new_ax(10, 4.8)
    ax.text(50, 94, "统一激活管理器：存储策略与模型代码解耦（张量粒度自由组合）",
            ha="center", fontsize=12, weight="bold", color=INK)
    box(ax, 8, 60, 22, 14, "为反向保存的\n张量（注解声明）", fc=NAVY, fs=11, rounding=0.05)
    backends = [("重计算", BLUE), ("FP8 块量化", TEAL), ("offload → CPU", AMBER),
                ("远程 offload → 别 rank", VIOLET)]
    for i, (nm, c) in enumerate(backends):
        box(ax, 55, 74 - i * 15, 38, 11, nm, fc=c, fs=11, rounding=0.05,
            tc="white" if c != AMBER else INK)
        arrow(ax, 30, 67, 55, 79.5 - i * 15, SLATE, 1.8, ms=12)
    box(ax, 12, 6, 76, 12, "单一内存池（主计算流）避免多 stream 碎片 · 层粒度预取与计算 overlap",
        fc=PANEL, tc=INK, fs=11, rounding=0.02)
    save(fig, "f_act_manager.png")


# =========================================================================
# 23. PP activation rebalance
# =========================================================================
def f_pp_rebalance():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ranks = [f"PP{i}" for i in range(6)]
    before = [10, 8, 6, 5, 3, 2]
    after = [6, 6, 5.5, 5.5, 5.5, 5.5]
    a1.bar(ranks, before, color=RED, alpha=0.8)
    a1.axhline(9, color=SLATE, ls="--"); a1.text(0.2, 9.2, "HBM 上限", color=SLATE, fontsize=9)
    a1.set_title("1F1B 暖机：越靠后 rank 驻留激活越少（峰值不均→OOM 风险）",
                 fontsize=9.5, weight="bold", color=INK)
    a1.set_ylabel("驻留激活"); a1.grid(alpha=0.2, axis="y")
    a2.bar(ranks, after, color=GREEN, alpha=0.85)
    a2.axhline(9, color=SLATE, ls="--")
    a2.set_title("Mooncake 远程 offload → 跨 PP rank 拉平峰值",
                 fontsize=9.5, weight="bold", color=INK)
    a2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    save(fig, "f_pp_rebalance.png")


# =========================================================================
# 24. P2P Muon vs all-gather
# =========================================================================
def f_p2p_muon():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200)
    for a in (a1, a2):
        a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off"); a.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    # left: naive all-gather (broadcast) — every rank holds full param buffer
    a1.set_title("朴素：全量 all-gather", fontsize=12, weight="bold", color=RED)
    for i in range(4):
        a1.add_patch(FancyBboxPatch((0.6 + i * 2.3, 6.5), 1.8, 1.6,
                     boxstyle="round,pad=0,rounding_size=0.15", fc=SLATE_L, ec=SLATE))
        a1.text(1.5 + i * 2.3, 7.3, f"rank{i}", ha="center", fontsize=9)
        # full buffer on each
        a1.add_patch(Rectangle((0.6 + i * 2.3, 3.0), 1.8, 2.6, fc=RED, alpha=0.4, ec=RED))
        a1.text(1.5 + i * 2.3, 4.3, "全参\nbuffer", ha="center", fontsize=8, color=RED)
    a1.text(5, 1.4, "显存 O(P)/rank + all-gather O(P) 通信", ha="center",
            fontsize=10, color=RED, weight="bold")
    # right: P2P targeted
    a2.set_title("K3：P2P 定向拉分片", fontsize=12, weight="bold", color=GREEN)
    for i in range(4):
        a2.add_patch(FancyBboxPatch((0.6 + i * 2.3, 6.5), 1.8, 1.6,
                     boxstyle="round,pad=0,rounding_size=0.15", fc=SLATE_L, ec=SLATE))
        a2.text(1.5 + i * 2.3, 7.3, f"rank{i}", ha="center", fontsize=9)
        a2.add_patch(Rectangle((0.6 + i * 2.3, 4.4), 1.8, 1.2, fc=GREEN, alpha=0.5, ec=GREEN))
        a2.text(1.5 + i * 2.3, 5.0, "仅自己\n负责的", ha="center", fontsize=7.5, color=INK)
    for i in range(3):
        a2.annotate("", xy=(1.5, 4.3), xytext=(1.5 + (i + 1) * 2.3, 4.3),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                    connectionstyle="arc3,rad=-0.3"))
    a2.text(5, 1.4, "buffer O(P/DP) · 消除全参 buffer · 通信流水隐藏", ha="center",
            fontsize=10, color=GREEN, weight="bold")
    fig.tight_layout()
    save(fig, "f_p2p_muon.png")


# =========================================================================
# 25. Bubble filling (ViT into pipeline bubbles)
# =========================================================================
def f_bubble_fill():
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 24); ax.set_ylim(0, 4)
    # text pipeline with bubbles, ViT fills them
    for s in range(3):
        y = 2 - s
        for m in range(5):
            x = s + m * 1.1
            ax.add_patch(Rectangle((x, y + 0.15), 1.0, 0.7, fc=NAVY, ec="white", lw=0.8))
            ax.text(x + 0.5, y + 0.5, "text", ha="center", va="center", fontsize=7, color="white")
        # bubbles filled by ViT
        for k in range(2):
            xb = 6.2 + s + k * 1.1
            ax.add_patch(Rectangle((xb, y + 0.15), 1.0, 0.7, fc=TEAL, ec="white", lw=0.8))
            ax.text(xb + 0.5, y + 0.5, "ViT", ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([i + 0.5 for i in range(3)])
    ax.set_yticklabels([f"PP{2-i}" for i in range(3)], fontsize=10)
    ax.set_xticks([])
    ax.set_title("Bubble filling：把 ViT 计算塞进 PP 流水气泡（承 K2.5 DEP · 思路同 Optimus）",
                 fontsize=11.5, weight="bold", color=INK)
    ax.text(11, -0.5, "ViT（可搬动、无跨 micro-batch 依赖）→ 有效开销基本被隐藏",
            ha="center", color=TEAL, fontsize=10.5, weight="bold")
    for sname in ["top", "right", "left", "bottom"]:
        ax.spines[sname].set_visible(False)
    save(fig, "f_bubble_fill.png")


# =========================================================================
# 26. KCP prefix scan
# =========================================================================
def f_kcp():
    fig, ax = new_ax(10, 4.6)
    ax.text(50, 94, "KDA Context Parallelism：转移矩阵分解 + 前缀扫描",
            ha="center", fontsize=12.5, weight="bold", color=INK)
    for i in range(4):
        x = 8 + i * 22
        box(ax, x, 55, 18, 12, f"rank {i}\n本地段", fc=TEAL, fs=11, rounding=0.05)
        # each computes two local quantities
        ax.text(x + 9, 49, r"算 $M^{T\leftarrow 1},\ \tilde{S}$", ha="center", fontsize=9.5, color=AMBER_D, weight="bold")
        if i < 3:
            arrow(ax, x + 18, 61, x + 22, 61, SLATE, 2)
    # prefix scan arrows
    ax.text(50, 36, "一次固定大小 all-gather 交换 (M, S̃) → 前缀扫描恢复各 rank 入态",
            ha="center", fontsize=11, color=INK)
    ax.text(50, 26, "线性注意力只传固定大小状态（softmax 需传随长度增长的 KV 块）",
            ha="center", fontsize=10.5, color=GREEN, weight="bold")
    ax.text(50, 16, "vanilla「从 S=0 求和」对 KDA 不成立：转移矩阵 M_t 要作用在入态上",
            ha="center", fontsize=10, color=SLATE)
    save(fig, "f_kcp.png")


if __name__ == "__main__":
    for fn in [f_threeaxis, f_hybrid, f_kda_state, f_kda_decay, f_attnres,
               f_blockattnres, f_latentmoe, f_situ, f_qb, f_compare, f_scaling,
               f_longctx, f_qat, f_parallel, f_schedule, f_1f1b, f_moonep_balance,
               f_moonep_proof, f_moonep_buffer, f_moonep_bench, f_mem_ledger,
               f_act_manager, f_pp_rebalance, f_p2p_muon, f_bubble_fill, f_kcp]:
        fn()
    print("ALL DONE")
