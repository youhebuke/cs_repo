# Kimi K3 技术分享材料

面向有大模型训练 infra / 算法背景同事的 **1 小时技术分享**。重点讲**结构设计背后的动机与数学原理、预训练、预训练 Infra**；模型结构基础从简，强化学习简化。

## 交付物

| 文件 | 说明 |
|---|---|
| [`slides_v2/kimi-k3-tech-share-v3.pptx`](slides_v2/kimi-k3-tech-share-v3.pptx) | **★ 推荐主交付物 · 可编辑 PowerPoint（42 页）**：对齐上传的参考 PPT『第二讲』六段结构；原生文本框 + 9 张可编辑表格；31 张可视化配图；每页详细演讲注释；少公式；**Infra 章节最强（14 页）** |
| [`slides_v2/kimi-k3-tech-share-v3.pdf`](slides_v2/kimi-k3-tech-share-v3.pdf) | v3 的 PDF 预览版 |
| [`slides_v2/make_figures.py`](slides_v2/make_figures.py) · [`slides_v2/build_pptx_v3.py`](slides_v2/build_pptx_v3.py) | v3 的图形生成脚本 + PPTX 构建脚本（可改内容后重生成） |
| [`slides_v2/kimi-k3-tech-share-v2.pptx`](slides_v2/kimi-k3-tech-share-v2.pptx) / [`.pdf`](slides_v2/kimi-k3-tech-share-v2.pdf) | v2 · 可编辑 PowerPoint（43 页，另一版结构；`build_pptx.py` 构建） |
| [`slides/kimi-k3-tech-share.pptx`](slides/kimi-k3-tech-share.pptx) / [`.pdf`](slides/kimi-k3-tech-share.pdf) / [`.html`](slides/kimi-k3-tech-share.html) / [`.md`](slides/kimi-k3-tech-share.md) | 初版 · Marp 幻灯片（57 页，图片式，不可逐字编辑） |
| [`DESIGN_NOTES.md`](DESIGN_NOTES.md) | **思路梳理文档（落盘版）**——比 PPT 更全的动机 / 数学 / 演进 / 对比 / 备问 |
| [`PRETRAIN_INFRA_DEEPDIVE.md`](PRETRAIN_INFRA_DEEPDIVE.md) | **预训练 Infra 深挖分析**——并行组合 / MoonEP（E/R 证明·零拷贝·静态形状）/ 显存组合拳 / 多模态 encoder，重原理与推导 |

> **各版 PPT 的区别**：`v3` 是最新推荐版——参照上传的参考 PPT（Kimi K3 第二讲）的六段结构与「痛点→选择→为什么」框架重做，原生可编辑（文本框 + 表格），配 31 张自绘可视化图 + 封面，少公式、每页详细演讲注释，并把预训练 Infra 扩到 14 页（FlashKDA 双 kernel/CHUNK=16、KCP、MoonEP 均衡/E-R 证明/零拷贝/基准、显存账本/激活管理器/PP 再平衡/P2P Muon、多模态 bubble filling、对比 DeepSeek 训练 infra）。`v2` 是上一版可编辑 PPT（结构略不同）。`slides/*` 是初版 Marp（图片式）。**对外分享请用 v3。**

## 内容主线

1. **定位与总纲**：为什么做 3T · 三轴信息流（KDA 跨 token / AttnRes 跨 layer / LatentMoE 跨 channel）
2. **结构 · 为什么这么设计**：KDA（下界化衰减、全秩门）· Gated MLA + NoPE · Attention Residuals · Stable LatentMoE（SiTU-GLU / Quantile Balancing）· Per-Head Muon · 与 DeepSeek-V3 / Qwen3 对比
3. **预训练**：Scaling Law（2.5× · cosine vs WSD 方法论）· 长上下文扩展 · MXFP4/8 QAT
4. **预训练 Infra（重点）**：KDA 协同（FlashKDA / intra-device CP / KCP）· MoonEP 完美负载均衡（E/R 上界证明）· 显存高效训练（统一激活管理 / ZeRO-2 / P2P Muon）
5. **RL（简化）+ 推理 Infra + 演进脉络 + 方法论启示**

## 重新构建幻灯片

### v3（可编辑 PPTX · 推荐）

```bash
cd slides_v2
pip install python-pptx Pillow matplotlib     # 首次
python3 make_figures.py            # 生成/更新 img/ 下的可视化图（需 CJK 字体，如 WenQuanYi Micro Hei）
python3 build_pptx_v3.py           # 生成 kimi-k3-tech-share-v3.pptx（含表格 + 演讲注释）
# 可选：转 PDF 预览（需 libreoffice-impress）
soffice --headless --convert-to pdf --outdir . kimi-k3-tech-share-v3.pptx
```

要改内容：直接在 PowerPoint / Keynote / WPS 里编辑 `.pptx`（文本框、表格、演讲者备注均可改）；
或改 `build_pptx_v3.py` 里 `build()` 中对应 `content(...)` 的 `blts` / `table` / `note` 后重跑。
（v2 同理，用 `build_pptx.py`。）

### 初版（Marp）

```bash
cd slides
npm install                       # 安装 Marp CLI（首次）
export CHROME_PATH=/usr/bin/google-chrome-stable   # PDF/PPTX 导出需要 Chrome
npx marp kimi-k3-tech-share.md -o kimi-k3-tech-share.pptx --pptx --allow-local-files
npx marp kimi-k3-tech-share.md -o kimi-k3-tech-share.pdf  --pdf  --allow-local-files
npx marp kimi-k3-tech-share.md -o kimi-k3-tech-share.html --html
```

## 资料来源

Kimi K3 技术报告（`k3_tech_report.pdf`）· Attention-Residuals（arXiv 2603.15031）· FlashKDA · MoonEP · Kimi Linear（arXiv 2510.26692）· 对比基线 DeepSeek-V3（2412.19437）/ Qwen3（2505.09388）。

> 注：`sources/`（第三方仓库与报告 PDF）未纳入版本库，见 `.gitignore`。
