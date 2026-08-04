# Kimi K3 技术分享材料

面向有大模型训练 infra / 算法背景同事的 **1 小时技术分享**。重点讲**结构设计背后的动机与数学原理、预训练、预训练 Infra**；模型结构基础从简，强化学习简化。

## 交付物

| 文件 | 说明 |
|---|---|
| [`slides/kimi-k3-tech-share.pptx`](slides/kimi-k3-tech-share.pptx) | **主交付物 · PowerPoint 幻灯片**（47 页，可直接放映） |
| [`slides/kimi-k3-tech-share.pdf`](slides/kimi-k3-tech-share.pdf) | PDF 版（便于预览 / 打印） |
| [`slides/kimi-k3-tech-share.html`](slides/kimi-k3-tech-share.html) | HTML 版（浏览器直接打开，支持演讲者模式） |
| [`slides/kimi-k3-tech-share.md`](slides/kimi-k3-tech-share.md) | 幻灯片 Marp 源码（改内容后可重渲染） |
| [`DESIGN_NOTES.md`](DESIGN_NOTES.md) | **思路梳理文档（落盘版）**——比 PPT 更全的动机 / 数学 / 演进 / 对比 / 备问 |
| [`PRETRAIN_INFRA_DEEPDIVE.md`](PRETRAIN_INFRA_DEEPDIVE.md) | **预训练 Infra 深挖分析**——并行组合 / MoonEP（E/R 证明·零拷贝·静态形状）/ 显存组合拳 / 多模态 encoder，重原理与推导 |

## 内容主线

1. **定位与总纲**：为什么做 3T · 三轴信息流（KDA 跨 token / AttnRes 跨 layer / LatentMoE 跨 channel）
2. **结构 · 为什么这么设计**：KDA（下界化衰减、全秩门）· Gated MLA + NoPE · Attention Residuals · Stable LatentMoE（SiTU-GLU / Quantile Balancing）· Per-Head Muon · 与 DeepSeek-V3 / Qwen3 对比
3. **预训练**：Scaling Law（2.5× · cosine vs WSD 方法论）· 长上下文扩展 · MXFP4/8 QAT
4. **预训练 Infra（重点）**：KDA 协同（FlashKDA / intra-device CP / KCP）· MoonEP 完美负载均衡（E/R 上界证明）· 显存高效训练（统一激活管理 / ZeRO-2 / P2P Muon）
5. **RL（简化）+ 推理 Infra + 演进脉络 + 方法论启示**

## 重新构建幻灯片

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
