#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an EDITABLE Kimi K3 tech-share deck with python-pptx.
- Native text boxes (fully editable in PowerPoint)
- Embedded schematic figures / charts (from make_figures.py) + AI hero cover
- Reduced formulas (math lives in figures), visual-first to avoid a wall of text
- Detailed speaker notes on every slide
Run: python3 build_pptx.py  ->  kimi-k3-tech-share-v2.pptx
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")

# ---- palette ----
NAVY = RGBColor(0x1E, 0x3A, 0x5F)
NAVY_D = RGBColor(0x0F, 0x24, 0x40)
TEAL = RGBColor(0x0D, 0x94, 0x88)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_BR = RGBColor(0xF5, 0x9E, 0x0B)
VIOLET = RGBColor(0x7C, 0x3A, 0xED)
INK = RGBColor(0x0F, 0x17, 0x2A)
SLATE = RGBColor(0x47, 0x55, 0x69)
GREY = RGBColor(0x64, 0x74, 0x8B)
LGREY = RGBColor(0xE2, 0xE8, 0xF0)
PANEL = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
BG = RGBColor(0xFB, 0xFC, 0xFE)

FONT = "Microsoft YaHei"       # CJK-capable; PowerPoint substitutes if absent
FONT_EN = "Segoe UI"

EMU_IN = 914400
SW, SH = 13.333, 7.5

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]


def set_ea(run, name=FONT):
    """Force East-Asian + latin + cs typeface so CJK renders with the chosen font."""
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        el = rPr.find(qn("a:" + tag))
        if el is None:
            el = rPr.makeelement(qn("a:" + tag), {})
            rPr.append(el)
        el.set("typeface", name)


def bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def rect(slide, x, y, w, h, color, line=None, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    return tb, tf


def add_para(tf, text, size=18, color=INK, bold=False, first=False,
             align=PP_ALIGN.LEFT, space_after=6, space_before=0, name=FONT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after); p.space_before = Pt(space_before)
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    set_ea(run, name)
    return p, run


def fit(path, max_w, max_h):
    im = Image.open(path); w, h = im.size
    r = min(max_w / w, max_h / h)
    return w * r, h * r


def place_image(slide, path, cx, cy, max_w, max_h, shadow=True):
    """Place image centered in (cx,cy) box region top-left, fitted to max_w/max_h."""
    fw, fh = fit(path, max_w, max_h)
    left = cx + (max_w - fw) / 2
    top = cy + (max_h - fh) / 2
    pic = slide.shapes.add_picture(path, Inches(left), Inches(top),
                                   Inches(fw), Inches(fh))
    return pic


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def bullets(tf, items, base=17):
    """items: list of (text, level). level 0/1/2. First para reuses paragraph[0]."""
    for i, (text, lvl) in enumerate(items):
        color = INK if lvl == 0 else SLATE
        size = base if lvl == 0 else base - 2
        bold = lvl == 0
        marker = {0: "●  ", 1: "–  ", 2: "·  "}[lvl]
        indent = {0: 0, 1: 0.28, 2: 0.56}[lvl]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(5 if lvl == 0 else 3)
        p.level = lvl
        # marker run (accent color)
        rm = p.add_run(); rm.text = marker
        rm.font.size = Pt(size); rm.font.bold = True
        rm.font.color.rgb = TEAL if lvl == 0 else GREY
        set_ea(rm)
        rr = p.add_run(); rr.text = text
        rr.font.size = Pt(size); rr.font.bold = bold; rr.font.color.rgb = color
        set_ea(rr)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(indent))))
        pPr.set("indent", "0")


def title_bar(slide, title, sub=None):
    rect(slide, 0.0, 0.0, 0.22, 1.42, TEAL)       # left accent
    tb, tf = textbox(slide, 0.55, 0.28, 12.2, 0.95)
    add_para(tf, title, size=27, color=NAVY, bold=True, first=True)
    rect(slide, 0.58, 1.14, 3.1, 0.045, AMBER_BR)  # underline
    if sub:
        tb2, tf2 = textbox(slide, 0.58, 1.24, 11.9, 0.34)
        add_para(tf2, sub, size=12.5, color=GREY, bold=False, first=True)


def footer(slide, idx):
    tb, tf = textbox(slide, 0.55, 7.08, 11.0, 0.35)
    add_para(tf, "Kimi K3 技术分享 · 结构设计 / 预训练 / 训练 Infra", size=9,
             color=GREY, first=True)
    tb2, tf2 = textbox(slide, 12.4, 7.08, 0.7, 0.35)
    add_para(tf2, str(idx), size=10, color=GREY, first=True, align=PP_ALIGN.RIGHT)


# =====================================================================
# slide builders
# =====================================================================
_idx = [0]


def slide_title():
    s = prs.slides.add_slide(BLANK); bg(s, NAVY_D)
    hero = os.path.join(IMG, "cover_hero.png")
    # full-bleed hero
    fw, fh = fit(hero, SW, SH)
    # cover the slide: scale to fill
    im = Image.open(hero); w, h = im.size
    r = max(SW / w, SH / h)
    pw, ph = w * r, h * r
    s.shapes.add_picture(hero, Inches((SW - pw) / 2), Inches((SH - ph) / 2),
                         Inches(pw), Inches(ph))
    # dark scrim panel for text legibility
    scrim = rect(s, 0.0, 2.5, 8.6, 3.1, NAVY_D)
    scrim.fill.fore_color.rgb = NAVY_D
    scrim.fill.transparency = 0  # keep solid-ish
    _set_alpha(scrim, 42)
    tb, tf = textbox(s, 0.7, 2.7, 8.0, 2.8)
    add_para(tf, "Kimi K3 技术分享", size=44, color=WHITE, bold=True, first=True, space_after=6)
    add_para(tf, "开源前沿智能 · 2.8T MoE / 1M 上下文 / 原生多模态", size=20,
             color=RGBColor(0x93, 0xC5, 0xFD), bold=True, space_after=14)
    add_para(tf, "重点：结构设计背后的「为什么」 · 预训练 · 预训练 Infra", size=16,
             color=WHITE, space_after=4)
    add_para(tf, "（结构基础已由前序分享覆盖，本场只讲动机与原理；强化学习简化）", size=12,
             color=RGBColor(0xCB, 0xD5, 0xE1))
    notes(s,
        "【开场 · 约1分钟】欢迎大家。今天这场分享的定位先说清楚：不是把 Kimi K3 的结构从头到尾再讲一遍——"
        "结构的基础介绍前序分享已经覆盖过了。今天我聚焦三件事：第一，结构设计背后的『为什么』，也就是每个"
        "模块解决什么问题、动机是什么；第二，预训练的过程与取舍；第三，也是今天篇幅最重的——预训练的 Infra（基础设施）。"
        "强化学习我会简化带过。\n\n"
        "一句话记住 K3：2.8 万亿总参数、1040 亿激活、100 万 token 上下文、原生多模态，相比上一代 K2 整体 scaling 效率提升约 2.5 倍。"
        "这张封面图其实就是今天的隐喻——信息在一个巨大的网络里，沿三个方向流动。等下第一部分就会展开这『三条轴』。")
    footer(s, 1); _idx[0] = 1


def _set_alpha(shape, pct):
    """Set solid fill transparency (pct 0-100)."""
    sp = shape.fill._xPr.find(qn("a:solidFill"))
    if sp is None:
        return
    srgb = sp.find(qn("a:srgbClr"))
    if srgb is None:
        return
    alpha = srgb.makeelement(qn("a:alpha"), {"val": str(int((100 - pct) * 1000))})
    srgb.append(alpha)


def slide_divider(title, sub, part):
    s = prs.slides.add_slide(BLANK); bg(s, NAVY_D)
    hero = os.path.join(IMG, "cover_hero.png")
    im = Image.open(hero); w, h = im.size
    r = max(SW / w, SH / h); pw, ph = w * r, h * r
    pic = s.shapes.add_picture(hero, Inches((SW - pw) / 2), Inches((SH - ph) / 2),
                               Inches(pw), Inches(ph))
    ov = rect(s, 0, 0, SW, SH, NAVY_D); _set_alpha(ov, 45)
    rect(s, 0.7, 2.75, 0.16, 2.0, AMBER_BR)
    tb, tf = textbox(s, 1.05, 2.7, 11.0, 2.4)
    add_para(tf, part, size=18, color=RGBColor(0x5E, 0xEA, 0xD4), bold=True, first=True, space_after=8)
    add_para(tf, title, size=40, color=WHITE, bold=True, space_after=10)
    add_para(tf, sub, size=17, color=RGBColor(0xCB, 0xD5, 0xE1))
    return s


def slide_content(title, sub, blts, image=None, layout="right", takeaway=None,
                  note=""):
    s = prs.slides.add_slide(BLANK); bg(s)
    title_bar(s, title, sub)
    ty = 1.55
    if layout == "right" and image:
        tb, tf = textbox(s, 0.6, ty, 5.9, 5.1)
        bullets(tf, blts, base=16)
        place_image(s, os.path.join(IMG, image), 6.65, 1.5, 6.35, 5.15)
    elif layout == "left" and image:
        place_image(s, os.path.join(IMG, image), 0.5, 1.5, 6.35, 5.15)
        tb, tf = textbox(s, 7.15, ty, 5.7, 5.1)
        bullets(tf, blts, base=16)
    elif layout == "big" and image:
        place_image(s, os.path.join(IMG, image), 0.6, 1.4, 12.1, 3.35)
        tb, tf = textbox(s, 0.6, 4.9, 12.1, 1.55)
        bullets(tf, blts, base=13)
    else:  # none
        tb, tf = textbox(s, 0.6, ty, 12.2, 5.1)
        bullets(tf, blts, base=18)
    if takeaway:
        rect(s, 0.6, 6.55, 0.12, 0.5, AMBER_BR)
        r = rect(s, 0.72, 6.55, 11.98, 0.5, PANEL)
        tb2, tf2 = textbox(s, 0.9, 6.57, 11.7, 0.46, anchor=MSO_ANCHOR.MIDDLE)
        add_para(tf2, "要点：" + takeaway, size=13, color=NAVY, bold=True, first=True)
    _idx[0] += 1
    footer(s, _idx[0])
    notes(s, note)
    return s


def divider(title, sub, part, note=""):
    s = slide_divider(title, sub, part)
    _idx[0] += 1
    notes(s, note)
    return s


# =====================================================================
# DECK CONTENT
# =====================================================================
def build():
    slide_title()

    # ---- roadmap ----
    slide_content(
        "本场路线图（约 1 小时）", "五个部分，Infra 篇幅最重",
        [("Part 0 · 定位与总纲（约 5 min）：为什么做 3T · 三轴信息流", 0),
         ("Part 1 · 结构：为什么这么设计（约 18 min）", 0),
         ("KDA / Gated MLA·NoPE · Attention Residuals · Stable LatentMoE · Per-Head Muon · 对比 DeepSeek/Qwen", 1),
         ("Part 2 · 预训练（约 10 min）：Scaling Law · 长上下文 · QAT", 0),
         ("Part 3 · 预训练 Infra（重点，约 20 min）", 0),
         ("并行组合 · KDA 协同(FlashKDA/KCP) · MoonEP 完美负载均衡 · 显存组合拳 · 多模态 encoder", 1),
         ("Part 4 · RL（简化，约 5 min）+ 收尾", 0),
         ("三个记忆钩子：KDA 管『跨 token』，AttnRes 管『跨 layer』，LatentMoE 管『跨 channel』", 0)],
        image=None, layout="none",
        takeaway="今天不重复结构基础，重点是『为什么这么设计』与预训练 Infra。",
        note="【路线图 · 约1分钟】给大家一个地图，方便你判断现在讲到哪、哪些是重点。"
             "五个部分：定位、结构动机、预训练、预训练 Infra、以及简化的 RL。"
             "时间分配上，Infra 是今天篇幅最重的一块，占到约三分之一。\n\n"
             "结尾这三个记忆钩子请先记住：KDA 负责跨 token 的信息流动，Attention Residuals 负责跨 layer（层与层之间），"
             "LatentMoE 负责跨 channel（专家/通道）。后面每一节其实都能挂到这三条轴上。如果中间听得有点晕，"
             "回到这三个钩子就不会迷路。")

    # ============ PART 0 ============
    divider("定位与总纲", "为什么是 3T？三轴信息流", "Part 0",
            note="【进入 Part 0】先讲定位和总纲——为什么要把底座做到 3T 级，以及贯穿全场的『三轴信息流』框架。"
                 "这部分是全场的『地基』，后面所有结构创新都会回到这个框架。")

    slide_content(
        "为什么把底座推到 3T 级？", "两条 scaling 轴，K3 选择同时推进",
        [("LLM 的 scaling 有两条轴", 0),
         ("轴一 · 预训练规模：更大模型 + 更多数据", 1),
         ("轴二 · test-time compute：RL + 推理 effort + 长程交互", 1),
         ("现状：开源在『轴二』猛进，但『轴一』停在 1T 级附近", 0),
         ("风险：都在相似规模底座上堆 RL → 开源互相收敛，与闭源差距扩大", 1),
         ("K3 的选择：两轴同时推到前沿", 0),
         ("底座到 3T 级，同时把 RL / effort / 交互推到 1M 上下文", 1),
         ("结果：2.8T 总参 / 104B 激活 / 1M 上下文，整体 scaling 效率相比 K2 约 2.5×", 0)],
        image=None, layout="none",
        takeaway="别人卷 test-time，K3 选择『底座 + test-time』两条轴一起卷到前沿。",
        note="【为什么 3T · 约2分钟】这页讲动机。大模型的 scaling 有两条轴：一条是传统的预训练规模（更大、更多数据）；"
             "另一条是这两年很火的 test-time compute——推理链更长、RL 更多、agentic 交互更长。\n\n"
             "关键观察：开源社区在第二条轴上进步飞快，但第一条轴几乎停在 1T 级。"
             "如果大家都在差不多规模的底座上堆 RL，那开源模型会互相趋同，而和闭源前沿的差距反而拉大。"
             "所以 Moonshot 的判断是：两条轴要一起推。底座做到史无前例的 3T 级，同时把上下文和 RL 推到 1M。"
             "最终数字：2.8T 总参、104B 激活、1M 上下文，相对 K2 的 scaling 效率约 2.5 倍——这个 2.5× 后面 Scaling Law 那页还会回来。")

    slide_content(
        "总纲：沿三个维度『扩展信息流』", "所有结构创新都在回答同一个问题",
        [("核心问题：信息如何更高效地流动？", 0),
         ("序列维（token）：Hybrid Attention = 3×KDA + 1×MLA → 长序列高效混合 + 全局兜底", 0),
         ("深度维（layer）：Attention Residuals → 每层选择性检索所有前层", 0),
         ("宽度维（channel）：Stable LatentMoE 896 选 16 → 稀疏扩展通道容量", 0),
         ("训练稳定：Per-Head Muon + QB + 有界化 → 在 2.8T/896专家/MXFP4 下稳住优化", 0),
         ("记忆钩子：KDA 跨 token · AttnRes 跨 layer · LatentMoE 跨 channel", 0)],
        image="f_threeaxis.png", layout="left",
        takeaway="记住这三条轴，后面每个模块都挂在它上面。",
        note="【三轴总纲 · 约2分钟 · 全场最重要的一页】这是理解 K3 结构的总框架，我建议大家把这张图记住。"
             "K3 所有结构创新，本质都在回答一个问题：信息如何更高效地流动？答案是沿三个正交方向扩展信息流。\n\n"
             "看右边这张图：横轴是序列维（token 之间），用 Hybrid Attention——3 层线性注意力 KDA 加 1 层全局 MLA；"
             "纵轴是深度维（层与层之间），用 Attention Residuals，让每一层能『选择性地』回看所有前面层；"
             "斜轴是宽度维（通道/专家），用 Stable LatentMoE，896 个专家里选 16 个。"
             "再加一条隐线：训练稳定性，靠 Per-Head Muon、Quantile Balancing 和各种有界化手段。"
             "这一页立住了，后面就是逐条展开每根轴的『为什么』。")

    slide_content(
        "K2 → K3 规格对照", "改了什么，一目了然",
        [("总参 / 激活：1.04T/32.6B → 2.78T/104B（+167% / +220%）", 0),
         ("层数：61 → 93；注意力：全 MLA → 69 KDA + 24 Gated MLA（3:1）", 0),
         ("路由专家/激活：384/8 → 896/16（+2 shared）", 0),
         ("Latent MoE 维：无 → 3584（0.5× hidden）", 1),
         ("激活函数：SwiGLU → SiTU-GLU；位置编码：RoPE → NoPE", 0),
         ("训练上下文：128K → 1M（8×）", 0),
         ("优化器：Muon → Per-Head Muon", 0),
         ("量化：无 → MXFP4 权重 / MXFP8 激活（QAT）", 0),
         ("视觉：新增 MoonViT-V2（401M，27 层，从零训练）", 1)],
        image=None, layout="none",
        takeaway="几乎每个模块都动了刀——这也是为什么需要一整套新 Infra。",
        note="【规格对照 · 约1.5分钟】这页是一张速查表，不用逐行念，挑重点：参数量翻了近三倍，激活参数翻了三倍多；"
             "注意力从全 MLA 变成 KDA 和 MLA 的 3:1 混合；专家从 384 选 8 变成 896 选 16，还引入了 latent 压缩维度；"
             "激活函数、位置编码、优化器、量化——几乎每个模块都动了刀。\n\n"
             "我想让大家记住的一点是：正因为改动这么全面、规模这么大，才逼出了后面一整套新的 Infra。"
             "结构的激进和系统的复杂是一体两面的——这也是今天为什么要花大篇幅讲 Infra。")

    # ============ PART 1 ============
    divider("模型结构：为什么这么设计", "基础从简，重点讲动机与原理", "Part 1",
            note="【进入 Part 1】现在展开结构。再强调一次：基础介绍从简，我重点讲每个模块『为什么这么设计』。"
                 "顺序按三轴走：先序列维（KDA/MLA），再深度维（AttnRes），再宽度维（LatentMoE），最后优化器和对比。")

    slide_content(
        "序列维：Hybrid Attention = 3×KDA + 1×MLA", "为什么是混合，而不是纯线性",
        [("每个 block：3 层 KDA（线性）+ 1 层 Gated MLA（全局），末尾再补 1 层 MLA", 0),
         ("为什么混合？", 0),
         ("KDA 承担绝大多数 token mixing，递归状态固定大小 → 长序列便宜", 1),
         ("线性注意力的软肋 = 长程精确召回 → 用 1/4 全局 MLA 兜底", 1),
         ("3:1 是『效率 vs 召回』的工程折中（继承自 Kimi Linear）", 1),
         ("直接收益：1M 上下文下 KV cache 与算力显著下降", 0)],
        image="f_hybrid.png", layout="right",
        takeaway="DeepSeek 用全 MLA、Qwen 用 GQA；K3 用『线性为主 + 全局兜底』换长上下文效率。",
        note="【Hybrid Attention · 约2分钟】序列维的第一根轴。结构上很简单：每个 block 三层 KDA 加一层全局 MLA，比例 3:1，"
             "网络最后再补一层 MLA，保证最后一层一定是全局注意力。\n\n"
             "重点是『为什么混合』。纯线性注意力最省，但它有个致命软肋——长程的精确召回（大海捞针类任务）做不好，"
             "因为它把历史压进一个固定大小的状态里。纯全局注意力召回强但 1M 上下文下 KV cache 和算力都吃不消。"
             "所以 K3 的折中是：让 KDA 干绝大多数活（便宜），每四层插一层全局 MLA 兜住召回。"
             "对比来看，DeepSeek-V3 是全 MLA，Qwen 是 GQA，K3 是唯一走『线性为主』的，这就是它 1M 上下文效率的来源。")

    slide_content(
        "KDA 是什么：带通道级遗忘门的 delta rule", "用固定大小 RNN 记忆做『可控遗忘 + 精确改写』",
        [("状态 S 是固定大小的递归记忆（不像 KV cache 随长度增长）", 0),
         ("delta rule：先『按 key 方向擦除旧关联』，再『写入新 k·v』", 0),
         ("通道级遗忘门 Diag(α)：每个特征维独立控制记忆衰减", 0),
         ("比 Gated DeltaNet / Mamba-2 的『头级标量门』更细粒度", 1),
         ("q/k/v 走 ShortConv + Swish，q/k 再 L2Norm（细节从略）", 1)],
        image="f_kda_state.png", layout="left",
        takeaway="固定大小状态 = 长序列便宜、易传输/复用（这点后面 Infra 会反复用到）。",
        note="【KDA 是什么 · 约2分钟】这页尽量少公式。KDA 是一种线性注意力，核心是用一个固定大小的状态 S 当记忆，"
             "而不是像 softmax 注意力那样缓存所有历史 KV。左图对比很直观：左边 KV cache 随长度越堆越高（内存 O(T)），"
             "右边 KDA 只有一个固定大小的 S（内存 O(1)）。\n\n"
             "更新规则叫 delta rule：来一个新 token，先沿着它的 key 方向把旧的关联『擦掉』，再写入新的关联。"
             "K3 的关键升级是遗忘门做到了『通道级』——每个特征维有自己独立的衰减率，比 Gated DeltaNet、Mamba-2 那种"
             "整个 head 共用一个标量门要精细得多。\n\n"
             "请记住『固定大小状态』这个点——它便宜、好传输、好复用，等到讲 Infra 的 KCP 和前缀缓存时会反复用到。")

    slide_content(
        "KDA 演进①：给衰减门『加下界』→ 系统大收益", "算法-系统协同的典型缩影",
        [("问题：chunkwise 里要用累计衰减的倒数重标定 key，倒数会无界增长、BF16 溢出", 0),
         ("Kimi Linear 的对角块只能逐位置对计算 → 上不了 Tensor Core，成瓶颈", 1),
         ("K3 改法：把 log-decay 从『无界负 Softplus』换成『带下界的缩放 sigmoid』", 0),
         ("数学后果：保留因子有下界 → 重标定因子落在 BF16 范围", 1),
         ("系统收益：对角块也能用稠密 Tensor Core GEMM，删掉标量路径", 0)],
        image="f_kda_decay.png", layout="right",
        takeaway="一个纯数学的『下界』，换来 kernel 的硬件友好——这类协同全场会反复出现。",
        note="【KDA 演进① · 约2.5分钟 · 重点】这是我特别想讲的一页，因为它是『算法-系统协同设计』最典型的例子。\n\n"
             "背景：KDA 的高效实现要分块（chunkwise），块内并行、块间串行。块内计算需要用『累计衰减的倒数』去重标定 key。"
             "问题是累计衰减是一堆 0 到 1 之间的数连乘，它的倒数会无界地放大，在 BF16 下就溢出了。"
             "上一代 Kimi Linear 的处理是把对角块单独用『逐位置对』的标量方式算——但这样对角块就上不了 Tensor Core，成了瓶颈。\n\n"
             "K3 的改动看似很小：把控制衰减的那个映射，从『无下界的负 Softplus』换成『有下界的缩放 sigmoid』，"
             "下界固定在 -5（看左图两条曲线，青色那条被压在 -5 以上）。这样每个保留因子都大于 e 的 -5 次方，"
             "一个 16-token 的小块累计下来落在可控范围，倒数就不会溢出 BF16。\n\n"
             "结果（右图）：对角块也能用稠密的 Tensor Core 矩阵乘了，那条慢的标量路径被彻底删掉。"
             "这就是我说的『一个纯数学的下界，换来 kernel 的硬件友好』。这种思路今天会反复出现，请留意。")

    slide_content(
        "KDA 演进② + Gated MLA + NoPE", "全秩输出门 · 全局层为什么不加位置编码",
        [("KDA 演进②：输出门从『低秩』换成『输入相关的全秩』投影", 0),
         ("让每个 token 更充分地调制从递归记忆读出的通道 → 表达力更强", 1),
         ("Gated MLA：把 KV 压成低维 latent 缓存，保留全局注意力（源自 DeepSeek-V2）", 0),
         ("NoPE：所有 MLA 层不加任何显式位置编码", 0),
         ("位置信息由中间的 KDA 层通过递归门控隐式提供", 1),
         ("副产品：扩上下文零改动——不用重调 RoPE、不用 YaRN → 直接外推 1M", 1)],
        image=None, layout="none",
        takeaway="NoPE 是 K3 能『直接外推到 1M』的关键之一。",
        note="【KDA② + MLA + NoPE · 约2分钟】两个小点合在一页。第一，KDA 的输出门从低秩换成全秩，"
             "简单说就是让每个 token 更充分地调制它从记忆里读出来的东西，表达力更强，也和 MLA 的门统一了参数化。\n\n"
             "第二，也是更值得强调的——NoPE。K3 的所有全局 MLA 层『不加任何位置编码』。为什么敢这么做？"
             "因为位置信息已经由中间的 KDA 层通过它的递归门控和衰减隐式提供了，MLA 只需要负责『无约束的全局内容交互』。"
             "这带来一个非常爽的副产品：扩上下文的时候什么都不用改——不用重新调 RoPE 的频率基、不用做 YaRN 插值，"
             "模型可以直接外推到 1M。这是 K3 长上下文能力的关键之一，等下 Part 2 讲长上下文时会呼应。")

    slide_content(
        "深度维：为什么需要 Attention Residuals", "标准残差 = 深度维的 RNN",
        [("洞见：PreNorm 残差 = 每层用固定单位权重把所有前层加起来 = 深度维 RNN", 0),
         ("三个后果", 0),
         ("无选择性访问：attention 层和 MLP 层拿到同一个聚合态", 1),
         ("不可逆损失：早层信息被埋，深层无法选择性找回", 1),
         ("输出膨胀：hidden 幅度随深度 O(L) 增长，稀释每层贡献、破坏稳定", 1),
         ("类比：序列维当年用 attention 取代 RNN 递归——深度维也该如此", 0)],
        image="f_attnres.png", layout="left",
        takeaway="把『序列维的 attention 革命』搬到深度维。",
        note="【AttnRes 动机 · 约2分钟】第二根轴，深度维。先讲动机，这个洞见很漂亮。"
             "我们习以为常的残差连接，如果把它展开，会发现：每一层其实是把前面所有层的输出用『固定的单位权重』加起来。"
             "这跟 RNN 在时间维上把历史压进一个状态，是一模一样的结构——所以标准残差本质是『深度维的 RNN』。\n\n"
             "这带来三个问题（看左图左半边）：第一，没有选择性——注意力层和 MLP 层拿到的是同一个混合态；"
             "第二，信息不可逆地损失——早期层的信息被后面层稀释掉，深层想找也找不回来；"
             "第三，输出膨胀——PreNorm 下 hidden 的幅度随层数线性增长，越深每层的相对贡献越小，还影响稳定性。\n\n"
             "解决思路很自然：序列维当年是用 attention 取代了 RNN 的递归，那深度维也照做——这就是 Attention Residuals。")

    slide_content(
        "Attention Residuals + Block 版", "深度维 softmax + 把开销降到 O(Nd)",
        [("Full AttnRes：每层用可学 pseudo-query，对『所有前层』做 softmax 选择", 0),
         ("理论：标准残差 = 深度维线性注意力；AttnRes = 深度维 softmax 注意力", 1),
         ("Block AttnRes：L 层切成 N 个 block，块内标准残差、块间才做注意力", 0),
         ("显存/通信 O(Ld) → O(Nd)；K3 用 N≈8（93 层切 8 块）", 1),
         ("收益：同算力 ≈ baseline 的 1.25× · hidden 幅度有界 · 梯度更均匀", 0),
         ("推理友好：block 间可并行 + online softmax 严格等价 → 延迟开销 <2%", 0)],
        image="f_blockattnres.png", layout="right",
        takeaway="N≈8 就能拿回大部分收益，且推理几乎不额外花钱。",
        note="【AttnRes 机制 + Block · 约2分钟】机制上（尽量不写公式）：每一层有一个可学习的『伪 query』，"
             "用它对『embedding 加上所有前面层的输出』做一次 softmax 注意力，选择性地聚合。"
             "理论上很优雅——标准残差等价于深度维的线性注意力，而 AttnRes 就是把它升级成深度维的 softmax 注意力。\n\n"
             "但 Full 版要保活所有层的输出，显存和跨 stage 通信是 O(Ld)。所以有 Block 版（右图）：把层分成 N 个块，"
             "块内还是普通残差求和，只在『块之间』做注意力。开销从 O(Ld) 降到 O(Nd)。实测 N≈8 就能拿回大部分收益，"
             "K3 就是 93 层切成 8 块。\n\n"
             "收益：同样算力能达到多花 1.25 倍算力的 baseline 效果；hidden 幅度不再膨胀、梯度在层间更均匀。"
             "而且推理很友好——块间可以并行、块内用 online softmax 合并且严格等价，实测推理延迟只多不到 2%。"
             "这条等下 Infra 讲显存时还会回来（Block 结构让它能压回标准残差的显存足迹）。")

    slide_content(
        "宽度维：Stable LatentMoE 让『896 选 16』可负担", "分离模型宽度与路由专家宽度",
        [("常规 MoE：每个被选专家吃完整 d 维 token → 通信/权重流量爆炸", 0),
         ("LatentMoE：shared 专家走全宽，routed 专家在压缩 latent 空间里算", 0),
         ("K3：shared=2，routed=896 选 16，稀疏度 ≈ 56", 1),
         ("极端稀疏放大两个失效：4 连乘病态→激活爆炸；近千专家→均衡失控", 0),
         ("三招治：Normalized（加 RMSNorm）· SiTU-GLU（有界）· Quantile Balancing", 0)],
        image="f_latentmoe.png", layout="left",
        takeaway="只有把 routed 专家压到 latent 空间，896 选 16 的通信/显存才付得起。",
        note="【LatentMoE · 约2分钟】第三根轴，宽度维。想扩专家数和激活数，能扩大专家特化的空间，但常规 MoE 里"
             "每个被选中的专家都要吃完整维度的 token，896 选 16 的话通信量和专家权重流量直接爆炸。\n\n"
             "LatentMoE 的巧思（看左图）：把两条路分开。共享专家走全宽路径处理共性；而路由专家在一个压缩的 latent 空间里算"
             "（K3 是 3584 维，正好是 hidden 的一半）。这样 896 选 16 的通信和显存才付得起，稀疏度做到约 56。\n\n"
             "但极端稀疏放大了两个问题：一是路由分支变成差不多四个矩阵连乘，病态，内部激活会爆炸；"
             "二是近一千个专家的负载均衡，超出了老方法能稳住的范围。K3 用三招治：加 RMSNorm、换有界的 SiTU-GLU 激活、"
             "以及 Quantile Balancing 做均衡。后两个我下面各用一页展开。")

    slide_content(
        "SiTU-GLU：有界，又保留 SwiGLU 的形状", "为 MXFP4/8 低精度铺路",
        [("背景：SwiGLU 的两个乘子都无界 → 大坐标产生激活离群点、低精度易溢出", 0),
         ("SiTU-GLU：对 Swish 线性因子和 up 分支各套一个平滑 cap（β·tanh）", 0),
         ("近原点一阶匹配 SwiGLU（保留局部行为）", 1),
         ("β1,β2→∞ 逐点恢复 SwiGLU；输出有界 ≤ β1·β2 = 100", 1),
         ("vs 硬 clamp：平滑 cap 在非饱和区保留非零梯度 → 训练更好", 0)],
        image="f_situ.png", layout="right",
        takeaway="有界激活是 MXFP4 权重 / MXFP8 激活能训得动的前提之一。",
        note="【SiTU-GLU · 约1.5分钟】治激活爆炸的这招。SwiGLU 大家都用，但它有个问题：两个相乘的因子都是无界的，"
             "一旦某些坐标同时很大，就会产生激活离群点，在低精度（FP8/FP4）下很容易溢出。\n\n"
             "SiTU-GLU 的做法是给这两个因子各套一个平滑的『软上限』——用 β·tanh(x/β) 把它压住。"
             "看右图三条曲线：灰色 SwiGLU 冲到天上去了，青色 SiTU-GLU 被稳稳压在 100 以下（100 = β1×β2 = 4×25）。"
             "但它在原点附近（看内嵌小图）和 SwiGLU 一阶重合，也就是保留了 SwiGLU 好用的局部形状；β 趋于无穷时就还原成 SwiGLU。"
             "相比直接硬截断（clamp），平滑 cap 的好处是在非饱和区还有非零梯度，训练更稳。\n\n"
             "一句话价值：这种有界激活，是后面 MXFP4 权重、MXFP8 激活能训得动的前提之一。")

    slide_content(
        "Quantile Balancing：从『符号步长』到『分位数』", "负载均衡升级为有理论最优解",
        [("背景：aux-loss-free 路由给分数加专家偏置 b_j，但 b_j 不进 softmax 权重", 0),
         ("DeepSeek-V3：固定步长符号更新 → 步长大则震荡、小则慢；896 专家更难稳", 1),
         ("QB：把均衡看作最优平衡指派 → 偏置的闭式解 = margin 分布的分位数", 0),
         ("无 learning-rate 类超参、几步就均衡", 1),
         ("工程：直方图全局估计——一次 all-reduce 求 bin 计数，与分片方式无关", 0)],
        image="f_qb.png", layout="left",
        takeaway="把启发式步长升级成『有最优解的分位数』+ 可扩展的直方图估计。",
        note="【Quantile Balancing · 约2分钟】治大规模均衡这招。先说背景：现在主流是 aux-loss-free 路由，"
             "给每个专家的 router 分数加一个偏置来影响 Top-k 选择，但这个偏置不进入真正的混合权重，所以不干扰 router 的梯度学习。"
             "DeepSeek-V3 用的是固定步长的符号更新——负载多了就减一点、少了就加一点，步长大了震荡、小了收敛慢，"
             "近一千个专家时更难调。\n\n"
             "QB 的做法是把负载均衡直接看成一个『最优平衡指派』问题，做对偶推导，发现专家偏置的闭式最优解，"
             "恰好就是每个专家 margin 分布的某个分位数（右图左边那条红线）。好处是：不需要 learning-rate 这种超参，几步就均衡。"
             "而且它和 DeepSeek 的符号更新是同一个目标——符号法只是它的一步近似。\n\n"
             "工程上（右图右边）：全局 batch 有几百万个 margin，没法精确算分位数，就每个专家维护一个直方图，"
             "一次 all-reduce 把各卡的 bin 计数加起来，就能恢复全局分位数。计数可加，所以和 token 怎么分片无关——这点很关键。")

    slide_content(
        "Per-Head Muon + 原生多模态", "两个『稳』字诀",
        [("Per-Head Muon：Q/K/V 投影按 head 切块、逐 head 做正交化", 0),
         ("整矩阵正交化会让大尺度 head 主导共享更新方向 → 逐 head 均衡各 head 更新", 1),
         ("顺带：高瘦 per-head 块的 Newton–Schulz 更便宜", 1),
         ("MoonViT-V2（401M）从零训练，不再用 SigLIP 初始化", 0),
         ("理由是稳定性：SigLIP-init 梯度范数持续偏高+尖峰；from-scratch 全程稳", 1),
         ("性能打平 → 大规模多模态 LLM 不必依赖对比预训练做初始化", 1)],
        image=None, layout="none",
        takeaway="优化器和视觉编码器的改动，共同主题都是『大规模下的稳定性』。",
        note="【Per-Head Muon + 视觉 · 约1.5分钟】两个都围绕『稳』字。优化器方面，K3 沿用 Muon（对矩阵参数做动量加正交化），"
             "改进是对注意力的 Q/K/V 投影，不再把整个矩阵一起正交化，而是按 head 切成块、每个 head 单独正交化。"
             "原因是：整矩阵正交化会让梯度尺度大的 head 主导共享的更新方向，小 head 就被带偏；逐 head 做能把各 head 的更新尺度拉平，"
             "大规模下更稳。副带一个好处，对高瘦的 per-head 块做 Newton–Schulz 迭代还更便宜。\n\n"
             "视觉方面，一个反直觉的选择：MoonViT-V2 是从零训练的，没有用 SigLIP 这类对比预训练来初始化。"
             "为什么？还是稳定性——用 SigLIP 初始化再联合训练，梯度范数持续偏高、频繁尖峰；从零训反而全程平稳。"
             "而且最终性能打平，说明大规模多模态 LLM 其实不需要对比预训练当初始化。")

    slide_content(
        "与 DeepSeek-V3 / Qwen3 的对比", "K3 = DeepSeek 地基 + 三层新机制",
        [("注意力：DeepSeek 全 MLA · Qwen GQA · K3 KDA+MLA 混合", 0),
         ("MoE：DeepSeek 256 选 8 · Qwen 128 选 8(无 shared) · K3 896 选 16 + latent 压缩", 0),
         ("均衡：DeepSeek 符号步长 · Qwen global-batch aux-loss · K3 Quantile Balancing", 0),
         ("位置编码：都用 RoPE · K3 用 NoPE；优化器：都用 AdamW · K3 用 Per-Head Muon", 0),
         ("优势：长上下文省 · scaling 效率高 · 低精度稳；代价：结构+系统复杂度显著上升", 0),
         ("Qwen 走保守稳健路线：工程简单，但 scaling/长上下文上限更受限", 1)],
        image="f_compare.png", layout="left",
        takeaway="激进换来 3T/1M，代价是复杂度——所以必须配套一整套 Infra。",
        note="【对比 · 约2分钟】把 K3 放到坐标系里。右图一句话概括：K3 = DeepSeek 系的地基（MLA、shared/routed 专家、aux-loss-free）"
             "再叠上三层新机制——KDA（序列维）、Attention Residuals（深度维）、LatentMoE（宽度维），"
             "外加 SiTU-GLU、QB、Per-Head Muon、NoPE、MXFP4 量化。\n\n"
             "逐项对比：注意力上 DeepSeek 全 MLA、Qwen GQA、K3 混合；MoE 上 K3 专家最多、还做了 latent 压缩；"
             "均衡上三家各不同；位置编码只有 K3 用 NoPE；优化器只有 K3 用 Muon。\n\n"
             "优劣要讲清楚：优势是长上下文省、scaling 效率高、低精度稳；代价是结构和系统复杂度显著上升。"
             "Qwen 走的是更保守稳健的路线，工程简单，但 scaling 和长上下文的上限更受限。"
             "这个『激进换来能力、复杂度靠 Infra 兜底』的判断，正好把我们引到今天的重头戏——预训练 Infra。")

    # ============ PART 2 ============
    divider("预训练", "Scaling Law · 长上下文 · QAT", "Part 2",
            note="【进入 Part 2】预训练这块讲三件事：Scaling Law（那个 2.5× 从哪来）、怎么扩到 1M 上下文、以及量化感知训练。"
                 "篇幅适中，重点为后面的 Infra 做铺垫。")

    slide_content(
        "预训练数据", "配方里的两个亮点",
        [("四大文本域：Web / Code / Math / Knowledge + 大规模视觉语料", 0),
         ("规则启发式 + 分类器质量打分 + 去重；小模型 ablation 定采样率", 1),
         ("亮点①：Rephrasing——对 knowledge/math 用多样化 prompt 改写，并对源文档保真校验", 0),
         ("亮点②：programmatic 多模态数据——代码片段配其渲染结果", 0),
         ("SVG / 3D / 网页 / 游戏 / CAD → 支撑『写代码看效果』的 vision-in-the-loop", 1)],
        image=None, layout="none",
        takeaway="programmatic 数据（代码↔渲染）是 K3 视觉编程能力的来源之一。",
        note="【数据 · 约1.5分钟】数据部分我挑两个亮点讲，其余从略。第一是 rephrasing：对知识和数学语料用风格、视角多样化的 prompt "
             "重写一遍，扩充有效数据，但关键是会对着源文档做保真校验，防止改写引入错误。\n\n"
             "第二个更有意思——programmatic 多模态数据：把代码片段和它渲染出来的视觉结果配对，涵盖 SVG、3D 资产、网页、游戏、CAD。"
             "这正是 K3『写代码、看渲染效果、再改』这种 vision-in-the-loop 能力的数据来源。视觉那边坐标监督还同时给绝对和归一化两种格式，"
             "兼顾精确和分辨率鲁棒。")

    slide_content(
        "Scaling Law：2.5× 与一条方法论", "比较训练技巧的正确姿势",
        [("结构/数据/训练一起改 → 最优训练 regime 也变 → 重调 batch/lr/TPP/模型 shape", 0),
         ("结论：整体 scaling 效率 ≈ 2.5×（held-out OOD 验证）", 0),
         ("cosine decay 一致优于 WSD——但关键在方法论", 0),
         ("两种调度的最优超参差别很大，用同一套超参比较会不公平地偏袒一方", 1),
         ("正确做法：各自独立做 scaling-law 搜索，在各自最优下再比", 1)],
        image="f_scaling.png", layout="right",
        takeaway="比较任何训练技巧，都要『各自调到最优再比』——否则结论不可信。",
        note="【Scaling Law · 约2分钟】那个反复出现的 2.5× 就来自这里（右图两条曲线的水平位移，等效于 2.5 倍算力）。"
             "因为结构、数据、训练都改了，最优的训练配置也会变，所以他们专门重新做了 scaling law，重调了 batch size、学习率、"
             "每参数 token 数（TPP）、以及模型形状。\n\n"
             "但我更想让大家带走的是一条方法论。他们发现 cosine 学习率一致优于 WSD，但强调了一个坑：这两种调度的最优超参差别很大。"
             "如果你用同一套超参去比，会不公平地偏袒其中一个。正确做法是——给每种调度各自独立做一遍 scaling law 搜索，"
             "在各自的最优点上再比。这个『各自调到最优再比』的原则，其实对我们自己做任何 A/B 训练技巧对比都适用，很实用。")

    slide_content(
        "长上下文扩展到 1M", "NoPE 外推 + 四阶段课程 + 数据合成",
        [("位置：NoPE → 直接外推 1M，无需任何位置编码改动", 0),
         ("数据：长文/长视频含大量低质 → 专门清洗（去重/帧感知哈希/结构校验）+ 上采样", 0),
         ("关键：长度 ≠ 长程能力", 0),
         ("合成『必须跨全 1M 才能解』的任务，把注意力训练在目标尺度，防退化成局部模式", 1),
         ("四阶段课程：预训练 8K→64K，cooldown 256K→1M", 0),
         ("把昂贵的长序列计算集中在训练预算的一小部分 → 经济且渐进", 1)],
        image="f_longctx.png", layout="left",
        takeaway="NoPE 负责『能外推』，数据合成负责『真的会用长上下文』。",
        note="【长上下文 · 约2分钟】怎么做到 1M。三件事。第一，位置编码用 NoPE，前面说过，好处是能直接外推，不用改任何东西。"
             "第二，数据要清洗——天然的长文和长视频里有大量近重复、二进制块、截断、无效日志，得专门清洗，视频还要用帧的感知哈希去重；"
             "而且真正连贯的长文很稀缺，要上采样。\n\n"
             "第三点最关键：长度不等于长程能力。就算你喂很长的序列，如果任务只需要看局部就能解，注意力会退化成只看附近。"
             "所以他们合成了一批『必须跨越整个 1M 上下文才能解』的任务（把多个文档和子任务排列拼接），"
             "强迫注意力在目标尺度上真正工作。\n\n"
             "最后看右图，扩展是分四阶段渐进的：预训练期 8K 到 64K，cooldown 期 256K 到 1M。"
             "把最贵的长序列计算集中在训练末尾的一小段，既经济又让模型逐步适应。")

    slide_content(
        "部署感知量化（QAT）", "训练即量化，消除 train-infer 失配",
        [("MoE 专家权重（参数显存大头）→ MXFP4，激活 → MXFP8", 0),
         ("非专家部分（注意力/latent 投影/shared 专家/router）保持高精度", 1),
         ("从 SFT 起全程 QAT（覆盖 SFT + RL）→ 模型适应量化精度损失", 0),
         ("RL 时 rollout 与训练用同一量化方案 → 消除 train-infer 失配", 0),
         ("能训得动的底气：SiTU-GLU 有界 + 各处 RMSNorm + KDA 门下界", 0)],
        image="f_qat.png", layout="right",
        takeaway="前面那些『有界化』手段，在这里一起兑现成了 MXFP4 可训。",
        note="【QAT · 约1.5分钟】部署感知的量化。策略是差异化的（右图）：占参数显存大头的 MoE 专家权重用 MXFP4、激活用 MXFP8；"
             "而注意力投影、latent 投影、共享专家、router 这些非专家部分保持高精度。\n\n"
             "关键点有两个。第一，从 SFT 阶段就开始全程量化感知训练，一直覆盖到 RL，让模型主动去适应量化带来的精度损失，"
             "而不是训完再量化。第二，RL 的时候 rollout（采样）和训练用完全相同的量化方案，这样就消除了训练和推理之间的失配——"
             "这个失配在 RL 里是很头疼的问题。\n\n"
             "顺便回收一个伏笔：为什么 MXFP4 这么激进的量化能训得动？就是靠前面讲的那些有界化手段——SiTU-GLU 有界、"
             "到处都有 RMSNorm、KDA 的门有下界。它们在这里一起兑现了。")

    # ============ PART 3 · INFRA ============
    divider("预训练 Infrastructure", "本场重点：把 3T/1M 训得动且不空转", "Part 3",
            note="【进入 Part 3 · 全场重点】现在进入今天篇幅最重的部分——预训练 Infra。"
                 "一句话立论：2.8T、1M、多模态放在一个模型里训，首先是个系统问题，不是算法问题。"
                 "接下来我会按『三大痛点 → 并行组合 → 调度 → MoonEP → 显存 → 多模态』的顺序讲，中间会不断呼应前面结构部分埋的伏笔。")

    slide_content(
        "为什么 2.8T 预训练首先是『系统问题』", "三个数量级事实定义了问题边界",
        [("参数显存：2.78T 参数即使 BF16 也要约 5.5TB → MXFP4 是压显存的前提", 0),
         ("激活显存：93 层 × 1M token × hidden 7168 → TB 级，单一手段不够", 0),
         ("专家通信：896 选 16 的 all-to-all dispatch/combine 是 EP 主带宽消耗", 0),
         ("三大痛点由此而来", 0),
         ("① EP 负载不均：token 路由数据相关、天然长尾", 1),
         ("② 显存超预算：参数+激活+梯度+优化器状态叠加", 1),
         ("③ ViT 计算方差：大图/长视频让 encoder 计算量剧烈波动，落在关键路径", 1)],
        image=None, layout="none",
        takeaway="核心哲学：每个算法选择都要先回答它对『通信/显存/kernel 形状』的影响。",
        note="【Infra 立论 · 约2分钟】先立一个论点：为什么说 2.8T 预训练首先是系统问题？三个数量级事实。"
             "参数：2.78 万亿即使 BF16 也要约 5.5TB，这就是为什么必须 MXFP4 量化专家权重。"
             "激活：93 层乘以百万 token 乘以 7168 隐藏维，是 TB 级的，任何单一手段都压不住，必须组合拳。"
             "通信：896 选 16 的专家 all-to-all，是 EP 最大的带宽消耗，负载一不均，最慢的卡就拖垮整轮。\n\n"
             "由此引出今天 Infra 要解决的三大痛点：EP 负载不均、显存超预算、ViT 计算方差暴露在关键路径。"
             "这三条会分别对应后面的 MoonEP、显存组合拳、和 bubble filling。\n\n"
             "还有一句贯穿全场的哲学，请记住：K3 每做一个算法选择，都会先问『它对通信量、显存账本、kernel 形状意味着什么』。"
             "最好的系统优化，往往始于一个小的算法改动——KDA 加下界那页就是例子。")

    slide_content(
        "3T 级预训练并行组合", "五维并行，各司其职又相互约束",
        [("PP（interleaved 1F1B + VPP）：93 层按 stage 切，VPP 压 bubble", 0),
         ("EP：896 专家按 rank 切，all-to-all dispatch/combine", 0),
         ("ZeRO-1 DP：优化器状态分片；Pipeline ZeRO-2：梯度分片 + 下沉 CPU", 0),
         ("CP（KCP）：1M 序列切分，KDA 递归态专门处理", 0),
         ("关键约束：EP all-to-all / PP P2P / DP reduce 争同一带宽", 0),
         ("→ 全篇主线：用计算盖住每一条通信（overlap）", 1)],
        image="f_parallel.png", layout="left",
        takeaway="这套组合不是『切得多』，而是切完后每条通信都能被某段计算盖住。",
        note="【并行组合 · 约2分钟】K3 用了五个维度的并行（右图）。流水并行 PP 把 93 层按 stage 切，配合虚拟阶段 VPP 压缩气泡；"
             "专家并行 EP 把 896 个专家分到各卡；ZeRO-1 分片优化器状态、Pipeline ZeRO-2 再分片梯度还下沉到 CPU；"
             "上下文并行 CP 把 1M 序列切开，其中 KDA 的递归态要用专门的 KCP 处理（一会儿单独讲）。\n\n"
             "这里我想强调的不是『切了多少维』，而是这几维之间是相互约束的：EP 的 all-to-all、PP 的点对点、DP 的 reduce，"
             "都在抢同一张网卡的带宽。所以这套组合能成立的前提，是『用计算把每一条通信都盖住』——overlap 是贯穿整个 Infra 的主线，"
             "下一页那张调度图就是这句话的具体化。")

    slide_content(
        "叠瓦式调度：每条通信被计算盖住", "Fig.11 调度复现",
        [("四条流并置：计算 / EP 通信 / NCCL 通信 / 激活 offload", 0),
         ("shared expert 拆成 SE1/SE2 派独立 stream → 填 all-to-all 空档", 0),
         ("reduce grad = reduce_scatter + onload + add + offload（Pipeline ZeRO-2 + CPU 梯度）", 0),
         ("EP-DR（dispatch 重算）在反向 → 对应 memory-efficient MoE", 0)],
        image="f_schedule.png", layout="big",
        takeaway="把通信藏进计算的缝隙里，是所有后续优化能成立的基础。",
        note="【调度图 · 约2分钟】这是把上一页那句『用计算盖住通信』落到实处的调度示意（复现报告 Fig.11）。"
             "四条泳道：最上是计算流（数据加载、ViT 前向、注意力、共享专家、MLP、算梯度、反向……），"
             "第二条是专家并行的通信（EP-D 派发、EP-C 合并、EP-DR 派发重算），第三条是 NCCL 通信（gather 参数、reduce 梯度），"
             "最下是激活的 offload/onload。\n\n"
             "看三个设计意图：第一，共享专家被拆成 SE1、SE2 两段放到独立的 stream，用来填专家 all-to-all 的通信空档；"
             "第二，那个 reduce grad 其实是个复合操作——reduce_scatter 之后 onload CPU 分片、相加、再 offload 回去，"
             "这正是 Pipeline ZeRO-2 加 CPU 梯度的落地；第三，反向里那个 EP-DR 是 dispatch 重算，对应后面要讲的 memory-efficient MoE。"
             "总之，通信全被藏进了计算的缝隙里——这是后面所有优化能成立的基础。")

    slide_content(
        "interleaved 1F1B + VPP", "压 bubble 的同时留下一个要治的副作用",
        [("GPipe bubble 占比 ≈ (p−1)/m（p=stage 数，m=micro-batch 数）", 0),
         ("1F1B：不改 bubble 比例，但每 stage 只驻留约 p 份在飞激活 → 降峰值", 0),
         ("VPP：每物理 stage 再拆 v 个虚拟块 → bubble ≈ (p−1)/(m·v)，代价是通信 ×v", 0),
         ("副作用：暖机期各 stage 驻留激活不均 → 后面用『PP 激活再平衡』治", 0)],
        image="f_1f1b.png", layout="big",
        takeaway="VPP 把 bubble 再除以 v，代价是暖机期激活不均（后面会治）。",
        note="【1F1B + VPP · 约1.5分钟】这页给对 PP 熟悉的同事一个定量直觉。GPipe 的气泡占比大约是 (p-1)/m。"
             "1F1B（一前向一反向）不改变气泡比例，但它让每个 stage 稳态下只需要驻留大约 p 份在飞的激活，而不是全部 m 份，"
             "所以能大幅降低激活峰值。VPP（虚拟流水）再进一步：把每个物理 stage 拆成 v 个虚拟块，气泡就降到大约 (p-1)/(m·v)，"
             "代价是通信次数变成 v 倍。K3 层多、机器多，所以这个 bubble 收益值得。\n\n"
             "但它留了个尾巴（看右图红色气泡的分布）：暖机期各 stage 驻留的激活是不均的，越靠前的 stage 压着越多在飞 micro-batch。"
             "这个副作用后面我用『PP 激活再平衡』那页来治，先记住。")

    slide_content(
        "MoonEP：把 EP 负载均衡从『缓解』做成『消灭』", "常规 EP 的三连击，其实同源",
        [("常规 EP 三连击", 0),
         ("① 热 rank 决定迭代时间（EP 是同步屏障，最慢 rank 拖住全部）", 1),
         ("② 动态形状 → 显存碎片 → 高不均时 OOM", 1),
         ("③ 每层 host↔device 同步：读回 cu_seqlens 才能按形状发射 grouped GEMM", 1),
         ("三点同源：都是『token→专家映射动态且不均』的后果", 0),
         ("治本 = 让映射结果对系统而言变成静态且均衡", 0)],
        image="f_moonep_balance.png", layout="left",
        takeaway="不去『缓解』不均，而是直接『消灭』它——后面所有红利都从这一步来。",
        note="【MoonEP 痛点 · 约2分钟 · 已开源】进入今天 Infra 的第一个重头戏 MoonEP，已经开源。"
             "先讲常规 EP（比如 DeepEP）痛在哪，我把它拆成三连击：第一，EP 是同步屏障，最慢的那张卡决定整轮时间，"
             "而负载天然长尾（右图左边，各卡 token 数很不均）；第二，每步每层路由到各专家的 token 数都在变，"
             "激活是动态形状，分配器反复申请释放不同大小的块，碎片累积，高不均时直接 OOM；"
             "第三，grouped GEMM 需要知道每个专家有多少 token 才能确定 kernel 形状，这个计数是设备上算出来的，"
             "host 必须先读回来——每层一次，CPU 卡在关键路径上。\n\n"
             "关键洞见：这三点其实同源，都是『token 到专家的映射是动态且不均』造成的。所以 MoonEP 不去缝缝补补地缓解，"
             "而是治本——想办法让这个映射的结果，对系统来说变成静态而且完全均衡（右图右边，每卡恰好 S×K）。"
             "一旦做到这一步，后面零拷贝、静态形状、免同步全是免费红利。")

    slide_content(
        "MoonEP 的地基：E/R 冗余专家上界（可证）", "这是它区别于 ECHO/UltraEP 的根本",
        [("手段：动态冗余专家——把热专家临时复制到别的 rank，让 token 可拆分摊平", 0),
         ("定理：任意路由下，每 rank ≤ E/R 个冗余专家的均衡方案必然存在（界基本紧）", 0),
         ("证明构造：反复用过载 rank 填满一个欠载 rank；每 rank 至多被填一次", 1),
         ("→ 远程 token 只来自单一 rank → 至多涉及 E/R 个专家", 1),
         ("含义：预留 E/R 冗余槽 → 规划永远可行、训练永不中断", 0),
         ("对比 ECHO/UltraEP：固定冗余数/token cap，无解时被迫停训 + 手调", 1)],
        image="f_moonep_proof.png", layout="left",
        takeaway="不是『预设一个够用的冗余数』，而是『证明了一个恒成立的上界』。",
        note="【MoonEP E/R 证明 · 约2.5分钟 · 重点】这是 MoonEP 最硬核、也是它和别人根本不同的地方。"
             "手段叫『动态冗余专家』：把一个热专家临时复制到别的卡上，它的 token 就能拆到多个副本去，把长尾摊平。\n\n"
             "关键问题是：到底要多少冗余专家，才能保证一定能均衡？MoonEP 证明了一个上界——任意路由分布下，"
             "每张卡最多 E/R 个冗余专家（E 是专家数、R 是 EP 卡数），均衡方案就一定存在，而且这个界基本是紧的。\n\n"
             "证明是构造性的（右图）：反复地『用一个过载的卡去填满一个欠载的卡，恰好填到 S×K』，"
             "每次填满一张、之后不再变，最多填 R-1 次；每张卡最多被填一次，所以它的远程 token 只来自单独一张卡，"
             "而那张卡只有 E/R 个本地专家，所以最多涉及 E/R 个专家。\n\n"
             "这个定理的工程含义很关键：只要每张卡预留 E/R 个冗余槽，在线规划就永远有可行解，训练永不因为不均而中断。"
             "对比 ECHO、UltraEP 这些方案，它们是预设一个固定冗余数或者 token 上限，一旦极端不均没有可行解，训练就被迫停，还得手调参数。"
             "MoonEP 是从数学上根除了这个风险。")

    slide_content(
        "MoonEP：均衡换来的三个免费红利", "零拷贝 · 静态形状 · 无碎片",
        [("零拷贝：规划 kernel 预算每 token 目的地 → 直接写到远端最终槽位，buffer 视图交给计算", 0),
         ("DeepEP 最坏要 S×K×R 的 buffer；MoonEP 完美均衡恒定 S×K，与倾斜无关", 1),
         ("静态形状：每层 MoE 形状静态已知 → 消除 host 同步、降 kernel 启动开销", 0),
         ("rank 内仍偏斜 → workload-aware GEMM 调度（代价模型+离线 autotune）+ 共享专家单独 stream", 1),
         ("无碎片：静态形状 × 单内存池 → 几乎不再碎片化（均衡直接反哺显存）", 0)],
        image="f_moonep_buffer.png", layout="right",
        takeaway="完美均衡是因，零拷贝/静态形状/无碎片都是果。",
        note="【MoonEP 红利 · 约2分钟】做到完美均衡之后，三个红利自动到手。"
             "第一，零拷贝：规划 kernel 提前算好每个 token 该去哪，token 直接写到远端按专家分好组的最终位置，"
             "通信 buffer 的视图直接交给计算，省掉了那次『通信缓冲区拷贝到用户缓冲区』的开销——而那步通常是主要开销。"
             "看右图这个对比很关键：DeepEP 要支持零拷贝，最坏情况下（所有卡都把 token 发给同一张卡）得预留 S×K×R 的 buffer；"
             "MoonEP 因为完美均衡，每卡恒定收 S×K，buffer 固定，和倾斜无关。\n\n"
             "第二，静态形状：每层 MoE 的计算形状变成静态已知，就不用每层把计数读回 host 了，消除同步、降启动开销。"
             "不过卡内部各专家之间还是偏斜的，这个用 workload-aware 的 GEMM 调度来治（分析型代价模型加离线 autotune），"
             "共享专家的 GEMM 单独放一个 stream overlap。\n\n"
             "第三，无碎片：动态形状本来是碎片的根源，消灭它之后再配单一内存池，几乎不再碎片化——"
             "所以你看，均衡不只是提吞吐，它直接反哺了显存。")

    slide_content(
        "MoonEP 基准（H20, EP=8）", "通信时间随不均『几乎平坦』",
        [("MoonEP：通信时间随 maxvio 几乎不变（buffer 不膨胀、路径无拷贝）", 0),
         ("DeepEP：随不均稳步恶化（延迟由最热 rank 决定），高不均直至 OOM", 0),
         ("端到端：MoonEP 各不均水平迭代时间持平、不 OOM", 0),
         ("代价：冗余专家要预取权重、反向要额外 reduce → 相对停训/OOM 可忽略", 1)],
        image="f_moonep_bench.png", layout="right",
        takeaway="用很小的额外开销，换来对负载不均的『免疫』。",
        note="【MoonEP 基准 · 约1分钟】用一张图收尾 MoonEP。右图是 H20、EP=8 上的表现：横轴是路由不均度，纵轴是通信时间。"
             "青色 MoonEP 几乎是平的——因为 buffer 不随不均膨胀、通信路径也没有拷贝；灰色 DeepEP 随不均稳步上升，"
             "因为它的延迟由最热的卡决定，到高不均那头还会 OOM（那个红点）。端到端训练也是一样，MoonEP 在各种不均水平下迭代时间持平、不 OOM。\n\n"
             "当然它不是完全免费的——冗余专家要预取权重、反向要额外做一次 reduce，但相比『训练停掉或者 OOM』，这点开销完全可以忽略。"
             "一句话：用很小的额外开销，换来了对负载不均的免疫。")

    slide_content(
        "显存组合拳：把 2.8T 塞进预算", "显存是全局零和，靠组合而非单一手段",
        [("统一激活管理器：重计算/FP8块量化/offload/远程offload 都是『存储策略』", 0),
         ("张量粒度自由组合、注解声明、与模型代码解耦；单内存池避免多流碎片", 1),
         ("Memory-efficient MoE：梯度数学变换免存前向输出；反向重算 dispatch 并 overlap", 0),
         ("Block AttnRes：块表示生成一次共享 + 整体 checkpoint → 回到标准残差显存足迹", 0),
         ("省下来的每 GB → 支撑更长序列 / RL 的外部 KV pool", 0)],
        image="f_mem_ledger.png", layout="left",
        takeaway="没有银弹——八种手段各省一点，合起来才把 2.8T 塞进预算。",
        note="【显存组合拳 · 约2分钟】显存是个全局零和游戏，省下来的每一 GB 都拿去支撑更长的序列或者 RL 的外部缓存。"
             "K3 不靠单一手段，而是一套可组合的策略框架（右图列了八种，各省一点）。\n\n"
             "先讲最核心的三个。第一，统一激活管理器：每个为反向保存的张量挂一个『存储后端』，重计算、FP8 块量化、offload 到 CPU、"
             "offload 到别的卡——这些都只是『存储策略』，可以在张量粒度自由组合，用注解声明，和模型代码解耦。"
             "还有个容易被忽略但很致命的细节：所有显存在单一内存池里分配，避免多个 stream 各自的分配器把显存割裂成碎片。\n\n"
             "第二，memory-efficient MoE：把某个梯度用数学变换改写成不依赖前向输出的形式，就不用存那个输出了；"
             "group GEMM 的输入也不存，反向的时候重算 dispatch 恢复，而且这个重算的通信能和反向计算 overlap——所以省了存储、代价近零。"
             "第三，回收前面的伏笔：Block AttnRes 因为块表示生成一次大家共享、整体又用 checkpoint 包起来，"
             "所以它每层为反向保存的激活，和普通残差架构一模一样——深度维注意力没有额外显存负担。")

    slide_content(
        "显存组合拳（续）：三个系统级手段", "把压力从 HBM 挪到 CPU / 别的卡 / P2P",
        [("统一激活管理器（图示）：张量 → 可插拔存储后端，与模型代码解耦", 0),
         ("PP 激活再平衡：1F1B 暖机让各 stage 激活不均 → Mooncake 远程 offload 削峰", 0),
         ("Pipeline ZeRO-2 + CPU 梯度：梯度分片下沉 CPU，GPU 只留 double buffer", 0),
         ("P2P Muon：正交化需完整矩阵，改为只 P2P 拉『自己负责更新的』分片", 0),
         ("消除全参 all-gather 的显存+通信双爆炸，按 model-chunk 流水隐藏", 1)],
        image="f_act_manager.png", layout="right",
        takeaway="削峰而非降总量——显存瓶颈是『单卡峰值』，不是『总量』。",
        note="【显存续 · 约2分钟】再讲三个系统级的手段，主题都是『把压力从 HBM 挪走』。右图是统一激活管理器的示意——"
             "一个张量可以路由到重计算、FP8 量化、offload、远程 offload 四种后端。\n\n"
             "第一，PP 激活再平衡，回收前面 1F1B 那个尾巴：暖机期越靠前的 stage 激活越多、越靠后越少，峰值不均容易 OOM。"
             "用 Mooncake Transfer Engine 把激活远程 offload 到别的 PP 卡的内存，把峰值削平。"
             "这里的关键认知是：显存瓶颈是『单卡峰值』而不是『总量』，把峰值从热卡挪到冷卡，等效于给你更大的可用空间。\n\n"
             "第二，Pipeline ZeRO-2 把梯度分片后下沉到 CPU，GPU 只留一个 double buffer。"
             "第三，P2P Muon——Muon 的正交化需要完整的参数矩阵，但优化器把参数分片了。朴素做法是全量 all-gather，"
             "显存和通信双爆炸。K3 改成每张卡只用点对点通信，拉回『自己负责更新的那些参数』的分片就行，"
             "消除了全参数缓冲区，通信量也降下来，再按 model-chunk 粒度流水隐藏。下一页专门画一下这个对比。")

    slide_content(
        "P2P Muon：定向通信 > 广播", "只有『要更新该参数的卡』才需要它的完整矩阵",
        [("约束：Newton–Schulz 是整矩阵多项式，必须完整矩阵，无法在分片上独立做", 0),
         ("朴素：全量 all-gather → 每卡多一份全参 buffer（显存）+ all-gather 全参（通信）", 0),
         ("K3：每卡只 P2P 拉回自己负责更新的参数分片（属主 → 更新者定向）", 0),
         ("临时 buffer 从 O(P) 降到 O(P/DP)，消除全参 buffer；通信流水隐藏", 1)],
        image="f_p2p_muon.png", layout="big",
        takeaway="想清楚『谁真正需要这份数据』，就能把 all-gather 降级为点对点。",
        note="【P2P Muon · 约1.5分钟】单独一页讲这个，因为它体现了一个很通用的思想。约束是：Newton–Schulz 正交化是整个矩阵的多项式运算，"
             "必须要完整矩阵，没法在参数分片上各算各的。朴素做法就是全量 all-gather，结果是每张卡都多存一份完整参数缓冲区（显存爆），"
             "还要把全部参数 all-gather 一遍（通信爆）。\n\n"
             "K3 的洞见很简单但很关键（看图对比）：其实只有『要更新某个参数的那张卡』才需要它的完整矩阵。"
             "所以 gather 应该是『属主卡定向发给更新卡』的点对点传输，而不是广播式的 all-gather。"
             "这样每张卡的临时缓冲区只需要它负责的那部分，从 O(P) 降到 O(P/DP)，全参缓冲区直接消失，通信也能按 model-chunk 流水藏起来。\n\n"
             "这个『想清楚谁真正需要数据，把广播降级成定向』的思路，其实对我们做很多分布式通信优化都有借鉴意义。")

    slide_content(
        "KDA 协同 + 多模态 encoder", "KCP 跨设备 · ViT 塞进流水气泡",
        [("KCP：线性注意力跨设备 CP 只传固定大小状态（softmax 要传随长度增长的 KV）", 0),
         ("vanilla『从 S=0 求和』对 KDA 不成立：转移矩阵 M_t 要作用在入态上", 1),
         ("KCP 把每段拆成两个本地可算量 → 前缀扫描恢复入态，一次固定大小 all-gather", 1),
         ("Bubble filling：把 ViT 前向/反向塞进 PP 流水气泡（承 K2.5 DEP，思路同 Optimus）", 0),
         ("ViT 可搬动、无跨 micro-batch 依赖 → 有效开销基本被隐藏", 1)],
        image="f_kcp.png", layout="right",
        takeaway="思想迁移：把 softmax 的 CP 用『转移矩阵分解+前缀扫描』搬到线性注意力。",
        note="【KCP + bubble filling · 约2分钟】最后两块 Infra。先是 KDA 的跨设备上下文并行 KCP（右图）。"
             "线性注意力做 CP 有个天然优势：跨卡只需要传固定大小的递归状态，而 softmax 注意力得传随长度增长的 KV 块。"
             "但有个坑：vanilla 线性注意力可以让每张卡『从 0 状态算本地、再求和』，这个对 KDA 不成立——"
             "因为 KDA 每步有个 token 相关的转移矩阵要作用在入态上，本地段的效果依赖于进入它的状态。\n\n"
             "KCP 的解法是把每段的效果拆成两个都能本地算的量：一个是累计转移矩阵，一个是从 0 生成的本地态；"
             "这两个量满足结合律，于是用『前缀扫描』就能恢复各卡的入态，全程只要一次固定大小的 all-gather。"
             "这其实是个漂亮的思想迁移——把 softmax 注意力的 CP 思路，用转移矩阵分解加前缀扫描，搬到了线性注意力上。\n\n"
             "最后是多模态 encoder：大图长视频让 ViT 计算量方差很大，串在关键路径上会拖慢流水。"
             "解法是 bubble filling——把 ViT 的前向反向计算塞进 PP 本来就空着的气泡里。ViT 是可搬动、跨 micro-batch 无依赖的计算，"
             "正好用来填气泡，所以它的有效开销基本被隐藏掉了。这个思路承自 K2.5 的 Decoupled Encoder Process，和 Optimus 一脉相承。")

    # ============ PART 4 · RL ============
    divider("后训练 / RL", "简化：三阶段 + 9 专家 + 蒸馏", "Part 4",
            note="【进入 Part 4 · 简化】RL 按要求简化，两页讲完：整体流程和几个关键机制，以及蒸馏与部署感知。")

    slide_content(
        "RL 全景：三阶段 + 9 个专家", "SFT → 按域×effort 做 RL → 蒸馏合并",
        [("SFT 冷启动 → 三大域各训一个专家 × 三档 effort = 9 个专家 → 合并", 0),
         ("三大域：general / general agents / coding agents", 1),
         ("partial rollout：λ 比例轨迹完成就暂停生成、让策略优化先走（避免长尾 straggler）", 0),
         ("长轨迹跨多迭代 → 数据 staleness → 逐 token 正则容忍极端 off-policy", 1),
         ("reasoning-effort RL：给 token 预算，超预算给 −1 → 得到 max/high/low 三档", 0),
         ("Agentic GRM：锦标赛式二元比较 + rubric 协议 + verbosity 预算防 reward hacking", 0)],
        image=None, layout="none",
        takeaway="不训单任务专用模型，而是『域 × effort』九宫格再蒸馏成一个。",
        note="【RL 全景 · 约2分钟】按要求简化。整体是三阶段：SFT 做冷启动，然后按域和 effort 做 RL 得到 9 个专家，最后蒸馏合并成一个模型。"
             "9 个专家 = 三大域（通用、通用 agent、编程 agent）乘以三档推理努力（low/high/max）。\n\n"
             "讲两个和 Infra 相关的机制。第一，partial rollout：不等所有轨迹都跑完，当 λ 比例的轨迹完成就暂停生成、让策略优化先走，"
             "避免个别超长轨迹拖成长尾。副作用是长轨迹会跨多个迭代，产生数据陈旧（staleness），这个靠逐 token 的正则来容忍极端 off-policy。"
             "第二，reasoning-effort RL：给每题一个 token 预算，超了就给 -1 惩罚，用课程从大预算退到小预算，得到 max/high/low 三档。"
             "不可验证的任务用 Agentic 生成式奖励模型，锦标赛式两两比较，还用长度预算防止模型靠『写得长』来刷分。")

    slide_content(
        "蒸馏合并 + 部署感知后训练", "把 9 个专家收敛成一个可部署模型",
        [("MOPD 多教师 on-policy 蒸馏：按域×effort 用对应教师，逐 token 稠密奖励", 0),
         ("稠密信号天然融入 RL 框架、支持 partial rollout", 1),
         ("Draft / EAGLE-3：把预训练 MTP 层微调成投机解码 draft", 0),
         ("输入融合第 1/4/末 AttnRes block 的低/中/高层特征", 1),
         ("LK loss：直接优化投机解码的『接受率』而非 KL 代理", 0),
         ("RL 环境：统一 white-box harness · 知识图谱任务合成 · kernel/AET/web dev", 1)],
        image=None, layout="none",
        takeaway="9 个专家最终蒸馏回一个模型，且 draft 复用了预训练的 MTP 层。",
        note="【蒸馏 + 部署 · 约1.5分钟】9 个专家怎么变回一个？用多教师 on-policy 蒸馏（MOPD）：训练时按当前的域和 effort，"
             "用对应的那个专家当老师，给学生一个逐 token 的稠密奖励信号，这个信号能天然融入 RL 框架，也支持 partial rollout。\n\n"
             "部署感知这块回收两个前面的点：一是把预训练时就带的 MTP 层微调成 EAGLE-3 的 draft 模型做投机解码，"
             "它的输入还融合了第 1、第 4、最后一个 AttnRes block 的低中高层特征——所以 Attention Residuals 的块结构在这里又被复用了；"
             "二是 draft 直接优化『接受率』（LK loss），而不是传统的 KL 代理，因为接受率才是投机解码真正的加速指标。\n\n"
             "RL 的环境和任务合成我就一句话带过：统一的 white-box harness 防止对单一 agent 框架过拟合，用知识图谱引导任务合成，"
             "覆盖 kernel 优化、自主执行、web 开发等。细节感兴趣可以看文档。")

    # ============ 收尾 ============
    slide_content(
        "评测速览", "紧随最强闭源，稳定领先其他开源",
        [("整体：紧随 Claude Fable 5 / GPT-5.6 Sol，稳定领先其他开源", 0),
         ("领先/SOTA：ProgramBench · SWE-Marathon · BrowseComp · OmniDocBench", 0),
         ("WebDev Arena 首个登顶的开源模型", 1),
         ("短板：research 级推理（HLE/CritPt）· 部分知识工作 Elo · hardened exploit", 0),
         ("成本效率：多个 coding/agentic 榜处于成本-效率前沿", 0),
         ("Case：GPU kernel 优化 · 从零写编译器 MiniTriton · 48h 自主设计推理芯片", 0)],
        image=None, layout="none",
        takeaway="开源前沿：能力接近闭源第一梯队，成本效率是亮点。",
        note="【评测 · 约1分钟】结果简单过一下。整体紧随最强的两个闭源模型，稳定领先其他开源；在 ProgramBench、SWE-Marathon、"
             "BrowseComp、OmniDocBench 这些上领先或 SOTA，还是首个登顶 WebDev Arena 的开源模型。"
             "短板也诚实列出来：research 级的难推理、部分知识工作、以及加固目标的端到端漏洞利用。"
             "一个亮点是成本效率——在多个 coding 和 agentic 榜上处于成本-效率前沿。"
             "case study 里那些也很能说明能力：优化 GPU kernel、从零写一个 Triton 类编译器、48 小时自主设计一个推理芯片原型。")

    slide_content(
        "演进脉络：每一代验证一个新机制", "K2 → K2.5 → Kimi Linear → AttnRes → K3",
        [("K2：全 MLA · 384 选 8 · Muon · SwiGLU · 128K", 0),
         ("K2.5：视觉 agentic · Decoupled Encoder Process", 0),
         ("Kimi Linear：提出 KDA · 3:1 KDA:MLA · 验证线性注意力可超全注意力", 0),
         ("Attention Residuals：深度维注意力 · 48B 规模验证", 0),
         ("Kimi K3：把 KDA + AttnRes + LatentMoE + Per-Head Muon + QAT + 全套 Infra 集成到 2.8T/1M", 0)],
        image=None, layout="none",
        takeaway="K3 是集大成——每一代先单独验证一个机制，再在 K3 集成。",
        note="【演进脉络 · 约1分钟】把 K3 放到时间线上看会更清楚它不是凭空出现的。K2 是全 MLA 的底子；"
             "K2.5 加了视觉 agentic 和解耦编码器；Kimi Linear 这篇专门提出并验证了 KDA、3:1 混合，证明线性注意力能超过全注意力；"
             "Attention Residuals 那篇在 48B 规模上验证了深度维注意力。K3 就是把这些前面各自验证过的机制，"
             "加上 Per-Head Muon、QAT 和一整套 Infra，集成到 2.8T、1M 的规模上。这种『每代先单独验证一个机制、再在旗舰上集成』"
             "的打法，本身也很值得我们学习。")

    slide_content(
        "收尾：五条可迁移的方法论", "给我们自己项目的启示",
        [("① 算法-系统协同是主线：改算法先想它对 kernel/通信/显存的影响", 0),
         ("KDA 加下界 → 对角块上 Tensor Core；MoonEP 完美均衡 → 静态形状+零拷贝+免同步", 1),
         ("② 把成熟维度的思想迁到新维度：attention 迁到深度维；CP 迁到线性注意力", 0),
         ("③ 有界化/归一化 = 极端稀疏+低精度下的稳定器（SiTU-GLU/RMSNorm/门下界）", 0),
         ("④ 比较训练技巧要『各自调到最优再比』（cosine vs WSD 的教训）", 0),
         ("⑤ 负载均衡从『启发式步长』升级为『有最优解的分位数』（QB + 直方图）", 0)],
        image=None, layout="none",
        takeaway="最好的系统优化，往往始于一个小的算法改动。",
        note="【收尾方法论 · 约2分钟】最后我把今天能带走的五条方法论收一下，这些对我们自己的项目都有借鉴意义。\n\n"
             "第一，也是贯穿全场的——算法和系统要协同设计。改算法之前先想清楚它对 kernel、通信、显存意味着什么。"
             "KDA 加下界让对角块能上 Tensor Core、MoonEP 的完美均衡换来静态形状零拷贝免同步，都是这个道理。"
             "第二，把成熟维度的思想迁到新维度：attention 从序列维迁到深度维（AttnRes）、CP 从 softmax 迁到线性注意力（KCP）。"
             "第三，有界化和归一化是极端稀疏加低精度下的稳定器——SiTU-GLU、到处的 RMSNorm、KDA 门下界。"
             "第四，比较训练技巧一定要各自调到最优再比，否则结论不可信。"
             "第五，负载均衡可以从启发式步长升级成有理论最优解的分位数，再用直方图做可扩展估计。\n\n"
             "一句话结束：最好的系统优化，往往始于一个小的算法改动。谢谢大家，欢迎提问。")

    # ---- Q&A ----
    s = prs.slides.add_slide(BLANK); bg(s, NAVY_D)
    hero = os.path.join(IMG, "cover_hero.png")
    im = Image.open(hero); w, h = im.size
    r = max(SW / w, SH / h); pw, ph = w * r, h * r
    s.shapes.add_picture(hero, Inches((SW - pw) / 2), Inches((SH - ph) / 2), Inches(pw), Inches(ph))
    ov = rect(s, 0, 0, SW, SH, NAVY_D); _set_alpha(ov, 40)
    tb, tf = textbox(s, 0.9, 2.9, 11.5, 2.3)
    add_para(tf, "Q & A · 谢谢！", size=44, color=WHITE, bold=True, first=True, space_after=10)
    add_para(tf, "备问：3:1 为何不全 KDA · NoPE 外推 · 896 选 16 如何训得动 · "
             "AttnRes 推理开销 · MoonEP 与 DeepEP 差异 · 与 DeepSeek 到底差在哪",
             size=14, color=RGBColor(0xCB, 0xD5, 0xE1))
    add_para(tf, "详见 DESIGN_NOTES.md 与 PRETRAIN_INFRA_DEEPDIVE.md", size=13,
             color=RGBColor(0x5E, 0xEA, 0xD4))
    _idx[0] += 1
    notes(s, "【Q&A】谢谢大家。这里列了几个我预期会被问到的问题，如果没人问我也可以主动补充。"
             "更细的推导和对比都在配套的两份文档里：DESIGN_NOTES 是全量思路，PRETRAIN_INFRA_DEEPDIVE 是 Infra 的深挖。")

    out = os.path.join(HERE, "kimi-k3-tech-share-v2.pptx")
    prs.save(out)
    print("saved:", out, "| slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    build()

