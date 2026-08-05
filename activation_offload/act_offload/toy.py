"""A tiny self-contained pre-norm Transformer stack for tests/benchmarks.

Deterministic (no dropout), plain PyTorch, so it runs anywhere and keeps the
offload library independent of Megatron for validation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, Dh]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, Dh]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(attn)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class TransformerStack(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
