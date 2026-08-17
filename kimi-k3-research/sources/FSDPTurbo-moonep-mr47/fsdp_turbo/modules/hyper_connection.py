import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4UnweightedRMSNorm

from fsdp_turbo.ops.sinkhorn import hc_split_sinkhorn


class HyperConnection(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps

        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())

        mixes = F.linear(flat, self.fn.float())

        pre, post, comb = hc_split_sinkhorn(
            mixes=mixes,
            hc_scale=self.scale,
            hc_base=self.base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=self.eps
        )

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)

        return post, comb, collapsed
