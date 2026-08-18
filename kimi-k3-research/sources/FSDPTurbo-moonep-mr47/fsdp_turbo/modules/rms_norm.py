import torch
from torch import nn

from fsdp_turbo.ops.rms_norm import rms_norm


class RMSNorm(nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        nn.Module.__init__(self) 
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
