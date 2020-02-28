import torch
from torch import nn
import torch.nn.functional as F


def gem(x: torch.Tensor, p: float = 3, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps)
    kernel_size = (x.size(-2), x.size(-1))
    return F.avg_pool2d(x.pow(p), kernel_size).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p: float = 3, eps: float = 1e-6):
        super(GeM,self).__init__()
        
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    
    def __repr__(self):
        name = self.__class__.__name__
        p = self.p.data.tolist()[0]
        
        return f'{name}(p={p:.4f}, eps={self.eps})'
