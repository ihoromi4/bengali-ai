import torch
from torch import nn


class OHEM(nn.CrossEntropyLoss):
    """ Online hard example mining. """
    
    def __init__(self, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, x, y):
        n_samples = x.shape[0]
        n_select = int(n_samples * self.ratio)
        
        probs = torch.gather(x, 1, y.unsqueeze(1)).squeeze()
        _, index = (-probs).topk(n_select)
        
        return super().forward(x[index], y[index])

