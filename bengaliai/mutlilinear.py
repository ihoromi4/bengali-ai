from torch import nn


class MultiLabelLinearClassfier(nn.Module):
    def __init__(self, in_dim: int, out_dims: list):
        super().__init__()
        
        modules = [nn.Linear(in_dim, dim) for dim in out_dims]
        self.linears = nn.ModuleList(modules)

    def forward(self, x):
        return [linear(x) for linear in self.linears]

