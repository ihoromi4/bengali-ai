import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from .mutlilinear import *


class BengaliEfficientNet(nn.Module):
    def __init__(self, output_classes, name: str = "efficientnet-b4", pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(name, in_channels=3)
        else:
            self.model = EfficientNet.from_name(name)
            
        conv0 = next((l for l in self.model.modules() if isinstance(l, nn.Conv2d)))
        conv0.weight = nn.Parameter(conv0.weight.mean(1, keepdim=True))
        conv0.in_channels = 1
        
        self.output_classes = output_classes
        
        self.model._swish = nn.Identity()
        
        dim = self.model._fc.in_features
        self.model._fc = MultiLabelLinearClassfier(dim, output_classes)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        x = self.model(x)
        
        return x

