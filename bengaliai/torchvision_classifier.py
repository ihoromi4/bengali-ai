import torch
from torch import nn
from .mutlilinear import *


class TorchVisionBengaliClassifier(nn.Module):
    def __init__(self, backbone: callable, output_classes, one_channel: bool = False, pretrained: bool = True):
        super().__init__()
        
        self.output_classes = output_classes
        self.one_channel = one_channel
        
        self.backbone = backbone(pretrained)
        self.backbone.classifier = nn.Identity()
        
        if one_channel:
            conv0 = next((l for l in backbone.features.modules() if isinstance(l, nn.Conv2d)))
            conv0.weight = nn.Parameter(conv0.weight.sum(1, keepdim=True))
        
        test_tensor = torch.zeros((1, 3, 128, 128), device=self.device)
        dim = self.backbone(test_tensor).shape[1]
        self.classifier = MultiLabelLinearClassfier(dim, output_classes)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        if not self.one_channel:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.backbone(x)
        x = self.classifier(x)
        
        return x

