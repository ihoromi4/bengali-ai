import torch
from torch import nn
import pretrainedmodels
from .mutlilinear import *


class PretrainedModelsBengaliClassifier(nn.Module):
    def __init__(self, backbone, output_classes, pretrained='imagenet', one_channel: bool = True):
        super().__init__()
        
        self.output_classes = output_classes
        self.one_channel = one_channel
        
        if callable(backbone):
            self.backbone = backbone(pretrained)
        else:
            self.backbone = getattr(pretrainedmodels, backbone)(pretrained=pretrained)
            
        dim = self.backbone.last_linear.in_features
        self.backbone.last_linear = nn.Identity()
        
        if one_channel:
            conv0 = next((l for l in self.backbone.modules() if isinstance(l, nn.Conv2d)))
            conv0.weight = nn.Parameter(conv0.weight.mean(1, keepdim=True))
            conv0.in_channels = 1
        
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

