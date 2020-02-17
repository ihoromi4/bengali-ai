import torch
from torch import nn
import pretrainedmodels
from .mutlilinear import *


class PretrainedModelsBengaliClassifier:
    def __init__(self, backbone, output_classes, pretrained='imagenet'):
        super().__init__()
        
        self.output_classes = output_classes
        
        if callable(backbone):
            self.backbone = backbone(pretrained)
        else:
            self.backbone = getattr(pretrainedmodels, backbone)(pretrained=pretrained)
            
        dim = self.backbone.last_linear.in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = MultiLabelLinearClassfier(dim, output_classes)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        
        x = self.backbone(x)
        x = torch.sum(x, dim=(-1, -2))
        x = self.classifier(x)
        
        return x

