from torch import nn
import pretrainedmodels
from bengaliai.models import pretrained_classifier


class MultiLabelLinearClassfier(nn.Module):
    def __init__(self, in_dim: int, out_dims: list):
        super().__init__()
        
        modules = [nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, dim)) for dim in out_dims]
        self.linears = nn.ModuleList(modules)

    def forward(self, x):
        return [linear(x) for linear in self.linears]


class PretrainedModelsBengaliClassifier(pretrained_classifier.PretrainedModelsBengaliClassifier):
    def __init__(self, backbone, output_classes, pretrained='imagenet', one_channel: bool = True):
        super().__init__()
        
        self.output_classes = output_classes
        self.one_channel = one_channel
        
        if callable(backbone):
            self.backbone = backbone(pretrained)
        else:
            self.backbone = getattr(pretrainedmodels, backbone)(pretrained=pretrained)
            
        dim = self.backbone.last_linear.in_features
        self.backbone.avg_pool = pretrained_classifier.GlobalMaxPooling()
        self.backbone.last_linear = nn.Identity()
        
        if one_channel:
            conv0 = next((l for l in self.backbone.modules() if isinstance(l, nn.Conv2d)))
            conv0.weight = nn.Parameter(conv0.weight.mean(1, keepdim=True))
            conv0.in_channels = 1
        
        self.classifier = MultiLabelLinearClassfier(2048, output_classes)

