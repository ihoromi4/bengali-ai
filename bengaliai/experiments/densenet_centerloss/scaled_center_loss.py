import torch
import torch.nn as nn


class ScaledCenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu: bool = True):
        super(ScaledCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        device = 'cuda' if use_gpu else None
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, device=device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.shape[0]
        sample_centers = self.centers[labels]
        dist = torch.sum((x - sample_centers)**2, dim=-1)
        
        with torch.no_grad():
            a = torch.sum(x**2, dim=-1)
            b = torch.sum(sample_centers**2, dim=-1)
            length = torch.max(a, b)
            
        normdist = dist / length
        dist = torch.clamp(dist, min=1e-12, max=1e+12)
        
        return torch.sum(dist) / batch_size

