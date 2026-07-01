import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Solves the extreme class imbalance in 15m financial time-series data.
    Down-weights easy majority classes (Hold/Noise) and heavily penalizes 
    missed minority classes (Momentum Breakouts).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Alpha acts as a static class weight multiplier. 
        # e.g., torch.tensor([0.2, 0.8, 0.8]) -> heavily weights Long/Short vs Hold
        self.alpha = alpha 

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate standard Cross Entropy (internally applies LogSoftmax)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Calculate the probability of the true class
        pt = torch.exp(-ce_loss)
        
        # Apply the Focal weighting sequence: (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss