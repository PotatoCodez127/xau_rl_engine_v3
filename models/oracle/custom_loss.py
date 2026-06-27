import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Focal Loss for imbalanced classification.
        alpha: 1D Tensor of weights for each class (e.g., giving 'Long' and 'Short' more weight than 'Hold').
        gamma: Focusing parameter. Higher values strictly penalize easily classified examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate standard Cross Entropy Loss (unreduced)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # Calculate the probability of the true class (pt)
        pt = torch.exp(-ce_loss)

        # Apply the focal modification: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
