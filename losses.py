#!/usr/bin/env python3
"""
losses.py — MedFed AI Loss Functions
Implements:
1. MultiLabelFocalLoss: Addresses heavy class imbalance in medical image findings.
2. FedProxLossWrapper: Incorporates proximal regularization term (mu/2) * ||w - w_global||^2
   for non-IID federated drift mitigation. When mu=0, exactly recovers plain Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelFocalLoss(nn.Module):
    """
    Numerically stable Multi-Label Focal Loss operating on raw logits.

    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is sigmoid probability for positive/negative class.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes] (unnormalized logits)
        targets: [batch_size, num_classes] (binary 0 or 1 labels)
        """
        targets = targets.float()

        # Standard Binary Cross Entropy with Logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )

        # Calculate p_t (probability of the true class)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t + 1e-8, self.gamma)

        # Alpha weighting factor
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_loss = alpha_t * modulating_factor * bce_loss
        else:
            focal_loss = modulating_factor * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FedProxLossWrapper:
    """
    Wraps a base loss function (e.g. MultiLabelFocalLoss) with the FedProx
    proximal term:
        Total_Loss = Base_Loss + (mu / 2) * ||w_local - w_global||^2

    When mu = 0 or w_global is None, proximal term is exactly 0.0.
    """
    def __init__(self, base_loss_fn, mu=0.0):
        self.base_loss_fn = base_loss_fn
        self.mu = mu
        self.global_weights = None

    def set_global_weights(self, model):
        """
        Caches a deep copy / snapshot of global model parameters at the start of local training round.
        """
        self.global_weights = [param.detach().clone() for param in model.parameters()]

    def __call__(self, logits, targets, model=None):
        base_loss = self.base_loss_fn(logits, targets)

        if self.mu > 0.0 and self.global_weights is not None and model is not None:
            proximal_term = 0.0
            for w_local, w_glob in zip(model.parameters(), self.global_weights):
                proximal_term += torch.sum((w_local - w_glob) ** 2)

            total_loss = base_loss + (self.mu / 2.0) * proximal_term
            return total_loss, base_loss.item(), ((self.mu / 2.0) * proximal_term).item()

        return base_loss, base_loss.item(), 0.0

if __name__ == "__main__":
    print("Testing MultiLabelFocalLoss & FedProxLossWrapper...")
    logits = torch.randn(4, 14, requires_grad=True)
    targets = torch.randint(0, 2, (4, 14)).float()

    focal_fn = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_fn(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    assert not torch.isnan(loss) and loss.item() > 0

    # Test FedProx shim with mu=0
    prox_shim_zero = FedProxLossWrapper(focal_fn, mu=0.0)
    loss_z, base_z, prox_z = prox_shim_zero(logits, targets)
    print(f"mu=0 => Total: {loss_z.item():.4f}, Base: {base_z:.4f}, Prox: {prox_z:.4f}")
    assert prox_z == 0.0

    print("All loss tests passed!")
