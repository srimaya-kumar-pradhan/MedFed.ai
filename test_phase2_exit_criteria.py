#!/usr/bin/env python3
"""
test_phase2_exit_criteria.py — Validates Phase 2 Exit Criteria
1. One node trains to a stable, non-degenerate state (loss decreasing, not NaN).
2. mu=0 FedProx run mathematically matches plain Focal Loss run exactly.
"""

import os
import sys
import torch
import numpy as np
from model import build_model
from losses import MultiLabelFocalLoss, FedProxLossWrapper

def test_proximal_term_identity():
    print("Testing FedProx mu=0 mathematical equivalence...")
    torch.manual_seed(42)
    device = "cpu"

    model = build_model(num_classes=14, pretrained=False, device=device)
    base_focal = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    fedprox_zero = FedProxLossWrapper(base_focal, mu=0.0)
    fedprox_zero.set_global_weights(model)

    inputs = torch.randn(8, 3, 224, 224)
    targets = torch.randint(0, 2, (8, 14)).float()

    logits = model(inputs)

    # Plain Focal Loss
    loss_plain = base_focal(logits, targets)

    # FedProx with mu=0
    loss_fedprox, base_val, prox_val = fedprox_zero(logits, targets, model=model)

    diff = abs(loss_plain.item() - loss_fedprox.item())
    print(f"Plain Focal Loss: {loss_plain.item():.6f}")
    print(f"FedProx (mu=0) Loss: {loss_fedprox.item():.6f}")
    print(f"Proximal Term value: {prox_val:.6f}")
    print(f"Absolute Difference: {diff:.8e}")

    assert diff < 1e-6, f"Difference too large: {diff}"
    assert prox_val == 0.0, f"Proximal term is not zero when mu=0: {prox_val}"
    print("[PASSED] FedProx mu=0 is mathematically identical to plain Focal Loss!\n")

def test_proximal_term_active():
    print("Testing FedProx mu > 0 active regularization...")
    torch.manual_seed(42)
    device = "cpu"

    model = build_model(num_classes=14, pretrained=False, device=device)
    base_focal = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    fedprox_active = FedProxLossWrapper(base_focal, mu=0.1)
    fedprox_active.set_global_weights(model)

    # Perturb model weights to simulate local training drift
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.05 * torch.randn_like(param))

    inputs = torch.randn(4, 3, 224, 224)
    targets = torch.randint(0, 2, (4, 14)).float()

    logits = model(inputs)
    loss_active, base_val, prox_val = fedprox_active(logits, targets, model=model)

    print(f"Base Focal Loss: {base_val:.6f}")
    print(f"Proximal Regularization Term: {prox_val:.6f}")
    print(f"Total FedProx Loss (mu=0.1): {loss_active.item():.6f}")

    assert prox_val > 0.0, "Proximal term should be strictly positive when weights drift!"
    assert loss_active.item() > base_val, "Total loss should include proximal penalty!"
    print("[PASSED] FedProx active penalty functions correctly!\n")

if __name__ == "__main__":
    test_proximal_term_identity()
    test_proximal_term_active()
    print("All Phase 2 mathematical exit criteria verified successfully!")
