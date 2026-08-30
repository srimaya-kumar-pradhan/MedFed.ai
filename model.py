#!/usr/bin/env python3
"""
model.py — MedFed AI DenseNet121 Architecture
Customized DenseNet121 for multi-label medical diagnostic classification.
Supports 14-class NIH Chest X-ray and parameterized for future domain transfer (Brain MRI).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

# Standard 14 Pathology Classes for Chest X-ray
DEFAULT_CHEST_XRAY_CLASSES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

class MedFedDenseNet(nn.Module):
    """
    DenseNet121 backbone with customized multi-label classification head.
    Outputs raw logits; sigmoid activation is used for multi-label probabilities.
    """
    def __init__(self, num_classes=14, pretrained=True, dropout_rate=0.2):
        super(MedFedDenseNet, self).__init__()
        self.num_classes = num_classes

        # Load DenseNet121 backbone
        if pretrained:
            weights = DenseNet121_Weights.DEFAULT
            self.densenet = models.densenet121(weights=weights)
        else:
            self.densenet = models.densenet121(weights=None)

        num_features = self.densenet.classifier.in_features # 1024

        # Replace classifier head with clinical multi-label architecture
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass returning raw logits (shape: [batch_size, num_classes]).
        """
        return self.densenet(x)

    def predict_proba(self, x):
        """
        Returns probabilities via Sigmoid activation for multi-label inference.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_features_layer(self):
        """
        Returns the target feature extraction layer for Grad-CAM visualization.
        """
        return self.densenet.features

    def get_last_conv_layer(self):
        """
        Returns the final convolutional layer of denseblock4 for Grad-CAM.
        """
        return self.densenet.features.denseblock4.denselayer16.conv2

def build_model(num_classes=14, pretrained=True, device="cpu"):
    """
    Factory helper to instantiate and place model on target compute device.
    """
    model = MedFedDenseNet(num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing MedFedDenseNet instantiation on {device}...")
    model = build_model(num_classes=14, pretrained=False, device=device)
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    print(f"Logits shape: {logits.shape} (Expected: [2, 14])")
    print(f"Probs range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    print("Model test passed successfully!")
