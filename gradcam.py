#!/usr/bin/env python3
"""
gradcam.py — MedFed AI Grad-CAM Visual Explanation Module
Provides explainable AI overlays for medical diagnostics on DenseNet121.
Mandatory constraint: No black-box output — every prediction must have Grad-CAM overlay.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# Standard 14 Pathology Classes
PATHOLOGY_CLASSES = [
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

class GradCAM:
    """
    Grad-CAM implementation tailored for MedFedDenseNet (DenseNet121).
    Extracts gradients and activations from the final denseblock/transition layer.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to the final convolutional layer of DenseNet121 features
            self.target_layer = model.densenet.features.denseblock4.denselayer16.conv2
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate 2D Grad-CAM heatmap for a given input tensor and target class index.

        Args:
            input_tensor: Tensor of shape [1, 3, H, W]
            class_idx: Integer index of target pathology class (0 to 13).
                       If None, uses the highest probability predicted class.

        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
            predicted_class_idx: Index of the explained class
            predicted_prob: Sigmoid probability of the explained class
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor) # [1, num_classes]
        probs = torch.sigmoid(logits)

        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()

        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        # Gradients: [1, C, H_feat, W_feat]
        # Activations: [1, C, H_feat, W_feat]
        gradients = self.gradients[0]     # [C, H_feat, W_feat]
        activations = self.activations[0] # [C, H_feat, W_feat]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2)) # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain only features with positive influence on class
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam.cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to input tensor dimensions
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_pil.resize((input_w, input_h), Image.BILINEAR)
        heatmap = np.array(cam_resized) / 255.0

        predicted_prob = probs[0, class_idx].item()
        return heatmap, class_idx, predicted_prob

    def overlay_heatmap(self, original_image, heatmap, alpha=0.45, colormap='jet'):
        """
        Overlays heatmap on top of original RGB/grayscale PIL Image.

        Args:
            original_image: PIL Image or numpy array [H, W, 3] (0-255)
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
            alpha: Transparency factor for heatmap (0.0 to 1.0)
            colormap: Matplotlib colormap name ('jet', 'turbo', 'inferno')

        Returns:
            blended_image: PIL Image of the composite overlay
        """
        if isinstance(original_image, Image.Image):
            orig_img = original_image.convert('RGB')
        else:
            orig_img = Image.fromarray(original_image.astype(np.uint8)).convert('RGB')

        w, h = orig_img.size
        # Resize heatmap if dimensions differ
        if heatmap.shape != (h, w):
            heat_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap = np.array(heat_pil.resize((w, h), Image.BILINEAR)) / 255.0

        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        color_heatmap = cmap(heatmap)[:, :, :3] # [H, W, 3], in range [0, 1]
        color_heatmap = (color_heatmap * 255).astype(np.uint8)

        orig_np = np.array(orig_img)
        blended = (orig_np * (1.0 - alpha) + color_heatmap * alpha).astype(np.uint8)
        return Image.fromarray(blended)

if __name__ == "__main__":
    from model import build_model
    print("Testing GradCAM module...")
    device = "cpu"
    model = build_model(num_classes=14, pretrained=False, device=device)
    gradcam = GradCAM(model)

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    heatmap, cls_idx, prob = gradcam.generate_heatmap(dummy_input, class_idx=0)
    print(f"Heatmap generated. Shape: {heatmap.shape}, Range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")
    print(f"Target Class: {PATHOLOGY_CLASSES[cls_idx]}, Prob: {prob:.4f}")

    blended = gradcam.overlay_heatmap(dummy_img, heatmap)
    print(f"Blended image size: {blended.size}")
    print("GradCAM sanity check passed successfully!")
