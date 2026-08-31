"""
Prediction service — orchestrates preprocessing, inference, and Grad-CAM
generation against the in-memory model. Read-only; never mutates state.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from backend.core import config
from backend.services.model_service import model_holder
from gradcam import GradCAM as _GradCAMEngine
from model import DEFAULT_CHEST_XRAY_CLASSES


_gradcam_engine: Optional[_GradCAMEngine] = None


def _get_gradcam_engine():
    global _gradcam_engine
    if _gradcam_engine is None:
        if model_holder.model is None:
            return None
        _gradcam_engine = _GradCAMEngine(model_holder.model)
    return _gradcam_engine


def predict(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """Run inference on a single PIL image and return structured results."""
    probs = model_holder.predict_proba(image)

    # Build full probability vector in fixed label order.
    label_probs = [
        {"label": cls, "probability": float(probs[i])}
        for i, cls in enumerate(DEFAULT_CHEST_XRAY_CLASSES)
    ]
    sorted_preds = sorted(label_probs, key=lambda x: x["probability"], reverse=True)
    top = sorted_preds[:top_k]

    return {
        "top_predictions": top,
        "all_predictions": label_probs,
        "model_version": model_holder.registry.get("current_version"),
        "model_round": model_holder.registry.get("round"),
    }


def explain(image: Image.Image, target_class: Optional[str] = None) -> Dict[str, Any]:
    """Return prediction + base64 Grad-CAM overlay for the specified class.

    If `target_class` is None, defaults to the highest-probability class.
    The choice is always surfaced in the response as `explained_class` so the
    frontend never has to guess which class the heatmap corresponds to.
    """
    # We need gradients enabled for Grad-CAM; use the dedicated method.
    probs, x = model_holder.predict_with_gradients(image)
    top_idx = int(np.argmax(probs))
    top_label = DEFAULT_CHEST_XRAY_CLASSES[top_idx]
    top_prob = float(probs[top_idx])

    # Resolve target class.
    if target_class is not None and target_class in DEFAULT_CHEST_XRAY_CLASSES:
        explained_idx = DEFAULT_CHEST_XRAY_CLASSES.index(target_class)
    else:
        explained_idx = top_idx
    explained_label = DEFAULT_CHEST_XRAY_CLASSES[explained_idx]
    explained_prob = float(probs[explained_idx])

    engine = _get_gradcam_engine()
    gradcam_b64: Optional[str] = None
    if engine is not None:
        heatmap, _, _ = engine.generate_heatmap(x, class_idx=explained_idx)
        overlay = engine.overlay_heatmap(image.convert("RGB"), heatmap, alpha=0.45)
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "predicted_label": top_label,
        "predicted_confidence": top_prob,
        "explained_class": explained_label,
        "explained_class_index": int(explained_idx),
        "explained_class_confidence": explained_prob,
        "gradcam_png_base64": gradcam_b64,
        "caption": (
            "Highlighted regions represent areas that contributed to the "
            "model's prediction for the selected class. This visualization "
            "is intended for model interpretability and does not replace "
            "clinical judgement."
        ),
        "disclaimer": (
            "This is an AI-generated explanation and does not constitute a "
            "clinical diagnosis. Clinical decision remains with the "
            "qualified healthcare professional."
        ),
    }
