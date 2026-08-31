"""
Model Service — loads the current model once at startup and serves it
in-memory for fast, read-only inference. Training must NEVER touch this
service except through the explicit deploy-model flow.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

from backend.core import config
from model import build_model, DEFAULT_CHEST_XRAY_CLASSES


# ────────────────────────────────────────────────────────────────────────────
# Persistent model registry (read once, written on deploy)
# ────────────────────────────────────────────────────────────────────────────
def _read_registry() -> Dict[str, Any]:
    if not config.REGISTRY_PATH.exists():
        return {}
    try:
        with open(config.REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_registry(registry: Dict[str, Any]) -> None:
    config.MODELS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _discover_existing_model() -> Optional[Path]:
    """Return the highest-priority model path that exists on disk."""
    candidates = [
        config.MODELS_CURRENT_DIR / "model.pth",
        config.MODELS_GLOBAL_DIR / "v1" / "model.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _initialize_registry(model_path: Path) -> Dict[str, Any]:
    """Read the actual checkpoint's stored metadata to populate the registry.

    No fake numbers — if the checkpoint lacks metrics they are reported as
    `null` so the UI can display N/A.
    """
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception as e:
        return {
            "current_version": "unknown",
            "path": str(model_path),
            "round": None,
            "status": "error_loading",
            "metrics": {"f1": None, "roc_auc": None, "loss": None},
            "error": str(e),
            "created_at": None,
        }

    if isinstance(ckpt, dict):
        round_ = ckpt.get("round")
        f1 = ckpt.get("global_macro_f1")
        auc = ckpt.get("global_roc_auc")
        loss = ckpt.get("global_loss")
        strategy = ckpt.get("strategy")
        privacy = ckpt.get("privacy")
    else:
        round_ = f1 = auc = loss = strategy = privacy = None

    return {
        "current_version": "global_v1",
        "path": str(model_path),
        "round": round_,
        "status": "ready",
        "metrics": {
            "f1": float(f1) if f1 is not None else None,
            "roc_auc": float(auc) if auc is not None else None,
            "loss": float(loss) if loss is not None else None,
        },
        "metadata": {
            "architecture": "densenet121",
            "num_classes": config.NUM_CLASSES,
            "task": "multi_label_classification",
            "study_type": "chest_xray",
            "dataset": "NIH Chest X-ray (prototype, 14-class)",
            "strategy": strategy,
            "privacy": privacy,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(model_path.stat().st_mtime)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ────────────────────────────────────────────────────────────────────────────
# Singleton holder
# ────────────────────────────────────────────────────────────────────────────
class _ModelHolder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model: Optional[nn.Module] = None
        self.transform: Optional[T.Compose] = None
        self.registry: Dict[str, Any] = {}
        self.device = "cpu"
        self._loaded_at: Optional[float] = None
        self._load_error: Optional[str] = None

    def initialize(self) -> None:
        """Load the current model exactly once at application startup."""
        with self._lock:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            model_path = _discover_existing_model()
            if model_path is None:
                self._load_error = (
                    "No persisted model found in models/global/. "
                    "Please run an explicit federated training run via the "
                    "Research Portal to produce a model."
                )
                self.registry = {
                    "current_version": None,
                    "path": None,
                    "round": None,
                    "status": "missing",
                    "metrics": {"f1": None, "roc_auc": None, "loss": None},
                    "metadata": {
                        "architecture": "densenet121",
                        "num_classes": config.NUM_CLASSES,
                        "task": "multi_label_classification",
                        "study_type": "chest_xray",
                    },
                    "created_at": None,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _write_registry(self.registry)
                return

            try:
                self.model = build_model(
                    num_classes=config.NUM_CLASSES,
                    pretrained=False,
                    device=self.device,
                )
                ckpt = torch.load(str(model_path), map_location=self.device, weights_only=True)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt, strict=False)
                self.model.eval()
                # Keep model parameters trainable in case Grad-CAM needs to
                # backprop. Forward paths wrap their own `torch.no_grad()`.

                self.transform = T.Compose([
                    T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                ])

                self.registry = _initialize_registry(model_path)
                _write_registry(self.registry)
                self._loaded_at = time.time()
                self._load_error = None
            except Exception as e:  # noqa: BLE001
                self._load_error = str(e)
                self.model = None

    def reload_from(self, version: str) -> None:
        """Hot-swap to a different model version. Used by deploy endpoint."""
        with self._lock:
            version_path = config.MODELS_GLOBAL_DIR / version / "model.pth"
            if not version_path.exists():
                raise FileNotFoundError(f"Model version '{version}' not found at {version_path}")
            self._swap_to_path(version_path, version=version)

    def _swap_to_path(self, model_path: Path, version: str) -> None:
        new_model = build_model(
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            device=self.device,
        )
        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            new_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            new_model.load_state_dict(ckpt, strict=False)
        new_model.eval()
        # Keep parameters trainable so Grad-CAM can backprop.
        for p in new_model.parameters():
            p.requires_grad_(True)

        # Atomic-ish swap via lock
        self.model = new_model
        self._loaded_at = time.time()

        # Update the current/ symlink-equivalent
        config.MODELS_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
        target = config.MODELS_CURRENT_DIR / "model.pth"
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        try:
            target.write_bytes(model_path.read_bytes())
        except OSError:
            pass

        self.registry = _initialize_registry(model_path)
        self.registry["current_version"] = version
        self.registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_registry(self.registry)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            available = self.model is not None
            loaded_for_seconds = (
                int(time.time() - self._loaded_at) if self._loaded_at else 0
            )
            return {
                "available": available,
                "device": self.device,
                "loaded_for_seconds": loaded_for_seconds,
                "error": self._load_error,
                "registry": self.registry,
            }

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Run inference on a single PIL image. Returns sigmoid probs of shape [num_classes]."""
        with self._lock:
            if self.model is None or self.transform is None:
                raise RuntimeError("Model unavailable. Please contact the system administrator.")
            x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            return probs

    def predict_with_gradients(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        """Run inference with gradient enabled, returning (probs, input_tensor) for Grad-CAM."""
        with self._lock:
            if self.model is None or self.transform is None:
                raise RuntimeError("Model unavailable.")
            x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            # Forward with grad enabled for Grad-CAM.
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            return probs, x


# Module-level singleton — same instance imported by every consumer.
model_holder = _ModelHolder()
