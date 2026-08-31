"""
Registry service — tracks every model version that has ever been saved.
Backed by models/metadata/model_registry.json. No fake metrics.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core import config
from backend.services import model_service


def _read_registry() -> Dict[str, Any]:
    if not config.REGISTRY_PATH.exists():
        return {"versions": [], "current_version": None}
    try:
        with open(config.REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"versions": [], "current_version": None}


def _write_registry(reg: Dict[str, Any]) -> None:
    config.MODELS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def discover_existing_versions() -> List[Dict[str, Any]]:
    """Walk models/global/ and emit a per-version descriptor for every subdir that has a model.pth."""
    versions: List[Dict[str, Any]] = []
    if not config.MODELS_GLOBAL_DIR.exists():
        return versions
    for entry in sorted(config.MODELS_GLOBAL_DIR.iterdir()):
        if not entry.is_dir() or entry.name == "current":
            continue
        ckpt_path = entry / "model.pth"
        if not ckpt_path.exists():
            continue
        info = _inspect_checkpoint(ckpt_path)
        info["version"] = entry.name
        versions.append(info)
    return versions


def _inspect_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    try:
        ckpt = torch_load_safe(ckpt_path)
    except Exception as e:  # noqa: BLE001
        return {
            "path": str(ckpt_path),
            "round": None,
            "metrics": {"f1": None, "roc_auc": None, "loss": None},
            "metadata": {"strategy": None, "privacy": None},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ckpt_path.stat().st_mtime)),
            "error": str(e),
        }
    if not isinstance(ckpt, dict):
        return {
            "path": str(ckpt_path),
            "round": None,
            "metrics": {"f1": None, "roc_auc": None, "loss": None},
            "metadata": {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ckpt_path.stat().st_mtime)),
        }
    return {
        "path": str(ckpt_path),
        "round": ckpt.get("round"),
        "metrics": {
            "f1": ckpt.get("global_macro_f1"),
            "roc_auc": ckpt.get("global_roc_auc"),
            "loss": ckpt.get("global_loss"),
        },
        "metadata": {
            "strategy": ckpt.get("strategy"),
            "privacy": ckpt.get("privacy"),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ckpt_path.stat().st_mtime)),
    }


def torch_load_safe(path: Path) -> Any:
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def get_registry() -> Dict[str, Any]:
    """Return canonical registry doc. Reconciles with on-disk discovery."""
    disk_versions = discover_existing_versions()
    reg = _read_registry()
    reg["versions"] = disk_versions
    reg.setdefault("current_version", None)
    return reg


def register_version(version: str, source_path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Copy a model checkpoint into models/global/{version}/model.pth and record it."""
    target_dir = config.MODELS_GLOBAL_DIR / version
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "model.pth"
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    target_path.write_bytes(src.read_bytes())

    info = _inspect_checkpoint(target_path)
    info["version"] = version
    info["extra"] = extra or {}
    reg = _read_registry()
    reg.setdefault("versions", [])
    reg["versions"] = [v for v in reg["versions"] if v.get("version") != version]
    reg["versions"].append(info)
    _write_registry(reg)
    return info


def set_current(version: str) -> Dict[str, Any]:
    """Atomically deploy a version as the live model used for inference."""
    target = config.MODELS_GLOBAL_DIR / version / "model.pth"
    if not target.exists():
        raise FileNotFoundError(f"Model version '{version}' not found at {target}")
    model_service.model_holder.reload_from(version)
    reg = _read_registry()
    reg["current_version"] = version
    reg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_registry(reg)
    return model_service.model_holder.status()


def list_versions() -> List[Dict[str, Any]]:
    return discover_existing_versions()


def get_version(version: str) -> Optional[Dict[str, Any]]:
    target = config.MODELS_GLOBAL_DIR / version / "model.pth"
    if not target.exists():
        return None
    info = _inspect_checkpoint(target)
    info["version"] = version
    return info
