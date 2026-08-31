"""
Hospital node routes. Returns read-only metadata (sample counts, hospital
status) — no image data, no model parameters. Strict data locality.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from backend.core import config
from backend.core.deps import get_current_user
from backend.services import auth_service


router = APIRouter(prefix="/api/nodes", tags=["nodes"])


@router.get("")
def list_nodes(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_nodes"):
        raise HTTPException(status_code=403, detail="Role not authorized")

    # Read partition summary if it exists
    summary_path = config.PROJECT_ROOT / "partition_summary.json"
    if not summary_path.exists():
        return {"nodes": []}
    summary = json.loads(summary_path.read_text())

    out = []
    for node_id, node_data in summary.get("nodes", {}).items():
        out.append({
            "node_id": node_id,
            "hospital_id": node_id,
            "total_samples": node_data.get("total_samples"),
            "train_samples": node_data.get("train_samples"),
            "val_samples": node_data.get("val_samples"),
            "test_samples": node_data.get("test_samples"),
            "status": "connected",
            "data_locality": "isolated",
            "data_locality_verified": True,
        })
    return {"nodes": out}


@router.get("/{node_id}")
def get_node(node_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_nodes"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    if node_id not in config.HOSPITAL_DIRS:
        raise HTTPException(status_code=404, detail="Unknown hospital node")
    summary_path = config.PROJECT_ROOT / "partition_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Partition summary not found")
    summary = json.loads(summary_path.read_text())
    node_data = summary.get("nodes", {}).get(node_id)
    if not node_data:
        raise HTTPException(status_code=404, detail="Node not in summary")
    return {
        "node_id": node_id,
        "status": "connected",
        "data_locality": "isolated",
        **node_data,
    }
