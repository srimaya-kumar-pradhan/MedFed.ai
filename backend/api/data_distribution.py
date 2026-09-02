"""
Data distribution endpoints. Returns per-hospital class-count distributions
and a computed skew assessment for the federated partition.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from backend.core import config
from backend.core.deps import get_current_user
from backend.services import auth_service

router = APIRouter(prefix="/api/data", tags=["data"])


def _load_summary() -> Dict[str, Any]:
    summary_path = config.PROJECT_ROOT / "partition_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Partition summary not found")
    return json.loads(summary_path.read_text())


def _ks_stat(counts: Dict[str, int]) -> float:
    """Kolmogorov-Smirnov statistic against a uniform distribution.

    Higher values indicate stronger non-IID skew.
    """
    vals = sorted(counts.values())
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    # Empirical CDF (sorted ascending)
    cdf = [sum(vals[: i + 1]) / total for i in range(n)]
    # Uniform CDF at each rank
    uni_cdf = [(i + 1) / n for i in range(n)]
    return max(abs(c - u) for c, u in zip(cdf, uni_cdf))


def _skew_verdict(ks: float) -> str:
    if ks < 0.12:
        return "normal"
    if ks < 0.25:
        return "moderate"
    return "high"


@router.get("/distribution")
def data_distribution(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_nodes"):
        raise HTTPException(status_code=403, detail="Role not authorized")

    summary = _load_summary()
    nodes_data = summary.get("nodes", {})
    all_classes = sorted(
        {cls for nd in nodes_data.values() for cls in nd.get("class_counts", {})}
    )

    node_distributions = {}
    ks_values = {}
    for node_id, node_data in nodes_data.items():
        cc = node_data.get("class_counts", {})
        node_distributions[node_id] = {
            "class_counts": {c: cc.get(c, 0) for c in all_classes},
            "total_samples": node_data.get("total_samples", 0),
            "train_samples": node_data.get("train_samples", 0),
            "val_samples": node_data.get("val_samples", 0),
            "test_samples": node_data.get("test_samples", 0),
        }
        ks_values[node_id] = round(_ks_stat(cc), 4)

    # Global class totals
    global_counts = {c: sum(nd.get("class_counts", {}).get(c, 0) for nd in nodes_data.values()) for c in all_classes}
    global_total = sum(global_counts.values())

    # Overall KS
    overall_ks = round(_ks_stat(global_counts), 4)

    verdict = _skew_verdict(overall_ks)

    return {
        "classes": all_classes,
        "nodes": node_distributions,
        "global_class_totals": global_counts,
        "global_total_samples": global_total,
        "ks_statistic": overall_ks,
        "skew_verdict": verdict,
        "ks_by_node": ks_values,
    }


@router.get("/distribution/{node_id}")
def node_distribution(node_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_nodes"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    if node_id not in config.HOSPITAL_DIRS:
        raise HTTPException(status_code=404, detail="Unknown hospital node")

    summary = _load_summary()
    nodes_data = summary.get("nodes", {})
    if node_id not in nodes_data:
        raise HTTPException(status_code=404, detail="Node not in partition summary")

    node_data = nodes_data[node_id]
    cc = node_data.get("class_counts", {})
    all_classes = sorted(cc.keys())

    return {
        "node_id": node_id,
        "class_counts": {c: cc.get(c, 0) for c in all_classes},
        "total_samples": node_data.get("total_samples", 0),
        "train_samples": node_data.get("train_samples", 0),
        "val_samples": node_data.get("val_samples", 0),
        "test_samples": node_data.get("test_samples", 0),
        "ks_statistic": round(_ks_stat(cc), 4),
        "skew_verdict": _skew_verdict(_ks_stat(cc)),
    }
