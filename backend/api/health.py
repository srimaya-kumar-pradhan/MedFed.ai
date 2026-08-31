"""
Health, model status, current model info.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from backend.core.deps import get_current_user
from backend.services.model_service import model_holder


router = APIRouter(tags=["system"])


@router.get("/api/health")
def health() -> Dict[str, Any]:
    """Liveness probe. No auth required. Returns 200 if the API process is up."""
    return {
        "status": "ok",
        "service": "MedFed AI",
        "version": "2.0.0",
        "study_type": "chest_xray",
        "dataset": "NIH Chest X-ray (prototype)",
    }


@router.get("/api/model/status")
def model_status(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Detailed model load status, including any startup errors."""
    return model_holder.status()


@router.get("/api/model/current")
def model_current(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the registry record for the live inference model."""
    status = model_holder.status()
    if not status["available"]:
        return {
            "available": False,
            "message": "Model unavailable. Please contact the system administrator.",
        }
    return {
        "available": True,
        **status,
    }
