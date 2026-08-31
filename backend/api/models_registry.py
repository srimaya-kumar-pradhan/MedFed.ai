"""
Model registry routes: list all versions, get one, deploy.
Deploy requires institution_admin or platform_admin.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.core.deps import get_current_user
from backend.services import auth_service, registry_service


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_model_registry"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    reg = registry_service.get_registry()
    return reg


@router.get("/{version}")
def get_model(version: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "view_model_registry"):
        raise HTTPException(status_code=403, detail="Role not authorized")
    info = registry_service.get_version(version)
    if not info:
        raise HTTPException(status_code=404, detail="Model version not found")
    return info


class DeployRequest(BaseModel):
    confirm: bool = False


@router.post("/{version}/deploy")
def deploy_model(
    version: str,
    req: DeployRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if not auth_service.has_permission(user, "deploy_model"):
        raise HTTPException(status_code=403, detail="Role not authorized to deploy models")
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Re-submit with confirm=true to deploy this version.",
        )
    try:
        result = registry_service.set_current(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
