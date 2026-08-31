"""
Authentication routes: /api/auth/login, /api/auth/me
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.deps import get_current_user
from backend.services import auth_service


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., description="Email or username")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    user = auth_service.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = auth_service.create_access_token({
        "sub": user["username"],
        "role": user["role"],
        "hospital_id": user["hospital_id"],
        "full_name": user["full_name"],
    })
    return LoginResponse(access_token=token, user=auth_service.public_user(user))


@router.get("/me")
def me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return auth_service.public_user(user)


@router.get("/roles")
def list_roles() -> Dict[str, Any]:
    """Expose role taxonomy (read-only) for the UI to render role-aware nav."""
    return auth_service.ROLES
