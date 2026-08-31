"""
FastAPI dependencies for authentication, RBAC, and current-user resolution.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from backend.services import auth_service


async def get_current_user(authorization: Annotated[Optional[str], Header()] = None) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(None, 1)[1].strip()
    payload = auth_service.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = payload.get("sub")
    user = auth_service.USERS_DB.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_permission(permission: str):
    async def _checker(user: Annotated[Dict[str, Any], Depends(get_current_user)] = None) -> Dict[str, Any]:
        if not auth_service.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' does not have permission '{permission}'",
            )
        return user
    return _checker
