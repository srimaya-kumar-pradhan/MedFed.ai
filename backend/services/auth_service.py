"""
Authentication & RBAC service.
Preserves the existing user database from clinical_portal/clinical_auth.py
but exposes a clean backend API surface.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("MEDFED_JWT_SECRET", "medfed-demo-secret-do-not-use-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8


# ────────────────────────────────────────────────────────────────────────────
# Roles
# ────────────────────────────────────────────────────────────────────────────
ROLES: Dict[str, Dict[str, Any]] = {
    "doctor": {
        "label": "Doctor",
        "description": "Run inference, view history, view explanations",
        "permissions": ["predict", "explain", "view_history", "view_model_info"],
    },
    "researcher": {
        "label": "Researcher",
        "description": "Start training, view training runs, evaluate models",
        "permissions": ["predict", "explain", "view_history", "view_model_info",
                        "start_training", "stop_training", "view_training_runs",
                        "view_model_registry", "view_nodes"],
    },
    "institution_admin": {
        "label": "Institution Admin",
        "description": "Deploy models, manage nodes, configure institution",
        "permissions": ["predict", "explain", "view_history", "view_model_info",
                        "start_training", "stop_training", "view_training_runs",
                        "view_model_registry", "deploy_model", "archive_model",
                        "view_nodes", "manage_nodes"],
    },
    "platform_admin": {
        "label": "Platform Admin",
        "description": "Full platform oversight",
        "permissions": ["*"],
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Demo user DB
# ────────────────────────────────────────────────────────────────────────────
USERS_DB: Dict[str, Dict[str, Any]] = {
    "dr.sharma@hospitala.com": {
        "username": "dr.sharma@hospitala.com",
        "full_name": "Dr. Arjun Sharma",
        "hospital_id": "Hospital_A",
        "role": "doctor",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
    },
    "dr.lee@hospitalc.com": {
        "username": "dr.lee@hospitalc.com",
        "full_name": "Dr. Soo-Min Lee",
        "hospital_id": "Hospital_C",
        "role": "doctor",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
    },
    "researcher@institution1.com": {
        "username": "researcher@institution1.com",
        "full_name": "Dr. Ravi Krishnan",
        "hospital_id": "Hospital_B",
        "role": "researcher",
        "password_hash": hashlib.sha256("research123".encode()).hexdigest(),
    },
    "admin@hospitala.com": {
        "username": "admin@hospitala.com",
        "full_name": "Dr. Meena Patel",
        "hospital_id": "Hospital_A",
        "role": "institution_admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
    },
    "platform@medfed.ai": {
        "username": "platform@medfed.ai",
        "full_name": "Platform Administrator",
        "hospital_id": "ALL",
        "role": "platform_admin",
        "password_hash": hashlib.sha256("platform123".encode()).hexdigest(),
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Token operations
# ────────────────────────────────────────────────────────────────────────────
def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Authentication
# ────────────────────────────────────────────────────────────────────────────
def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = USERS_DB.get(username)
    if not user:
        return None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash != user["password_hash"]:
        return None
    return user


def has_permission(user: Dict[str, Any], permission: str) -> bool:
    role = user.get("role", "")
    perms = ROLES.get(role, {}).get("permissions", [])
    return "*" in perms or permission in perms


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": user["username"],
        "full_name": user["full_name"],
        "hospital_id": user["hospital_id"],
        "role": user["role"],
        "role_label": ROLES.get(user["role"], {}).get("label", user["role"]),
    }
