#!/usr/bin/env python3
"""
clinical_auth.py — JWT Authentication & RBAC for MedFed Clinical Portal
Implements hospital-scoped tenancy, role-based access control, and session management.
"""

import os
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# JWT implementation using python-jose (already installed)
# ─────────────────────────────────────────────────────────────────────────────
from jose import JWTError, jwt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("MEDFED_JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8-hour sessions

# ─────────────────────────────────────────────────────────────────────────────
# Role definitions
# ─────────────────────────────────────────────────────────────────────────────
ROLES = {
    "clinician": {
        "label": "Clinician",
        "description": "Analyze images",
        "permissions": ["upload_image", "validate_image", "analyze_image",
                        "view_results", "confirm_decision", "override_decision",
                        "view_history", "generate_report"]
    },
    "researcher": {
        "label": "Researcher / Institution",
        "description": "Train / contribute data",
        "permissions": ["view_datasets", "start_local_training",
                        "view_contribution_dashboard", "register_institution",
                        "view_privacy_status"]
    },
    "admin": {
        "label": "Admin",
        "description": "Infrastructure & node management",
        "permissions": ["view_federated_status", "view_node_health",
                        "view_privacy_dashboard", "manage_users",
                        "view_aggregation_metrics"]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Demo user database (in production, replace with hashed DB + bcrypt)
# ─────────────────────────────────────────────────────────────────────────────
USERS_DB = {
    "dr.sharma@hospitalA.com": {
        "username": "dr.sharma@hospitalA.com",
        "full_name": "Dr. Arjun Sharma",
        "hospital_id": "Hospital_A",
        "role": "clinician",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest()
    },
    "dr.patel@hospitalA.com": {
        "username": "dr.patel@hospitalA.com",
        "full_name": "Dr. Meena Patel",
        "hospital_id": "Hospital_A",
        "role": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest()
    },
    "researcher@institution1.com": {
        "username": "researcher@institution1.com",
        "full_name": "Dr. Ravi Krishnan",
        "hospital_id": "Hospital_B",
        "role": "researcher",
        "password_hash": hashlib.sha256("research123".encode()).hexdigest()
    },
    "dr.lee@hospitalC.com": {
        "username": "dr.lee@hospitalC.com",
        "full_name": "Dr. Soo-Min Lee",
        "hospital_id": "Hospital_C",
        "role": "clinician",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest()
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Token operations
# ─────────────────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Authentication operations
# ─────────────────────────────────────────────────────────────────────────────
def authenticate_user(username: str, password: str) -> dict | None:
    """
    Authenticate against demo user database.
    Returns user dict on success, None on failure.
    """
    user = USERS_DB.get(username)
    if not user:
        return None
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash != user["password_hash"]:
        return None
    return user

def has_permission(user: dict, permission: str) -> bool:
    """Check if a user has a specific permission."""
    role = user.get("role", "")
    role_perms = ROLES.get(role, {}).get("permissions", [])
    return permission in role_perms

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization for Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state(st):
    """Initialize all auth-related session state variables."""
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "session_mode" not in st.session_state:
        st.session_state.session_mode = None
    if "patient_cases" not in st.session_state:
        st.session_state.patient_cases = {}  # For in-memory case tracking

def login(st, username: str, password: str):
    """Authenticate and create session."""
    user = authenticate_user(username, password)
    if user is None:
        return False, "Invalid username or password."

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"],
        "hospital_id": user["hospital_id"],
        "full_name": user["full_name"]
    })
    st.session_state.auth_token = token
    st.session_state.user_info = user
    st.session_state.session_mode = None
    return True, "Login successful."

def logout(st):
    """Clear session state."""
    st.session_state.auth_token = None
    st.session_state.user_info = None
    st.session_state.session_mode = None

def get_current_user(st) -> dict | None:
    """Return current logged-in user or None."""
    if st.session_state.auth_token and st.session_state.user_info:
        payload = decode_access_token(st.session_state.auth_token)
        if payload:
            return st.session_state.user_info
    return None

if __name__ == "__main__":
    # Simple unit test
    u = authenticate_user("dr.sharma@hospitalA.com", "demo123")
    assert u is not None
    assert has_permission(u, "analyze_image")
    assert not has_permission(u, "view_node_health")
    token = create_access_token({"sub": u["username"], "role": u["role"]})
    payload = decode_access_token(token)
    assert payload["sub"] == u["username"]
    print("All auth unit tests passed.")
