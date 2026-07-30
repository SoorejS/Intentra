import os
import time
import hashlib
import json
import base64
from typing import Optional

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "intentra_secret_key_change_in_production_2026")

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = "intentra_salt_v3"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed version."""
    return hash_password(plain_password) == hashed_password

def create_token(user_id: int, email: str) -> str:
    """Create a lightweight signed token for user session."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": int(time.time()) + (86400 * 30) # 30 days
    }
    payload_str = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    signature = hashlib.sha256((payload_str + SECRET_KEY).encode("utf-8")).hexdigest()
    return f"{payload_str}.{signature}"

def decode_token(token: str) -> Optional[dict]:
    """Decode and verify signed token."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_str, signature = parts
        expected_sig = hashlib.sha256((payload_str + SECRET_KEY).encode("utf-8")).hexdigest()
        if signature != expected_sig:
            return None
        payload = json.loads(base64.b64decode(payload_str.encode("utf-8")).decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
