from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.core.config import settings

_HASHER = PasswordHasher()
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INSECURE_SECRETS = frozenset(
    {
        "",
        "changeme",
        "secret",
        "replace-with-a-long-random-secret-at-least-32-chars",
    }
)
_MAX_FAILED_LOGINS = 10
_LOCKOUT_SECONDS = 15 * 60
_failed_logins: dict[str, list[float]] = {}
_failed_lock = threading.Lock()


class AuthConfigError(RuntimeError):
    """JWT settings are missing or too weak."""


def normalize_email(value: str) -> str:
    return value.strip().lower()


def is_valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(value))) and len(normalize_email(value)) <= 320


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return _HASHER.verify(password_hash, password)
    except (VerifyMismatchError, ValueError, TypeError):
        return False


def hash_secret(value: str) -> str:
    secret, _algorithm, _expire_minutes = require_jwt_config()
    return hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def jwt_is_configured() -> bool:
    secret = settings.jwt_secret
    algorithm = settings.jwt_algorithm
    return (
        len(secret) >= 32
        and secret not in _INSECURE_SECRETS
        and algorithm == "HS256"
        and settings.jwt_expire_minutes > 0
    )


def require_jwt_config() -> tuple[str, str, int]:
    if not jwt_is_configured():
        raise AuthConfigError("JWT is not configured")
    return settings.jwt_secret, settings.jwt_algorithm, settings.jwt_expire_minutes


def create_access_token(user_id: UUID, expires_delta: timedelta | None = None) -> str:
    secret, algorithm, expire_minutes = require_jwt_config()
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta if expires_delta is not None else timedelta(minutes=expire_minutes))
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_access_token(token: str) -> UUID:
    secret, algorithm, _expire_minutes = require_jwt_config()
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError as exc:
        raise ValueError("invalid token") from exc
    subject = payload.get("sub")
    if not subject:
        raise ValueError("invalid token")
    try:
        return UUID(str(subject))
    except ValueError as exc:
        raise ValueError("invalid token") from exc


def login_is_locked(email: str) -> bool:
    key = normalize_email(email)
    cutoff = time.monotonic() - _LOCKOUT_SECONDS
    with _failed_lock:
        stamps = [item for item in _failed_logins.get(key, []) if item >= cutoff]
        _failed_logins[key] = stamps
        return len(stamps) >= _MAX_FAILED_LOGINS


def record_failed_login(email: str) -> None:
    key = normalize_email(email)
    with _failed_lock:
        stamps = _failed_logins.setdefault(key, [])
        stamps.append(time.monotonic())


def clear_failed_logins(email: str) -> None:
    key = normalize_email(email)
    with _failed_lock:
        _failed_logins.pop(key, None)
