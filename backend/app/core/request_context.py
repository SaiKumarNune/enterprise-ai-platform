"""Request-scoped correlation identifiers."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def validate_request_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


def set_request_id(value: str) -> None:
    _request_id.set(value)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def bind_request_id(value: str):
    return _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()
