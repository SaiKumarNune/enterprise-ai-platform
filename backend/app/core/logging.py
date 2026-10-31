"""JSON request logging without secrets or document content."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.core.config import settings
from app.core.request_context import get_request_id

REDACTED_KEYS = {
    "authorization",
    "password",
    "passwd",
    "secret",
    "token",
    "jwt",
    "api_key",
    "apikey",
    "cookie",
    "set-cookie",
    "database_url",
    "postgres_password",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "service": settings.service_name,
            "event": getattr(record, "event", record.getMessage()[:80]),
            "logger": record.name,
        }
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for key, value in getattr(record, "fields", {}).items():
            if str(key).lower() in REDACTED_KEYS:
                continue
            payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(handler, logging.StreamHandler) and isinstance(handler.formatter, JsonLogFormatter) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    if os_env_json():
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def os_env_json() -> bool:
    import os

    raw = os.getenv("LOG_JSON")
    if raw is None:
        return settings.log_json
    return raw.strip().lower() in {"1", "true", "yes"}


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: object) -> None:
    logger.log(level, event, extra={"event": event, "fields": fields, "request_id": get_request_id()})
