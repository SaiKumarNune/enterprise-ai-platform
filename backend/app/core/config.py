"""Minimal application settings using the Python standard library."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_title: str
    service_name: str


def _load_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "Enterprise AI Platform"),
        service_name=os.getenv("SERVICE_NAME", "enterprise-ai-platform"),
    )


settings = _load_settings()
