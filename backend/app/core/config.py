"""Application settings loaded from the project-root .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_title: str
    service_name: str
    postgres_db: str
    postgres_user: str
    postgres_password: str = field(repr=False)
    postgres_host: str
    postgres_port: int
    upload_dir: Path
    max_upload_bytes: int

    @property
    def database_url(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _resolve_upload_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "Enterprise AI Platform"),
        service_name=os.getenv("SERVICE_NAME", "enterprise-ai-platform"),
        postgres_db=os.getenv("POSTGRES_DB", "enterprise_ai"),
        postgres_user=os.getenv("POSTGRES_USER", "enterprise_user"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        postgres_host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        upload_dir=_resolve_upload_dir(os.getenv("UPLOAD_DIR", "uploads")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
    )


settings = _load_settings()
