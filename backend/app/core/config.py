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
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimensions: int
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float
    rag_context_chars: int
    rag_min_similarity: float
    rag_retrieve_multiplier: int
    agent_max_steps: int
    jwt_secret: str = field(repr=False)
    jwt_algorithm: str
    jwt_expire_minutes: int
    default_tenant_id: str

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
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "384")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
        rag_context_chars=int(os.getenv("RAG_CONTEXT_CHARS", "6000")),
        rag_min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.2")),
        rag_retrieve_multiplier=int(os.getenv("RAG_RETRIEVE_MULTIPLIER", "2")),
        agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        jwt_secret=os.getenv("JWT_SECRET", ""),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
        default_tenant_id=os.getenv("DEFAULT_TENANT_ID", ""),
    )


settings = _load_settings()
