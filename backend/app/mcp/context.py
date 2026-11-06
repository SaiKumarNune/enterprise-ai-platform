"""Request-scoped MCP execution context.

Identity is bound here from the authenticated FastAPI/LangGraph principal.
Tool arguments must never carry user IDs, tenant IDs, roles, or authorization flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services.authorization import Principal
from app.services.embeddings import EmbeddingBackend

FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "user_id",
        "tenant_id",
        "role",
        "authorized",
        "auth_user_id",
        "auth_tenant_id",
        "principal",
        "is_admin",
        "is_active",
        "owner_id",
    }
)


@dataclass(frozen=True)
class McpExecutionContext:
    principal: Principal
    db: Session | None = None
    embedder: EmbeddingBackend | None = None
