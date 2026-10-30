from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import create_engine, pool

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
backend_path = str(BACKEND_DIR)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.core.config import settings
from app.core.database import Base
from app.models import AgentWorkflow, Document, DocumentChunk, User  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL from migration scripts without opening a database connection."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations to PostgreSQL using a short-lived connection."""
    connectable = create_engine(settings.database_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
