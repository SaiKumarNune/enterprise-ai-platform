"""Guards and fixtures for an isolated PostgreSQL test database.

Never drop, truncate, or migrate the development database named enterprise_ai.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import quote_plus

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
FORBIDDEN_DATABASES = frozenset({"enterprise_ai", "postgres", "template0", "template1"})

SKIP_ISOLATED_POSTGRES = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason=(
        "Isolated PostgreSQL tests require TEST_DATABASE_URL pointing at a dedicated "
        "database whose name ends with _test; skipped to protect enterprise_ai."
    ),
)


def database_name(url: str) -> str:
    parsed = make_url(url)
    name = parsed.database
    if not name:
        raise RuntimeError("database URL must include a database name")
    return name


def require_isolated_url(url: str, *, development_db: str | None = None) -> str:
    name = database_name(url)
    lowered = name.lower()
    if lowered in FORBIDDEN_DATABASES or lowered == "enterprise_ai":
        raise RuntimeError("Refusing to use the development or system PostgreSQL database for tests")
    if not lowered.endswith("_test"):
        raise RuntimeError("TEST_DATABASE_URL database name must end with _test")
    expected_dev = (development_db or settings.postgres_db or "enterprise_ai").lower()
    if lowered == expected_dev:
        raise RuntimeError("TEST_DATABASE_URL must not use the same database as the application")
    return url


def assert_safe_test_database(url: str) -> str:
    return require_isolated_url(url)


@pytest.fixture()
def postgres_client() -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    url = assert_safe_test_database(TEST_DATABASE_URL)
    engine = create_engine(url, poolclass=NullPool)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()


def build_admin_url() -> str:
    user = quote_plus(settings.postgres_user)
    password = quote_plus(settings.postgres_password)
    return (
        f"postgresql+psycopg://{user}:{password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/postgres"
    )


def ensure_isolated_database(name: str = "enterprise_ai_acl_test") -> str:
    """Create the isolated test database if missing. Does not drop enterprise_ai."""
    if not name.endswith("_test") or name.lower() in FORBIDDEN_DATABASES:
        raise RuntimeError("refusing to create a non-test database")
    admin = create_engine(build_admin_url(), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": name},
            ).scalar()
            if not exists:
                connection.execute(
                    text(f'CREATE DATABASE "{name}" TEMPLATE template0')
                )
    finally:
        admin.dispose()
    user = quote_plus(settings.postgres_user)
    password = quote_plus(settings.postgres_password)
    url = (
        f"postgresql+psycopg://{user}:{password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{name}"
    )
    assert_safe_test_database(url)
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    finally:
        engine.dispose()
    return url
