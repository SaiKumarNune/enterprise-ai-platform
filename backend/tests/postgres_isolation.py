"""Shared skip/guard helpers for isolated PostgreSQL tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

SKIP_ISOLATED_POSTGRES = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason=(
        "Isolated PostgreSQL tests require TEST_DATABASE_URL pointing at a dedicated "
        "test database; skipped to protect the development database."
    ),
)


def require_isolated_url(url: str) -> str:
    lowered = url.lower()
    if "enterprise_ai" in lowered and "test" not in lowered:
        raise RuntimeError(
            "Refusing TEST_DATABASE_URL that looks like the development database. "
            "Use a dedicated database whose name includes 'test'."
        )
    return url


@pytest.fixture()
def postgres_client() -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    url = require_isolated_url(TEST_DATABASE_URL)
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
