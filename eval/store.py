"""Isolated PostgreSQL session factory for live evaluation only."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.database import Base
from tests.postgres_isolation import assert_safe_test_database


def isolated_session_factory(url: str) -> tuple[sessionmaker[Session], object]:
    safe = assert_safe_test_database(url)
    engine = create_engine(safe, poolclass=NullPool)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False), engine


def close_isolated(engine) -> None:
    Base.metadata.drop_all(engine)
    engine.dispose()


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
