"""Database engine/session configuration for V12.

The application uses DATABASE_URL so the same ORM/repository layer can run
against local SQLite or PostgreSQL without changing API code.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./platform_data.db").strip()


def _engine_kwargs(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {
            "connect_args": {"check_same_thread": False},
            "pool_pre_ping": True,
        }
    return {"pool_pre_ping": True, "pool_recycle": 1800}


def create_db_engine(database_url: str | None = None) -> Engine:
    url = (database_url or get_database_url()).strip()
    if not url:
        raise ValueError("DATABASE_URL cannot be empty.")
    return create_engine(url, future=True, **_engine_kwargs(url))


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """Enable FK enforcement for SQLite only."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


@contextmanager
def session_scope(session_factory: sessionmaker = SessionLocal) -> Iterator[Session]:
    """Commit on success and rollback on failure."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
