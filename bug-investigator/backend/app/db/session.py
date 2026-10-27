from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def configure(engine) -> None:
    """Point the app at a specific engine (used by tests)."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    get_engine()
    assert _SessionLocal is not None
    with _SessionLocal() as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Standalone session for background jobs and CLIs."""
    get_engine()
    assert _SessionLocal is not None
    with _SessionLocal() as session:
        yield session
