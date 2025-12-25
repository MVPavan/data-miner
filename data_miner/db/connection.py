"""
Database connection and session management.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlmodel import Session, create_engine, SQLModel

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/data_miner")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)


def create_tables() -> None:
    """Create all tables in the database."""
    # Import models to register them with SQLModel metadata
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
