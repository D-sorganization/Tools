import logging
from collections.abc import Generator
from typing import Any

from settings import P1AMSettings, get_settings
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

# Set up logging conforming to user guidelines
logger = logging.getLogger("dcs_backend.database")

DB_FILE = "dcs_scada.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"


def _synchronous_mode(settings: P1AMSettings | None = None) -> str:
    """Resolve PRAGMA synchronous from P1AM_SQLITE_SYNCHRONOUS (default NORMAL)."""
    return str((settings or get_settings()).sqlite_synchronous)


# Connect args needed for SQLite threaded async access
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _configure_sqlite_connection(dbapi_connection: Any, _record: Any) -> None:
    """Apply performance pragmas to EVERY new SQLite connection.

    ``synchronous`` and ``busy_timeout`` are per-connection settings, so they
    must be set on each connect — not once at startup. WAL journaling plus
    ``synchronous=NORMAL`` makes the 10 Hz historian write path ~7x cheaper than
    the default rollback-journal + FULL-sync, and lets readers (trend/export
    queries) run without blocking the writer.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA synchronous={_synchronous_mode()}")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def init_db() -> None:
    """Initialize database tables using SQLModel metadata.

    Raises:
        Exception: If connection or table creation fails.
    """
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_session() -> Generator[Session, None, None]:
    """Generate a database session for request scopes.

    Yields:
        Session: Active SQLModel session.
    """
    with Session(engine) as session:
        yield session
