import logging
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

# Set up logging conforming to user guidelines
logger = logging.getLogger("dcs_backend.database")

DB_FILE = "dcs_scada.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Connect args needed for SQLite threaded async access
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


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
