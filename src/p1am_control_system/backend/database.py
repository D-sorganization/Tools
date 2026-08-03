import logging
from collections.abc import Generator
from typing import Any

from audit_log import install_append_only_guards
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
    queries) run without blocking the writer. The read-performance pragmas
    (``mmap_size``, ``cache_size``, ``temp_store``) are likewise per-connection
    and only speed up reads / bound memory — they do not weaken durability.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA synchronous={_synchronous_mode()}")
        cursor.execute("PRAGMA busy_timeout=5000")
        # Cap the WAL file: a long-held read transaction (e.g. a slow trend/export
        # query) can grow the WAL indefinitely; without a size limit the file
        # never shrinks on disk after a checkpoint. 64 MiB keeps it bounded on the
        # Pi's storage while leaving ample headroom for burst writes.
        cursor.execute("PRAGMA journal_size_limit=67108864")
        # --- Read-performance pragmas (per-connection, safe for a single writer) ---
        # These only affect read throughput / memory use; they never relax the
        # durability guarantees that matter for a safety-critical historian.
        #
        # mmap_size=256 MiB: memory-map the DB so hot trend/export range scans read
        # pages straight from the page cache instead of via pread() syscalls. This
        # is an upper bound, not a reservation — SQLite maps lazily up to this size.
        cursor.execute("PRAGMA mmap_size=268435456")
        # cache_size=-65536: negative => KiB, so 65536 KiB = 64 MiB of page cache
        # per connection. A larger cache keeps the (tag_name, timestamp) index and
        # recently-scanned leaf pages resident, cutting I/O on repeated trend reads.
        cursor.execute("PRAGMA cache_size=-65536")
        # temp_store=MEMORY (2): keep transient B-trees / materializations (sorts,
        # temp tables from aggregate/export queries) in RAM rather than spilling to
        # the Pi's slower storage.
        cursor.execute("PRAGMA temp_store=MEMORY")
    finally:
        cursor.close()


def init_db() -> None:
    """Initialize database tables using SQLModel metadata, then migrate.

    ``create_all`` only creates *missing* tables — it does not add indexes to a
    pre-existing table. So the historian-read performance index and the WAL
    reclaim are applied explicitly here (idempotently) so an already-populated
    ``dcs_scada.db`` gets them on the next boot.

    Raises:
        Exception: If connection or table creation fails.
    """
    try:
        SQLModel.metadata.create_all(engine)
        install_append_only_guards(engine)
        _migrate_historian_quality_columns()
        _migrate_historian_indexes()
        _optimize_planner_statistics()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def _migrate_historian_quality_columns() -> None:
    """Add signal provenance columns without discarding legacy historian rows."""
    from sqlalchemy import text

    definitions = {
        "source_timestamp": "DATETIME",
        "quality": "VARCHAR NOT NULL DEFAULT 'uncertain'",
        "diagnostic_reason": "VARCHAR DEFAULT 'legacy_unqualified'",
        "sequence": "INTEGER NOT NULL DEFAULT 0",
        "source": "VARCHAR NOT NULL DEFAULT 'legacy.adapter'",
    }
    with engine.begin() as connection:
        columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(taglog)")
        }
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(
                    text(f"ALTER TABLE taglog ADD COLUMN {name} {definition}")
                )
        connection.execute(
            text(
                "UPDATE taglog SET source_timestamp = timestamp "
                "WHERE source_timestamp IS NULL"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_taglog_quality ON taglog (quality)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_taglog_sequence ON taglog (sequence)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_taglog_source ON taglog (source)")
        )


def _migrate_historian_indexes() -> None:
    """Ensure the composite trend-query index exists and reclaim WAL space.

    The trend/export/Data-Explorer read paths all filter
    ``WHERE tag_name = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp``.
    A composite ``(tag_name, timestamp)`` index turns that from an index-scan +
    temp-B-tree sort (measured ~3.9 s on a 6 M-row DB) into a pure indexed range
    scan (~0.58 s, ~7x). The composite fully covers ``tag_name``-only lookups, so
    the redundant single-column index is dropped to cut write overhead. A one-off
    ``wal_checkpoint(TRUNCATE)`` reclaims a WAL that a prior long reader bloated.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_taglog_tag_name_timestamp "
                "ON taglog (tag_name, timestamp)"
            )
        )
        conn.execute(text("DROP INDEX IF EXISTS ix_taglog_tag_name"))
    # Checkpoint outside the transaction; TRUNCATE shrinks the WAL file on disk.
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as exc:  # pragma: no cover - best-effort reclaim
            logger.warning("WAL truncate checkpoint skipped: %s", exc)


def _optimize_planner_statistics() -> None:
    """Refresh the query-planner statistics once at startup (best-effort).

    ``PRAGMA optimize`` lets SQLite record/update ``sqlite_stat*`` data so the
    planner picks the right index for the trend/export hot path — especially
    important right after the ``(tag_name, timestamp)`` migration adds a new
    index the planner has never seen stats for. It is cheap, idempotent, and
    non-essential: wrap it so a failure only warns and never blocks boot of the
    safety-critical controller.
    """
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("PRAGMA optimize")
        except Exception as exc:  # pragma: no cover - best-effort stats refresh
            logger.warning("PRAGMA optimize skipped: %s", exc)


def get_session() -> Generator[Session, None, None]:
    """Generate a database session for request scopes.

    Yields:
        Session: Active SQLModel session.
    """
    with Session(engine) as session:
        yield session
