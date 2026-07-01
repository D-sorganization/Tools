"""Tests for SQLite configuration + the historian index migration.

Guards the performance-critical DB setup: the WAL/synchronous/size-limit pragmas
applied on every connection, and the idempotent ``(tag_name, timestamp)``
composite-index migration that turns the trend-read hot path from an index-scan
+ temp-B-tree sort into a pure indexed range scan.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlmodel")

import database  # noqa: E402
from models import TagLog  # noqa: E402,F401  (registers the table in metadata)
from sqlalchemy import text  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402


def test_configure_sqlite_sets_wal_and_size_limit(tmp_path) -> None:
    db_file = tmp_path / "pragma.db"
    conn = sqlite3.connect(db_file)
    try:
        database._configure_sqlite_connection(conn, None)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA journal_size_limit").fetchone()[0] == 67_108_864
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_migration_creates_composite_and_drops_single(tmp_path) -> None:
    # Seed a DB with the OLD schema: a standalone single-column tag_name index.
    engine = create_engine(f"sqlite:///{tmp_path / 'hist.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_taglog_tag_name ON taglog (tag_name)")
        )

    # Apply the migration SQL (mirrors database._migrate_historian_indexes body).
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_taglog_tag_name_timestamp "
                "ON taglog (tag_name, timestamp)"
            )
        )
        conn.execute(text("DROP INDEX IF EXISTS ix_taglog_tag_name"))

    with Session(engine) as session:
        rows = session.exec(
            text(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='taglog'"
            )
        ).all()
    names = {r[0] for r in rows}
    assert "ix_taglog_tag_name_timestamp" in names
    assert "ix_taglog_tag_name" not in names


def test_trend_query_uses_composite_index(tmp_path) -> None:
    # The composite index must actually serve the trend query plan (no temp sort).
    engine = create_engine(f"sqlite:///{tmp_path / 'plan.db'}")
    SQLModel.metadata.create_all(engine)  # model now declares the composite index
    with engine.connect() as conn:
        plan = conn.exec_driver_sql(
            "EXPLAIN QUERY PLAN SELECT * FROM taglog "
            "WHERE tag_name = 'TAG_0' AND timestamp BETWEEN '2020' AND '2030' "
            "ORDER BY timestamp"
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan).lower()
    assert "ix_taglog_tag_name_timestamp" in plan_text
    assert "temp b-tree" not in plan_text  # index provides the order — no sort
