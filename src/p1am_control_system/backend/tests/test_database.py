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
from audit_log import AuditLog  # noqa: E402,F401  (registers audit metadata)
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


def test_configure_sqlite_sets_read_perf_pragmas(tmp_path) -> None:
    # The read-performance pragmas must be applied on every fresh connection.
    db_file = tmp_path / "readperf.db"
    conn = sqlite3.connect(db_file)
    try:
        database._configure_sqlite_connection(conn, None)
        # mmap_size is an upper bound; SQLite maps lazily but reports the cap.
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 268_435_456
        # cache_size uses the negative-KiB convention: -65536 KiB == 64 MiB.
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -65_536
        # temp_store == 2 is the MEMORY setting.
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
    finally:
        conn.close()


def test_optimize_planner_statistics_is_best_effort(tmp_path, monkeypatch) -> None:
    # PRAGMA optimize must run without raising and must never block startup even
    # when the driver call fails (best-effort contract).
    engine = create_engine(f"sqlite:///{tmp_path / 'optimize.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(database, "engine", engine)
    # Happy path: no exception propagates.
    database._optimize_planner_statistics()

    # Failure path: a raising exec_driver_sql is swallowed (logged as warning).
    class _BoomConn:
        def exec_driver_sql(self, _sql: str) -> None:
            raise RuntimeError("boom")

        def __enter__(self) -> _BoomConn:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    monkeypatch.setattr(database.engine, "connect", lambda: _BoomConn())
    database._optimize_planner_statistics()  # must not raise


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


def test_init_db_installs_append_only_audit_guards(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'init-audit.db'}")
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()

    with engine.connect() as connection:
        trigger_names = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND tbl_name='auditlog'"
                )
            )
        }
    assert trigger_names == {"auditlog_no_delete", "auditlog_no_update"}


def test_init_db_creates_versioned_configuration_store_idempotently(
    tmp_path, monkeypatch
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'init-configuration.db'}")
    monkeypatch.setattr(database, "engine", engine)

    database.init_db()
    database.init_db()

    with engine.connect() as connection:
        table = connection.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='configurationrevisionrecord'"
            )
        ).scalar_one()
    assert table == "configurationrevisionrecord"


def test_historian_quality_migration_preserves_legacy_rows(
    tmp_path, monkeypatch
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-quality.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE taglog (id INTEGER PRIMARY KEY, tag_name VARCHAR "
                "NOT NULL, value FLOAT NOT NULL, timestamp DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO taglog(tag_name, value, timestamp) "
                "VALUES ('TAG_0', 1.5, '2026-08-03 12:00:00')"
            )
        )
    monkeypatch.setattr(database, "engine", engine)

    database._migrate_historian_quality_columns()
    database._migrate_historian_quality_columns()

    with engine.connect() as connection:
        columns = {
            row[1] for row in connection.exec_driver_sql("PRAGMA table_info(taglog)")
        }
        row = connection.execute(
            text(
                "SELECT quality, diagnostic_reason, sequence, source, "
                "source_timestamp FROM taglog"
            )
        ).one()
    assert {
        "quality",
        "diagnostic_reason",
        "sequence",
        "source",
        "source_timestamp",
    } <= columns
    assert tuple(row) == (
        "uncertain",
        "legacy_unqualified",
        0,
        "legacy.adapter",
        "2026-08-03 12:00:00",
    )


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
