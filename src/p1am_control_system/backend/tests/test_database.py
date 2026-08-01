"""Tests for SQLite configuration + the historian index migration.

Guards the performance-critical DB setup: the WAL/synchronous/size-limit pragmas
applied on every connection, and the idempotent ``(tag_name, timestamp)``
composite-index migration that turns the trend-read hot path from an index-scan
+ temp-B-tree sort into a pure indexed range scan.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

import database  # noqa: E402
from models import TagLog  # noqa: E402,F401  (registers the table in metadata)
from sqlalchemy import text  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402


def test_db_file_is_anchored_to_the_backend_package(monkeypatch) -> None:
    # The historian path must NOT depend on the process CWD: a relative
    # "dcs_scada.db" forks the DB into one file per launch directory (and drops
    # a stray untracked copy at the repo root during a test run).
    monkeypatch.delenv("P1AM_DB_PATH", raising=False)
    resolved = Path(database._resolve_db_file())
    assert resolved.is_absolute()
    assert resolved.name == database.DB_FILENAME
    assert resolved.parent == Path(database.__file__).resolve().parent


def test_db_path_env_override_is_honoured(tmp_path, monkeypatch) -> None:
    # Deployments that keep the historian on separate storage override the path.
    target = tmp_path / "historian" / "bench.db"
    target.parent.mkdir()
    monkeypatch.setenv("P1AM_DB_PATH", str(target))
    assert Path(database._resolve_db_file()) == target.resolve()


def test_database_url_is_a_wellformed_absolute_sqlite_url() -> None:
    # Windows drive paths must use posix separators inside the URL, otherwise
    # SQLAlchemy sees escapes instead of a drive-absolute path.
    assert "\\" not in database.DATABASE_URL
    assert database.DATABASE_URL == f"sqlite:///{Path(database.DB_FILE).as_posix()}"
    assert Path(database.DB_FILE).is_absolute()


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
