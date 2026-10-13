"""Contract and lint tests for TimescaleDB plant historian schema migrations.

H3 / #5542:
1. Migrations are contiguously numbered, each idempotent-marked, and reference the
   declared table/column names.
2. Hypertable, compression, one continuous aggregate, and retention each present
   exactly once.
3. No credentials in SQL; roles/passwords come from env at deploy time.
4. Down/rollback notes present in migrations.
5. Integration test (skipped by default) applies migrations twice without error.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# Repository root: tests/scada/ -> root
ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "deploy" / "historian" / "timescale"

# Regex patterns for validation
_MIGRATION_FILE_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")
_PASSWORD_OR_SECRET_RE = re.compile(
    r"(?:password\s*=\s*['\"][^'\"]+['\"]"
    r"|identified\s+by\s+['\"][^'\"]+['\"]"
    r"|postgres(?:ql)?://[^:]+:[^@]+@"
    r"|secret\s*=\s*['\"][^'\"]+['\"])",
    re.IGNORECASE,
)


def _get_migration_files() -> list[Path]:
    """Return all .sql migration files sorted by prefix number."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    files = [
        p
        for p in MIGRATIONS_DIR.iterdir()
        if p.is_file() and _MIGRATION_FILE_RE.match(p.name)
    ]
    return sorted(
        files,
        key=lambda p: int(_MIGRATION_FILE_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )


def test_migrations_directory_exists_and_has_files() -> None:
    """deploy/historian/timescale/ must exist and contain migration files."""
    assert MIGRATIONS_DIR.is_dir(), f"Directory not found: {MIGRATIONS_DIR}"
    files = _get_migration_files()
    assert len(files) >= 4, (
        f"Expected at least 4 migration files in {MIGRATIONS_DIR}, found {len(files)}"
    )


def test_migrations_are_contiguously_numbered_starting_at_one() -> None:
    """Migrations must be contiguously numbered starting at 001."""
    files = _get_migration_files()
    assert files, "No migration files found"
    numbers = [
        int(_MIGRATION_FILE_RE.match(f.name).group(1))  # type: ignore[union-attr]
        for f in files
    ]
    expected = list(range(1, len(numbers) + 1))
    assert numbers == expected, (
        f"Migration numbers must be {expected}, but got {numbers}"
    )


def test_each_migration_is_idempotent() -> None:
    """Every migration file must contain idempotency guards."""
    files = _get_migration_files()
    assert files, "No migration files found"
    for file_path in files:
        sql = file_path.read_text(encoding="utf-8")
        assert (
            "IF NOT EXISTS" in sql.upper()
            or "if_not_exists" in sql.lower()
            or "DO $$" in sql
        ), (
            f"Migration {file_path.name} lacks idempotency markers "
            "(IF NOT EXISTS or if_not_exists => TRUE)"
        )


def test_tag_sample_table_and_column_contract() -> None:
    """Migration 001 must declare tag_sample with required schema and quality."""
    files = _get_migration_files()
    assert files, "No migration files found"
    sql = files[0].read_text(encoding="utf-8")

    # Table declaration
    assert re.search(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+tag_sample", sql, re.IGNORECASE
    ), "tag_sample table must be created with CREATE TABLE IF NOT EXISTS"

    # Columns: ts TIMESTAMPTZ, tag_id, value DOUBLE PRECISION,
    # quality SMALLINT DEFAULT 192
    assert re.search(r"\bts\b\s+TIMESTAMPTZ", sql, re.IGNORECASE), (
        "tag_sample must contain 'ts TIMESTAMPTZ' column"
    )
    assert re.search(r"\btag_id\b", sql, re.IGNORECASE), (
        "tag_sample must contain 'tag_id' column"
    )
    assert re.search(r"\bvalue\b\s+DOUBLE\s+PRECISION", sql, re.IGNORECASE), (
        "tag_sample must contain 'value DOUBLE PRECISION' column"
    )
    assert re.search(r"\bquality\b\s+SMALLINT.*?DEFAULT\s+192", sql, re.IGNORECASE), (
        "tag_sample must contain 'quality SMALLINT DEFAULT 192' column"
    )


def test_core_timescale_features_present_exactly_once() -> None:
    """Hypertable, compression, CAGG, and retention each present exactly once."""
    files = _get_migration_files()
    assert files, "No migration files found"

    all_sql = {f.name: f.read_text(encoding="utf-8") for f in files}

    # 1. Hypertable on tag_sample
    hypertable_files = [
        name
        for name, sql in all_sql.items()
        if re.search(
            r"create_hypertable\s*\(\s*['\"]tag_sample['\"]",
            sql,
            re.IGNORECASE,
        )
    ]
    assert len(hypertable_files) == 1, (
        "create_hypertable for tag_sample must appear exactly once, "
        f"found in: {hypertable_files}"
    )

    # 2. Continuous aggregate on tag_sample
    cagg_files = [
        name
        for name, sql in all_sql.items()
        if re.search(r"timescaledb\.continuous", sql, re.IGNORECASE)
        or re.search(
            r"add_continuous_aggregate_policy\s*\(\s*['\"]tag_sample_1m['\"]",
            sql,
            re.IGNORECASE,
        )
    ]
    assert len(cagg_files) == 1, (
        "Continuous aggregate for tag_sample must appear in exactly one migration, "
        f"found in: {cagg_files}"
    )

    # 3. Compression policy on tag_sample
    compression_files = [
        name
        for name, sql in all_sql.items()
        if re.search(
            r"add_compression_policy\s*\(\s*['\"]tag_sample['\"]",
            sql,
            re.IGNORECASE,
        )
    ]
    assert len(compression_files) == 1, (
        "add_compression_policy for tag_sample must appear exactly once, "
        f"found in: {compression_files}"
    )

    # 4. Retention policy on tag_sample
    retention_files = [
        name
        for name, sql in all_sql.items()
        if re.search(
            r"add_retention_policy\s*\(\s*['\"]tag_sample['\"]",
            sql,
            re.IGNORECASE,
        )
    ]
    assert len(retention_files) == 1, (
        "add_retention_policy for tag_sample must appear exactly once, "
        f"found in: {retention_files}"
    )


def test_no_hardcoded_credentials_in_sql() -> None:
    """Migrations must not contain plaintext passwords or credentials."""
    files = _get_migration_files()
    assert files, "No migration files found"
    for file_path in files:
        sql = file_path.read_text(encoding="utf-8")
        matches = _PASSWORD_OR_SECRET_RE.findall(sql)
        assert not matches, (
            f"Possible hardcoded credentials found in {file_path.name}: {matches}"
        )


def test_down_or_rollback_notes_present() -> None:
    """Migrations or README must document rollback / down procedures."""
    files = _get_migration_files()
    assert files, "No migration files found"

    readme_path = MIGRATIONS_DIR / "README.md"
    readme_has_rollback = (
        readme_path.is_file()
        and "rollback" in readme_path.read_text(encoding="utf-8").lower()
    )

    files_with_rollback = [
        f.name
        for f in files
        if "rollback" in f.read_text(encoding="utf-8").lower()
        or "drop" in f.read_text(encoding="utf-8").lower()
        or "remove_" in f.read_text(encoding="utf-8").lower()
    ]

    assert readme_has_rollback or len(files_with_rollback) == len(files), (
        "Rollback / down procedures must be documented in README.md "
        "or in migration comments"
    )


@pytest.mark.skipif(
    "TIMESCALE_DSN" not in os.environ,
    reason="Requires live TimescaleDB; set TIMESCALE_DSN to run live integration test",
)
def test_live_timescale_migration_idempotent() -> None:
    """Apply migrations twice to a live TimescaleDB database without error."""
    dsn = os.environ["TIMESCALE_DSN"]
    import psycopg  # type: ignore[import-untyped]

    files = _get_migration_files()
    assert files, "No migration files found"

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Pass 1: Apply all migrations
            for f in files:
                cur.execute(f.read_text(encoding="utf-8"))

            # Pass 2: Re-apply all migrations (idempotency check)
            for f in files:
                cur.execute(f.read_text(encoding="utf-8"))

            # Verify hypertable exists
            cur.execute(
                "SELECT hypertable_name FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = 'tag_sample';"
            )
            row = cur.fetchone()
            assert row is not None, (
                "tag_sample hypertable not found in timescaledb_information.hypertables"
            )
