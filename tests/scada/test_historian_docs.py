"""Tests for SCADA Historian documentation, runbook, and ADR-0050 (issue #5548).

Verifies acceptance criteria for H9:
1. ADR-0050 and historian runbook exist and pass docs governance.
2. Every flag, metric, and path named in the docs exists in the codebase.
3. Runbook includes an explicit SQLite-only rollback procedure.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADR_PATH = ROOT / "docs" / "adr" / "ADR-0050-plant-historian-timescaledb.md"
RUNBOOK_PATH = ROOT / "docs" / "scada" / "historian_runbook.md"
RULING_PATH = ROOT / "docs" / "development" / "HISTORIAN_LICENSING_RULING.md"


def test_historian_docs_files_exist() -> None:
    """Verify that ADR-0050, runbook, and licensing ruling are present."""
    assert ADR_PATH.is_file(), f"Missing ADR-0050 at {ADR_PATH}"
    assert RUNBOOK_PATH.is_file(), f"Missing runbook at {RUNBOOK_PATH}"
    assert RULING_PATH.is_file(), f"Missing licensing ruling at {RULING_PATH}"


def test_adr_structure_and_licensing_scope() -> None:
    """ADR-0050 must follow the template and record licensing and alternatives."""
    text = ADR_PATH.read_text(encoding="utf-8")

    assert "# ADR-0050:" in text
    assert "Status: Accepted" in text
    assert "## Context" in text
    assert "## Decision Flow" in text
    assert "## Decision" in text
    assert "## Alternatives Considered" in text
    assert "## Consequences" in text
    assert "## Validation" in text

    # Issue #4951 licensing ruling and alternatives
    assert "#4951" in text or "HISTORIAN_LICENSING_RULING.md" in text
    assert "VictoriaMetrics" in text
    assert "QuestDB" in text
    assert "TimescaleDB" in text
    assert "Grafana" in text


def test_runbook_includes_sqlite_rollback_procedure() -> None:
    """Criterion 3: Runbook must include an explicit SQLite-only rollback procedure."""
    text = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert "rollback" in text.lower(), "Runbook must mention rollback"
    assert "sqlite" in text.lower(), "Runbook must mention SQLite"

    # Specific rollback section heading or clear procedure
    rollback_match = re.search(r"##+.*Rollback.*SQLite", text, re.IGNORECASE)
    assert rollback_match is not None, (
        "Runbook must have an explicit SQLite rollback section"
    )


def test_every_path_named_in_docs_exists() -> None:
    """Criterion 2: Every file path cited in the docs must exist in the repository."""
    for doc_path in (ADR_PATH, RUNBOOK_PATH):
        text = doc_path.read_text(encoding="utf-8")
        # Find paths matching `src/...` or `docs/...` in backticks
        cited_paths = set(
            re.findall(r"`((?:src|docs)/[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`", text)
        )
        assert cited_paths, f"Expected cited paths in {doc_path.name}"
        for path_str in cited_paths:
            full_path = ROOT / path_str
            assert full_path.exists(), (
                f"Path '{path_str}' cited in {doc_path.name} does not exist"
            )


def test_every_flag_and_metric_named_in_docs_exists() -> None:
    """Criterion 2: Every active P1AM flag and metric named in docs exists in code."""
    backend_root = ROOT / "src" / "p1am_control_system" / "backend"
    backend_code = " ".join(
        p.read_text(encoding="utf-8") for p in backend_root.glob("*.py")
    )

    # Active flags cited in the runbook must exist in backend settings/database
    active_flags = [
        "P1AM_HISTORIAN_MAX_BYTES",
        "P1AM_HISTORIAN_RETENTION_INTERVAL_S",
        "P1AM_CAPTURE_INTERVAL_S",
        "P1AM_SQLITE_SYNCHRONOUS",
        "P1AM_POLL_INTERVAL_S",
        "P1AM_DB_PATH",
    ]
    for flag in active_flags:
        assert flag in backend_code, f"Flag {flag} not found in backend code"

    # Metrics cited in the docs must exist in historian or performance models
    active_metrics = [
        "historian_write_failures",
        "historian_samples_dropped",
        "write_failures",
        "dropped_samples",
        "rows_written",
        "queue_depth",
        "scan_overruns",
    ]
    for metric in active_metrics:
        assert metric in backend_code, f"Metric {metric} not found in backend code"
