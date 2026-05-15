"""Tests for data_explorer_service — lazy pandas import (#2770) and
cancellable row count (#2772)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Issue #2770 — lazy pandas import
# ---------------------------------------------------------------------------


def test_import_data_explorer_service_does_not_pull_in_pandas(tmp_path: Path) -> None:
    """Importing data_explorer_service must NOT add pandas to sys.modules.

    Runs in a subprocess so the parent process's already-imported pandas
    cannot produce a false negative.  The parent's sys.path is forwarded via
    PYTHONPATH so the package is importable in the child.
    """
    worktree_src = str(Path(__file__).parents[5] / "src" / "shared" / "python")
    path_entries = [worktree_src] + [p for p in sys.path if p and p != worktree_src]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path_entries)}
    script = (
        "import sys\n"
        "from upstream_drift_tools.ui.tools_sidebar import data_explorer_service\n"
        "if 'pandas' in sys.modules:\n"
        "    print('FAIL', flush=True); import sys; sys.exit(1)\n"
        "print('OK', flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        "Lazy-import subprocess failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Issue #2772 — cancellable / progress-reporting row count
# ---------------------------------------------------------------------------

from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (  # noqa: E402
    _count_delimited_rows,
)


def _write_csv(path: Path, n_data_rows: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("col_a,col_b\n")
        for i in range(n_data_rows):
            fh.write(f"{i},{i * 2}\n")


@pytest.mark.unit
def test_count_delimited_rows_returns_correct_total(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, 100)
    assert _count_delimited_rows(csv_path) == 100


@pytest.mark.unit
def test_count_delimited_rows_cancel_returns_early(tmp_path: Path) -> None:
    """Setting cancel_event before calling must cause an early exit."""
    csv_path = tmp_path / "large.csv"
    _write_csv(csv_path, 50_000)

    cancel = threading.Event()
    cancel.set()  # set before the call — stops at first interval check

    result = _count_delimited_rows(csv_path, cancel_event=cancel)
    # With cancel already set, iteration stops within the first 10 000 rows.
    assert result < 50_000


@pytest.mark.unit
def test_count_delimited_rows_cancel_mid_way(tmp_path: Path) -> None:
    """Cancel from another thread stops iteration before reaching the end."""
    csv_path = tmp_path / "large.csv"
    _write_csv(csv_path, 50_000)

    cancel = threading.Event()
    results: list[int] = []

    def _run() -> None:
        count = _count_delimited_rows(csv_path, cancel_event=cancel)
        results.append(count)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    cancel.set()
    t.join(timeout=10)

    assert not t.is_alive(), "thread did not finish within timeout"
    assert len(results) == 1
    assert results[0] < 50_000


@pytest.mark.unit
def test_count_delimited_rows_progress_callback_fires(tmp_path: Path) -> None:
    """progress_cb must be called at least once for a file with >= 10 000 data rows."""
    csv_path = tmp_path / "medium.csv"
    _write_csv(csv_path, 25_000)

    calls: list[int] = []
    _count_delimited_rows(csv_path, progress_cb=calls.append)

    assert len(calls) >= 2  # 25 000 rows -> progress at 10 000 and 20 000
    for reported in calls:
        assert reported % 10_000 == 0


@pytest.mark.unit
def test_count_delimited_rows_no_cancel_full_count(tmp_path: Path) -> None:
    """Without cancel_event, all rows must be counted correctly."""
    csv_path = tmp_path / "exact.csv"
    _write_csv(csv_path, 15_000)
    assert _count_delimited_rows(csv_path) == 15_000


@pytest.mark.unit
def test_count_delimited_rows_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("col_a,col_b\n", encoding="utf-8")
    assert _count_delimited_rows(csv_path) == 0
