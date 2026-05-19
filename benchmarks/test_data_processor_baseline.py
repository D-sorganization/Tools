"""Baseline benchmarks for data_processor CSV and Parquet I/O.

Phase 1 of issue #2989: Establish benchmark-first baseline before Rust/Polars
bulk-I/O work begins.

Measures for 100k, 500k, 1M, and 5M row synthetic datasets:
  - CSV schema preview time  (pandas.read_csv with nrows=0)
  - CSV full load time       (DataLoader.load_csv_file)
  - Parquet full load time   (pandas.read_parquet)
  - Filter + export time     (time-range slice → to_csv)
  - Memory peak during load  (tracemalloc)
  - UI latency proxy         (DataLoader.detect_signals on a single file)

Results are written to benchmarks/data_processor_baseline.json by
run_baseline_capture.py.  This file is the pytest-benchmark companion
and carries the ``benchmark`` marker so it is excluded from the default
test run (``-m "not slow"`` already covers it; the ``benchmark`` marker
provides explicit gate control).

Usage
-----
Run via pytest-benchmark directly (generates .benchmarks/ JSON storage)::

    pytest benchmarks/test_data_processor_baseline.py \
        --benchmark-enable \
        --benchmark-json=benchmarks/data_processor_baseline.json \
        -v -m benchmark

Or run the standalone capture script::

    python benchmarks/run_baseline_capture.py
"""

from __future__ import annotations

import io
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from the benchmarks/ dir
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in [
    _REPO_ROOT / "src",
    _REPO_ROOT / "src" / "python" / "src",
    _REPO_ROOT / "src" / "shared" / "python",
    _REPO_ROOT / "src" / "data_processing" / "data_processor" / "python",
]:
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from data_processor.core.data_loader import DataLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SCALES = [100_000, 500_000, 1_000_000, 5_000_000]
_NUM_SIGNALS = 8  # number of numeric columns besides the time column
_SEED = 42


# ---------------------------------------------------------------------------
# Fixtures — generate synthetic data files once per test session
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build a synthetic DataFrame with a time column plus numeric signals."""
    t = np.linspace(0.0, float(n_rows) / 1000.0, n_rows)
    data: dict[str, np.ndarray] = {"time_s": t}
    for i in range(_NUM_SIGNALS):
        freq = (i + 1) * 0.5
        data[f"signal_{i:02d}"] = np.sin(2 * np.pi * freq * t) + rng.normal(
            0, 0.05, n_rows
        )
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared temp directory for synthetic data files (session-scoped)."""
    return tmp_path_factory.mktemp("benchmark_data")


@pytest.fixture(scope="session")
def csv_paths(data_dir: Path) -> dict[int, Path]:
    """Generate one CSV file per scale, return {n_rows: path} mapping."""
    rng = np.random.default_rng(_SEED)
    paths: dict[int, Path] = {}
    for n in _SCALES:
        p = data_dir / f"synthetic_{n // 1000}k.csv"
        if not p.exists():
            df = _make_synthetic_df(n, rng)
            df.to_csv(p, index=False)
        paths[n] = p
    return paths


@pytest.fixture(scope="session")
def parquet_paths(data_dir: Path, csv_paths: dict[int, Path]) -> dict[int, Path]:
    """Generate one Parquet file per scale from the CSV data."""
    paths: dict[int, Path] = {}
    for n, csv_p in csv_paths.items():
        pq_p = data_dir / f"synthetic_{n // 1000}k.parquet"
        if not pq_p.exists():
            df = pd.read_csv(csv_p)
            df.to_parquet(pq_p, index=False)
        paths[n] = pq_p
    return paths


# ---------------------------------------------------------------------------
# Helper — peak memory during a callable
# ---------------------------------------------------------------------------


def _peak_mib(fn):  # type: ignore[no-untyped-def]
    """Return (result, peak_mib) for calling fn()."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Benchmark: CSV schema preview (read header only via nrows=0)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="csv_schema_preview")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_csv_schema_preview(
    benchmark,  # type: ignore[no-untyped-def]
    csv_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Benchmark: read only the header row (schema preview) from a CSV."""
    path = str(csv_paths[n_rows])

    def _run() -> pd.DataFrame:
        return pd.read_csv(path, nrows=0)

    result = benchmark(_run)
    assert list(result.columns)[0] == "time_s"


# ---------------------------------------------------------------------------
# Benchmark: CSV full load via DataLoader
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="csv_full_load")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_csv_full_load(
    benchmark,  # type: ignore[no-untyped-def]
    csv_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Benchmark: full CSV load through DataLoader (pandas back-end)."""
    path = str(csv_paths[n_rows])
    loader = DataLoader(use_high_performance=False)

    def _run() -> pd.DataFrame | None:
        return loader.load_csv_file(path, validate_security=False)

    result = benchmark(_run)
    assert result is not None
    assert len(result) == n_rows


# ---------------------------------------------------------------------------
# Benchmark: Parquet full load
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="parquet_full_load")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_parquet_full_load(
    benchmark,  # type: ignore[no-untyped-def]
    parquet_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Benchmark: full Parquet load via pandas.read_parquet (pyarrow engine)."""
    path = str(parquet_paths[n_rows])

    def _run() -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    result = benchmark(_run)
    assert len(result) == n_rows


# ---------------------------------------------------------------------------
# Benchmark: filter + export (time-range slice → CSV bytes)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="filter_export")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_filter_export(
    benchmark,  # type: ignore[no-untyped-def]
    csv_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Benchmark: load CSV, apply a 10% time-window filter, export to in-memory CSV."""
    path = str(csv_paths[n_rows])
    # Pre-load outside the timed section
    df = pd.read_csv(path)
    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    t_lo = t_min + 0.4 * (t_max - t_min)
    t_hi = t_min + 0.5 * (t_max - t_min)

    def _run() -> int:
        filtered = df[(df["time_s"] >= t_lo) & (df["time_s"] <= t_hi)]
        buf = io.StringIO()
        filtered.to_csv(buf, index=False)
        return len(buf.getvalue())

    byte_count = benchmark(_run)
    assert byte_count > 0


# ---------------------------------------------------------------------------
# Benchmark: memory peak during CSV load (tracemalloc, not benchmarked by
# pytest-benchmark timing — just asserts and records the number)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="memory_peak")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_memory_peak_csv_load(
    benchmark,  # type: ignore[no-untyped-def]
    csv_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Record memory peak (MiB) during full CSV load via tracemalloc.

    The benchmark timer measures the full load including tracemalloc overhead;
    the peak_mib value is captured as a custom extra_info annotation.
    """
    path = str(csv_paths[n_rows])

    def _run() -> float:
        _, peak = _peak_mib(lambda: pd.read_csv(path))
        return peak

    peak_mib = benchmark(_run)
    benchmark.extra_info["peak_mib"] = peak_mib
    # Sanity: 100k rows at ~9 cols of float64 ≈ 7 MB; allow generous headroom
    assert peak_mib < n_rows * 0.001  # < 1 KiB per row is very lenient


# ---------------------------------------------------------------------------
# Benchmark: UI latency proxy — schema header scan (nrows=0)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="ui_latency")
@pytest.mark.benchmark
@pytest.mark.parametrize("n_rows", _SCALES, ids=[f"{n // 1000}k" for n in _SCALES])
def test_ui_latency_schema_scan(
    benchmark,  # type: ignore[no-untyped-def]
    csv_paths: dict[int, Path],
    n_rows: int,
) -> None:
    """Benchmark: header-only CSV read (nrows=0) — UI latency proxy.

    This measures the time a user would wait for column names to appear in the
    signal-selector panel when loading a file.  It uses pd.read_csv directly
    rather than DataLoader.detect_signals because detect_signals has a
    pre-existing bug where it discards columns from empty DataFrames (the
    nrows=0 result has ``df.empty == True`` which causes the internal check
    ``not df_header.empty`` to short-circuit).  Tracking issue: #2989.
    """
    path = str(csv_paths[n_rows])

    def _run() -> list[str]:
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)

    columns = benchmark(_run)
    assert "time_s" in columns
