"""Standalone baseline capture script for issue #2989 Phase 1.

Generates synthetic test data at 100k / 500k / 1M / 5M rows,
runs timed measurements for CSV and Parquet I/O, and writes
benchmarks/data_processor_baseline.json.

This script is intentionally *not* pytest-benchmark instrumented so it can
be executed directly (``python benchmarks/run_baseline_capture.py``) without
a full pytest session and without requiring CI access.  It is a companion to
test_data_processor_baseline.py, which wraps the same workloads as proper
pytest-benchmark fixtures.

Usage::

    # From repo root
    python benchmarks/run_baseline_capture.py

    # Override output path
    python benchmarks/run_baseline_capture.py --output /tmp/my_baseline.json
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
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
# Config
# ---------------------------------------------------------------------------
_SCALES = [100_000, 500_000, 1_000_000, 5_000_000]
_NUM_SIGNALS = 8
_SEED = 42
_REPEATS = 3  # number of timed repetitions; report median


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    t = np.linspace(0.0, float(n_rows) / 1000.0, n_rows)
    data: dict[str, np.ndarray] = {"time_s": t}
    for i in range(_NUM_SIGNALS):
        freq = (i + 1) * 0.5
        data[f"signal_{i:02d}"] = np.sin(2 * np.pi * freq * t) + rng.normal(
            0, 0.05, n_rows
        )
    return pd.DataFrame(data)


def _generate_files(work_dir: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    """Write CSV and Parquet files; return (csv_paths, parquet_paths)."""
    rng = np.random.default_rng(_SEED)
    csv_paths: dict[int, Path] = {}
    parquet_paths: dict[int, Path] = {}
    for n in _SCALES:
        tag = f"{n // 1000}k"
        csv_p = work_dir / f"synthetic_{tag}.csv"
        pq_p = work_dir / f"synthetic_{tag}.parquet"
        print(f"  generating {tag} rows ...", flush=True)
        df = _make_synthetic_df(n, rng)
        df.to_csv(csv_p, index=False)
        df.to_parquet(pq_p, index=False)
        csv_paths[n] = csv_p
        parquet_paths[n] = pq_p
    return csv_paths, parquet_paths


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _median_seconds(fn, repeats: int = _REPEATS) -> float:
    """Return median elapsed seconds over `repeats` calls."""
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _peak_mib(fn) -> tuple[object, float]:  # type: ignore[type-arg]
    """Return (result, peak_mib) using tracemalloc."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------


def bench_csv_schema_preview(csv_paths: dict[int, Path]) -> dict[str, float]:
    results: dict[str, float] = {}
    for n, p in csv_paths.items():
        path = str(p)
        t = _median_seconds(lambda: pd.read_csv(path, nrows=0))
        results[str(n)] = round(t * 1000, 3)  # ms
        print(f"    csv_schema_preview {n // 1000}k: {results[str(n)]:.1f} ms")
    return results


def bench_csv_full_load(
    csv_paths: dict[int, Path], loader: DataLoader
) -> dict[str, float]:
    results: dict[str, float] = {}
    for n, p in csv_paths.items():
        path = str(p)
        t = _median_seconds(lambda: loader.load_csv_file(path, validate_security=False))
        results[str(n)] = round(t * 1000, 3)
        print(f"    csv_full_load {n // 1000}k: {results[str(n)]:.1f} ms")
    return results


def bench_parquet_full_load(parquet_paths: dict[int, Path]) -> dict[str, float]:
    results: dict[str, float] = {}
    for n, p in parquet_paths.items():
        path = str(p)
        t = _median_seconds(lambda: pd.read_parquet(path, engine="pyarrow"))
        results[str(n)] = round(t * 1000, 3)
        print(f"    parquet_full_load {n // 1000}k: {results[str(n)]:.1f} ms")
    return results


def bench_filter_export(csv_paths: dict[int, Path]) -> dict[str, float]:
    results: dict[str, float] = {}
    for n, p in csv_paths.items():
        df = pd.read_csv(str(p))
        t_min = df["time_s"].min()
        t_max = df["time_s"].max()
        t_lo = t_min + 0.4 * (t_max - t_min)
        t_hi = t_min + 0.5 * (t_max - t_min)

        def _run() -> None:
            filtered = df[(df["time_s"] >= t_lo) & (df["time_s"] <= t_hi)]
            buf = io.StringIO()
            filtered.to_csv(buf, index=False)

        t = _median_seconds(_run)
        results[str(n)] = round(t * 1000, 3)
        print(f"    filter_export {n // 1000}k: {results[str(n)]:.1f} ms")
    return results


def bench_memory_peak_csv(csv_paths: dict[int, Path]) -> dict[str, float]:
    results: dict[str, float] = {}
    for n, p in csv_paths.items():
        path = str(p)
        _, peak = _peak_mib(lambda: pd.read_csv(path))
        results[str(n)] = round(peak, 2)
        print(f"    memory_peak_csv {n // 1000}k: {results[str(n)]:.1f} MiB")
    return results


def bench_ui_latency(csv_paths: dict[int, Path]) -> dict[str, float]:
    """Header-only schema scan (nrows=0) — UI latency proxy.

    Note: DataLoader.detect_signals has a pre-existing bug where it discards
    columns from nrows=0 DataFrames (``df.empty == True``).  We benchmark
    pd.read_csv(nrows=0) directly as the correct equivalent.  Tracking: #2989.
    """
    results: dict[str, float] = {}
    for n, p in csv_paths.items():
        path = str(p)
        t = _median_seconds(lambda: pd.read_csv(path, nrows=0))
        results[str(n)] = round(t * 1000, 3)
        print(f"    ui_latency_schema_scan {n // 1000}k: {results[str(n)]:.1f} ms")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture data processor baseline")
    default_out = _REPO_ROOT / "benchmarks" / "data_processor_baseline.json"
    parser.add_argument(
        "--output",
        default=str(default_out),
        help="Output JSON path (default: benchmarks/data_processor_baseline.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=_REPEATS,
        help=f"Repetitions per measurement (default: {_REPEATS})",
    )
    args = parser.parse_args()

    print("=== Data Processor Baseline Capture (issue #2989 Phase 1) ===")
    print(f"  scales: {[f'{n // 1000}k' for n in _SCALES]}")
    print(f"  repeats: {args.repeats}")
    print()

    with tempfile.TemporaryDirectory(prefix="dp_bench_") as tmp:
        work_dir = Path(tmp)
        print("Generating synthetic datasets ...")
        csv_paths, parquet_paths = _generate_files(work_dir)

        loader = DataLoader(use_high_performance=False)

        print("\n[1/6] CSV schema preview ...")
        csv_preview = bench_csv_schema_preview(csv_paths)

        print("\n[2/6] CSV full load ...")
        csv_load = bench_csv_full_load(csv_paths, loader)

        print("\n[3/6] Parquet full load ...")
        pq_load = bench_parquet_full_load(parquet_paths)

        print("\n[4/6] Filter + export ...")
        filt_exp = bench_filter_export(csv_paths)

        print("\n[5/6] Memory peak (CSV) ...")
        mem_peak = bench_memory_peak_csv(csv_paths)

        print("\n[6/6] UI latency (schema scan nrows=0) ...")
        ui_lat = bench_ui_latency(csv_paths)

    import platform

    baseline = {
        "meta": {
            "issue": "#2989",
            "phase": 1,
            "description": "Baseline Python/pandas I/O before Rust/Polars engine",
            "units": {
                "csv_schema_preview_ms": "milliseconds (median of repeats)",
                "csv_full_load_ms": "milliseconds (median of repeats)",
                "parquet_full_load_ms": "milliseconds (median of repeats)",
                "filter_export_ms": "milliseconds (median of repeats)",
                "memory_peak_csv_mib": "MiB peak (tracemalloc)",
                "ui_latency_schema_scan_ms": "milliseconds (median of repeats, nrows=0 header read)",
            },
            "repeats": args.repeats,
            "scales_rows": _SCALES,
            "num_signals": _NUM_SIGNALS,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "csv_schema_preview_ms": csv_preview,
        "csv_full_load_ms": csv_load,
        "parquet_full_load_ms": pq_load,
        "filter_export_ms": filt_exp,
        "memory_peak_csv_mib": mem_peak,
        "ui_latency_schema_scan_ms": ui_lat,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline written to: {out_path}")

    # Print a quick summary table
    print("\n=== Summary (ms, median) ===")
    print(
        f"{'Scale':<10} {'CSV preview':>12} {'CSV load':>10} {'Parquet':>10} "
        f"{'Filter+exp':>12} {'Mem (MiB)':>10} {'UI scan':>10}"
    )
    for n in _SCALES:
        tag = f"{n // 1000}k"
        k = str(n)
        print(
            f"{tag:<10} "
            f"{csv_preview[k]:>12.1f} "
            f"{csv_load[k]:>10.1f} "
            f"{pq_load[k]:>10.1f} "
            f"{filt_exp[k]:>12.1f} "
            f"{mem_peak[k]:>10.1f} "
            f"{ui_lat[k]:>10.1f}"
        )


if __name__ == "__main__":
    main()
