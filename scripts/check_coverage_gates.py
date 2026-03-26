#!/usr/bin/env python3
"""Coverage gate enforcement script.

Checks that per-module test coverage does not regress below configured thresholds.
Exit code 0 if all gates pass, 1 if any gate fails.

Usage:
    python3 scripts/check_coverage_gates.py [--coverage-file .coverage]

Design by Contract:
    Precondition: .coverage file exists (run pytest --cov first)
    Postcondition: Returns 0 iff every gate module meets its threshold
    Invariant: Gate thresholds are monotonically non-decreasing over time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Per-module coverage gates.  Format: (module_path_prefix, min_percent)
# These thresholds should only ever increase, never decrease.
COVERAGE_GATES: list[tuple[str, float]] = [
    ("src/shared/python/upstream_drift_tools/calculators/conversion", 60.0),
    ("src/shared/python/upstream_drift_tools/data_processing", 40.0),
    ("src/shared/python/upstream_drift_tools/utils", 50.0),
    ("src/shared/python/upstream_drift_tools/process_calculators/wgs_reactor", 50.0),
    ("src/shared/python/upstream_drift_tools/process_calculators/baghouse", 40.0),
    ("src/shared/python/upstream_drift_tools/process_calculators/flare", 40.0),
]


def parse_coverage_json(coverage_json_path: Path) -> dict[str, dict[str, int]]:
    """Parse coverage.json into a mapping of file -> {covered, total}."""
    data = json.loads(coverage_json_path.read_text())
    result: dict[str, dict[str, int]] = {}
    for filepath, file_data in data.get("files", {}).items():
        summary = file_data.get("summary", {})
        result[filepath] = {
            "covered_lines": summary.get("covered_lines", 0),
            "num_statements": summary.get("num_statements", 0),
        }
    return result


def check_gates(
    file_coverage: dict[str, dict[str, int]],
    gates: list[tuple[str, float]],
) -> list[tuple[str, float, float]]:
    """Check coverage gates. Returns list of (module, actual%, threshold%) failures."""
    failures: list[tuple[str, float, float]] = []
    for prefix, threshold in gates:
        total_stmts = 0
        covered_stmts = 0
        for filepath, stats in file_coverage.items():
            if prefix in filepath:
                total_stmts += stats["num_statements"]
                covered_stmts += stats["covered_lines"]
        if total_stmts == 0:
            continue  # Module not in coverage data (possibly not collected)
        actual = (covered_stmts / total_stmts) * 100
        if actual < threshold:
            failures.append((prefix, actual, threshold))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check coverage gates")
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("coverage.json"),
        help="Path to coverage.json (from pytest --cov --cov-report=json)",
    )
    args = parser.parse_args()

    if not args.coverage_json.exists():
        print(f"ERROR: {args.coverage_json} not found. Run pytest --cov first.")
        return 1

    file_coverage = parse_coverage_json(args.coverage_json)
    failures = check_gates(file_coverage, COVERAGE_GATES)

    if failures:
        print("COVERAGE GATE FAILURES:")
        for prefix, actual, threshold in failures:
            print(f"  FAIL: {prefix} = {actual:.1f}% (minimum: {threshold:.1f}%)")
        return 1

    print("All coverage gates passed.")
    for prefix, threshold in COVERAGE_GATES:
        total = sum(s["num_statements"] for f, s in file_coverage.items() if prefix in f)
        covered = sum(s["covered_lines"] for f, s in file_coverage.items() if prefix in f)
        if total > 0:
            pct = (covered / total) * 100
            print(f"  OK: {prefix} = {pct:.1f}% (>= {threshold:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
