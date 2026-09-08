#!/usr/bin/env python3
"""CLI check gate for Tools calculation freshness and reverse-impact governance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.tools_calculation_freshness_contract import (
    CalculationFreshnessError,
    verify_calculation_freshness,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify calculation freshness and reverse-impact integrity",
    )
    parser.add_argument(
        "--diff-base",
        type=str,
        default=None,
        help="Optional git diff base for diff-aware checking",
    )
    args = parser.parse_args()
    changed_files = None
    if args.diff_base:
        import subprocess

        try:
            res = subprocess.run(
                ["git", "diff", "--name-only", args.diff_base],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [
                line.strip() for line in res.stdout.splitlines() if line.strip()
            ]
        except Exception as exc:
            print(
                f"ERROR: failed to get git diff from {args.diff_base}: {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        summary = verify_calculation_freshness(ROOT, changed_files=changed_files)
    except CalculationFreshnessError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(
        f"Calculation freshness verified: {summary.calculation_count} calculation(s), "
        f"{summary.nominal_count} nominal, {summary.boundary_count} boundary, "
        f"{summary.failure_count} failure examples, {summary.documented_value_count} documented values, "
        f"{summary.table_count} tables, {summary.active_exemption_count} active exemptions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
