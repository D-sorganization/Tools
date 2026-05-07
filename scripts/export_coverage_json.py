#!/usr/bin/env python3
"""Export coverage reports to JSON format for tracking over time.

This script:
1. Generates coverage.json from the existing coverage data
2. Optionally saves it to a tracked location for historical tracking
3. Is called during CI to ensure coverage.json is always available

Issue: #2354 (coverage.json tracking)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export coverage to JSON format"
    )
    ap.add_argument(
        "--output",
        default="coverage.json",
        help="Output file for coverage JSON (default: coverage.json)",
    )
    ap.add_argument(
        "--save-to-config",
        action="store_true",
        help="Also save a timestamped copy to config/ for historical tracking",
    )
    args = ap.parse_args()

    try:
        import coverage
    except ImportError:
        print("Error: coverage module not found. Install with: pip install coverage")
        return 1

    # Load the coverage data from .coverage file
    cov = coverage.Coverage()
    cov.load()

    # Generate JSON report
    output_path = Path(args.output)
    cov.json_report(outfile=str(output_path), pretty_print=True)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            num_files = len(data.get("files", {}))
            total_coverage = data.get("totals", {}).get("percent_covered", 0)
            print(
                f"✓ Coverage JSON exported to {output_path} "
                f"({num_files} files, {total_coverage:.2f}% coverage)"
            )
    else:
        print(f"Error: Failed to create {output_path}")
        return 1

    if args.save_to_config:
        from datetime import datetime

        config_dir = Path("config")
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d")
        config_path = config_dir / f"coverage-{timestamp}.json"

        with open(output_path, "r", encoding="utf-8") as src:
            with open(config_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

        print(f"✓ Also saved timestamped copy to {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
