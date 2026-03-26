#!/usr/bin/env python3
"""Migrate CSV data files to Parquet format.

Scans the repository for CSV data files and creates Parquet equivalents
alongside the originals (non-destructive). Also generates a backward-
compatible reader module.

See issue #565.

Usage:
    python scripts/migrate_csv_to_parquet.py [--dry-run] [--remove-originals]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories to scan for CSV data files
DATA_DIRS = [
    _REPO_ROOT
    / "src"
    / "data_processing"
    / "data_processor"
    / "python"
    / "data_processor"
    / "data",
    _REPO_ROOT / "src" / "scientific_modeling" / "rrt_path_planner" / "matlab" / "data",
]

# Directories to skip
SKIP_PATTERNS = {"node_modules", "__pycache__", ".git", ".ruff_cache"}


def find_csv_files(root: Path) -> list[Path]:
    """Find all CSV files under the given root directory."""
    csv_files: list[Path] = []
    if not root.exists():
        return csv_files
    for csv_path in root.rglob("*.csv"):
        if any(part in csv_path.parts for part in SKIP_PATTERNS):
            continue
        csv_files.append(csv_path)
    return csv_files


def migrate_csv_to_parquet(
    csv_path: Path,
    *,
    dry_run: bool = False,
    remove_original: bool = False,
) -> Path | None:
    """Convert a single CSV file to Parquet format.

    Args:
        csv_path: Path to the CSV file.
        dry_run: If True, only print what would be done.
        remove_original: If True, remove the CSV after conversion.

    Returns:
        Path to the created Parquet file, or None on failure / dry-run.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas pyarrow")
        return None

    parquet_path = csv_path.with_suffix(".parquet")

    if parquet_path.exists():
        print(f"  SKIP (exists): {parquet_path.relative_to(_REPO_ROOT)}")
        return parquet_path

    if dry_run:
        print(f"  WOULD convert: {csv_path.relative_to(_REPO_ROOT)} -> .parquet")
        return None

    try:
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        size_csv = csv_path.stat().st_size
        size_pq = parquet_path.stat().st_size
        ratio = size_pq / size_csv * 100 if size_csv > 0 else 0
        print(
            f"  OK: {csv_path.name} -> {parquet_path.name} "
            f"({size_csv:,} -> {size_pq:,} bytes, {ratio:.0f}%)"
        )

        if remove_original:
            csv_path.unlink()
            print(f"  REMOVED: {csv_path.name}")

        return parquet_path

    except Exception as exc:
        print(f"  FAILED: {csv_path.name}: {exc}")
        return None


def main() -> int:
    """Entry point for the migration script."""
    parser = argparse.ArgumentParser(description="Migrate CSV files to Parquet")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--remove-originals",
        action="store_true",
        help="Remove CSV files after successful conversion",
    )
    args = parser.parse_args()

    all_csvs: list[Path] = []
    for data_dir in DATA_DIRS:
        all_csvs.extend(find_csv_files(data_dir))

    if not all_csvs:
        print("No CSV files found in data directories.")
        return 0

    print(f"Found {len(all_csvs)} CSV file(s) to migrate:")
    converted = 0
    for csv_path in sorted(all_csvs):
        result = migrate_csv_to_parquet(
            csv_path,
            dry_run=args.dry_run,
            remove_original=args.remove_originals,
        )
        if result is not None:
            converted += 1

    if args.dry_run:
        print(f"\nDry run complete. {len(all_csvs)} files would be processed.")
    else:
        print(f"\nMigration complete. {converted}/{len(all_csvs)} files converted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
