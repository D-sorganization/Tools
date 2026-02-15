# ruff: noqa: T201
"""Migrate CSV files to Parquet format.

Usage:
    python scripts/migrate_csv_to_parquet.py [--dry-run] [path]

This script finds all CSV files under a directory and converts them
to Parquet format using the upstream_drift_tools.data_io module.
Original CSV files are preserved by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_csv_files(directory: Path) -> list[Path]:
    """Find all CSV files under a directory recursively.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of CSV file paths.
    """
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    csv_files = sorted(directory.rglob("*.csv"))
    logger.info("Found %d CSV files under %s", len(csv_files), directory)
    return csv_files


def migrate_csv_to_parquet(
    csv_path: Path,
    *,
    dry_run: bool = False,
) -> Path | None:
    """Convert a single CSV file to Parquet format.

    Args:
        csv_path: Path to the CSV file.
        dry_run: If True, only log what would be done.

    Returns:
        Path to the new Parquet file, or None if dry_run.
    """
    parquet_path = csv_path.with_suffix(".parquet")

    if parquet_path.exists():
        logger.info("Skipping %s: Parquet file already exists", csv_path.name)
        return parquet_path

    if dry_run:
        logger.info(
            "[DRY RUN] Would convert: %s -> %s", csv_path.name, parquet_path.name
        )
        return None

    try:
        from upstream_drift_tools.data_io import read_data, write_data

        df = read_data(csv_path, prefer_parquet=False)
        write_data(df, parquet_path)
        logger.info(
            "Converted: %s -> %s (%d rows)", csv_path.name, parquet_path.name, len(df)
        )
        return parquet_path
    except Exception as exc:
        logger.error("Failed to convert %s: %s", csv_path.name, exc)
        return None


def main() -> None:
    """CLI entry point for CSV-to-Parquet migration."""
    parser = argparse.ArgumentParser(
        description="Migrate CSV files to Parquet format.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to search for CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    directory = Path(args.path).resolve()

    csv_files = find_csv_files(directory)
    converted = 0
    for csv_path in csv_files:
        result = migrate_csv_to_parquet(csv_path, dry_run=args.dry_run)
        if result is not None:
            converted += 1

    print(f"\nMigration complete: {converted}/{len(csv_files)} files converted")


if __name__ == "__main__":
    main()
