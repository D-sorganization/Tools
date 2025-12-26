import argparse
import logging
import sys
from pathlib import Path

from .deduper import DuplicateFinder
from .extractor import extract_metadata
from .renamer import Renamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk rename PDF files based on metadata."
    )
    parser.add_argument("directory", type=Path, help="Directory containing PDF files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without renaming"
    )
    parser.add_argument(
        "--delete-duplicates",
        action="store_true",
        help="Delete duplicate files automatically",
    )
    parser.add_argument(
        "--style",
        choices=["standard", "snake_case", "kebab_case"],
        default="standard",
        help="Naming style for renamed files (default: standard)",
    )

    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Invalid directory: {directory}")
        sys.exit(1)

    # 1. Handle Duplicates
    logger.info("Scanning for duplicates...")
    finder = DuplicateFinder(directory)
    duplicates = finder.find_duplicates()

    if duplicates:
        logger.info(f"Found {len(duplicates)} sets of duplicates.")
        for file_hash, paths in duplicates.items():
            logger.info(f"Duplicate set (Hash: {file_hash}):")
            for p in paths:
                logger.info(f"  - {p}")

            if args.delete_duplicates:
                # Keep the first one, delete the rest
                # Sort to be deterministic (e.g., shortest filename, or lexicographical)
                sorted_paths = sorted(paths, key=lambda p: (len(str(p)), p.name))
                keep = sorted_paths[0]
                to_delete = sorted_paths[1:]

                logger.info(f"  Keeping: {keep}")
                for p in to_delete:
                    if args.dry_run:
                        logger.info(f"  [DRY RUN] Would delete: {p}")
                    else:
                        try:
                            p.unlink()
                            logger.info(f"  Deleted: {p}")
                        except OSError as e:
                            logger.error(f"  Failed to delete {p}: {e}")
            else:
                logger.info("  (Use --delete-duplicates to remove extra copies)")
    else:
        logger.info("No duplicates found.")

    # 2. Rename Files
    logger.info(f"Starting renaming process using style: {args.style}...")
    renamer = Renamer(dry_run=args.dry_run, style=args.style)

    # scan again in case files were deleted
    for file_path in directory.glob("**/*.pdf"):
        if not file_path.exists():
            continue

        author, title = extract_metadata(file_path)

        if not author or not title:
            logger.warning(
                f"Skipping {file_path.name}: Missing metadata "
                f"(Author: {author}, Title: {title})"
            )
            continue

        new_filename = renamer.generate_new_filename(author, title)
        renamer.rename_file(file_path, new_filename)

    logger.info("Done.")


if __name__ == "__main__":
    main()
