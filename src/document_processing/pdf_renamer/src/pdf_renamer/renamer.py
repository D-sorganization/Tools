"""File renaming engine that applies extracted titles to PDF filenames."""

import logging
from pathlib import Path

from .utils import (
    get_last_name,
    sanitize_filename,
    to_kebab_case,
    to_snake_case,
    to_title_case,
)

# Import extractor later to avoid circular dependency issues if any, but ideally here.
# For now, I'll assume extractor is passed or imported.

logger = logging.getLogger(__name__)


class Renamer:
    def __init__(self, dry_run: bool = False, style: str = "standard"):
        self.dry_run = dry_run
        self.style = style

    def generate_new_filename(self, author: str, title: str) -> str:
        last_name = get_last_name(author)

        if self.style == "snake_case":
            # author_last_title.pdf
            clean_title = to_snake_case(title)
            clean_author = to_snake_case(last_name)
            sep = "_"
        elif self.style == "kebab_case":
            # author-last-title.pdf
            clean_title = to_kebab_case(title)
            clean_author = to_kebab_case(last_name)
            sep = "-"
        else:
            # Standard: Author Last - Title.pdf
            clean_title = sanitize_filename(to_title_case(title))
            clean_author = sanitize_filename(last_name)
            sep = " - "

        if not clean_title:
            is_computer_friendly = self.style in ("snake_case", "kebab_case")
            clean_title = "untitled" if is_computer_friendly else "Untitled"

        if not clean_author:
            is_computer_friendly = self.style in ("snake_case", "kebab_case")
            clean_author = "unknown" if is_computer_friendly else "Unknown"

        return f"{clean_author}{sep}{clean_title}.pdf"

    def rename_file(self, original_path: Path, new_filename: str) -> str | None:
        """
        Rename a file to the new filename.

        Args:
            original_path: Path to the original file
            new_filename: New filename to use

        Returns:
            Success message if renamed, None if failed or skipped
        """
        if not original_path.exists():
            logger.error(f"File not found: {original_path}")
            return None

        target_path = original_path.parent / new_filename

        # Handle filename collision
        counter = 1
        stem = target_path.stem
        suffix = target_path.suffix
        while target_path.exists() and target_path != original_path:
            target_path = original_path.parent / f"{stem}_{counter}{suffix}"
            counter += 1

        if target_path == original_path:
            logger.info(f"Skipping {original_path.name} (already named correctly)")
            return f"Skipped: {original_path.name} (already named correctly)"

        logger.info(f"Renaming '{original_path.name}' -> '{target_path.name}'")

        if not self.dry_run:
            try:
                original_path.rename(target_path)
                return f"Renamed: {original_path.name} -> {target_path.name}"
            except OSError as e:
                logger.error(f"Failed to rename {original_path}: {e}")
                return None
        else:
            return f"[DRY RUN] Would rename: {original_path.name} -> {target_path.name}"
