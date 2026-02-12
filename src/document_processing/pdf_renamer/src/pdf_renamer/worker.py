"""Thread-safe worker module for parallel PDF processing."""

import logging
import threading
from pathlib import Path

from .cache import ResultCache
from .core import extract_title
from .extractors import TitleLLM, author_from_metadata
from .transaction_log import TransactionLog
from .types import TitleResult
from .utils import (
    get_last_name,
    sanitize_filename,
    sha256_file,
    to_kebab_case,
    to_snake_case,
    to_title_case,
)

logger = logging.getLogger(__name__)

# Global lock for file operations to prevent TOCTOU race conditions
_file_operation_lock = threading.Lock()


class ProcessingResult:
    """Result of processing a single PDF file."""

    def __init__(
        self,
        file_path: Path,
        success: bool,
        message: str,
        title_result: TitleResult | None = None,
        new_path: Path | None = None,
    ):
        self.file_path = file_path
        self.success = success
        self.message = message
        self.title_result = title_result
        self.new_path = new_path


def process_single_file(
    file_path: Path,
    cache: ResultCache,
    transaction_log: TransactionLog,
    llm: TitleLLM | None = None,
    dry_run: bool = True,
    style: str = "standard",
    include_author: bool = False,
    move_failed: bool = True,
    failed_folder: str = "failed_renames",
) -> ProcessingResult:
    """
    Process a single PDF file: extract title and rename.

    Args:
        file_path: Path to PDF file
        cache: ResultCache instance
        transaction_log: TransactionLog instance
        llm: Optional LLM for title extraction
        dry_run: If True, don't actually rename files
        style: Naming style ('standard', 'snake_case', 'kebab_case')
        include_author: If True, include author in filename
        move_failed: If True, move failed files to subfolder
        failed_folder: Name of subfolder for failed files

    Returns:
        ProcessingResult with operation details
    """
    try:
        if not file_path.exists():
            return ProcessingResult(file_path, False, f"File not found: {file_path}")

        # 1. Calculate hash
        file_hash = sha256_file(file_path)

        # 2. Check cache
        cached = cache.get(file_hash)
        result: TitleResult

        if cached and cached.title:
            result = cached
            logger.debug(f"[CACHE] {file_path.name} -> {result.title}")
        else:
            # 3. Extract title
            result = extract_title(file_path, llm)

            # 4. Save to cache
            model_name = getattr(llm, "DEFAULT_MODEL", "unknown") if llm else "local"
            cache.save(
                file_hash,
                file_path,
                result,
                provider="gemini" if llm else "local",
                model=model_name if llm else "heuristic",
            )
            logger.info(
                f"[{result.method.upper()}] {file_path.name} -> {result.title} "
                f"({result.confidence:.2f})"
            )

        # 5. Check if we have a valid title
        if not result.title:
            # Move to failed folder if enabled
            if move_failed and not dry_run:
                failed_path = _move_to_failed_folder(
                    file_path, failed_folder, transaction_log
                )
                if failed_path:
                    return ProcessingResult(
                        file_path,
                        False,
                        f"Could not extract title from {file_path.name}: {result.details}. Moved to {failed_folder}/",
                        result,
                        failed_path,
                    )

            return ProcessingResult(
                file_path,
                False,
                f"Could not extract title from {file_path.name}: {result.details}",
                result,
            )

        # 6. Extract author if needed
        author = ""
        if include_author:
            author = author_from_metadata(file_path) or ""

        # 7. Generate new filename
        new_filename = _generate_filename(result.title, author, style, include_author)

        target_path = file_path.parent / new_filename

        # 8. Check if already correctly named
        if target_path == file_path:
            return ProcessingResult(
                file_path,
                True,
                f"Already correctly named: {file_path.name}",
                result,
                target_path,
            )

        # 9. Rename with thread safety
        if not dry_run:
            with _file_operation_lock:
                # Handle collisions
                if target_path.exists():
                    # Add hash suffix to prevent collisions
                    short_hash = file_hash[:6]
                    stem = target_path.stem
                    target_path = file_path.parent / f"{stem}_{short_hash}.pdf"

                    if target_path.exists():
                        return ProcessingResult(
                            file_path,
                            False,
                            f"Target exists and collision resolution failed: {target_path.name}",
                            result,
                        )

                # Perform rename
                try:
                    file_path.rename(target_path)
                    transaction_log.log_rename(file_path, target_path, True)
                    return ProcessingResult(
                        file_path,
                        True,
                        f"Renamed: {file_path.name} -> {target_path.name}",
                        result,
                        target_path,
                    )
                except OSError as e:
                    transaction_log.log_rename(file_path, target_path, False, str(e))
                    return ProcessingResult(
                        file_path,
                        False,
                        f"Failed to rename {file_path.name}: {e}",
                        result,
                    )
        else:
            return ProcessingResult(
                file_path,
                True,
                f"[DRY RUN] Would rename: {file_path.name} -> {new_filename}",
                result,
                target_path,
            )

    except (PermissionError, OSError) as e:
        logger.error(f"Error processing {file_path}: {e}")
        return ProcessingResult(
            file_path, False, f"Error processing {file_path.name}: {e}"
        )


def _generate_filename(
    title: str, author: str, style: str, include_author: bool
) -> str:
    """
    Generate filename based on title, author, and style.

    Args:
        title: Document title
        author: Document author
        style: Naming style
        include_author: Whether to include author in filename

    Returns:
        Generated filename with .pdf extension
    """
    if style == "snake_case":
        clean_title = to_snake_case(title)
        if include_author and author:
            clean_author = to_snake_case(get_last_name(author))
            return f"{clean_author}_{clean_title}.pdf"
        return f"{clean_title}.pdf"

    elif style == "kebab_case":
        clean_title = to_kebab_case(title)
        if include_author and author:
            clean_author = to_kebab_case(get_last_name(author))
            return f"{clean_author}-{clean_title}.pdf"
        return f"{clean_title}.pdf"

    else:  # standard
        clean_title = sanitize_filename(to_title_case(title))
        if include_author and author:
            clean_author = sanitize_filename(get_last_name(author))
            return f"{clean_author} - {clean_title}.pdf"
        return f"{clean_title}.pdf"


def _move_to_failed_folder(
    file_path: Path, failed_folder: str, transaction_log: TransactionLog
) -> Path | None:
    """
    Move a file to the failed processing folder.

    Args:
        file_path: Original file path
        failed_folder: Name of the failed folder
        transaction_log: Transaction log instance

    Returns:
        New path if successful, None if failed
    """
    try:
        failed_dir = file_path.parent / failed_folder
        failed_dir.mkdir(exist_ok=True)

        target_path = failed_dir / file_path.name

        # Handle name collisions
        counter = 1
        while target_path.exists():
            stem = file_path.stem
            target_path = failed_dir / f"{stem}_{counter}.pdf"
            counter += 1

        with _file_operation_lock:
            file_path.rename(target_path)
            transaction_log.log_rename(
                file_path, target_path, True, "Moved to failed folder"
            )

        return target_path

    except (PermissionError, OSError) as e:
        logger.error(f"Failed to move {file_path} to failed folder: {e}")
        return None
