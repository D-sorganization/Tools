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
        assert file_path is not None, "file_path must be provided"
        self.file_path = file_path
        self.success = success
        self.message = message
        self.title_result = title_result
        self.new_path = new_path


def _extract_or_load_title(
    file_path: Path,
    file_hash: str,
    cache: ResultCache,
    llm: TitleLLM | None,
) -> TitleResult:
    """Extract title from PDF, using cache if available.

    Args:
        file_path: Path to the PDF file
        file_hash: SHA-256 hash of the file
        cache: ResultCache instance
        llm: Optional LLM for title extraction

    Returns:
        TitleResult from cache or fresh extraction
    """
    assert file_path is not None, "file_path must be provided"
    cached = cache.get(file_hash)
    if cached and cached.title:
        logger.debug(f"[CACHE] {file_path.name} -> {cached.title}")
        return cached

    result = extract_title(file_path, llm)
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
    return result


def _handle_missing_title(
    file_path: Path,
    result: TitleResult,
    move_failed: bool,
    dry_run: bool,
    failed_folder: str,
    transaction_log: TransactionLog,
) -> ProcessingResult:
    """Handle the case where no title could be extracted.

    Args:
        file_path: Path to the PDF file
        result: TitleResult with empty title
        move_failed: Whether to move the file to a failed folder
        dry_run: Whether this is a dry run
        failed_folder: Name of the failed folder
        transaction_log: TransactionLog instance

    Returns:
        ProcessingResult indicating failure
    """
    assert file_path is not None, "file_path must be provided"
    if move_failed and not dry_run:
        failed_path = _move_to_failed_folder(file_path, failed_folder, transaction_log)
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


def _rename_file_with_collision_handling(
    file_path: Path,
    target_path: Path,
    file_hash: str,
    result: TitleResult,
    transaction_log: TransactionLog,
) -> ProcessingResult:
    """Rename a file with thread-safe collision handling.

    Args:
        file_path: Original file path
        target_path: Desired target path
        file_hash: SHA-256 hash for collision resolution
        result: TitleResult for the file
        transaction_log: TransactionLog instance

    Returns:
        ProcessingResult indicating success or failure
    """
    with _file_operation_lock:
        if target_path.exists():
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

    Orchestrates title extraction, filename generation, and renaming.

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

        file_hash = sha256_file(file_path)
        result = _extract_or_load_title(file_path, file_hash, cache, llm)

        if not result.title:
            return _handle_missing_title(
                file_path, result, move_failed, dry_run, failed_folder, transaction_log
            )

        author = author_from_metadata(file_path) or "" if include_author else ""
        new_filename = _generate_filename(result.title, author, style, include_author)
        target_path = file_path.parent / new_filename

        if target_path == file_path:
            return ProcessingResult(
                file_path,
                True,
                f"Already correctly named: {file_path.name}",
                result,
                target_path,
            )

        if dry_run:
            return ProcessingResult(
                file_path,
                True,
                f"[DRY RUN] Would rename: {file_path.name} -> {new_filename}",
                result,
                target_path,
            )

        return _rename_file_with_collision_handling(
            file_path, target_path, file_hash, result, transaction_log
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
