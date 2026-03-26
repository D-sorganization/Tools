"""ArchiveOperationsMixin -- Archive extraction methods."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from Folders_Tool_r0 import (
    MAX_ARCHIVE_SIZE_RATIO,
    MAX_FILE_SIZE_MB,
)

logger = logging.getLogger(__name__)


class ArchiveOperationsMixin:
    """Archive extraction and validation methods."""

    def safe_extract_archive(self, archive_path: str) -> tuple[bool, str]:
        """Safely extracts an archive with validation.

        Args:
            archive_path: Path to the archive file to extract [str]
                - must exist and be readable

        Returns:
            Tuple of (success: bool, message: str)
                - success indicates extraction completed without errors

        Raises:
            ValueError: If archive_path is empty or invalid
            FileNotFoundError: If archive file does not exist
            PermissionError: If insufficient permissions to read archive or write to
                extract directory
            OSError: If file system operations fail
            Exception: If extraction process fails
        """
        if not (archive_path is not None):
            raise ValueError("archive_path must be provided")
        archive_path_obj, archive_size = self._validate_archive_input(archive_path)

        # Generate unique extraction directory
        extract_dir = self._get_unique_path(os.path.splitext(archive_path)[0])
        extract_dir_obj = Path(extract_dir)

        try:
            self._prepare_extraction_directory(extract_dir, extract_dir_obj)

            # Extract archive
            logger.info(f"Extracting archive: {archive_path} -> {extract_dir}")
            shutil.unpack_archive(archive_path, extract_dir)

            # Validate extraction if safe mode is enabled
            if self.safe_extract_var.get():
                self._validate_extraction_result(extract_dir, extract_dir_obj, archive_size)

            # Only delete original if extraction was successful
            self._cleanup_original_archive(archive_path_obj)

            return (
                True,
                f"Successfully extracted and deleted '{Path(archive_path).name}'",
            )

        except (PermissionError, OSError) as e:
            self._cleanup_failed_extraction(extract_dir_obj, extract_dir)
            return False, f"Failed to extract '{Path(archive_path).name}': {e}"

    def _validate_archive_input(self, archive_path: str) -> tuple[Path, int]:
        """Validate archive path, accessibility, size, and format.

        Returns:
            Tuple of (archive_path_obj, archive_size).

        Raises:
            ValueError: If path is empty or not a file.
            FileNotFoundError: If file does not exist.
            PermissionError: If file is not readable.
        """
        if not archive_path or not isinstance(archive_path, str):
            raise ValueError(
                f"Archive path must be non-empty string, got {type(archive_path)}",
            )

        archive_path_obj = Path(archive_path)

        if not archive_path_obj.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        if not archive_path_obj.is_file():
            raise ValueError(f"Archive path is not a file: {archive_path}")
        if not os.access(archive_path, os.R_OK):
            raise PermissionError(f"Cannot read archive file: {archive_path}")

        archive_size = archive_path_obj.stat().st_size
        if archive_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.warning(
                f"Archive file exceeds maximum size limit: {archive_path} "
                f"({archive_size / (1024 * 1024):.1f} MB)",
            )

        supported_formats = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}
        archive_ext = archive_path_obj.suffix.lower()
        if archive_ext not in supported_formats:
            logger.warning(
                f"Unsupported archive format: {archive_ext} for {archive_path}",
            )

        return archive_path_obj, archive_size

    def _prepare_extraction_directory(self, extract_dir: str, extract_dir_obj: Path) -> None:
        """Create extraction directory and verify it is writable."""
        extract_dir_obj.mkdir(parents=True, exist_ok=True)

        if not extract_dir_obj.exists():
            raise Exception("Failed to create extraction directory")
        if not os.access(extract_dir, os.W_OK):
            raise PermissionError(
                f"Cannot write to extraction directory: {extract_dir}",
            )

    def _validate_extraction_result(
        self, extract_dir: str, extract_dir_obj: Path, archive_size: int
    ) -> None:
        """Validate that extraction produced expected files and sizes."""
        if not extract_dir_obj.exists():
            raise Exception(
                "Extraction failed - destination folder was not created",
            )
        if not any(extract_dir_obj.iterdir()):
            raise Exception("Extraction failed - destination folder is empty")

        extracted_files = []
        total_extracted_size = 0
        for root, _dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_extracted_size += os.path.getsize(file_path)
                    extracted_files.append(file_path)
                except OSError as e:
                    logger.warning(
                        f"Cannot access extracted file size: {file_path} - {e}",
                    )

        if not extracted_files:
            raise Exception(
                "Extraction failed - no files found in extracted folder",
            )

        if total_extracted_size < archive_size * MAX_ARCHIVE_SIZE_RATIO:
            logger.warning(
                f"Extracted size ({total_extracted_size}) seems small "
                f"compared to archive size ({archive_size})",
            )

        logger.info(
            f"Extraction validation passed: {len(extracted_files)} files, "
            f"{total_extracted_size} bytes",
        )

    def _cleanup_original_archive(self, archive_path_obj: Path) -> None:
        """Delete the original archive after successful extraction."""
        try:
            archive_path_obj.unlink()
            logger.info(f"Deleted original archive: {archive_path_obj}")
        except OSError as e:
            logger.warning(
                f"Failed to delete original archive: {archive_path_obj} - {e}",
            )

    def _cleanup_failed_extraction(self, extract_dir_obj: Path, extract_dir: str) -> None:
        """Remove partially extracted directory on failure."""
        if extract_dir_obj.exists():
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
                logger.info(
                    f"Cleaned up failed extraction directory: {extract_dir}",
                )
            except (PermissionError, OSError) as cleanup_error:
                logger.warning(
                    f"Failed to cleanup extraction directory: {extract_dir} - " f"{cleanup_error}",
                )

    def _bulk_unzip_enhanced(self) -> list[str]:
        """Enhanced bulk extraction with better validation."""
        log = ["Starting enhanced bulk extraction..."]
        extracted_count = 0
        failed_count = 0

        # Find all archives
        archives = []
        for source_folder in self.source_folders:
            for root, _dirs, files in os.walk(source_folder):
                for file in files:
                    if file.lower().endswith((".zip", ".rar", ".7z")):
                        archives.append(Path(root) / file)

        if not archives:
            return ["No archives found to extract."]

        for i, archive_path in enumerate(archives):
            if self.cancel_operation:
                break

            self.update_progress(
                20 + (i / len(archives)) * 10,
                f"Extracting {Path(archive_path).name}...",
            )

            if not Path(archive_path).exists():
                continue

            success, message = self.safe_extract_archive(str(archive_path))
            log.append(message)

            if success:
                extracted_count += 1
            else:
                failed_count += 1

        summary = f"Processed {len(archives)} archive(s). "
        summary += f"Successfully extracted: {extracted_count}, Failed: {failed_count}"
        return [summary, *log[1:]]
