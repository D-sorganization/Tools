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
        # Input validation
        if not archive_path or not isinstance(archive_path, str):
            raise ValueError(
                f"Archive path must be non-empty string, got {type(archive_path)}",
            )

        archive_path_obj = Path(archive_path)

        # Validate archive file exists and is accessible
        if not archive_path_obj.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        if not archive_path_obj.is_file():
            raise ValueError(f"Archive path is not a file: {archive_path}")
        if not os.access(archive_path, os.R_OK):
            raise PermissionError(f"Cannot read archive file: {archive_path}")

        # Validate archive file size
        try:
            archive_size = archive_path_obj.stat().st_size
            if archive_size == 0:
                return False, f"Archive file is empty: {archive_path}"
            if archive_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(
                    f"Archive file exceeds maximum size limit: {archive_path} "
                    f"({archive_size / (1024 * 1024):.1f} MB)",
                )
        except OSError as e:
            return False, f"Cannot access archive file: {e}"

        # Validate archive file extension
        archive_ext = archive_path_obj.suffix.lower()
        supported_formats = {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}
        if archive_ext not in supported_formats:
            logger.warning(
                f"Unsupported archive format: {archive_ext} for {archive_path}",
            )

        # Generate unique extraction directory
        extract_dir = self._get_unique_path(os.path.splitext(archive_path)[0])
        extract_dir_obj = Path(extract_dir)

        try:
            # Create extraction directory
            extract_dir_obj.mkdir(parents=True, exist_ok=True)

            # Verify directory was created and is writable
            if not extract_dir_obj.exists():
                raise Exception("Failed to create extraction directory")
            if not os.access(extract_dir, os.W_OK):
                raise PermissionError(
                    f"Cannot write to extraction directory: {extract_dir}",
                )

            # Extract archive
            logger.info(f"Extracting archive: {archive_path} -> {extract_dir}")
            shutil.unpack_archive(archive_path, extract_dir)

            # Validate extraction if safe mode is enabled
            if self.safe_extract_var.get():
                if not extract_dir_obj.exists():
                    raise Exception(
                        "Extraction failed - destination folder was not created",
                    )

                if not any(extract_dir_obj.iterdir()):
                    raise Exception("Extraction failed - destination folder is empty")

                # Check if any files were actually extracted
                extracted_files = []
                total_extracted_size = 0

                for root, _dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            file_size = os.path.getsize(file_path)
                            extracted_files.append(file_path)
                            total_extracted_size += file_size
                        except OSError as e:
                            logger.warning(
                                f"Cannot access extracted file size: {file_path} - {e}",
                            )

                if not extracted_files:
                    raise Exception(
                        "Extraction failed - no files found in extracted folder",
                    )

                # Verify total extracted size is reasonable
                if total_extracted_size < archive_size * MAX_ARCHIVE_SIZE_RATIO:
                    logger.warning(
                        f"Extracted size ({total_extracted_size}) seems small "
                        f"compared to archive size ({archive_size})",
                    )

                logger.info(
                    f"Extraction validation passed: {len(extracted_files)} files, "
                    f"{total_extracted_size} bytes",
                )

            # Only delete original if extraction was successful
            try:
                archive_path_obj.unlink()
                logger.info(f"Deleted original archive: {archive_path}")
            except OSError as e:
                logger.warning(
                    f"Failed to delete original archive: {archive_path} - {e}",
                )
                # Don't fail the operation if cleanup fails

            return (
                True,
                f"Successfully extracted and deleted "
                f"'{Path(archive_path).name}' "
                f"({'known' if 'extracted_files' in locals() else 'unknown'} "
                "files)",
            )

        except (IOError, PermissionError, OSError) as e:
            # Clean up failed extraction directory
            if extract_dir_obj.exists():
                try:
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    logger.info(
                        f"Cleaned up failed extraction directory: {extract_dir}",
                    )
                except (IOError, PermissionError, OSError) as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup extraction directory: {extract_dir} - "
                        f"{cleanup_error}",
                    )

            return False, f"Failed to extract '{Path(archive_path).name}': {e}"

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
