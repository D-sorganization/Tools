"""ArchiveMixin -- ZIP archive creation for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path

from Folders_Tool_r0 import (
    MAX_UI_UPDATE_FREQUENCY,
    PROGRESS_START_ZIP,
    PROGRESS_ZIP_PERCENT,
)

logger = logging.getLogger(__name__)


class ArchiveMixin:
    """ZIP archive creation for FolderProcessorApp.

    Expects the host class to provide:
    - self.dest_folder: str
    - self.cancel_operation: bool
    - self._get_unique_path(path: str) -> str
    - self.update_progress(value: float, status: str) -> None
    """

    def create_output_zip(self) -> str:
        """Creates a ZIP archive of the destination folder.

        Returns:
            Path to the created ZIP file [str] - absolute path to the created archive

        Raises:
            ValueError: If destination folder path is empty or invalid
            FileNotFoundError: If destination folder does not exist
            PermissionError: If insufficient permissions to read destination or
                write ZIP
            OSError: If file system operations fail during ZIP creation
        """
        dest_path_obj = self._validate_zip_destination()
        zip_path = self._resolve_zip_path(dest_path_obj)

        logger.info(f"Creating ZIP archive: {zip_path}")

        try:
            total_files, total_size = self._count_zip_contents()
            logger.info(
                f"ZIP will contain {total_files} files, "
                f"{total_size / (1024 * 1024):.1f} MB",
            )
            self._write_zip_archive(zip_path, total_files)
        except (IOError, PermissionError, OSError) as e:
            self._cleanup_failed_zip(zip_path)
            logger.error(f"Failed to create ZIP archive: {e}")
            raise RuntimeError(f"Failed to create ZIP archive: {e}") from e

        return str(zip_path)

    def _validate_zip_destination(self) -> Path:
        """Validate destination folder exists, is a directory, and is non-empty."""
        if not self.dest_folder or not isinstance(self.dest_folder, str):
            raise ValueError("Destination folder not set or invalid")

        dest_path_obj = Path(self.dest_folder)

        if not dest_path_obj.exists():
            raise FileNotFoundError(
                f"Destination folder does not exist: {self.dest_folder}",
            )
        if not dest_path_obj.is_dir():
            raise ValueError(f"Destination path is not a directory: {self.dest_folder}")
        if not os.access(self.dest_folder, os.R_OK):
            raise PermissionError(f"Cannot read destination folder: {self.dest_folder}")

        try:
            if not list(dest_path_obj.iterdir()):
                raise ValueError("Destination folder is empty - nothing to archive")
        except (OSError, PermissionError) as e:
            raise PermissionError(
                f"Cannot access destination folder contents: {self.dest_folder} - {e}",
            ) from e

        return dest_path_obj

    def _resolve_zip_path(self, dest_path_obj: Path) -> Path:
        """Generate a unique ZIP file path in the parent of the destination."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"

        try:
            zip_path = dest_path_obj.parent / zip_filename
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot determine ZIP location: {e}") from e

        if zip_path.exists():
            zip_path = Path(self._get_unique_path(str(zip_path)))

        return zip_path

    def _count_zip_contents(self) -> tuple[int, int]:
        """Count accessible files and total size in the destination folder."""
        total_files = 0
        total_size = 0
        for root, _dirs, files in os.walk(self.dest_folder):
            for file in files:
                file_path = Path(root) / file
                try:
                    if file_path.exists() and os.access(file_path, os.R_OK):
                        total_files += 1
                        total_size += os.path.getsize(file_path)
                except (OSError, PermissionError):
                    continue

        if total_files == 0:
            raise ValueError("No accessible files found in destination folder")
        return total_files, total_size

    def _write_zip_archive(self, zip_path: Path, total_files: int) -> None:
        """Write all destination files into a ZIP archive with progress updates."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            processed_files = 0
            processed_size = 0
            failed_files = 0

            for root, _dirs, files in os.walk(self.dest_folder):
                for file in files:
                    if self.cancel_operation:
                        raise RuntimeError("ZIP creation cancelled by user")

                    file_path = Path(root) / file
                    try:
                        if not file_path.exists() or not os.access(file_path, os.R_OK):
                            failed_files += 1
                            continue

                        arcname = os.path.relpath(file_path, self.dest_folder)
                        zipf.write(file_path, arcname)
                        processed_files += 1
                        processed_size += os.path.getsize(file_path)

                        if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:
                            progress = (
                                PROGRESS_START_ZIP
                                + (processed_files / total_files) * PROGRESS_ZIP_PERCENT
                            )
                            self.update_progress(
                                progress,
                                f"Added {processed_files}/{total_files} files to ZIP",
                            )

                    except (IOError, PermissionError, OSError) as e:
                        failed_files += 1
                        logger.warning(f"Failed to add file to ZIP: {file_path} - {e}")

            self._verify_zip(zip_path, processed_files, processed_size, failed_files)

    def _verify_zip(
        self, zip_path: Path, processed: int, size: int, failed: int
    ) -> None:
        """Verify the created ZIP file is valid and log summary."""
        if not zip_path.exists():
            raise RuntimeError("ZIP file was not created")
        try:
            zip_size = zip_path.stat().st_size
            if zip_size == 0:
                raise RuntimeError("ZIP file is empty")
            logger.info(
                f"ZIP archive created: {zip_path} ({processed} files, "
                f"{size / (1024 * 1024):.1f} MB, "
                f"ZIP size: {zip_size / (1024 * 1024):.1f} MB)",
            )
        except OSError as e:
            logger.warning(f"Cannot verify ZIP file size: {e}")

        if failed > 0:
            logger.warning(f"ZIP creation completed with {failed} failed files")
        else:
            logger.info("ZIP creation completed successfully")

    def _cleanup_failed_zip(self, zip_path: Path) -> None:
        """Remove a partially-created ZIP file on failure."""
        if zip_path.exists():
            try:
                zip_path.unlink()
                logger.info(f"Cleaned up failed ZIP file: {zip_path}")
            except OSError as cleanup_error:
                logger.warning(
                    f"Failed to cleanup failed ZIP file: {zip_path} - {cleanup_error}",
                )
