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
            Exception: If ZIP creation fails for other reasons
        """
        # Input validation
        if not self.dest_folder:
            raise ValueError("Destination folder not set")
        if not isinstance(self.dest_folder, str):
            raise ValueError(
                f"Destination folder must be a string, got {type(self.dest_folder)}",
            )

        dest_path_obj = Path(self.dest_folder)

        # Validate destination folder exists and is accessible
        if not dest_path_obj.exists():
            raise FileNotFoundError(
                f"Destination folder does not exist: {self.dest_folder}",
            )
        if not dest_path_obj.is_dir():
            raise ValueError(f"Destination path is not a directory: {self.dest_folder}")
        if not os.access(self.dest_folder, os.R_OK):
            raise PermissionError(f"Cannot read destination folder: {self.dest_folder}")

        # Check if destination folder is empty
        try:
            folder_contents = list(dest_path_obj.iterdir())
            if not folder_contents:
                raise ValueError("Destination folder is empty - nothing to archive")
        except (OSError, PermissionError) as e:
            raise PermissionError(
                f"Cannot access destination folder contents: {self.dest_folder} - {e}",
            ) from e

        # Generate ZIP filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"

        # Create ZIP in parent directory of destination
        try:
            zip_path = dest_path_obj.parent / zip_filename
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            raise ValueError(f"Cannot determine ZIP location: {e}") from e

        # Check if ZIP file already exists and generate unique name
        if zip_path.exists():
            zip_path = Path(self._get_unique_path(str(zip_path)))

        logger.info(f"Creating ZIP archive: {zip_path}")

        try:
            # Count total files and size for progress tracking
            total_files = 0
            total_size = 0

            for root, _dirs, files in os.walk(self.dest_folder):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        if Path(file_path).exists() and os.access(file_path, os.R_OK):
                            total_files += 1
                            total_size += os.path.getsize(file_path)
                    except (OSError, PermissionError):
                        continue

            if total_files == 0:
                raise ValueError("No accessible files found in destination folder")

            logger.info(
                f"ZIP will contain {total_files} files, "
                f"{total_size / (1024 * 1024):.1f} MB",
            )

            # Create ZIP archive
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                processed_files = 0
                processed_size = 0
                failed_files = 0

                for root, _dirs, files in os.walk(self.dest_folder):
                    for file in files:
                        if self.cancel_operation:
                            raise Exception("ZIP creation cancelled by user")

                        file_path = Path(root) / file

                        # Validate file before adding to ZIP
                        try:
                            if not Path(file_path).exists():
                                failed_files += 1
                                logger.warning(f"File no longer exists: {file_path}")
                                continue
                            if not os.access(file_path, os.R_OK):
                                failed_files += 1
                                logger.warning(f"Cannot read file: {file_path}")
                                continue

                            # Calculate relative path for archive
                            arcname = os.path.relpath(file_path, self.dest_folder)

                            # Add file to ZIP
                            zipf.write(file_path, arcname)
                            processed_files += 1
                            processed_size += os.path.getsize(file_path)

                            # Update progress every N files
                            if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:
                                progress = (
                                    PROGRESS_START_ZIP
                                    + (processed_files / total_files)
                                    * PROGRESS_ZIP_PERCENT
                                )
                                self.update_progress(
                                    progress,
                                    f"Added {processed_files}/{total_files} files "
                                    "to ZIP",
                                )

                        except (IOError, PermissionError, OSError) as e:
                            failed_files += 1
                            logger.warning(
                                f"Failed to add file to ZIP: {file_path} - {e}",
                            )
                            continue

                # Verify ZIP was created successfully
                if not zip_path.exists():
                    raise Exception("ZIP file was not created")

                # Verify ZIP size is reasonable
                try:
                    zip_size = zip_path.stat().st_size
                    if zip_size == 0:
                        raise Exception("ZIP file is empty")
                    logger.info(
                        f"ZIP archive created: {zip_path} ({processed_files} files, "
                        f"{processed_size / (1024 * 1024):.1f} MB, "
                        f"ZIP size: {zip_size / (1024 * 1024):.1f} MB)",
                    )
                except OSError as e:
                    logger.warning(f"Cannot verify ZIP file size: {e}")

                # Final summary
                if failed_files > 0:
                    logger.warning(
                        f"ZIP creation completed with {failed_files} failed files",
                    )
                else:
                    logger.info("ZIP creation completed successfully")

        except (IOError, PermissionError, OSError) as e:
            # Cleanup failed ZIP file
            if zip_path.exists():
                try:
                    zip_path.unlink()
                    logger.info(f"Cleaned up failed ZIP file: {zip_path}")
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup failed ZIP file: {zip_path} - "
                        f"{cleanup_error}",
                    )

            logger.error(f"Failed to create ZIP archive: {e}")
            raise Exception(f"Failed to create ZIP archive: {e}") from e

        return str(zip_path)
