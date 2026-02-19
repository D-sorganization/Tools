"""BackupCopyMixin -- Backup creation and safe file copy methods."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from Folders_Tool_r0 import (
    MAX_COUNTER_ATTEMPTS,
    MAX_FILE_SIZE_MB,
    MAX_RETRY_ATTEMPTS,
    PROGRESS_BACKUP_PERCENT,
)

logger = logging.getLogger(__name__)


class BackupCopyMixin:
    """Backup creation, safe file copy, and unique path generation methods."""

    def create_backup(self) -> str | None:
        """Creates a backup of source folders before processing.

        Returns:
            Path to backup directory if successful [str], None if failed

        Raises:
            ValueError: If source_folders list is empty or invalid
            OSError: If file system operations fail during backup creation
            PermissionError: If insufficient permissions to create backup
            Exception: If backup process fails for other reasons
        """
        # Input validation
        if not self.source_folders:
            raise ValueError("No source folders to backup")
        if not isinstance(self.source_folders, list):
            raise ValueError(
                f"Source folders must be a list, got {type(self.source_folders)}",
            )

        # Validate each source folder
        valid_source_folders = []
        for folder in self.source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
                continue
            if not Path(folder).exists():
                logger.warning(f"Source folder no longer exists: {folder}")
                continue
            if not os.access(folder, os.R_OK):
                logger.warning(f"Cannot access source folder: {folder}")
                continue
            valid_source_folders.append(folder)

        if not valid_source_folders:
            raise ValueError("No valid source folders to backup")

        # Generate backup directory name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_base_name = f"backup_{timestamp}"

        # Create backup in parent directory of first source folder
        try:
            first_source_parent = Path(valid_source_folders[0]).parent
            backup_base = first_source_parent / backup_base_name
        except (IOError, PermissionError, OSError) as e:
            raise ValueError(f"Cannot determine backup location: {e}") from e

        self.update_status("Creating backup...")
        logger.info(f"Creating backup at: {backup_base}")

        try:
            # Create backup base directory
            backup_base.mkdir(parents=True, exist_ok=True)

            # Verify directory was created and is writable
            if not backup_base.exists():
                raise Exception("Failed to create backup base directory")
            if not os.access(backup_base, os.W_OK):
                raise PermissionError(
                    f"Cannot write to backup directory: {backup_base}",
                )

            total_folders = len(valid_source_folders)
            successful_backups = 0
            failed_backups = 0

            for i, folder in enumerate(valid_source_folders):
                if self.cancel_operation:
                    logger.info("Backup operation cancelled by user")
                    return None

                if not Path(folder).exists():
                    logger.warning(f"Source folder no longer exists: {folder}")
                    continue

                # Create backup path
                try:
                    folder_name = Path(folder).name
                    backup_path = backup_base / folder_name

                    # Ensure backup path is unique
                    if backup_path.exists():
                        unique_path = self._get_unique_path(str(backup_path))
                        backup_path = Path(unique_path)
                except (IOError, PermissionError, OSError) as e:
                    logger.error(f"Failed to create backup path for {folder}: {e}")
                    failed_backups += 1
                    continue

                try:
                    # Create backup
                    shutil.copytree(folder, backup_path)
                    successful_backups += 1
                    logger.info(f"Backed up folder: {folder} -> {backup_path}")

                    # Verify backup was created successfully
                    if not backup_path.exists():
                        raise Exception("Backup directory was not created")
                    if not any(backup_path.iterdir()):
                        raise Exception("Backup directory is empty")

                except (IOError, PermissionError, OSError) as e:
                    failed_backups += 1
                    logger.error(f"Failed to backup folder {folder}: {e}")

                    # Clean up failed backup
                    if backup_path.exists():
                        try:
                            shutil.rmtree(backup_path, ignore_errors=True)
                            logger.info(f"Cleaned up failed backup: {backup_path}")
                        except (IOError, PermissionError, OSError) as cleanup_error:
                            logger.warning(
                                f"Failed to cleanup failed backup: {backup_path} - "
                                f"{cleanup_error}",
                            )

                    # Continue with other folders
                    continue

                # Update progress
                progress = (
                    (i + 1) / total_folders * PROGRESS_BACKUP_PERCENT
                )  # PROGRESS_BACKUP_PERCENT% for backup
                self.update_progress(
                    progress,
                    f"Backing up folder {i + 1}/{total_folders}",
                )

            # Verify overall backup success
            if successful_backups == 0:
                logger.error("No folders were successfully backed up")
                # Clean up empty backup directory
                if backup_base.exists():
                    try:
                        shutil.rmtree(backup_base, ignore_errors=True)
                        logger.info(f"Cleaned up empty backup directory: {backup_base}")
                    except (IOError, PermissionError, OSError) as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup empty backup directory: "
                            f"{backup_base} - {cleanup_error}",
                        )
                return None

            # Final verification
            if backup_base.exists() and any(backup_base.iterdir()):
                logger.info(f"Backup completed successfully: {backup_base}")
                logger.info(
                    f"Backup summary: {successful_backups} successful, "
                    f"{failed_backups} failed",
                )
                return str(backup_base)
            else:
                logger.error("Backup directory is empty or was not created")
                return None

        except (IOError, PermissionError, OSError) as e:
            logger.error(f"Backup creation failed: {e}")
            # Cleanup failed backup
            if backup_base.exists():
                try:
                    shutil.rmtree(backup_base, ignore_errors=True)
                    logger.info(f"Cleaned up failed backup: {backup_base}")
                except (IOError, PermissionError, OSError) as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup failed backup: {backup_base} - "
                        f"{cleanup_error}",
                    )
            raise

    def _safe_copy_file(self, source_path: str, dest_path: str) -> bool:
        """Safely copy a file with retry logic and error handling.

        Args:
            source_path: Source file path [str] - must exist and be readable
            dest_path: Destination file path [str]
                - parent directory will be created if needed

        Returns:
            True if copy successful, False otherwise

        Raises:
            OSError: If file operations fail after all retry attempts
            IOError: If file I/O operations fail
            ValueError: If source_path is empty or invalid
            FileNotFoundError: If source file does not exist
            PermissionError: If insufficient permissions to read source or write
                destination
        """
        # Input validation
        if not source_path or not isinstance(source_path, str):
            raise ValueError(
                f"Source path must be non-empty string, got {type(source_path)}",
            )
        if not dest_path or not isinstance(dest_path, str):
            raise ValueError(
                f"Destination path must be non-empty string, got {type(dest_path)}",
            )

        source_path_obj = Path(source_path)
        dest_path_obj = Path(dest_path)

        # Validate source file exists and is accessible
        if not source_path_obj.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        if not source_path_obj.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        if not os.access(source_path, os.R_OK):
            raise PermissionError(f"Cannot read source file: {source_path}")

        # Validate source file size
        try:
            source_size = source_path_obj.stat().st_size
            if source_size == 0:
                logger.warning(f"Source file is empty: {source_path}")
            elif source_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(
                    f"Source file exceeds maximum size limit: {source_path} "
                    f"({source_size / (1024 * 1024):.1f} MB)",
                )
        except OSError as e:
            logger.warning(f"Cannot access source file size: {source_path} - {e}")

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Ensure destination directory exists
                dest_dir = dest_path_obj.parent
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Check if destination directory is writable
                if not os.access(dest_dir, os.W_OK):
                    raise PermissionError(
                        f"Cannot write to destination directory: {dest_dir}",
                    )

                # Copy file with metadata preservation
                shutil.copy2(source_path, dest_path)

                # Verify copy was successful
                if dest_path_obj.exists():
                    try:
                        source_size = source_path_obj.stat().st_size
                        dest_size = dest_path_obj.stat().st_size
                        if source_size == dest_size:
                            logger.debug(
                                f"Successfully copied {source_path} -> {dest_path} "
                                f"({source_size} bytes)",
                            )
                            return True
                        else:
                            logger.warning(
                                f"Size mismatch after copy: source={source_size}, "
                                f"dest={dest_size}",
                            )
                            # Size mismatch, remove failed copy and retry
                            if dest_path_obj.exists():
                                dest_path_obj.unlink()
                            if attempt < MAX_RETRY_ATTEMPTS - 1:
                                logger.info(
                                    "Retrying copy due to size mismatch "
                                    f"(attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})",
                                )
                                continue
                    except OSError as e:
                        logger.warning(f"Failed to verify copy sizes: {e}")
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            continue
                else:
                    logger.error(f"Destination file was not created: {dest_path}")
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        continue

            except (OSError, PermissionError) as e:
                logger.warning(f"Copy attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Wait before retry (exponential backoff)
                    time.sleep(0.1 * (2**attempt))
                    continue
                else:
                    logger.error(
                        f"Failed to copy {source_path} after "
                        f"{MAX_RETRY_ATTEMPTS} attempts: {e}",
                    )
                    raise

        return False

    def _get_unique_path(self, path: str) -> str:
        """Generate a unique path by appending counter if path exists.

        Args:
            path: Original file or directory path [str]
                - will be analyzed for name and extension

        Returns:
            Unique path that doesn't exist [str]
                - original path or path with counter suffix

        Raises:
            ValueError: If path is empty or invalid
            OSError: If file system operations fail during path checking
            PermissionError: If insufficient permissions to check path existence
        """
        # Input validation
        if not path or not isinstance(path, str):
            raise ValueError(f"Path must be non-empty string, got {type(path)}")

        path_obj = Path(path)

        # Validate path format
        try:
            # Check if path is absolute or relative
            if path_obj.is_absolute():
                # Ensure drive exists on Windows
                if sys.platform == "win32" and len(path_obj.parts) > 0:
                    drive = path_obj.parts[0]
                    if not Path(drive).exists():
                        raise ValueError(f"Drive does not exist: {drive}")
        except (IOError, PermissionError, OSError) as e:
            raise ValueError(f"Invalid path format: {path} - {e}") from e

        # Check if path already exists
        try:
            if not path_obj.exists():
                return path
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot check if path exists: {path} - {e}")
            # Assume it doesn't exist and return original path
            return path

        # Path exists, generate unique version
        parent = path_obj.parent
        name = path_obj.name

        # Determine if this is a file or directory
        try:
            is_file = path_obj.is_file()
        except (OSError, PermissionError):
            # If we can't determine, assume it's a file if it has an extension
            is_file = "." in name and not name.endswith(".")

        if is_file:
            filename = path_obj.stem
            ext = path_obj.suffix
        else:
            filename = name
            ext = ""

        # Generate unique path with counter
        counter = 1

        while counter <= MAX_COUNTER_ATTEMPTS:
            new_name = f"{filename} ({counter}){ext}"
            new_path = parent / new_name

            try:
                if not new_path.exists():
                    logger.debug(f"Generated unique path: {path} -> {new_path}")
                    return str(new_path)
            except (OSError, PermissionError) as e:
                logger.warning(
                    f"Cannot check if generated path exists: {new_path} - {e}",
                )
                # If we can't check, assume it's safe to use
                return str(new_path)

            counter += 1

        # If we've exhausted all reasonable attempts, append timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_name = f"{filename}_{timestamp}{ext}"
        fallback_path = parent / fallback_name

        logger.warning(
            f"Exhausted counter attempts, using timestamp fallback: {fallback_path}",
        )
        return str(fallback_path)
