# ruff: noqa: E501
# mypy: ignore-errors
"""BackupCopyMixin -- Backup creation and safe file copy methods."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import timezone, datetime
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

    @staticmethod
    def _validated_source_folders(source_folders: list[str]) -> list[str]:
        """Return only accessible folders from *source_folders*.

        Raises:
            ValueError: If no valid folders remain.
        """
        if not source_folders:
            raise ValueError("No source folders to backup")
        if not isinstance(source_folders, list):
            raise ValueError(
                f"Source folders must be a list, got {type(source_folders)}",
            )

        valid = []
        for folder in source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
            elif not Path(folder).exists():
                logger.warning(f"Source folder no longer exists: {folder}")
            elif not os.access(folder, os.R_OK):
                logger.warning(f"Cannot access source folder: {folder}")
            else:
                valid.append(folder)

        if not valid:
            raise ValueError("No valid source folders to backup")
        return valid

    def _backup_single_folder(self, folder: str, backup_base: Path) -> bool:
        """Backup one folder into *backup_base*. Returns True on success."""
        if folder is None:
            raise ValueError("folder must be provided")
        if not Path(folder).exists():
            logger.warning(f"Source folder no longer exists: {folder}")
            return False

        try:
            folder_name = Path(folder).name
            backup_path = backup_base / folder_name
            if backup_path.exists():
                backup_path = Path(self._get_unique_path(str(backup_path)))
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to create backup path for {folder}: {e}")
            return False

        try:
            shutil.copytree(folder, backup_path)
            logger.info(f"Backed up folder: {folder} -> {backup_path}")
            if not backup_path.exists() or not any(backup_path.iterdir()):
                raise OSError("Backup directory was not created or is empty")
            return True
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to backup folder {folder}: {e}")
            if backup_path.exists():
                try:
                    shutil.rmtree(backup_path, ignore_errors=True)
                except (PermissionError, OSError) as ce:
                    logger.warning(f"Cleanup failed for {backup_path}: {ce}")
            return False

    @staticmethod
    def _cleanup_backup_dir(backup_base: Path) -> None:
        """Remove an empty/failed backup directory."""
        if backup_base.exists():
            try:
                shutil.rmtree(backup_base, ignore_errors=True)
                logger.info(f"Cleaned up backup directory: {backup_base}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to cleanup {backup_base}: {e}")

    def create_backup(self) -> str | None:
        """Creates a backup of source folders before processing.

        Returns:
            Path to backup directory if successful [str], None if failed.
        """
        valid_folders = self._validated_source_folders(self.source_folders)  # type: ignore[attr-defined]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        try:
            backup_base = Path(valid_folders[0]).parent / f"backup_{timestamp}"
        except (PermissionError, OSError) as e:
            raise ValueError(f"Cannot determine backup location: {e}") from e

        self.update_status("Creating backup...")  # type: ignore[attr-defined]
        logger.info(f"Creating backup at: {backup_base}")

        try:
            backup_base.mkdir(parents=True, exist_ok=True)
            if not backup_base.exists():
                raise OSError("Failed to create backup base directory")
            if not os.access(backup_base, os.W_OK):
                raise PermissionError(
                    f"Cannot write to backup directory: {backup_base}",
                )

            successful = 0
            failed = 0
            for i, folder in enumerate(valid_folders):
                if self.cancel_operation:  # type: ignore[attr-defined]
                    logger.info("Backup operation cancelled by user")
                    return None

                if self._backup_single_folder(folder, backup_base):
                    successful += 1
                else:
                    failed += 1

                progress = (i + 1) / len(valid_folders) * PROGRESS_BACKUP_PERCENT
                self.update_progress(  # type: ignore[attr-defined]
                    progress,
                    f"Backing up folder {i + 1}/{len(valid_folders)}",
                )

            if successful == 0:
                logger.error("No folders were successfully backed up")
                self._cleanup_backup_dir(backup_base)
                return None

            if backup_base.exists() and any(backup_base.iterdir()):
                logger.info(
                    f"Backup completed: {successful} successful, {failed} failed",
                )
                return str(backup_base)

            logger.error("Backup directory is empty or was not created")
            return None

        except (PermissionError, OSError) as e:
            logger.error(f"Backup creation failed: {e}")
            self._cleanup_backup_dir(backup_base)
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
        if source_path is None:
            raise ValueError("source_path must be provided")
        source_path_obj, dest_path_obj = self._validate_copy_inputs(
            source_path, dest_path
        )

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                self._prepare_dest_directory(dest_path_obj)
                shutil.copy2(source_path, dest_path)

                if self._verify_copy(
                    source_path_obj, dest_path_obj, source_path, dest_path
                ):
                    return True

                # Verification failed; clean up and potentially retry
                try:
                    if dest_path_obj.exists():
                        dest_path_obj.unlink()
                except OSError:
                    pass
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    logger.info(
                        "Retrying copy due to verification failure "
                        f"(attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})",
                    )
                    continue

            except (OSError, PermissionError) as e:
                logger.warning(f"Copy attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(0.1 * (2**attempt))
                    continue
                else:
                    logger.error(
                        f"Failed to copy {source_path} after "
                        f"{MAX_RETRY_ATTEMPTS} attempts: {e}",
                    )
                    raise

        return False

    def _validate_copy_inputs(
        self, source_path: str, dest_path: str
    ) -> tuple[Path, Path]:
        """Validate source and destination paths for a copy operation.

        Returns:
            Tuple of (source_path_obj, dest_path_obj).
        """
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

        if not source_path_obj.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        if not source_path_obj.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        if not os.access(source_path, os.R_OK):
            raise PermissionError(f"Cannot read source file: {source_path}")

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

        return source_path_obj, dest_path_obj

    @staticmethod
    def _prepare_dest_directory(dest_path_obj: Path) -> None:
        """Ensure the destination directory exists and is writable."""
        dest_dir = dest_path_obj.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(dest_dir, os.W_OK):
            raise PermissionError(
                f"Cannot write to destination directory: {dest_dir}",
            )

    @staticmethod
    def _verify_copy(
        source_path_obj: Path,
        dest_path_obj: Path,
        source_path: str,
        dest_path: str,
    ) -> bool:
        """Verify that the copied file matches the source in size.

        Returns:
            True if sizes match, False otherwise.
        """
        if source_path_obj is None:
            raise ValueError("source_path_obj must be provided")
        try:
            if not dest_path_obj.exists():
                logger.error(f"Destination file was not created: {dest_path}")
                return False
            source_size = source_path_obj.stat().st_size
            dest_size = dest_path_obj.stat().st_size
            if source_size == dest_size:
                logger.debug(
                    f"Successfully copied {source_path} -> {dest_path} "
                    f"({source_size} bytes)",
                )
                return True
            logger.warning(
                f"Size mismatch after copy: source={source_size}, dest={dest_size}",
            )
        except OSError as e:
            logger.warning(f"Failed to verify copy sizes: {e}")
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
        except (PermissionError, OSError) as e:
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
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fallback_name = f"{filename}_{timestamp}{ext}"
        fallback_path = parent / fallback_name

        logger.warning(
            f"Exhausted counter attempts, using timestamp fallback: {fallback_path}",
        )
        return str(fallback_path)
