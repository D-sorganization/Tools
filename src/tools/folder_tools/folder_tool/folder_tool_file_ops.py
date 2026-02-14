"""FileOperationsMixin -- File operation methods for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

from Folders_Tool_r0 import (
    MAX_ARCHIVE_SIZE_RATIO,
    MAX_COUNTER_ATTEMPTS,
    MAX_FILE_SIZE_MB,
    MAX_LOG_ENTRIES,
    MAX_RETRY_ATTEMPTS,
    MAX_UI_UPDATE_FREQUENCY,
    PROGRESS_BACKUP_PERCENT,
    PROGRESS_MAIN_OP_PERCENT,
    PROGRESS_START_MAIN,
)

logger = logging.getLogger(__name__)


class FileOperationsMixin:
    """File operation methods for FolderProcessorApp."""

    def validate_file_filters(self, file_path: str) -> bool:
        """Validates if a file meets the filtering criteria.

        Args:
            file_path: Path to the file to validate [str] - must be absolute path

        Returns:
            True if file passes all filters, False otherwise

        Raises:
            OSError: If file system operations fail
            ValueError: If file size validation fails
        """
        if self.cancel_operation:
            return False

        # Extension filter
        extensions = self.filter_extensions.get().strip()
        if extensions:
            ext_list = [ext.strip().lower() for ext in extensions.split(",")]
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ext_list:
                return False

        # Size filter
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)

            # Validate minimum size
            min_size_mb = float(self.min_file_size.get() or 0)
            if min_size_mb < 0:
                min_size_mb = 0  # Reset invalid negative values
                self.min_file_size.set("0")
            if file_size_mb < min_size_mb:
                return False

            # Validate maximum size
            max_size_str = self.max_file_size.get().strip()
            if max_size_str:
                try:
                    max_size_mb = float(max_size_str)
                    if max_size_mb < 0:
                        max_size_mb = MAX_FILE_SIZE_MB  # Reset invalid negative values
                        self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                    if file_size_mb > max_size_mb:
                        return False

                    # Validate against absolute maximum
                    if max_size_mb > MAX_FILE_SIZE_MB:
                        max_size_mb = MAX_FILE_SIZE_MB
                        self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                        return False
                except ValueError:
                    # Invalid input, reset to empty
                    self.max_file_size.set("")
                    return False

        except (ValueError, OSError):
            return False

        return True

    def validate_size_inputs(self) -> bool:
        """Validates file size input fields and provides user feedback.

        Returns:
            True if inputs are valid, False otherwise
        """
        try:
            # Validate minimum size
            min_size_str = self.min_file_size.get().strip()
            if min_size_str:
                min_size_mb = float(min_size_str)
                if min_size_mb < 0:
                    messagebox.showwarning(
                        "Invalid Input",
                        "Minimum file size cannot be negative. Setting to 0 MB.",
                    )
                    self.min_file_size.set("0")
                    return False
                if min_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Minimum file size cannot exceed {MAX_FILE_SIZE_MB} MB. "
                        "Setting to 0 MB.",
                    )
                    self.min_file_size.set("0")
                    return False

            # Validate maximum size
            max_size_str = self.max_file_size.get().strip()
            if max_size_str:
                max_size_mb = float(max_size_str)
                if max_size_mb < 0:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Maximum file size cannot be negative. Setting to "
                        f"{MAX_FILE_SIZE_MB} MB.",
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                    return False
                if max_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Maximum file size cannot exceed {MAX_FILE_SIZE_MB} MB. "
                        f"Setting to {MAX_FILE_SIZE_MB} MB.",
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                    return False

                # Check if min > max
                if min_size_str and float(min_size_str) > max_size_mb:
                    messagebox.showwarning(
                        "Invalid Input",
                        "Minimum file size cannot be greater than maximum file size.",
                    )
                    return False

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numeric values for file sizes.",
            )
            return False

        return True

    def get_organized_path(self, file_path: str, dest_base: str) -> str:
        """Returns the organized destination path based on organization options.

        Args:
            file_path: Source file path [str]
                - used to determine file type and modification date
            dest_base: Base destination directory [str]
                - where organized files will be placed

        Returns:
            Organized destination path [str]
                - includes type/date subdirectories if enabled

        Raises:
            OSError: If file system operations fail during path construction
        """
        filename = Path(file_path).name
        dest_path = dest_base

        # Organize by type
        if self.organize_by_type_var.get():
            file_ext = Path(filename).suffix.lower()
            type_mapping = {
                ".jpg": "Images",
                ".jpeg": "Images",
                ".png": "Images",
                ".gif": "Images",
                ".bmp": "Images",
                ".mp4": "Videos",
                ".avi": "Videos",
                ".mov": "Videos",
                ".wmv": "Videos",
                ".mkv": "Videos",
                ".mp3": "Audio",
                ".wav": "Audio",
                ".flac": "Audio",
                ".aac": "Audio",
                ".pdf": "Documents",
                ".doc": "Documents",
                ".docx": "Documents",
                ".txt": "Documents",
                ".zip": "Archives",
                ".rar": "Archives",
                ".7z": "Archives",
                ".tar": "Archives",
            }
            file_type = type_mapping.get(file_ext, "Other")
            dest_path = Path(dest_path) / file_type

        # Organize by date
        if self.organize_by_date_var.get():
            try:
                mtime = os.path.getmtime(file_path)
                date_folder = datetime.fromtimestamp(mtime).strftime("%Y/%m")
                dest_path = Path(dest_path) / date_folder
            except OSError:
                dest_path = Path(dest_path) / "Unknown_Date"

        return Path(dest_path) / filename

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

    def _combine_folders_enhanced(self) -> list[str]:
        """Enhanced combine operation with filtering and organization."""
        log = []
        file_count = 0
        renamed_count = 0
        skipped_count = 0
        failed_count = 0

        Path(self.dest_folder).mkdir(parents=True, exist_ok=True)

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:
            if self.cancel_operation:
                break

            for root, _dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:
                        break  # type: ignore[unreachable]

                    source_path = Path(root) / file

                    # Apply filters
                    if not self.validate_file_filters(source_path):
                        skipped_count += 1
                        processed_files += 1
                        continue

                    # Get organized destination path
                    dest_path = self.get_organized_path(source_path, self.dest_folder)
                    dest_dir = Path(dest_path).parent

                    # Create destination directory if needed
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{Path(final_dest_path).name}'",
                        )
                        renamed_count += 1

                    try:
                        if not self.preview_mode_var.get():
                            if self._safe_copy_file(source_path, final_dest_path):
                                file_count += 1
                            else:
                                failed_count += 1
                                log.append(f"FAILED to copy '{file}' after retries")
                        else:
                            file_count += 1  # Count in preview mode
                    except (KeyError, ValueError, TypeError) as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")

                    processed_files += 1
                    if (
                        processed_files % MAX_UI_UPDATE_FREQUENCY == 0
                    ):  # Update progress every N files
                        progress = (
                            PROGRESS_START_MAIN
                            + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
                        )
                        self.update_progress(
                            progress,
                            f"Processed {processed_files}/{total_files} files",
                        )

        summary = [
            f"Processed {file_count} files.",
            f"Renamed {renamed_count} files due to duplicates.",
            f"Skipped {skipped_count} files due to filters.",
        ]

        if failed_count > 0:
            summary.append(f"Failed to copy {failed_count} files.")

        if self.preview_mode_var.get():
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]

    # --- Keep existing methods for compatibility ---

    def _perform_deduplication(self, target_folder: str) -> list[str]:
        """Core logic to find and delete renamed duplicates in a single
        target folder.
        """
        log = []
        deleted_count = 0
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")

        if not self.preview_mode_var.get():
            confirm = messagebox.askyesno(
                "Confirm Deletion",
                f"This will permanently delete duplicate files in:\n{target_folder}\n\n"
                "It keeps the newest version of files like 'file (1).txt'. "
                "This cannot be undone. Are you sure?",
            )
            if not confirm:
                return ["Deduplication cancelled by user."]

        log.append(f"Processing folder: {target_folder}")
        for dirpath, _, filenames in os.walk(target_folder):
            if self.cancel_operation:
                break

            files_by_base_name: dict[str, list[str]] = {}
            for filename in filenames:
                match = pattern.match(filename)
                if match:
                    base, _, ext = match.groups()
                    base_name = f"{base}{ext}"
                    files_by_base_name.setdefault(base_name, []).append(
                        str(Path(dirpath) / filename),
                    )

            for base_name, files in files_by_base_name.items():
                if len(files) > 1:
                    try:
                        file_to_keep = max(files, key=lambda f: Path(f).stat().st_mtime)
                    except FileNotFoundError:
                        continue

                    log.append(
                        f"Duplicate set for '{base_name}': Keeping "
                        f"'{Path(file_to_keep).name}'",
                    )

                    for file_path in files:
                        if file_path != file_to_keep:
                            try:
                                if not self.preview_mode_var.get():
                                    Path(file_path).unlink()
                                mode_str = (
                                    "WOULD DELETE"
                                    if self.preview_mode_var.get()
                                    else "DEL"
                                )
                                log.append(
                                    f"  - {mode_str}: '{Path(file_path).name}'",
                                )
                                deleted_count += 1
                            except OSError as e:
                                log.append(
                                    f"  - FAILED to delete '{Path(file_path).name}': "
                                    f"{e}",
                                )

        summary = [
            f"Deduplication "
            f"{'preview' if self.preview_mode_var.get() else 'complete'}.",
            f"{'Would delete' if self.preview_mode_var.get() else 'Deleted'} a total "
            f"of {deleted_count} files.",
            *log[:MAX_LOG_ENTRIES],
        ]

        if len(log) > MAX_LOG_ENTRIES:
            summary.append("... (see log for full details)")

        return summary

    # Keep other existing methods...

    def _run_deduplicate_main_op(self) -> list[str]:
        """Run deduplication as a main, in-place operation on source folders."""
        full_log = []
        for folder in self.source_folders:
            if self.cancel_operation:
                break
            folder_log = self._perform_deduplication(folder)
            full_log.extend(folder_log)
            full_log.append("---")
        return full_log

    def _flatten_folders(self) -> list[str]:
        """Flatten folder structure by moving all files to root level of destination.

        Returns:
            List of log messages describing the operation results
        """
        log = []
        moved_count = 0
        skipped_count = 0
        failed_count = 0

        os.makedirs(self.dest_folder, exist_ok=True)

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:
            if self.cancel_operation:
                break

            for root, _dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:
                        break  # type: ignore[unreachable]

                    source_path = Path(root) / file

                    # Apply filters
                    if not self.validate_file_filters(source_path):
                        skipped_count += 1
                        processed_files += 1
                        continue

                    # Get organized destination path (flattened to root)
                    dest_path = self.get_organized_path(source_path, self.dest_folder)
                    dest_dir = Path(dest_path).parent

                    # Create destination directory if needed
                    os.makedirs(dest_dir, exist_ok=True)

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{Path(final_dest_path).name}'",
                        )

                    try:
                        if not self.preview_mode_var.get():
                            if self._safe_copy_file(source_path, final_dest_path):
                                moved_count += 1
                            else:
                                failed_count += 1
                                log.append(f"FAILED to copy '{file}' after retries")
                        else:
                            moved_count += 1  # Count in preview mode
                    except (KeyError, ValueError, TypeError) as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")

                    processed_files += 1
                    if (
                        processed_files % MAX_UI_UPDATE_FREQUENCY == 0
                    ):  # Update progress every N files
                        progress = (
                            PROGRESS_START_MAIN
                            + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
                        )
                        self.update_progress(
                            progress,
                            f"Processed {processed_files}/{total_files} files",
                        )

        summary = [
            f"Flattened {moved_count} files to destination root level.",
            f"Skipped {skipped_count} files due to filters.",
        ]

        if failed_count > 0:
            summary.append(f"Failed to copy {failed_count} files.")

        if self.preview_mode_var.get():
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]

    def _prune_empty_folders(self) -> list[str]:
        """Copy source folders to destination while preserving structure but
        skipping empty sub-folders.

        Returns:
            List of log messages describing the operation results
        """
        log = []
        file_count = 0
        processed_folders = 0
        empty_folders_skipped = 0
        failed_count = 0

        os.makedirs(self.dest_folder, exist_ok=True)

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:
            if self.cancel_operation:
                break

            src_name = Path(src).name
            dest_src_path = Path(self.dest_folder) / src_name

            for root, dirs, files in os.walk(src):
                if self.cancel_operation:
                    break  # type: ignore[unreachable]

                # Skip empty folders
                if not files and not any(
                    any(Path(root, d).iterdir())
                    for d in dirs
                    if Path(Path(root).exists() / d)
                ):
                    empty_folders_skipped += 1
                    continue

                # Calculate relative path from source root
                rel_path = os.path.relpath(root, src)
                dest_path = Path(dest_src_path) / rel_path

                # Create destination directory
                os.makedirs(dest_path, exist_ok=True)

                # Copy files in this directory
                for file in files:
                    if self.cancel_operation:
                        break  # type: ignore[unreachable]

                    source_file_path = Path(root) / file

                    # Apply filters
                    if not self.validate_file_filters(source_file_path):
                        processed_files += 1
                        continue

                    dest_file_path = Path(dest_path) / file

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_file_path)
                    if final_dest_path != dest_file_path:
                        log.append(
                            f"Renamed: '{file}' to '{Path(final_dest_path).name}'",
                        )

                    try:
                        if not self.preview_mode_var.get():
                            if self._safe_copy_file(source_file_path, final_dest_path):
                                file_count += 1
                            else:
                                failed_count += 1
                                log.append(f"FAILED to copy '{file}' after retries")
                        else:
                            file_count += 1  # Count in preview mode
                    except (KeyError, ValueError, TypeError) as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")

                    processed_files += 1
                    if (
                        processed_files % MAX_UI_UPDATE_FREQUENCY == 0
                    ):  # Update progress every N files
                        progress = (
                            PROGRESS_START_MAIN
                            + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
                        )
                        self.update_progress(
                            progress,
                            f"Processed {processed_files}/{total_files} files",
                        )

                processed_folders += 1

        summary = [
            f"Processed {processed_folders} non-empty source folder(s).",
            f"Copied a total of {file_count} files.",
            f"Skipped {empty_folders_skipped} empty folders.",
        ]

        if failed_count > 0:
            summary.append(f"Failed to copy {failed_count} files.")

        if self.preview_mode_var.get():
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]
