"""AnalysisMixin -- Analysis report and input validation for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from Folders_Tool_r0 import (
    MAX_FILE_SIZE_MB,
    MIN_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Analysis and validation methods for FolderProcessorApp.

    Expects the host class to provide:
    - self.source_folders: list[str]
    - self.dest_folder: str
    - self.cancel_operation: bool
    - self.filter_extensions: tk.StringVar
    - self.min_file_size: tk.StringVar
    - self.max_file_size: tk.StringVar
    - self._validate_constants() -> None
    - self.validate_size_inputs() -> bool
    """

    def generate_analysis_report(self) -> str | None:
        """Generates a comprehensive analysis report.

        Returns:
            Formatted analysis report [str] if successful, None if cancelled or failed

        Raises:
            ValueError: If source_folders list is empty or invalid
            OSError: If file system operations fail during analysis
            PermissionError: If insufficient permissions to access source folders
        """
        valid_source_folders = self._validate_source_folders()

        report = ["=== FOLDER ANALYSIS REPORT ===", f"Generated: {datetime.now()}", ""]
        logger.info(f"Starting analysis of {len(valid_source_folders)} source folders")

        total_files = 0
        total_size = 0
        file_types: dict[str, int] = defaultdict(int)
        size_by_type: dict[str, int] = defaultdict(int)
        largest_files: list[tuple[Path, int]] = []
        analysis_errors: list[str] = []

        for folder in valid_source_folders:
            if self.cancel_operation:
                logger.info("Analysis cancelled by user")
                return None

            result = self._analyze_single_folder(folder, report)
            if result is None:
                continue
            folder_files, folder_size, folder_largest, folder_errors_list = result

            total_files += folder_files
            total_size += folder_size
            for fp, fs in folder_largest:
                file_ext = Path(fp).suffix.lower() or "no_extension"
                file_types[file_ext] += 1
                size_by_type[file_ext] += fs
            largest_files.extend(folder_largest)
            largest_files.sort(key=lambda x: x[1], reverse=True)
            largest_files = largest_files[:10]
            analysis_errors.extend(folder_errors_list)

        self._append_report_summary(
            report,
            total_files,
            total_size,
            file_types,
            size_by_type,
            largest_files,
            analysis_errors,
            valid_source_folders,
        )

        logger.info(
            f"Analysis completed: {total_files} files, "
            f"{total_size / (1024 * 1024):.1f} MB",
        )
        if analysis_errors:
            logger.warning(f"Analysis completed with {len(analysis_errors)} errors")

        return "\n".join(report)

    def _validate_source_folders(self) -> list[str]:
        """Validate and filter source folders, raising on empty input."""
        if not self.source_folders:
            raise ValueError("No source folders to analyze")
        if not isinstance(self.source_folders, list):
            raise ValueError(
                f"Source folders must be a list, got {type(self.source_folders)}",
            )

        valid = []
        for folder in self.source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
            elif not Path(folder).exists():
                logger.warning(f"Source folder no longer exists: {folder}")
            elif not os.access(folder, os.R_OK):
                logger.warning(f"Cannot access source folder: {folder}")
            else:
                valid.append(folder)

        if not valid:
            raise ValueError("No valid source folders to analyze")
        return valid

    def _analyze_single_folder(
        self, folder: str, report: list[str]
    ) -> tuple[int, int, list[tuple[Path, int]], list[str]] | None:
        """Analyze one folder, appending per-folder lines to *report*.

        Returns (files, size, largest_files, errors) or None on OS error.
        """
        report.append(f"Analyzing: {folder}")
        folder_files = 0
        folder_size = 0
        folder_errors = 0
        largest: list[tuple[Path, int]] = []

        try:
            for root, _dirs, files in os.walk(folder):
                if self.cancel_operation:
                    break  # type: ignore[unreachable]

                for file in files:
                    if self.cancel_operation:
                        break  # type: ignore[unreachable]

                    file_path = Path(root) / file
                    try:
                        if not file_path.exists() or not os.access(file_path, os.R_OK):
                            folder_errors += 1
                            continue

                        file_size = os.path.getsize(file_path)
                        if file_size < MIN_FILE_SIZE_BYTES:
                            continue
                        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                            logger.warning(
                                f"File exceeds maximum size: {file_path} "
                                f"({file_size / (1024 * 1024):.1f} MB)",
                            )

                        folder_files += 1
                        folder_size += file_size
                        largest.append((file_path, file_size))
                        if len(largest) > 10:
                            largest.sort(key=lambda x: x[1], reverse=True)
                            largest = largest[:10]

                    except (OSError, PermissionError) as e:
                        folder_errors += 1
                        logger.debug(f"Cannot access file {file_path}: {e}")

            errors_list: list[str] = []
            if folder_errors > 0:
                report.append(
                    f"  Files: {folder_files}, "
                    f"Size: {folder_size / (1024 * 1024):.1f} MB, "
                    f"Errors: {folder_errors}",
                )
                errors_list.append(f"Folder {folder}: {folder_errors} access errors")
            else:
                report.append(
                    f"  Files: {folder_files}, "
                    f"Size: {folder_size / (1024 * 1024):.1f} MB",
                )
            return folder_files, folder_size, largest, errors_list

        except (OSError, PermissionError) as e:
            error_msg = f"Error accessing folder {folder}: {e}"
            report.append(f"  ERROR: {error_msg}")
            logger.error(error_msg)
            return None

    def _append_report_summary(
        self,
        report: list[str],
        total_files: int,
        total_size: int,
        file_types: dict[str, int],
        size_by_type: dict[str, int],
        largest_files: list[tuple[Path, int]],
        analysis_errors: list[str],
        valid_source_folders: list[str],
    ) -> None:
        """Append summary statistics, largest files, and metadata to report."""
        report.extend(
            [
                "",
                f"TOTAL FILES: {total_files}",
                f"TOTAL SIZE: {total_size / (1024 * 1024):.1f} MB",
                "",
                "FILE TYPES:",
            ],
        )

        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            size_mb = size_by_type[ext] / (1024 * 1024)
            report.append(f"  {ext}: {count} files, {size_mb:.1f} MB")

        report.extend(["", "LARGEST FILES:"])
        for file_path, size in sorted(largest_files, key=lambda x: x[1], reverse=True):
            size_mb = size / (1024 * 1024)
            report.append(f"  {Path(file_path).name}: {size_mb:.1f} MB")

        if analysis_errors:
            report.extend(["", "ANALYSIS ERRORS:", *analysis_errors])

        report.extend(
            [
                "",
                "ANALYSIS METADATA:",
                f"  Source folders processed: {len(valid_source_folders)}",
                f"  Total folders analyzed: {len(valid_source_folders)}",
                f"  Analysis timestamp: {datetime.now()}",
                f"  File size limits: {MIN_FILE_SIZE_BYTES} bytes - "
                f"{MAX_FILE_SIZE_MB} MB",
            ],
        )

    def validate_inputs(self, check_destination: bool = True) -> bool:
        """Validate user inputs before processing.

        Args:
            check_destination: Whether to validate destination folder selection [bool]
                - defaults to True

        Returns:
            True if inputs are valid, False otherwise

        Raises:
            ValueError: If file size inputs are invalid
            Exception: If extension filter validation fails
        """
        from tkinter import messagebox

        if not self.source_folders:
            messagebox.showerror("Error", "Please add at least one source folder.")
            return False

        if check_destination:
            if not self.dest_folder:
                messagebox.showerror("Error", "Please select a destination folder.")
                return False
            if any(src == self.dest_folder for src in self.source_folders):
                messagebox.showerror(
                    "Error",
                    "The destination folder cannot be a source folder.",
                )
                return False

        # Validate file size inputs
        if not self.validate_size_inputs():
            return False

        # Validate extension filter format
        extensions = self.filter_extensions.get().strip()
        if extensions:
            try:
                ext_list = [ext.strip().lower() for ext in extensions.split(",")]
                # Validate each extension starts with a dot
                for ext in ext_list:
                    if ext and not ext.startswith("."):
                        messagebox.showwarning(
                            "Invalid Extension Format",
                            f"Extension '{ext}' should start with a dot "
                            "(e.g., '.txt').",
                        )
                        return False
            except (KeyError, ValueError, TypeError):
                messagebox.showerror(
                    "Error",
                    "Invalid extension filter format. Use comma-separated values "
                    "like '.txt,.pdf'.",
                )
                return False

        return True

    def validate_application_state(self) -> dict[str, bool]:
        """Validates the current application state and returns validation results.

        Returns:
            Dictionary mapping validation checks to their results [dict]
                - True if valid, False if invalid

        Example:
            {
                'source_folders_exist': True,
                'destination_writable': False,
                'constants_valid': True
            }
        """
        validation_results: dict[str, bool] = {}

        # Check source folders
        validation_results["source_folders_exist"] = (
            all(Path(folder).exists() for folder in self.source_folders)
            if self.source_folders
            else True
        )

        validation_results["source_folders_readable"] = (
            all(os.access(folder, os.R_OK) for folder in self.source_folders)
            if self.source_folders
            else True
        )

        # Check destination folder
        if self.dest_folder:
            validation_results["destination_exists"] = Path(self.dest_folder).exists()
            validation_results["destination_writable"] = os.access(
                self.dest_folder,
                os.W_OK,
            )
        else:
            validation_results["destination_exists"] = (
                True  # Not required for all modes
            )
            validation_results["destination_writable"] = (
                True  # Not required for all modes
            )

        # Check file size inputs
        try:
            min_size = float(self.min_file_size.get() or 0)
            max_size = float(self.max_file_size.get() or MAX_FILE_SIZE_MB)
            validation_results["size_inputs_valid"] = (
                0 <= min_size <= MAX_FILE_SIZE_MB
                and 0 <= max_size <= MAX_FILE_SIZE_MB
                and min_size <= max_size
            )
        except ValueError:
            validation_results["size_inputs_valid"] = False

        # Check extension filter format
        extensions = self.filter_extensions.get().strip()
        if extensions:
            try:
                ext_list = [ext.strip().lower() for ext in extensions.split(",")]
                validation_results["extension_filter_valid"] = all(
                    ext.startswith(".") for ext in ext_list if ext
                )
            except (KeyError, ValueError, TypeError):
                validation_results["extension_filter_valid"] = False
        else:
            validation_results["extension_filter_valid"] = True

        # Check constants
        try:
            self._validate_constants()
            validation_results["constants_valid"] = True
        except ValueError:
            validation_results["constants_valid"] = False

        return validation_results
