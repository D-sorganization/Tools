"""FileValidationMixin -- File validation methods for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

from Folders_Tool_r0 import (
    MAX_FILE_SIZE_MB,
)

logger = logging.getLogger(__name__)


class FileValidationMixin:
    """File validation and path organization methods."""

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
        assert file_path is not None, "file_path must be provided"
        if self.cancel_operation:  # type: ignore[attr-defined]
            return False

        # Extension filter
        extensions = self.filter_extensions.get().strip()  # type: ignore[attr-defined]
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
            min_size_mb = float(self.min_file_size.get() or 0)  # type: ignore[attr-defined]
            if min_size_mb < 0:
                min_size_mb = 0  # Reset invalid negative values
                self.min_file_size.set("0")  # type: ignore[attr-defined]
            if file_size_mb < min_size_mb:
                return False

            # Validate maximum size
            max_size_str = self.max_file_size.get().strip()  # type: ignore[attr-defined]
            if max_size_str:
                try:
                    max_size_mb = float(max_size_str)
                    if max_size_mb < 0:
                        max_size_mb = MAX_FILE_SIZE_MB  # Reset invalid negative values
                        self.max_file_size.set(str(MAX_FILE_SIZE_MB))  # type: ignore[attr-defined]
                    if file_size_mb > max_size_mb:
                        return False

                    # Validate against absolute maximum
                    if max_size_mb > MAX_FILE_SIZE_MB:
                        max_size_mb = MAX_FILE_SIZE_MB
                        self.max_file_size.set(str(MAX_FILE_SIZE_MB))  # type: ignore[attr-defined]
                        return False
                except ValueError:
                    # Invalid input, reset to empty
                    self.max_file_size.set("")  # type: ignore[attr-defined]
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
            min_size_str = self.min_file_size.get().strip()  # type: ignore[attr-defined]
            if min_size_str:
                min_size_mb = float(min_size_str)
                if min_size_mb < 0:
                    messagebox.showwarning(
                        "Invalid Input",
                        "Minimum file size cannot be negative. Setting to 0 MB.",
                    )
                    self.min_file_size.set("0")  # type: ignore[attr-defined]
                    return False
                if min_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Minimum file size cannot exceed {MAX_FILE_SIZE_MB} MB. "
                        "Setting to 0 MB.",
                    )
                    self.min_file_size.set("0")  # type: ignore[attr-defined]
                    return False

            # Validate maximum size
            max_size_str = self.max_file_size.get().strip()  # type: ignore[attr-defined]
            if max_size_str:
                max_size_mb = float(max_size_str)
                if max_size_mb < 0:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Maximum file size cannot be negative. Setting to "
                        f"{MAX_FILE_SIZE_MB} MB.",
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))  # type: ignore[attr-defined]
                    return False
                if max_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Maximum file size cannot exceed {MAX_FILE_SIZE_MB} MB. "
                        f"Setting to {MAX_FILE_SIZE_MB} MB.",
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))  # type: ignore[attr-defined]
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
        assert file_path is not None, "file_path must be provided"
        filename = Path(file_path).name
        dest_path = dest_base

        # Organize by type
        if self.organize_by_type_var.get():  # type: ignore[attr-defined]
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
            dest_path = Path(dest_path) / file_type  # type: ignore[assignment]

        # Organize by date
        if self.organize_by_date_var.get():  # type: ignore[attr-defined]
            try:
                mtime = os.path.getmtime(file_path)
                date_folder = datetime.fromtimestamp(mtime).strftime("%Y/%m")
                dest_path = Path(dest_path) / date_folder  # type: ignore[assignment]
            except OSError:
                dest_path = Path(dest_path) / "Unknown_Date"  # type: ignore[assignment]

        return str(Path(dest_path) / filename)
