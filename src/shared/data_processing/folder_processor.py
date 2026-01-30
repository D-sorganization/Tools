from __future__ import annotations

import os
import shutil
import logging
import re
import heapq
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Callable, Any


class FolderProcessor:
    """Handles folder processing logic."""

    def __init__(
        self,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[float], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.cancel_flag = False

    def update_progress(self, value: float) -> None:
        if self.progress_callback:
            self.progress_callback(value)

    def update_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)

    def log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def process_folders(
        self,
        mode: str,
        source_folders: list[str | Path],
        destination_folder: str | Path | None,
        options: dict[str, Any],
    ) -> None:
        """
        Perform the folder processing operation.

        Args:
            mode: Operation mode ('combine', 'flatten', 'prune', 'deduplicate', 'analyze')
            source_folders: List of source folder paths
            destination_folder: Destination folder path (can be None for some modes)
            options: Dictionary of options
        """
        try:
            self.cancel_flag = False

            if mode == "combine":
                self._combine_operation(source_folders, destination_folder, options)
            elif mode == "flatten":
                self._flatten_operation(source_folders, destination_folder, options)
            elif mode == "prune":
                self._prune_operation(source_folders, destination_folder, options)
            elif mode == "deduplicate":
                self._deduplicate_operation(source_folders, options)
            elif mode == "analyze":
                report = self._analyze_operation(source_folders)
                # For analyze, we might want to return the report or handle it via callback
                if "report_callback" in options and options["report_callback"]:
                    options["report_callback"](report)

            if not self.cancel_flag:
                self.update_status("Processing complete")
                self.update_progress(1.0)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.update_status(error_msg)
            raise

    def _combine_operation(
        self,
        source_folders: list[str | Path],
        destination_folder: str | Path,
        options: dict[str, Any],
    ) -> None:
        """Combine operation - copy all files from source folders to destination."""
        # Create destination directory
        if not options.get("preview_mode", False):
            os.makedirs(destination_folder, exist_ok=True)

        # Collect all file paths
        all_file_paths = []
        for src in source_folders:
            for root, _dirs, files in os.walk(src):
                for file in files:
                    all_file_paths.append(Path(root) / file)

        total_files = len(all_file_paths)
        if total_files == 0:
            self.update_status("No files found in source folders")
            return

        processed_files = 0
        copied_count = 0
        renamed_count = 0
        skipped_count = 0

        for source_path in all_file_paths:
            if self.cancel_flag:
                break

            # Apply file filters
            if not self._validate_file_filters(source_path, options):
                skipped_count += 1
                processed_files += 1
                continue

            # Get organized destination path
            dest_path = self._get_organized_path(
                source_path, destination_folder, options
            )
            dest_dir = Path(dest_path).parent

            # Create destination directory if needed
            if not options.get("preview_mode", False):
                os.makedirs(dest_dir, exist_ok=True)

            # Handle naming conflicts
            final_dest_path = self._get_unique_path(dest_path)
            if final_dest_path != dest_path:
                renamed_count += 1

            try:
                if not options.get("preview_mode", False):
                    shutil.copy2(source_path, final_dest_path)
                copied_count += 1
            except Exception as e:
                self.log(f"Error copying '{Path(source_path).name}': {e}")

            processed_files += 1
            if processed_files % 10 == 0:
                self.update_progress(processed_files / total_files)
                self.update_status(f"Processed {processed_files}/{total_files} files")

        if options.get("preview_mode", False):
            status = (
                f"PREVIEW: Would copy {copied_count} files, "
                f"rename {renamed_count}, skip {skipped_count}"
            )
        else:
            status = (
                f"Copied {copied_count} files,"
                f"renamed {renamed_count}, skipped {skipped_count}"
            )
        self.update_status(status)

    def _flatten_operation(
        self,
        source_folders: list[str | Path],
        destination_folder: str | Path,
        options: dict[str, Any],
    ) -> None:
        """Flatten operation - copy files from nested folders to top level."""
        # Create destination directory
        if not options.get("preview_mode", False):
            os.makedirs(destination_folder, exist_ok=True)

        all_file_paths = []
        for src in source_folders:
            for root, _dirs, files in os.walk(src):
                for file in files:
                    all_file_paths.append((Path(root) / file, file))

        total_files = len(all_file_paths)
        if total_files == 0:
            self.update_status("No files found in source folders")
            return

        processed_files = 0
        copied_count = 0
        renamed_count = 0
        skipped_count = 0

        for source_path, file in all_file_paths:
            if self.cancel_flag:
                break

            # Apply file filters
            if not self._validate_file_filters(source_path, options):
                skipped_count += 1
                processed_files += 1
                continue

            # For flatten operation, files go directly to destination root
            dest_path = Path(destination_folder) / file

            # Handle naming conflicts
            final_dest_path = self._get_unique_path(dest_path)
            if final_dest_path != dest_path:
                renamed_count += 1

            try:
                if not options.get("preview_mode", False):
                    shutil.copy2(source_path, final_dest_path)
                copied_count += 1
            except Exception as e:
                self.log(f"Error copying '{file}': {e}")

            processed_files += 1
            if processed_files % 10 == 0:
                self.update_progress(processed_files / total_files)
                self.update_status(f"Processed {processed_files}/{total_files} files")

        if options.get("preview_mode", False):
            status = (
                f"PREVIEW: Would flatten {copied_count} files, "
                f"rename {renamed_count}, skip {skipped_count}"
            )
        else:
            status = (
                f"Flattened {copied_count} files, "
                f"renamed {renamed_count}, skipped {skipped_count}"
            )
        self.update_status(status)

    def _prune_operation(
        self,
        source_folders: list[str | Path],
        destination_folder: str | Path,
        options: dict[str, Any],
    ) -> None:
        """Prune operation - copy folders but skip empty subfolders."""
        if not options.get("preview_mode", False):
            os.makedirs(destination_folder, exist_ok=True)

        all_file_data = []
        for src in source_folders:
            for root, _dirs, files in os.walk(src):
                if files:  # Skip empty directories
                    for file in files:
                        source_path = Path(root) / file
                        all_file_data.append((source_path, file, src, root))

        total_files = len(all_file_data)
        if total_files == 0:
            self.update_status("No files found in source folders")
            return

        processed_files = 0
        copied_count = 0
        skipped_count = 0
        created_dirs = set()

        for source_path, file, src, root in all_file_data:
            if self.cancel_flag:
                break

            src_name = Path(src).name
            dest_src_path = Path(destination_folder) / src_name
            rel_path = os.path.relpath(root, src)
            dest_dir = Path(dest_src_path) / rel_path

            if dest_dir not in created_dirs:
                if not options.get("preview_mode", False):
                    os.makedirs(dest_dir, exist_ok=True)
                created_dirs.add(dest_dir)

            if not self._validate_file_filters(source_path, options):
                skipped_count += 1
                processed_files += 1
                continue

            dest_path = Path(dest_dir) / file

            try:
                if not options.get("preview_mode", False):
                    shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                self.log(f"Error copying '{file}': {e}")

            processed_files += 1
            if processed_files % 10 == 0:
                self.update_progress(processed_files / total_files)
                self.update_status(f"Processed {processed_files}/{total_files} files")

        if options.get("preview_mode", False):
            status = (
                f"PREVIEW: Would copy {copied_count} files, "
                f"skip {skipped_count} (pruned empty folders)"
            )
        else:
            status = (
                f"Copied {copied_count} files, "
                f"skipped {skipped_count} (pruned empty folders)"
            )
        self.update_status(status)

    def _deduplicate_operation(
        self, source_folders: list[str | Path], options: dict[str, Any]
    ) -> None:
        """Deduplicate operation - Remove renamed duplicates in source folders."""
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")
        all_dir_files = []
        total_files = 0

        for src in source_folders:
            for root, _dirs, files in os.walk(src):
                if files:
                    all_dir_files.append((root, files))
                    total_files += len(files)

        if total_files == 0:
            self.update_status("No files found in source folders")
            return

        processed_files = 0
        deleted_count = 0

        for root, files in all_dir_files:
            if self.cancel_flag:
                break

            files_by_base_name = {}
            for filename in files:
                match = pattern.match(filename)
                if match:
                    base, _, ext = match.groups()
                    base_name = f"{base}{ext}"
                    files_by_base_name.setdefault(base_name, []).append(
                        Path(root) / filename
                    )

            for _base_name, file_list in files_by_base_name.items():
                if len(file_list) > 1:
                    try:
                        file_to_keep = max(file_list, key=lambda f: os.path.getmtime(f))
                    except (OSError, FileNotFoundError):
                        continue

                    for file_path in file_list:
                        if file_path != file_to_keep:
                            try:
                                if not options.get("preview_mode", False):
                                    os.remove(file_path)
                                deleted_count += 1
                            except OSError as e:
                                logging.warning("Failed to delete file: %s", str(e))

                        processed_files += 1
                        if processed_files % 50 == 0:
                            self.update_status(
                                f"Processed {processed_files}/{total_files} files"
                            )

        if options.get("preview_mode", False):
            status = f"PREVIEW: Would delete {deleted_count} duplicate files"
        else:
            status = f"Deleted {deleted_count} duplicate files"
        self.update_status(status)

    def _analyze_operation(self, source_folders: list[str | Path]) -> str:
        """Analyze operation - generate detailed report."""
        total_files = 0
        for src in source_folders:
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        if total_files == 0:
            self.update_status("No files found in source folders")
            return "No files found"

        processed_files = 0
        total_size = 0
        file_types = defaultdict(int)
        size_by_type = defaultdict(int)
        largest_files = []

        report_lines = [
            "=== FOLDER ANALYSIS REPORT ===",
            f"Generated: {datetime.now()}",
            "",
        ]

        for src in source_folders:
            if self.cancel_flag:
                break

            report_lines.append(f"Analyzing: {src}")
            folder_files = 0
            folder_size = 0

            for root, _dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_flag:
                        break

                    file_path = Path(root) / file
                    try:
                        file_size = os.path.getsize(file_path)
                        file_ext = Path(file).suffix.lower() or "no_extension"

                        total_size += file_size
                        folder_files += 1
                        folder_size += file_size
                        file_types[file_ext] += 1
                        size_by_type[file_ext] += file_size

                        largest_files.append((file_path, file_size))

                    except OSError:
                        continue

                    processed_files += 1
                    if processed_files % 10 == 0:
                        self.update_progress(processed_files / total_files)
                        self.update_status(
                            f"Processed {processed_files}/{total_files} files"
                        )

            report_lines.append(
                f"  Files: {folder_files}, "
                f"Size: {folder_size / (1024 * 1024):.1f} MB"
            )

        report_lines.extend(
            [
                "",
                f"TOTAL FILES: {processed_files}",
                f"TOTAL SIZE: {total_size / (1024 * 1024):.1f} MB",
                "",
                "FILE TYPES:",
            ]
        )

        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            size_mb = size_by_type[ext] / (1024 * 1024)
            report_lines.append(f"  {ext}: {count} files, {size_mb:.1f} MB")

        report_lines.extend(["", "LARGEST FILES:"])
        top_10_files = heapq.nlargest(10, largest_files, key=lambda x: x[1])
        for file_path, size in top_10_files:
            size_mb = size / (1024 * 1024)
            report_lines.append(f"  {Path(file_path).name}: {size_mb:.1f} MB")

        return "\n".join(report_lines)

    def _validate_file_filters(
        self, file_path: str | Path, options: dict[str, Any]
    ) -> bool:
        """Validate if a file meets the filtering criteria."""
        if self.cancel_flag:
            return False

        # Extension filter
        extensions = options.get("filter_extensions", "")
        if extensions:
            if isinstance(extensions, str):
                ext_list = [ext.strip().lower() for ext in extensions.split(",")]
            else:
                ext_list = [ext.strip().lower() for ext in extensions]

            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ext_list:
                return False

        # Size filter
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            min_size = float(options.get("min_file_size", 0) or 0)
            if file_size_mb < min_size:
                return False

            max_size_val = options.get("max_file_size", "")
            if max_size_val:
                max_size = float(max_size_val)
                if file_size_mb > max_size:
                    return False
        except (ValueError, OSError):
            return False

        return True

    def _get_organized_path(
        self, file_path: str | Path, dest_base: str | Path, options: dict[str, Any]
    ) -> Path:
        """Returns the organized destination path."""
        filename = Path(file_path).name
        dest_path = dest_base

        if options.get("organize_by_type", False):
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

        if options.get("organize_by_date", False):
            try:
                mtime = os.path.getmtime(file_path)
                date_folder = datetime.fromtimestamp(mtime).strftime("%Y/%m")
                dest_path = Path(dest_path) / date_folder
            except OSError:
                dest_path = Path(dest_path) / "Unknown_Date"

        return Path(dest_path) / filename

    def _get_unique_path(self, path: str | Path) -> Path:
        """Get a unique path by adding a number if the file already exists."""
        if not Path(path).exists():
            return Path(path)
        parent, name = os.path.split(path)
        is_file = "." in name and not os.path.isdir(path)
        filename = Path(name).stem if is_file else name
        ext = Path(name).suffix if is_file else ""
        counter = 1
        new_path = Path(parent) / f"{filename}_{counter}{ext}"
        while Path(new_path).exists():
            counter += 1
            new_path = Path(parent) / f"{filename}_{counter}{ext}"
        return new_path
