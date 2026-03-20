"""FolderOperationsMixin -- Folder-level operations for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from tkinter import messagebox

from Folders_Tool_r0 import (
    MAX_LOG_ENTRIES,
    MAX_UI_UPDATE_FREQUENCY,
    PROGRESS_MAIN_OP_PERCENT,
    PROGRESS_START_MAIN,
)

logger = logging.getLogger(__name__)


class FolderOperationsMixin:
    """Folder-level operations: combine, deduplicate, flatten, prune."""

    def _combine_folders_enhanced(self) -> list[str]:
        """Enhanced combine operation with filtering and organization."""
        log = []
        file_count = 0
        renamed_count = 0
        skipped_count = 0
        failed_count = 0

        Path(self.dest_folder).mkdir(parents=True, exist_ok=True)  # type: ignore

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:  # type: ignore
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:  # type: ignore
            if self.cancel_operation:  # type: ignore
                break

            for root, _dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:  # type: ignore
                        break    # type: ignore

                    source_path = Path(root) / file

                    # Apply filters
                    if not self.validate_file_filters(source_path):  # type: ignore
                        skipped_count += 1
                        processed_files += 1
                        continue

                    # Get organized destination path
                    dest_path = self.get_organized_path(source_path, self.dest_folder)  # type: ignore
                    dest_dir = Path(dest_path).parent

                    # Create destination directory if needed
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)  # type: ignore
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{Path(final_dest_path).name}'",
                        )
                        renamed_count += 1

                    try:
                        if not self.preview_mode_var.get():  # type: ignore
                            if self._safe_copy_file(source_path, final_dest_path):  # type: ignore
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
                        self.update_progress(  # type: ignore
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

        if self.preview_mode_var.get():  # type: ignore
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]

    # --- Keep existing methods for compatibility ---

    def _perform_deduplication(self, target_folder: str) -> list[str]:
        """Core logic to find and delete renamed duplicates in a single
        target folder.
        """
        assert target_folder is not None, "target_folder must be provided"
        log = []
        deleted_count = 0
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")

        if not self.preview_mode_var.get():  # type: ignore
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
            if self.cancel_operation:  # type: ignore
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
                                if not self.preview_mode_var.get():  # type: ignore
                                    Path(file_path).unlink()
                                mode_str = (
                                    "WOULD DELETE"
                                    if self.preview_mode_var.get()  # type: ignore
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
            f"{'preview' if self.preview_mode_var.get() else 'complete'}.",  # type: ignore
            f"{'Would delete' if self.preview_mode_var.get() else 'Deleted'} a total "  # type: ignore
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
        for folder in self.source_folders:  # type: ignore
            if self.cancel_operation:  # type: ignore
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

        os.makedirs(self.dest_folder, exist_ok=True)  # type: ignore

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:  # type: ignore
            for _root, _dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:  # type: ignore
            if self.cancel_operation:  # type: ignore
                break

            for root, _dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:  # type: ignore
                        break    # type: ignore

                    source_path = Path(root) / file

                    # Apply filters
                    if not self.validate_file_filters(source_path):  # type: ignore
                        skipped_count += 1
                        processed_files += 1
                        continue

                    # Get organized destination path (flattened to root)
                    dest_path = self.get_organized_path(source_path, self.dest_folder)  # type: ignore
                    dest_dir = Path(dest_path).parent

                    # Create destination directory if needed
                    os.makedirs(dest_dir, exist_ok=True)

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)  # type: ignore
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{Path(final_dest_path).name}'",
                        )

                    try:
                        if not self.preview_mode_var.get():  # type: ignore
                            if self._safe_copy_file(source_path, final_dest_path):  # type: ignore
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
                        self.update_progress(  # type: ignore
                            progress,
                            f"Processed {processed_files}/{total_files} files",
                        )

        summary = [
            f"Flattened {moved_count} files to destination root level.",
            f"Skipped {skipped_count} files due to filters.",
        ]

        if failed_count > 0:
            summary.append(f"Failed to copy {failed_count} files.")

        if self.preview_mode_var.get():  # type: ignore
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]

    def _count_total_files(self) -> int:
        """Count total files across all source folders for progress tracking."""
        total = 0
        for src in self.source_folders:  # type: ignore
            for _root, _dirs, files in os.walk(src):
                total += len(files)
        return total

    def _copy_single_file_in_prune(
        self,
        source_file_path: Path,
        dest_path: Path,
        file: str,
        log: list[str],
    ) -> tuple[int, int]:
        """Copy a single file during prune operation, handling conflicts.

        Args:
            source_file_path: Source file path
            dest_path: Destination directory
            file: Filename
            log: Log list to append messages to

        Returns:
            Tuple of (files_copied, files_failed)
        """
        assert source_file_path is not None, "source_file_path must be provided"
        if not self.validate_file_filters(source_file_path):  # type: ignore
            return 0, 0

        dest_file_path = Path(dest_path) / file
        final_dest_path = self._get_unique_path(dest_file_path)  # type: ignore
        if final_dest_path != dest_file_path:
            log.append(f"Renamed: '{file}' to '{Path(final_dest_path).name}'")

        try:
            if not self.preview_mode_var.get():  # type: ignore
                if self._safe_copy_file(source_file_path, final_dest_path):  # type: ignore
                    return 1, 0
                else:
                    log.append(f"FAILED to copy '{file}' after retries")
                    return 0, 1
            else:
                return 1, 0  # Count in preview mode
        except (KeyError, ValueError, TypeError) as e:
            log.append(f"ERROR copying '{file}': {e}")
            return 0, 1

    def _prune_empty_folders(self) -> list[str]:
        """Copy source folders to destination while preserving structure but
        skipping empty sub-folders.

        Returns:
            List of log messages describing the operation results
        """
        log: list[str] = []
        file_count = 0
        processed_folders = 0
        empty_folders_skipped = 0
        failed_count = 0

        os.makedirs(self.dest_folder, exist_ok=True)  # type: ignore
        total_files = self._count_total_files()
        processed_files = 0

        for src in self.source_folders:  # type: ignore
            if self.cancel_operation:  # type: ignore
                break

            src_name = Path(src).name
            dest_src_path = Path(self.dest_folder) / src_name  # type: ignore

            for root, dirs, files in os.walk(src):
                if self.cancel_operation:  # type: ignore
                    break    # type: ignore

                if not files and not any(
                    any(Path(root, d).iterdir())
                    for d in dirs
                    if (Path(root) / d).exists()
                ):
                    empty_folders_skipped += 1
                    continue

                rel_path = os.path.relpath(root, src)
                dest_path = Path(dest_src_path) / rel_path
                os.makedirs(dest_path, exist_ok=True)

                for file in files:
                    if self.cancel_operation:  # type: ignore
                        break    # type: ignore

                    copied, failed = self._copy_single_file_in_prune(
                        Path(root) / file,
                        dest_path,
                        file,
                        log,
                    )
                    file_count += copied
                    failed_count += failed

                    processed_files += 1
                    if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:
                        progress = (
                            PROGRESS_START_MAIN
                            + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
                        )
                        self.update_progress(  # type: ignore
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

        if self.preview_mode_var.get():  # type: ignore
            summary.insert(0, "PREVIEW MODE - No files were actually copied.")

        return summary + log[:MAX_LOG_ENTRIES]
