"""Pack/Unpack operation runners for Folder Packer Pro.

Extracted from app.py to decompose the monolithic main window class.
These mixin classes handle the threaded pack and unpack workflows.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING, Any

from .file_ops import (
    collect_folder_stats,
    format_size,
    get_file_type,
    should_exclude,
)
from .pack_engine import (
    collect_files,
    inspect_package,
    pack_files,
    unpack_files,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ScanPreviewMixin:
    """Mixin providing folder scanning and file preview functionality."""

    def _scan_folder(self) -> None:
        """Scan source folder and display statistics."""
        if not self.pack_source_entry.get():
            return

        source_path = Path(self.pack_source_entry.get())
        if not source_path.exists():
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        def scan() -> None:
            """Background task to scan folder statistics."""
            stats = collect_folder_stats(
                source_path,
                self.exclude_patterns,
                self.include_git_var.get(),
            )
            self.root.after(0, lambda: self._display_stats(stats))

        threading.Thread(target=scan, daemon=True).start()

    def _display_stats(self, stats: dict[str, Any]) -> None:
        """Display folder statistics in the stats text widget.

        Args:
            stats: Dictionary with folder statistics.
        """
        if not (stats is not None):
            raise ValueError("stats must be provided")
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")

        output = "Project Statistics\n\n"
        output += f"Total Files: {stats['total_files']:,}\n"
        output += f"Total Size: {format_size(stats['total_size'])}\n"
        output += f"Excluded Files: {stats['excluded_files']:,}\n\n"

        output += "File Types:\n"
        for ext, count in sorted(
            stats["file_types"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]:
            percentage = (
                (count / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
            )
            output += f"  {ext:20s} {count:5,} files ({percentage:5.1f}%)\n"

        self.stats_text.insert("1.0", output)
        self.stats_text.configure(state="disabled")

        self._update_preview_tree()

    def _update_preview_tree(self) -> None:
        """Update preview tree with files to be packed."""
        self.preview_tree.delete(*self.preview_tree.get_children())

        if not self.pack_source_entry.get():
            return

        source_path = Path(self.pack_source_entry.get())
        if not source_path.exists():
            return

        def scan() -> None:
            """Background task to scan files for preview."""
            files = []
            for root_dir, dirs, filenames in os.walk(source_path):
                dirs[:] = [
                    d
                    for d in dirs
                    if not should_exclude(
                        Path(root_dir) / d,
                        self.exclude_patterns,
                        self.include_git_var.get(),
                    )
                ]

                for filename in filenames:
                    file_path = Path(root_dir) / filename
                    if not should_exclude(
                        file_path,
                        self.exclude_patterns,
                        self.include_git_var.get(),
                    ):
                        try:
                            stat = file_path.stat()
                            files.append((file_path, stat))
                            if len(files) >= 500:
                                break
                        except (OSError, PermissionError):
                            logger.exception("Error scanning %s", file_path)
                if len(files) >= 500:
                    break

            self.root.after(0, lambda: self._populate_tree(files, source_path))

        threading.Thread(target=scan, daemon=True).start()

    def _populate_tree(
        self, files: list[tuple[Path, os.stat_result]], base_path: Path
    ) -> None:
        """Populate tree with file list.

        Args:
            files: List of (path, stat_result) tuples.
            base_path: Root path for relative path calculation.
        """
        if not (files is not None):
            raise ValueError("files must be provided")
        from datetime import datetime as dt

        for file_path, stat in files:
            rel_path = file_path.relative_to(base_path)
            size = format_size(stat.st_size)
            file_type = get_file_type(file_path)
            modified = dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            self.preview_tree.insert(
                "",
                "end",
                text=str(rel_path),
                values=(size, file_type, modified),
                tags=(str(file_path),),
            )


class PackOperationMixin:
    """Mixin providing pack operation workflow."""

    def _start_pack(self) -> None:
        """Start packing operation."""
        if not self.pack_source_entry.get():
            messagebox.showwarning("No Source", "Please select a source folder.")
            return

        if not self.pack_output_entry.get():
            messagebox.showwarning("No Output", "Please select an output file.")
            return

        if self.encrypt_var.get():
            password = self.pack_password_entry.get()
            confirm = self.pack_password_confirm.get()

            if not password:
                messagebox.showwarning(
                    "No Password",
                    "Please enter an encryption password.",
                )
                return

            if password != confirm:
                messagebox.showwarning("Password Mismatch", "Passwords do not match.")
                return

        self.cancel_operation = False
        self.pack_btn.configure(state="disabled")
        self.pack_cancel_btn.configure(state="normal")
        self.pack_progress_var.set(0)

        threading.Thread(target=self._run_pack, daemon=True).start()

    def _run_pack(self) -> None:
        """Run pack operation in background."""
        try:
            source_path = Path(self.pack_source_entry.get())
            output_path = Path(self.pack_output_entry.get())

            self._update_pack_status("Collecting files...")

            files_to_pack = collect_files(
                source_path,
                self.exclude_patterns,
                self.include_git_var.get(),
                cancel_check=lambda: self.cancel_operation,
            )

            if self.cancel_operation:
                self._log_message("Pack operation cancelled", "warning")
                return

            total_files = len(files_to_pack)
            self._log_message(f"Packing {total_files} files...", "info")

            def progress_callback(filename: str, current: int, total: int) -> None:
                """Report pack progress to UI."""
                if not (filename is not None):
                    raise ValueError("filename must be provided")
                progress = (current / total) * 100
                self.root.after(
                    0,
                    lambda p=progress: self.pack_progress_var.set(float(p)),  # type: ignore[misc]
                )
                self._update_pack_status(f"Packing {filename} ({current}/{total})")

            result = pack_files(
                source_path=source_path,
                output_path=output_path,
                files_to_pack=files_to_pack,
                compression=self.compression_var.get(),
                encrypt=self.encrypt_var.get(),
                password=(
                    self.pack_password_entry.get() if self.encrypt_var.get() else ""
                ),
                create_manifest=self.create_manifest_var.get(),
                progress_callback=progress_callback,
                cancel_check=lambda: self.cancel_operation,
            )

            for error in result.errors:
                self._log_message(error, "error")

            if result.success:
                self._log_message(
                    f"Package created successfully: {output_path}", "success"
                )
                self._log_message(
                    f"Package size: {format_size(result.package_size)}",
                    "info",
                )
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Success",
                        f"Package created successfully!\n\n"
                        f"Files: {result.total_files}\n"
                        f"Size: {format_size(result.package_size)}",
                    ),
                )
            else:
                self._log_message(f"Pack operation failed: {result.error}", "error")
                error_msg = result.error or "Unknown error"
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Pack failed:\n\n{error_msg}"
                    ),
                )

        except (PermissionError, OSError) as e:
            logger.exception("Pack operation failed")
            self._log_message(f"Pack operation failed: {e}", "error")
            error_msg = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Pack failed:\n\n{error_msg}"),
            )

        finally:
            self.root.after(0, self._pack_finished)


class UnpackOperationMixin:
    """Mixin providing unpack operation workflow."""

    def _start_unpack(self) -> None:
        """Start unpacking operation."""
        if not self.unpack_source_entry.get():
            messagebox.showwarning("No Package", "Please select a package file.")
            return

        if not self.unpack_dest_entry.get():
            messagebox.showwarning(
                "No Destination",
                "Please select a destination folder.",
            )
            return

        if self.encrypted_var.get():
            password = self.unpack_password_entry.get()
            if not password:
                messagebox.showwarning(
                    "No Password",
                    "Please enter the decryption password.",
                )
                return

        self.cancel_operation = False
        self.unpack_btn.configure(state="disabled")
        self.unpack_cancel_btn.configure(state="normal")
        self.unpack_progress_var.set(0)

        threading.Thread(target=self._run_unpack, daemon=True).start()

    def _run_unpack(self) -> None:
        """Run unpack operation in background."""
        try:
            package_path = Path(self.unpack_source_entry.get())
            dest_path = Path(self.unpack_dest_entry.get())

            self._update_unpack_status("Reading package...")

            def progress_callback(filename: str, current: int, total: int) -> None:
                """Report unpack progress to UI."""
                if not (filename is not None):
                    raise ValueError("filename must be provided")
                progress = (current / total) * 100
                self.root.after(
                    0,
                    lambda p=progress: self.unpack_progress_var.set(float(p)),  # type: ignore[misc]
                )
                self._update_unpack_status(f"Extracting {filename} ({current}/{total})")

            result = unpack_files(
                package_path=package_path,
                dest_path=dest_path,
                encrypted=self.encrypted_var.get(),
                password=(
                    self.unpack_password_entry.get() if self.encrypted_var.get() else ""
                ),
                progress_callback=progress_callback,
                cancel_check=lambda: self.cancel_operation,
            )

            for error in result.errors:
                self._log_message(error, "error")

            if result.success:
                self._log_message(
                    f"Package extracted successfully to: {dest_path}",
                    "success",
                )
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Success",
                        f"Package extracted successfully!\n\n"
                        f"Files: {result.total_files}\n"
                        f"Location: {dest_path}",
                    ),
                )
            else:
                self._log_message(f"Unpack operation failed: {result.error}", "error")
                error_msg = result.error or "Unknown error"
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Unpack failed:\n\n{error_msg}"
                    ),
                )

        except (PermissionError, OSError) as e:
            logger.exception("Unpack operation failed")
            self._log_message(f"Unpack operation failed: {e}", "error")
            error_msg = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Unpack failed:\n\n{error_msg}"),
            )

        finally:
            self.root.after(0, self._unpack_finished)

    def _inspect_package(self) -> None:
        """Inspect package file and show information."""
        package_path = self.unpack_source_entry.get()
        if not package_path:
            messagebox.showwarning("No Package", "Please select a package file first.")
            return

        try:
            info = inspect_package(Path(package_path))

            self.package_info_text.configure(state="normal")
            self.package_info_text.delete("1.0", "end")

            output = "Package Information\n\n"
            output += f"File: {info['file']}\n"
            output += f"Size: {info['size_formatted']}\n"
            output += f"Encrypted: {'Yes' if info['encrypted'] else 'No'}\n\n"

            if not info["encrypted"] and info["metadata"]:
                metadata = info["metadata"]
                output += f"Created: {metadata.get('created_at', 'Unknown')}\n"
                output += f"Total Files: {metadata.get('total_files', 0)}\n"
                output += f"Compression: {metadata.get('compression', 'Unknown')}\n"

            self.package_info_text.insert("1.0", output)
            self.package_info_text.configure(state="disabled")

        except (
            OSError,
            ValueError,
        ) as e:
            messagebox.showerror("Error", f"Failed to inspect package:\n\n{e}")
