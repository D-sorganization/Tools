"""Folder Tool Tab and Logic for Data Processor."""

from __future__ import annotations

import heapq
import logging
import os
import shutil
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

logger = logging.getLogger(__name__)


class FolderToolMixin:
    """Mixin containing UI and logic for the Folder Tool tab."""

    # Initialized in main app
    folder_source_folders: list[str]
    folder_destination: str
    folder_cancel_flag: bool

    # UI elements
    folder_source_listbox: ctk.CTkTextbox
    folder_source_info_label: ctk.CTkLabel
    folder_dest_label: ctk.CTkLabel
    folder_status_var: ctk.StringVar
    folder_progress_bar: ctk.CTkProgressBar
    folder_run_button: ctk.CTkButton
    folder_cancel_button: ctk.CTkButton
    folder_operation_mode: ctk.StringVar
    folder_mode_description: ctk.CTkLabel
    folder_filter_extensions: ctk.StringVar
    folder_min_file_size: ctk.StringVar
    folder_max_file_size: ctk.StringVar
    folder_organize_by_type_var: ctk.BooleanVar
    folder_organize_by_date_var: ctk.BooleanVar
    folder_deduplicate_var: ctk.BooleanVar
    folder_zip_output_var: ctk.BooleanVar
    folder_preview_mode_var: ctk.BooleanVar
    folder_backup_before_var: ctk.BooleanVar

    def create_folder_tool_tab(self, parent_tab: ctk.CTkFrame) -> None:
        """Create the folder tool tab with integrated folder processor functionality."""
        if not (parent_tab is not None):
            raise ValueError("parent_tab must be provided")
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        # Create scrollable frame for the folder tool
        folder_scrollable_frame = ctk.CTkScrollableFrame(parent_tab)
        folder_scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        folder_scrollable_frame.grid_columnconfigure(0, weight=1)

        # Create UI sections
        self._create_folder_source_section(folder_scrollable_frame)
        self._create_folder_destination_section(folder_scrollable_frame)
        self._create_folder_filtering_section(folder_scrollable_frame)
        self._create_folder_operation_section(folder_scrollable_frame)
        self._create_folder_organization_section(folder_scrollable_frame)
        self._create_folder_output_section(folder_scrollable_frame)
        self._create_folder_progress_section(folder_scrollable_frame)
        self._create_folder_run_section(folder_scrollable_frame)

        # Initialize mode description
        self._update_folder_mode_description()

    def _create_folder_source_section(self, parent) -> None:
        """Create the source folders section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        source_frame = ctk.CTkFrame(parent)
        source_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        source_frame.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            source_frame,
            text="1. Select Folder(s) to Process",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Source folder listbox
        self.folder_source_listbox = ctk.CTkTextbox(source_frame, height=120)
        self.folder_source_listbox.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Buttons
        button_frame = ctk.CTkFrame(source_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkButton(
            button_frame,
            text="Add Folder(s)",
            command=self._folder_select_source_folders,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame,
            text="Remove Selected",
            command=self._folder_remove_selected_source,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame, text="Clear All", command=self._folder_clear_source_folders
        ).pack(side="left", padx=5)

        # Info label
        self.folder_source_info_label = ctk.CTkLabel(
            source_frame, text="No folders selected", text_color="gray"
        )
        self.folder_source_info_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

    def _create_folder_destination_section(self, parent):
        """Create the destination folder section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        dest_frame = ctk.CTkFrame(parent)
        dest_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        dest_frame.grid_columnconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            dest_frame,
            text="2. Select Final Destination Folder",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Destination label
        self.folder_dest_label = ctk.CTkLabel(
            dest_frame, text="No destination selected", text_color="gray"
        )
        self.folder_dest_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # Set destination button
        ctk.CTkButton(
            dest_frame, text="Set Destination", command=self._folder_select_dest_folder
        ).grid(row=1, column=1, padx=10, pady=5)

    def _create_folder_filtering_section(self, parent):
        """Create the file filtering section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        filter_frame = ctk.CTkFrame(parent)
        filter_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        filter_frame.grid_columnconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            filter_frame,
            text="3. File Filtering Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Extensions filter
        ctk.CTkLabel(
            filter_frame, text="Include only extensions (comma-separated):"
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkEntry(
            filter_frame,
            textvariable=self.folder_filter_extensions,
            placeholder_text=".jpg,.png,.pdf",
        ).grid(row=1, column=1, sticky="ew", padx=10, pady=2)

        # File size filters
        ctk.CTkLabel(filter_frame, text="Min size (MB):").grid(
            row=2, column=0, sticky="w", padx=10, pady=2
        )
        ctk.CTkEntry(
            filter_frame, textvariable=self.folder_min_file_size, width=100
        ).grid(row=2, column=1, sticky="w", padx=10, pady=2)

        ctk.CTkLabel(filter_frame, text="Max size (MB):").grid(
            row=3, column=0, sticky="w", padx=10, pady=2
        )
        ctk.CTkEntry(
            filter_frame, textvariable=self.folder_max_file_size, width=100
        ).grid(row=3, column=1, sticky="w", padx=10, pady=2)

        # Help text
        ctk.CTkLabel(
            filter_frame,
            text="Example: .jpg,.png,.pdf (leave empty for all files)",
            text_color="gray",
            font=ctk.CTkFont(size=12),
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def _create_folder_operation_section(self, parent):
        """Create the main operation section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        operation_frame = ctk.CTkFrame(parent)
        operation_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            operation_frame,
            text="4. Choose Main Operation",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Radio buttons
        operations = [
            ("Combine & Copy", "combine"),
            ("Flatten & Tidy", "flatten"),
            ("Copy & Prune Empty Folders", "prune"),
            ("Deduplicate Files (In-Place)", "deduplicate"),
            ("Analyze & Report Only", "analyze"),
        ]

        for i, (text, value) in enumerate(operations):
            ctk.CTkRadioButton(
                operation_frame,
                text=text,
                variable=self.folder_operation_mode,
                value=value,
                command=self._update_folder_mode_description,
            ).grid(row=i + 1, column=0, sticky="w", padx=10, pady=2)

        # Mode description
        self.folder_mode_description = ctk.CTkLabel(
            operation_frame, text="", wraplength=600, text_color="blue"
        )
        self.folder_mode_description.grid(
            row=len(operations) + 1, column=0, sticky="w", padx=10, pady=10
        )

    def _create_folder_organization_section(self, parent):
        """Create the organization options section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        org_frame = ctk.CTkFrame(parent)
        org_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            org_frame,
            text="5. File Organization Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Checkboxes
        ctk.CTkCheckBox(
            org_frame,
            text="Organize files by type (create subfolders)",
            variable=self.folder_organize_by_type_var,
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            org_frame,
            text="Organize files by date (YYYY/MM folders)",
            variable=self.folder_organize_by_date_var,
        ).grid(row=2, column=0, sticky="w", padx=10, pady=2)

    def _create_folder_output_section(self, parent):
        """Create the output options section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        output_frame = ctk.CTkFrame(parent)
        output_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            output_frame,
            text="6. Output Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Checkboxes
        ctk.CTkCheckBox(
            output_frame,
            text="Deduplicate renamed files in destination folder after copy",
            variable=self.folder_deduplicate_var,
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Create ZIP archive of final result",
            variable=self.folder_zip_output_var,
        ).grid(row=2, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Preview mode (show what would be done without executing)",
            variable=self.folder_preview_mode_var,
        ).grid(row=3, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Create backup before processing",
            variable=self.folder_backup_before_var,
        ).grid(row=4, column=0, sticky="w", padx=10, pady=2)

    def _create_folder_progress_section(self, parent):
        """Create the progress section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            progress_frame, text="Progress", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Progress bar
        self.folder_progress_bar = ctk.CTkProgressBar(progress_frame)
        self.folder_progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.folder_progress_bar.set(0)

        # Status label
        self.folder_status_label = ctk.CTkLabel(
            progress_frame, textvariable=self.folder_status_var
        )
        self.folder_status_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

    def _create_folder_run_section(self, parent):
        """Create the run button section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        run_frame = ctk.CTkFrame(parent)
        run_frame.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        # Buttons
        self.folder_run_button = ctk.CTkButton(
            run_frame,
            text="Run Folder Process",
            command=self._folder_run_processing,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.folder_run_button.grid(row=0, column=0, padx=10, pady=10)

        self.folder_cancel_button = ctk.CTkButton(
            run_frame,
            text="Cancel",
            command=self._folder_cancel_processing,
            state="disabled",
        )
        self.folder_cancel_button.grid(row=0, column=1, padx=10, pady=10)

    def _update_folder_mode_description(self):
        """Update the mode description based on selected operation."""
        mode = self.folder_operation_mode.get()
        descriptions = {
            "combine": (
                "Copies all files from source folders"
                " into the single destination folder."
            ),
            "flatten": (
                "Finds deeply nested folders and copies"
                " them to the top level of the destination."
            ),
            "prune": (
                "Copies source folders to the destination,"
                " preserving structure but skipping empty"
                " sub-folders."
            ),
            "deduplicate": (
                "Deletes renamed duplicates within the"
                " source folder(s), keeping the newest"
                " version."
            ),
            "analyze": (
                "Analyzes folder contents and generates a"
                " detailed report without making changes."
            ),
        }
        self.folder_mode_description.configure(text=descriptions.get(mode, ""))

    def _folder_select_source_folders(self) -> None:
        """Select source folders for processing."""
        try:
            folder = filedialog.askdirectory(title="Select Source Folders")
            if folder:
                self.folder_source_folders.append(folder)
                self._folder_update_source_display()
        except (OSError, RuntimeError) as e:
            messagebox.showerror("Error", f"Failed to select source folders: {str(e)}")

    def _folder_remove_selected_source(self):
        """Remove selected source folder from the list."""
        if self.folder_source_folders:
            self.folder_source_folders.pop()
            self._folder_update_source_display()

    def _folder_clear_source_folders(self):
        """Clear all source folders."""
        self.folder_source_folders = []
        self._folder_update_source_display()

    def _folder_update_source_display(self):
        """Update the source folders display."""
        self.folder_source_listbox.delete("1.0", "end")
        if self.folder_source_folders:
            for folder in self.folder_source_folders:
                self.folder_source_listbox.insert("end", f"{folder}\n")
            self.folder_source_info_label.configure(
                text=f"{len(self.folder_source_folders)} folder(s) selected"
            )
        else:
            self.folder_source_info_label.configure(text="No folders selected")

    def _folder_select_dest_folder(self):
        """Select destination folder."""
        try:
            folder = filedialog.askdirectory(title="Select Destination Folder")
            if folder:
                self.folder_destination = folder
                self.folder_dest_label.configure(text=folder)
        except (OSError, RuntimeError) as e:
            messagebox.showerror(
                "Error", f"Failed to select destination folder: {str(e)}"
            )

    def _folder_run_processing(self) -> None:
        """Start the folder processing operation."""
        if not self.folder_source_folders:
            messagebox.showwarning(
                "No Source Folders", "Please select at least one source folder."
            )
            return

        mode = self.folder_operation_mode.get()
        if mode not in ["deduplicate", "analyze"] and not self.folder_destination:
            messagebox.showwarning(
                "No Destination", "Please select a destination folder."
            )
            return

        self.folder_cancel_flag = False
        threading.Thread(target=self._folder_perform_processing, daemon=True).start()

        self.folder_run_button.configure(state="disabled")
        self.folder_cancel_button.configure(state="normal")
        self.folder_status_var.set("Processing...")

    def _folder_cancel_processing(self) -> None:
        """Cancel the folder processing operation."""
        self.folder_cancel_flag = True
        self.folder_status_var.set("Cancelled")
        self.folder_progress_bar.set(0)
        self.folder_run_button.configure(state="normal")
        self.folder_cancel_button.configure(state="disabled")

    def _folder_perform_processing(self) -> None:
        """Perform the actual folder processing operation.

        Contracts (DbC):
        - Precondition: folder_source_folders must not be empty.
        - Precondition: mode must be valid.
        """
        if not (self.folder_source_folders):
            raise ValueError("No source folders selected")

        try:
            mode = self.folder_operation_mode.get()
            if mode == "combine":
                self._folder_combine_operation()
            elif mode == "flatten":
                self._folder_flatten_operation()
            elif mode == "prune":
                self._folder_prune_operation()
            elif mode == "deduplicate":
                self._folder_deduplicate_operation()
            elif mode == "analyze":
                self._folder_analyze_operation()

            self.after(0, lambda: self.folder_status_var.set("Processing complete"))  # type: ignore
            self.after(0, lambda: self.folder_progress_bar.set(1.0))  # type: ignore
            self.after(0, lambda: self.folder_run_button.configure(state="normal"))  # type: ignore
            self.after(0, lambda: self.folder_cancel_button.configure(state="disabled"))  # type: ignore
        except (OSError, PermissionError, ValueError) as exc:
            msg = f"Error: {exc}"
            self.after(0, lambda m=msg: self.folder_status_var.set(m))  # type: ignore
            self.after(0, lambda: self.folder_run_button.configure(state="normal"))  # type: ignore
            self.after(0, lambda: self.folder_cancel_button.configure(state="disabled"))  # type: ignore

    def _folder_combine_operation(self) -> None:
        """Combine operation - copy all files from source folders to destination."""
        try:
            os.makedirs(self.folder_destination, exist_ok=True)
            all_file_paths = []
            for src in self.folder_source_folders:
                for root, _, files in os.walk(src):
                    for file in files:
                        all_file_paths.append(Path(root) / file)

            total_files = len(all_file_paths)
            if total_files == 0:
                self.after(0, lambda: self.folder_status_var.set("No files found"))  # type: ignore
                return

            processed_files = 0
            for source_path in all_file_paths:
                if self.folder_cancel_flag:
                    break
                if self._folder_validate_file_filters(source_path):
                    dest_path = self._folder_get_organized_path(
                        source_path, self.folder_destination
                    )
                    os.makedirs(dest_path.parent, exist_ok=True)
                    final_dest = self._folder_get_unique_path(dest_path)
                    if not self.folder_preview_mode_var.get():
                        shutil.copy2(source_path, final_dest)

                processed_files += 1
                if processed_files % 10 == 0:
                    pct = processed_files / total_files
                    n, tot = processed_files, total_files
                    self.after(  # type: ignore
                        0,
                        lambda p=pct: self.folder_progress_bar.set(p),
                    )
                    self.after(  # type: ignore
                        0,
                        lambda p=n, t=tot: self.folder_status_var.set(
                            f"Processed {p}/{t}"
                        ),
                    )
        except (OSError, PermissionError) as e:
            logger.error(f"Combine failed: {e}")

    def _folder_flatten_operation(self) -> None:
        """Flatten operation - copy files from nested folders to top level."""
        try:
            os.makedirs(self.folder_destination, exist_ok=True)
            all_files = []
            for src in self.folder_source_folders:
                for root, _, files in os.walk(src):
                    for f in files:
                        all_files.append((Path(root) / f, f))

            total = len(all_files)
            if total == 0:
                return

            for i, (src_path, filename) in enumerate(all_files):
                if self.folder_cancel_flag:
                    break
                if self._folder_validate_file_filters(src_path):
                    dest_path = Path(self.folder_destination) / filename
                    final_dest = self._folder_get_unique_path(dest_path)
                    if not self.folder_preview_mode_var.get():
                        shutil.copy2(src_path, final_dest)

                if (i + 1) % 10 == 0:
                    self.after(
                        0, lambda p=(i + 1) / total: self.folder_progress_bar.set(p)
                    )  # type: ignore
        except (OSError, PermissionError) as e:
            logger.error(f"Flatten failed: {e}")

    def _folder_prune_operation(self) -> None:
        """Prune operation - copy folders but skip empty subfolders."""
        try:
            os.makedirs(self.folder_destination, exist_ok=True)
            for src in self.folder_source_folders:
                for root, _, files in os.walk(src):
                    if not files:
                        continue
                    rel = os.path.relpath(root, src)
                    dest_dir = Path(self.folder_destination) / Path(src).name / rel
                    if not self.folder_preview_mode_var.get():
                        os.makedirs(dest_dir, exist_ok=True)
                    for f in files:
                        src_path = Path(root) / f
                        if self._folder_validate_file_filters(src_path):
                            if not self.folder_preview_mode_var.get():
                                shutil.copy2(src_path, dest_dir / f)
        except (OSError, PermissionError) as e:
            logger.error(f"Prune failed: {e}")

    def _folder_deduplicate_operation(self) -> None:
        """Deduplicate operation."""
        import re

        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")
        for src in self.folder_source_folders:
            for root, _, files in os.walk(src):
                files_by_base = {}
                for f in files:
                    m = pattern.match(f)
                    if m:
                        base, _, ext = m.groups()
                        files_by_base.setdefault(f"{base}{ext}", []).append(
                            Path(root) / f
                        )

                for file_list in files_by_base.values():
                    if len(file_list) > 1:
                        keep = max(file_list, key=lambda f: os.path.getmtime(f))
                        for f in file_list:
                            if f != keep and not self.folder_preview_mode_var.get():
                                os.remove(f)

    def _folder_analyze_operation(self) -> None:
        """Analyze operation."""

        info = []
        total_size = 0
        for src in self.folder_source_folders:
            for root, _, files in os.walk(src):
                for f in files:
                    p = Path(root) / f
                    sz = os.path.getsize(p)
                    total_size += sz
                    info.append((p, sz))

        size_mb = total_size / 1e6
        report = (
            f"Analysis Report\nTotal Files: {len(info)}\nTotal Size: {size_mb:.2f} MB\n"
        )
        largest = heapq.nlargest(10, info, key=lambda x: x[1])
        report += "\nLargest Files:\n"
        for p, sz in largest:
            report += f"{p.name}: {sz / 1e6:.2f} MB\n"

        self.after(0, lambda: self._show_folder_analysis_report(report))  # type: ignore

    def _show_folder_analysis_report(self, text: str) -> None:
        """Show report."""
        if not (text is not None):
            raise ValueError("text must be provided")
        dialog = ctk.CTkToplevel(self)  # type: ignore
        dialog.title("Analysis Report")
        t = ctk.CTkTextbox(dialog)
        t.pack(fill="both", expand=True)
        t.insert("1.0", text)

    def _folder_validate_file_filters(self, file_path: Path) -> bool:
        """Validate filters."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        exts = self.folder_filter_extensions.get().strip().lower()
        if exts:
            if file_path.suffix.lower() not in [e.strip() for e in exts.split(",")]:
                return False

        sz_mb = os.path.getsize(file_path) / 1e6
        min_sz = float(self.folder_min_file_size.get() or 0)
        max_sz = float(self.folder_max_file_size.get() or 0)
        if min_sz and sz_mb < min_sz:
            return False
        if max_sz and sz_mb > max_sz:
            return False
        return True

    def _folder_get_organized_path(self, file_path: Path, dest_base: str) -> Path:
        """Get organized path."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        dest = Path(dest_base)
        if self.folder_organize_by_type_var.get():
            dest = dest / "Organized"
        if self.folder_organize_by_date_var.get():
            import datetime

            dt = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            dest = dest / dt.strftime("%Y-%m-%d")
        return dest / file_path.name

    def _folder_get_unique_path(self, path: Path) -> Path:
        """Get unique path."""
        if not (path is not None):
            raise ValueError("path must be provided")
        if not path.exists():
            return path
        i = 1
        while (path.parent / f"{path.stem}_{i}{path.suffix}").exists():
            i += 1
        return path.parent / f"{path.stem}_{i}{path.suffix}"

    def after(self, ms: int, func: Any) -> None: ...
