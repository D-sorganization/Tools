"""Folder tool tab for data processing operations."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .threads import create_processing_thread


class FolderToolTab:
    """Tab for folder processing operations."""

    def __init__(self, parent: tk.Frame) -> None:
        """Initialize the folder tool tab.

        Args:
            parent: Parent frame widget

        """
        self.parent = parent
        self.source_folders: list[Path] = []
        self.dest_folder: Path | None = None
        self.processing_thread = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the user interface for the folder tool tab."""
        # Main frame
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Folder Processing Tool",
            font=("Arial", 14, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Source folders section
        ttk.Label(main_frame, text="Source Folders:").grid(
            row=1,
            column=0,
            sticky=tk.W,
            pady=(0, 5),
        )

        # Source folders listbox
        self.folders_listbox = tk.Listbox(main_frame, height=6, selectmode=tk.EXTENDED)
        self.folders_listbox.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky=(tk.W, tk.E),
            pady=(0, 10),
        )

        # Scrollbar for folders listbox
        folders_scrollbar = ttk.Scrollbar(
            main_frame,
            orient=tk.VERTICAL,
            command=self.folders_listbox.yview,
        )
        folders_scrollbar.grid(row=2, column=2, sticky=(tk.N, tk.S))
        self.folders_listbox.configure(yscrollcommand=folders_scrollbar.set)

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=(0, 20))

        ttk.Button(
            buttons_frame,
            text="Add Folder",
            command=self.folder_select_source_folders,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            buttons_frame,
            text="Remove Selected",
            command=self.folder_remove_selected_source,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            buttons_frame,
            text="Clear All",
            command=self.folder_clear_source_folders,
        ).pack(side=tk.LEFT)

        # Operation selection
        ttk.Label(main_frame, text="Operation:").grid(
            row=4,
            column=0,
            sticky=tk.W,
            pady=(0, 5),
        )

        self.operation_var = tk.StringVar(value="combine")
        operation_frame = ttk.Frame(main_frame)
        operation_frame.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky=(tk.W, tk.E),
            pady=(0, 20),
        )

        ttk.Radiobutton(
            operation_frame,
            text="Combine",
            variable=self.operation_var,
            value="combine",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            operation_frame,
            text="Flatten",
            variable=self.operation_var,
            value="flatten",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            operation_frame,
            text="Prune",
            variable=self.operation_var,
            value="prune",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            operation_frame,
            text="Deduplicate",
            variable=self.operation_var,
            value="deduplicate",
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Radiobutton(
            operation_frame,
            text="Analyze",
            variable=self.operation_var,
            value="analyze",
        ).pack(side=tk.LEFT)

        # Destination folder section
        ttk.Label(main_frame, text="Destination Folder:").grid(
            row=6,
            column=0,
            sticky=tk.W,
            pady=(0, 5),
        )

        # Destination folder entry and browse button
        dest_frame = ttk.Frame(main_frame)
        dest_frame.grid(
            row=7,
            column=0,
            columnspan=3,
            sticky=(tk.W, tk.E),
            pady=(0, 20),
        )
        dest_frame.columnconfigure(0, weight=1)

        self.dest_entry = ttk.Entry(dest_frame)
        self.dest_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        ttk.Button(
            dest_frame,
            text="Browse",
            command=self.folder_select_dest_folder,
        ).grid(row=0, column=1)

        # Process button
        self.process_button = ttk.Button(
            main_frame,
            text="🚀 Process Folders",
            command=self.folder_run_processing,
        )
        self.process_button.grid(row=8, column=0, columnspan=3, pady=(0, 20))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(
            row=9,
            column=0,
            columnspan=3,
            sticky=(tk.W, tk.E),
            pady=(0, 10),
        )

        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Ready to process folders",
            font=("Arial", 9),
        )
        self.status_label.grid(row=10, column=0, columnspan=3)

    def folder_select_source_folders(self) -> None:
        """Select source folders for processing."""
        folder_paths = filedialog.askdirectory(
            title="Select Source Folders",
            multiple=True,
        )

        if folder_paths:
            for folder_path in folder_paths:
                path = Path(folder_path)
                if path not in self.source_folders:
                    self.source_folders.append(path)
                    self.folders_listbox.insert(tk.END, str(path))

            self.update_folder_progress(f"Added {len(folder_paths)} folder(s)")

    def folder_remove_selected_source(self) -> None:
        """Remove selected source folders."""
        selected_indices = self.folders_listbox.curselection()

        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select folders to remove")
            return

        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            folder_path = Path(self.folders_listbox.get(index))
            self.source_folders.remove(folder_path)
            self.folders_listbox.delete(index)

        self.update_folder_progress(f"Removed {len(selected_indices)} folder(s)")

    def folder_clear_source_folders(self) -> None:
        """Clear all source folders."""
        if self.source_folders:
            self.source_folders.clear()
            self.folders_listbox.delete(0, tk.END)
            self.update_folder_progress("Cleared all folders")

    def on_folder_operation_changed(self) -> None:
        """Handle operation selection change."""
        operation = self.operation_var.get()
        self.update_folder_progress(f"Operation changed to: {operation}")

    def folder_select_dest_folder(self) -> None:
        """Select destination folder for processing."""
        dest_path = filedialog.askdirectory(title="Select Destination Folder")

        if dest_path:
            self.dest_folder = Path(dest_path)
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, str(self.dest_folder))
            self.update_folder_progress(f"Destination: {self.dest_folder.name}")

    def folder_run_processing(self) -> None:
        """Start folder processing operation."""
        if not self.source_folders:
            messagebox.showwarning(
                "No Folders",
                "Please add at least one source folder",
            )
            return

        if not self.dest_folder:
            messagebox.showwarning(
                "No Destination",
                "Please select a destination folder",
            )
            return

        # Get selected operation
        operation = self.operation_var.get()

        # Start processing
        self.progress.start()
        self.update_folder_progress(f"Processing folders with {operation} operation...")

        # Create and start processing thread
        self.processing_thread = create_processing_thread(self, operation)
        self.processing_thread.start()

        # Update UI
        self.process_button.config(state="disabled")
        self.status_label.config(text="Processing...")

    def get_folder_source_list(self) -> list[Path]:
        """Get list of source folders.

        Returns:
            List[Path]: List of source folder paths

        """
        return self.source_folders.copy()

    def update_folder_progress(self, message: str) -> None:
        """Update progress and status information.

        Args:
            message: Status message to display

        """
        self.status_label.config(text=message)
        self.parent.update_idletasks()

    def folder_processing_finished(self) -> None:
        """Handle completion of folder processing."""
        self.progress.stop()
        self.process_button.config(state="normal")
        self.update_folder_progress("Processing completed")

        if self.processing_thread:
            self.processing_thread = None
