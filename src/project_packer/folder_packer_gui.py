"""Folder Packer GUI application for organizing and packaging folders."""

import datetime
import logging
import shutil
from pathlib import Path

# Use shared path utilities
from utils.path_helpers import ensure_utils_in_path

ensure_utils_in_path()

from constants import (  # noqa: E402
    BOLD_HEADER_FONT_SIZE,
    DEFAULT_LISTBOX_HEIGHT,
    DEFAULT_PADDING,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    GRID_WEIGHT_MAIN,
    HEADER_FONT_SIZE,
    SMALL_PADDING,
    STATUS_TEXT_HEIGHT,
    TINY_PADDING,
    TITLE_FONT_SIZE,
)

# Import UTC from compatibility module
from utils.compatibility import UTC  # noqa: E402

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    # If tkinter is not available (e.g. headless CI), mock it for tests
    from unittest.mock import MagicMock

    tk = MagicMock()
    filedialog = MagicMock()
    messagebox = MagicMock()
    ttk = MagicMock()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for file filtering
INCLUDE_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
    ".xlsx",
    ".xls",
    ".pdf",
    ".doc",
    ".docx",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".sql",
    ".xml",
}

EXCLUDE_PATTERNS = {
    "__pycache__",
    ".git",
    ".svn",
    ".DS_Store",
    "Thumbs.db",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "*.egg-info",
}


class FolderPackerGUI:
    """GUI application for packing and organizing folders."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the Folder Packer GUI.

        Args:
            root: Main Tkinter root window.

        """
        if not (root is not None):
            raise ValueError("root must be provided")
        self.root = root
        self.root.title("Folder Packer")
        self.root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
        self.root.resizable(width=True, height=True)

        # Initialize data
        self.source_folders: list[str] = []
        self.output_directory: str = ""

        # Set up UI
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self) -> None:
        """Set up the main user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=str(DEFAULT_PADDING))
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=GRID_WEIGHT_MAIN)
        self.root.rowconfigure(0, weight=GRID_WEIGHT_MAIN)
        main_frame.columnconfigure(2, weight=GRID_WEIGHT_MAIN)

        self._setup_header(main_frame)
        self._setup_source_section(main_frame)
        self._setup_output_section(main_frame)
        self._setup_actions(main_frame)
        self._setup_status_section(main_frame)

    def _setup_header(self, parent: ttk.Frame) -> None:
        """Set up the header section."""
        # Title
        if not (parent is not None):
            raise ValueError("parent must be provided")
        title_label = ttk.Label(
            parent,
            text="📁 Folder Packer",
            font=("Arial", TITLE_FONT_SIZE, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, DEFAULT_PADDING))

        # Description
        desc_label = ttk.Label(
            parent,
            text="Select source folders and pack them to a destination directory",
            font=("Arial", HEADER_FONT_SIZE),
        )
        desc_label.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(0, DEFAULT_PADDING),
        )

    def _setup_source_section(self, parent: ttk.Frame) -> None:
        """Set up the source folders section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        source_frame = ttk.LabelFrame(parent, text="Source Folders", padding=str(SMALL_PADDING))
        source_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, DEFAULT_PADDING))
        source_frame.columnconfigure(1, weight=GRID_WEIGHT_MAIN)

        # Source folders listbox
        self.folders_listbox = tk.Listbox(source_frame, height=DEFAULT_LISTBOX_HEIGHT)
        self.folders_listbox.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, SMALL_PADDING),
        )

        # Source folder buttons
        ttk.Button(
            source_frame,
            text="Add Folder",
            command=self.add_folder,
        ).grid(row=1, column=0, sticky="w", pady=(0, TINY_PADDING))

        ttk.Button(
            source_frame,
            text="Remove Selected",
            command=self.remove_selected_folders,
        ).grid(row=1, column=1, sticky="w", pady=(0, TINY_PADDING))

    def _setup_output_section(self, parent: ttk.Frame) -> None:
        """Set up the output directory section."""
        # Output directory section
        if not (parent is not None):
            raise ValueError("parent must be provided")
        output_label = ttk.Label(
            parent,
            text="Output Directory:",
            font=("Arial", BOLD_HEADER_FONT_SIZE, "bold"),
        )
        output_label.grid(row=3, column=0, sticky="w", pady=(0, TINY_PADDING))

        # Output directory entry and browse button
        output_frame = ttk.Frame(parent)
        output_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, DEFAULT_PADDING))
        output_frame.columnconfigure(0, weight=GRID_WEIGHT_MAIN)

        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, SMALL_PADDING))

        ttk.Button(
            output_frame,
            text="Browse",
            command=self.browse_output,
        ).grid(row=0, column=1)

    def _setup_actions(self, parent: ttk.Frame) -> None:
        """Set up action buttons."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        pack_button = ttk.Button(
            parent,
            text="Pack Folders",
            command=self.pack_folders,
            style="Accent.TButton",
        )
        pack_button.grid(row=5, column=0, columnspan=3, pady=(0, DEFAULT_PADDING))

    def _setup_status_section(self, parent: ttk.Frame) -> None:
        """Set up status display section."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        status_frame = ttk.LabelFrame(parent, text="Status", padding=str(SMALL_PADDING))
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        status_frame.columnconfigure(0, weight=GRID_WEIGHT_MAIN)

        self.status_text = tk.Text(status_frame, height=STATUS_TEXT_HEIGHT, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky="ew")

        # Scrollbar for status text
        status_scrollbar = ttk.Scrollbar(
            status_frame,
            orient="vertical",
            command=self.status_text.yview,
        )
        status_scrollbar.grid(row=0, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

    def setup_styles(self) -> None:
        """Set up custom styles for the application."""
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", BOLD_HEADER_FONT_SIZE, "bold"))

    def add_folder(self) -> None:
        """Add a folder to the source folders list."""
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder and folder not in self.source_folders:
            self.source_folders.append(folder)
            self.folders_listbox.insert(tk.END, folder)

    def remove_selected_folders(self) -> None:
        """Remove selected folders from the source folders list."""
        selection = self.folders_listbox.curselection()
        for index in reversed(selection):
            folder = self.folders_listbox.get(index)
            self.source_folders.remove(folder)
            self.folders_listbox.delete(index)

    def browse_output(self) -> None:
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory = directory
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, directory)

    def pack_folders(self) -> None:
        """Pack the selected folders to the output directory."""
        if not self.source_folders:
            messagebox.showwarning(
                "Warning",
                "Please select at least one source folder.",
            )
            return

        if not self.output_directory:
            messagebox.showwarning("Warning", "Please select an output directory.")
            return

        try:
            output_path = Path(self.output_directory)
            output_path.mkdir(parents=True, exist_ok=True)

            success_count = 0
            total_count = len(self.source_folders)

            for folder in self.source_folders:
                self.update_status(f"Packing: {folder}")
                if self.pack_single_folder(folder):
                    success_count += 1
                    self.update_status(f"✅ Successfully packed: {folder}")
                else:
                    self.update_status(f"❌ Failed to pack: {folder}")

            if success_count == total_count:
                messagebox.showinfo(
                    "Success",
                    f"All {success_count} folders packed successfully to:\n{output_path}",
                )
            else:
                messagebox.showwarning(
                    "Partial Success",
                    f"Packed {success_count}/{total_count} folders successfully.\n"
                    f"Check status for details.",
                )

        except OSError as e:
            messagebox.showerror(
                "Error",
                f"An error occurred while packing: {e}",
            )
            self.update_status("Error occurred during packing")

    def pack_single_folder(self, source_folder: str) -> bool:
        """Pack a single folder to the output directory.

        Args:
            source_folder: Path to the source folder.

        Returns:
            bool: True if packing was successful, False otherwise.

        """
        try:
            source_path = Path(source_folder)
            if not source_path.exists():
                logger.error("Source folder does not exist: %s", source_folder)
                return False

            # Create destination path
            dest_path = Path(self.output_directory) / source_path.name
            if dest_path.exists():
                shutil.rmtree(dest_path)

            # Copy folder contents
            self.copy_folder_contents(source_path, dest_path)
        except OSError:
            logger.exception("Error packing %s", source_folder)
            return False
        else:
            return True

    def copy_folder_contents(self, source: Path, destination: Path) -> None:
        """Copy folder contents with filtering.

        Args:
            source: Source folder path.
            destination: Destination folder path.

        """
        if not (source is not None):
            raise ValueError("source must be provided")
        destination.mkdir(parents=True, exist_ok=True)

        for item in source.iterdir():
            if item.is_file() and self.should_include_file(item):
                shutil.copy2(item, destination / item.name)
            elif item.is_dir() and self.should_include_directory(item):
                new_dest = destination / item.name
                self.copy_folder_contents(item, new_dest)

    def should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included in the packed output.

        Args:
            file_path: Path to the file to check.

        Returns:
            bool: True if the file should be included.

        """
        # Check if it's a configuration file (these are always included)
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        config_extensions = {".env", ".config", ".conf", ".cfg", ".ini", ".toml"}
        if file_path.suffix.lower() in config_extensions:
            return True

        # Check if file extension is in the include list
        return file_path.suffix.lower() in INCLUDE_EXTENSIONS

    def should_include_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be included in the packed output.

        Args:
            dir_path: Path to the directory to check.

        Returns:
            bool: True if the directory should be included.

        """
        if not (dir_path is not None):
            raise ValueError("dir_path must be provided")
        dir_name = dir_path.name.lower()

        # Always exclude certain patterns
        if any(pattern.lower() in dir_name for pattern in EXCLUDE_PATTERNS):
            return False

        # Check if any files in the directory should be included
        return any(self._should_include_item(item) for item in dir_path.iterdir())

    def _should_include_item(self, item: Path) -> bool:
        """Check if an item (file or dir) should be included."""
        if not (item is not None):
            raise ValueError("item must be provided")
        if item.is_file():
            return self.should_include_file(item)
        if item.is_dir():
            return self.should_include_directory(item)
        return False

    def update_status(self, message: str) -> None:
        """Update the status display with a new message.

        Args:
            message: Status message to display.

        """
        if not (message is not None):
            raise ValueError("message must be provided")
        timestamp = datetime.datetime.now(tz=UTC).strftime("%H:%M:%S")
        status_line = f"[{timestamp}] {message}\n"
        self.status_text.insert(tk.END, status_line)
        self.status_text.see(tk.END)
        self.root.update_idletasks()


def main() -> None:
    """Run the Folder Packer application."""
    # Create main window
    root = tk.Tk()
    FolderPackerGUI(root)

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()
