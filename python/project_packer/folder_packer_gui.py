"""Folder Packer GUI application for organizing and packaging folders."""

import datetime
import logging
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
        self.root = root
        self.root.title("Folder Packer")
        self.root.geometry("600x500")
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
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="📁 Folder Packer",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Select source folders and pack them to a destination directory",
            font=("Arial", 10),
        )
        desc_label.grid(
            row=1,
            column=0,
            columnspan=3,
            pady=(0, 20),
        )

        # Source folders section
        source_frame = ttk.LabelFrame(main_frame, text="Source Folders", padding="10")
        source_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 20))
        source_frame.columnconfigure(1, weight=1)

        # Source folders listbox
        self.folders_listbox = tk.Listbox(source_frame, height=6)
        self.folders_listbox.grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10),
        )

        # Source folder buttons
        ttk.Button(
            source_frame,
            text="Add Folder",
            command=self.add_folder,
        ).grid(row=1, column=0, sticky="w", pady=(0, 5))

        ttk.Button(
            source_frame,
            text="Remove Selected",
            command=self.remove_selected_folders,
        ).grid(row=1, column=1, sticky="w", pady=(0, 5))

        # Output directory section
        output_label = ttk.Label(
            main_frame,
            text="Output Directory:",
            font=("Arial", 10, "bold"),
        )
        output_label.grid(row=3, column=0, sticky="w", pady=(0, 5))

        # Output directory entry and browse button
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 20))
        output_frame.columnconfigure(0, weight=1)

        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ttk.Button(
            output_frame,
            text="Browse",
            command=self.browse_output,
        ).grid(row=0, column=1)

        # Pack button
        pack_button = ttk.Button(
            main_frame,
            text="Pack Folders",
            command=self.pack_folders,
            style="Accent.TButton",
        )
        pack_button.grid(row=5, column=0, columnspan=3, pady=(0, 20))

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        status_frame.columnconfigure(0, weight=1)

        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
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
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))

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
                "Warning", "Please select at least one source folder.",
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
        dir_name = dir_path.name.lower()

        # Always exclude certain patterns
        if any(pattern.lower() in dir_name for pattern in EXCLUDE_PATTERNS):
            return False

        # Check if any files in the directory should be included
        for item in dir_path.iterdir():
            if item.is_file() and self.should_include_file(item):
                return True
            if item.is_dir() and self.should_include_directory(item):
                return True

        return False

    def update_status(self, message: str) -> None:
        """Update the status display with a new message.

        Args:
            message: Status message to display.

        """
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%H:%M:%S")
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
