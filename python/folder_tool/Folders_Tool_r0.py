# Standard library imports
import ctypes
import logging
import os
import re
import shutil
import sys
import threading
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Final

# Third-party imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Constants for configuration with sources
MAX_LOG_ENTRIES: Final[int] = 20  # Maximum number of log entries to display per operation
PROGRESS_INCREMENT: Final[int] = 10  # Progress bar increment percentage for operations
MAX_FILE_SIZE_MB: Final[int] = 1024  # Maximum file size limit [MB] - reasonable limit for most systems
MIN_FILE_SIZE_BYTES: Final[int] = 1  # Minimum file size [bytes] - 1 byte minimum
DEFAULT_CHUNK_SIZE: Final[int] = 8192  # File copy chunk size [bytes] - optimal for most systems per Python docs
MAX_RETRY_ATTEMPTS: Final[int] = 3  # Maximum retry attempts for file operations
ICON_SIZES: Final[Tuple[int, ...]] = (16, 32, 48, 64)  # Standard icon sizes [pixels] per Windows guidelines

# Set up logging to capture detailed information
log_filename = "folder_processor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w")],
)

# Get logger for this module
logger = logging.getLogger(__name__)


class FolderProcessorApp:
    """
    An enhanced GUI application for comprehensive folder processing tasks.
    """

    def __init__(self, root_window: tk.Tk) -> None:
        """
        Initializes the application's user interface.
        
        Args:
            root_window: The root Tkinter window
        """
        self.root = root_window
        self.root.title("Folder Fix - Enhanced Folder Processor v2.0")
        self.root.geometry("700x900")
        self.root.minsize(600, 800)

        # Set application icon
        self._setup_application_icon()

        # --- UI Variables ---
        self.source_folders = []
        self.dest_folder = ""
        self.unzip_var = tk.BooleanVar(value=False)
        self.safe_extract_var = tk.BooleanVar(value=True)
        self.deduplicate_var = tk.BooleanVar(value=False)
        self.operation_mode = tk.StringVar(value="combine")

        # New feature variables
        self.zip_output_var = tk.BooleanVar(value=False)
        self.filter_extensions = tk.StringVar(value="")
        self.organize_by_type_var = tk.BooleanVar(value=False)
        self.organize_by_date_var = tk.BooleanVar(value=False)
        self.min_file_size = tk.StringVar(value="0")
        self.max_file_size = tk.StringVar(value="")
        self.preview_mode_var = tk.BooleanVar(value=False)
        self.backup_before_var = tk.BooleanVar(value=False)

        # Progress tracking
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.cancel_operation = False

        # --- UI Style ---
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")
        style.configure("TLabel", padding=5)

        # --- Main Frame with Scrollable Content ---
        self.create_scrollable_interface()

    def _setup_application_icon(self) -> None:
        """Sets up the application icon with fallback options."""
        try:
            # Get the directory where the script/executable is located
            if getattr(sys, "frozen", False):
                # Running as compiled executable
                base_dir = sys._MEIPASS
            else:
                # Running as script
                base_dir = os.path.dirname(__file__)

            # On Windows, set the app ID FIRST for better taskbar behavior
            self._set_windows_app_id()

            # Try ICO file first (best for Windows)
            ico_path = os.path.join(base_dir, "paper_plane_icon.ico")
            if os.path.exists(ico_path):
                self._load_ico_icon(ico_path)
            else:
                # Fallback to PNG if ICO doesn't exist
                self._load_png_fallback(base_dir)

        except Exception as e:
            logger.error(f"Could not load icon: {e}")

    def _set_windows_app_id(self) -> None:
        """Sets the Windows app user model ID for taskbar grouping."""
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "FolderFix.Tool.2.0",
            )
            logger.info("Set Windows App User Model ID for taskbar grouping")
        except Exception as e:
            logger.warning(f"Could not set app ID: {e}")

    def _load_ico_icon(self, ico_path: str) -> None:
        """Loads and sets the ICO icon for the application."""
        # Use iconbitmap for Windows taskbar integration
        self.root.iconbitmap(ico_path)
        logger.info(f"Loaded ICO icon for taskbar: {ico_path}")

        # Also set iconphoto with multiple sizes for better display
        try:
            from PIL import Image, ImageTk

            # Load the ICO file which now has multiple sizes
            image = Image.open(ico_path)

            # Create PhotoImage objects for different sizes using constants
            photos = []

            for size in ICON_SIZES:
                try:
                    # Try to get exact size from ICO, or resize
                    resized = image.resize(
                        (size, size),
                        Image.Resampling.LANCZOS,
                    )
                    if resized.mode != "RGBA":
                        resized = resized.convert("RGBA")
                    photo = ImageTk.PhotoImage(resized)
                    photos.append(photo)
                except Exception as e:
                    logger.warning(f"Could not create {size}x{size} icon: {e}")

            # Set all sizes at once for best scaling
            if photos:
                self.root.iconphoto(True, *photos)
                # Keep references to prevent garbage collection
                self.icon_photos = photos
                logger.info(f"Set iconphoto with {len(photos)} different sizes")

        except Exception as e:
            logger.warning(f"Could not set iconphoto from ICO: {e}")

    def _load_png_fallback(self, base_dir: str) -> None:
        """Loads PNG icon as fallback when ICO is not available."""
        png_path = os.path.join(base_dir, "paper_plane_icon.png")
        if os.path.exists(png_path):
            logger.info("ICO not found, using PNG fallback")
            from PIL import Image, ImageTk

            image = Image.open(png_path)

            # Convert to RGBA for transparency support
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # Create multiple sizes for better scaling using constants
            photos = []
            for size in ICON_SIZES:
                resized = image.resize((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(resized)
                photos.append(photo)

            if photos:
                self.root.iconphoto(True, *photos)
                # Keep references to prevent garbage collection
                self.icon_photos = photos
                logger.info(f"Loaded PNG icon: {png_path}")
        else:
            logger.warning(
                "No icon files found (paper_plane_icon.ico or paper_plane_icon.png)",
            )

    def create_scrollable_interface(self) -> None:
        """Creates a scrollable main interface."""
        # Create canvas and scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            """Handle mouse wheel scrolling for the canvas.
            
            Args:
                event: Mouse wheel event containing delta information
            """
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- UI SECTIONS ---
        self.create_source_widgets(main_frame)
        self.create_destination_widgets(main_frame)
        self.create_filtering_widgets(main_frame)
        self.create_preprocessing_widgets(main_frame)
        self.create_main_operation_widgets(main_frame)
        self.create_organization_widgets(main_frame)
        self.create_postprocessing_widgets(main_frame)
        self.create_output_options_widgets(main_frame)
        self.create_advanced_options_widgets(main_frame)
        self.create_progress_widgets(main_frame)
        self.create_run_button(main_frame)

        self.on_mode_change()  # Initial UI setup

    def create_source_widgets(self, parent: tk.Widget) -> None:
        """Create source folder selection widgets.
        
        Args:
            parent: Parent widget to contain the source widgets
        """
        self.source_frame = ttk.LabelFrame(
            parent,
            text="1. Select Folder(s) to Process",
            padding="10",
        )
        self.source_frame.pack(fill=tk.X, pady=5)

        # Source folder listbox with scrollbar
        listbox_frame = ttk.Frame(self.source_frame)
        listbox_frame.pack(fill=tk.X, expand=True)

        self.source_listbox = tk.Listbox(listbox_frame, height=6)
        source_scrollbar = ttk.Scrollbar(
            listbox_frame,
            orient="vertical",
            command=self.source_listbox.yview,
        )
        self.source_listbox.configure(yscrollcommand=source_scrollbar.set)

        self.source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        source_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self.source_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            button_frame,
            text="Add Folder(s)",
            command=self.select_source_folders,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_selected_source,
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        # Add folder info label
        self.source_info_label = ttk.Label(
            self.source_frame,
            text="",
            foreground="blue",
        )
        self.source_info_label.pack(fill=tk.X, pady=2)

    def create_destination_widgets(self, parent: tk.Widget) -> None:
        """Create destination folder selection widgets.
        
        Args:
            parent: Parent widget to contain the destination widgets
        """
        self.dest_frame = ttk.LabelFrame(
            parent,
            text="2. Select Final Destination Folder",
            padding="10",
        )
        self.dest_frame.pack(fill=tk.X, pady=5)
        self.dest_label = ttk.Label(
            self.dest_frame,
            text="No destination selected.",
            foreground="grey",
        )
        self.dest_label.pack(fill=tk.X, expand=True, side=tk.LEFT)
        ttk.Button(
            self.dest_frame,
            text="Set Destination",
            command=self.select_dest_folder,
        ).pack(side=tk.RIGHT)

    def create_filtering_widgets(self, parent: tk.Widget) -> None:
        """Create file filtering configuration widgets.
        
        Args:
            parent: Parent widget to contain the filtering widgets
        """
        filter_frame = ttk.LabelFrame(
            parent,
            text="3. File Filtering Options",
            padding="10",
        )
        filter_frame.pack(fill=tk.X, pady=5)

        # File extensions filter
        ext_frame = ttk.Frame(filter_frame)
        ext_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ext_frame, text="Include only extensions (comma-separated):").pack(
            side=tk.LEFT,
        )
        ttk.Entry(ext_frame, textvariable=self.filter_extensions, width=30).pack(
            side=tk.RIGHT,
        )
        ttk.Label(
            filter_frame,
            text="Example: .jpg,.png,.pdf (leave empty for all files)",
            foreground="grey",
        ).pack(anchor=tk.W)

        # File size filters
        size_frame = ttk.Frame(filter_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Min size (MB):").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.min_file_size, width=10).pack(
            side=tk.LEFT,
            padx=5,
        )
        ttk.Label(size_frame, text="Max size (MB):").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(size_frame, textvariable=self.max_file_size, width=10).pack(
            side=tk.LEFT,
            padx=5,
        )

    def create_preprocessing_widgets(self, parent: tk.Widget) -> None:
        """Create preprocessing configuration widgets.
        
        Args:
            parent: Parent widget to contain the preprocessing widgets
        """
        self.pre_process_frame = ttk.LabelFrame(
            parent,
            text="4. Pre-processing Options (On Source)",
            padding="10",
        )
        self.pre_process_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            self.pre_process_frame,
            text="Bulk extract archives (.zip, .rar, .7z)",
            variable=self.unzip_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            self.pre_process_frame,
            text="Safe extraction (verify before deleting originals)",
            variable=self.safe_extract_var,
        ).pack(anchor=tk.W, padx=(20, 0))

    def create_main_operation_widgets(self, parent: tk.Widget) -> None:
        """Create main operation selection widgets.
        
        Args:
            parent: Parent widget to contain the main operation widgets
        """
        self.mode_frame = ttk.LabelFrame(
            parent,
            text="5. Choose Main Operation",
            padding="10",
        )
        self.mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            self.mode_frame,
            text="Combine & Copy",
            variable=self.operation_mode,
            value="combine",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Flatten & Tidy",
            variable=self.operation_mode,
            value="flatten",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Copy & Prune Empty Folders",
            variable=self.operation_mode,
            value="prune",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Deduplicate Files (In-Place)",
            variable=self.operation_mode,
            value="deduplicate",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            self.mode_frame,
            text="Analyze & Report Only",
            variable=self.operation_mode,
            value="analyze",
            command=self.on_mode_change,
        ).pack(anchor=tk.W)

        self.mode_description = ttk.Label(
            self.mode_frame,
            text="",
            wraplength=600,
            justify=tk.LEFT,
        )
        self.mode_description.pack(fill=tk.X, pady=(5, 0))

    def create_organization_widgets(self, parent: tk.Widget) -> None:
        """Create file organization configuration widgets.
        
        Args:
            parent: Parent widget to contain the organization widgets
        """
        org_frame = ttk.LabelFrame(
            parent,
            text="6. File Organization Options",
            padding="10",
        )
        org_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            org_frame,
            text="Organize files by type (create subfolders)",
            variable=self.organize_by_type_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            org_frame,
            text="Organize files by date (YYYY/MM folders)",
            variable=self.organize_by_date_var,
        ).pack(anchor=tk.W)

    def create_postprocessing_widgets(self, parent: tk.Widget) -> None:
        """Create postprocessing configuration widgets.
        
        Args:
            parent: Parent widget to contain the postprocessing widgets
        """
        self.post_process_frame = ttk.LabelFrame(
            parent,
            text="7. Post-processing Options (On Destination)",
            padding="10",
        )
        self.post_process_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            self.post_process_frame,
            text="Deduplicate renamed files in destination folder after copy",
            variable=self.deduplicate_var,
        ).pack(anchor=tk.W)

    def create_output_options_widgets(self, parent: tk.Widget) -> None:
        """Create output options configuration widgets.
        
        Args:
            parent: Parent widget to contain the output options widgets
        """
        output_frame = ttk.LabelFrame(parent, text="8. Output Options", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            output_frame,
            text="Create ZIP archive of final result",
            variable=self.zip_output_var,
        ).pack(anchor=tk.W)

    def create_advanced_options_widgets(self, parent: tk.Widget) -> None:
        """Create advanced options configuration widgets.
        
        Args:
            parent: Parent widget to contain the advanced options widgets
        """
        advanced_frame = ttk.LabelFrame(
            parent,
            text="9. Advanced Options",
            padding="10",
        )
        advanced_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(
            advanced_frame,
            text="Preview mode (show what would be done without executing)",
            variable=self.preview_mode_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            advanced_frame,
            text="Create backup before processing",
            variable=self.backup_before_var,
        ).pack(anchor=tk.W)

    def create_progress_widgets(self, parent: tk.Widget) -> None:
        """Create progress tracking widgets.
        
        Args:
            parent: Parent widget to contain the progress widgets
        """
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W)

    def create_run_button(self, parent: tk.Widget) -> None:
        """Create the main run button widget.
        
        Args:
            parent: Parent widget to contain the run button
        """
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 5))

        self.run_button = ttk.Button(
            button_frame,
            text="Run Process",
            command=self.run_processing_threaded,
            style="Accent.TButton",
        )
        self.run_button.pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(0, 5),
            ipady=10,
        )

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_processing,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0), ipady=10)

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))

    def on_mode_change(self) -> None:
        """Updates UI descriptions and widget states based on the selected operation mode."""
        mode = self.operation_mode.get()

        # Update description
        descriptions = {
            "combine": "Copies all files from source folders into the single destination folder.",
            "flatten": (
                "Finds deeply nested folders and copies them to the top level of the destination."
            ),
            "prune": (
                "Copies source folders to the destination, preserving structure but "
                "skipping empty sub-folders."
            ),
            "deduplicate": (
                "Deletes renamed duplicates like 'file (1).txt' within the source folder(s), "
                "keeping the newest version."
            ),
            "analyze": (
                "Analyzes folder contents and generates a detailed report without making changes."
            ),
        }
        self.mode_description.config(text=descriptions.get(mode, ""))

        # Enable/disable widgets
        is_deduplicate_or_analyze = mode in ["deduplicate", "analyze"]
        new_state = tk.DISABLED if is_deduplicate_or_analyze else tk.NORMAL

        frames_to_toggle = [
            self.dest_frame,
            self.pre_process_frame,
            self.post_process_frame,
        ]
        for frame in frames_to_toggle:
            for child in frame.winfo_children():
                if hasattr(child, "configure"):
                    child.configure(state=new_state)

    def update_source_info(self) -> None:
        """Updates the source folder information display."""
        if not self.source_folders:
            self.source_info_label.config(text="")
            return

        total_size = 0
        total_files = 0

        for folder in self.source_folders:
            try:
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
                            total_files += 1
            except (OSError, PermissionError):
                continue

        size_mb = total_size / (1024 * 1024)
        info_text = f"Total: {total_files} files, {size_mb:.1f} MB"
        self.source_info_label.config(text=info_text)

    def run_processing_threaded(self) -> None:
        """Runs the processing in a separate thread to keep UI responsive."""
        self.cancel_operation = False
        self.run_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)

        def processing_thread() -> None:
            """Run the processing operation in a separate thread."""
            try:
                self.run_processing()
            finally:
                self.root.after(0, self.processing_complete)

        thread = threading.Thread(target=processing_thread, daemon=True)
        thread.start()

    def cancel_processing(self) -> None:
        """Cancels the current operation."""
        self.cancel_operation = True
        self.update_status("Cancelling operation...")

    def processing_complete(self) -> None:
        """Called when processing is complete to reset UI state."""
        self.run_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.update_status("Ready")

    def update_progress(self, value: int, status: str = "") -> None:
        """Updates the progress bar and status."""
        self.progress_var.set(value)
        if status:
            self.update_status(status)
        self.root.update_idletasks()

    def update_status(self, status: str) -> None:
        """Updates the status label."""
        self.status_var.set(status)
        self.root.update_idletasks()

    def validate_file_filters(self, file_path: str) -> bool:
        """Validates if a file meets the filtering criteria."""
        if self.cancel_operation:
            return False

        # Extension filter
        extensions = self.filter_extensions.get().strip()
        if extensions:
            ext_list = [ext.strip().lower() for ext in extensions.split(",")]
            file_ext = os.path.splitext(file_path)[1].lower()
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
                        f"Minimum file size cannot be negative. Setting to 0 MB."
                    )
                    self.min_file_size.set("0")
                    return False
                if min_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Minimum file size cannot exceed {MAX_FILE_SIZE_MB} MB. Setting to 0 MB."
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
                        f"Maximum file size cannot be negative. Setting to {MAX_FILE_SIZE_MB} MB."
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                    return False
                if max_size_mb > MAX_FILE_SIZE_MB:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Maximum file size cannot exceed {MAX_FILE_SIZE_MB} MB. Setting to {MAX_FILE_SIZE_MB} MB."
                    )
                    self.max_file_size.set(str(MAX_FILE_SIZE_MB))
                    return False
                
                # Check if min > max
                if min_size_str and float(min_size_str) > max_size_mb:
                    messagebox.showwarning(
                        "Invalid Input",
                        "Minimum file size cannot be greater than maximum file size."
                    )
                    return False
                    
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numeric values for file sizes."
            )
            return False
            
        return True

    def get_organized_path(self, file_path: str, dest_base: str) -> str:
        """Returns the organized destination path based on organization options."""
        filename = os.path.basename(file_path)
        dest_path = dest_base

        # Organize by type
        if self.organize_by_type_var.get():
            file_ext = os.path.splitext(filename)[1].lower()
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
            dest_path = os.path.join(dest_path, file_type)

        # Organize by date
        if self.organize_by_date_var.get():
            try:
                mtime = os.path.getmtime(file_path)
                date_folder = datetime.fromtimestamp(mtime).strftime("%Y/%m")
                dest_path = os.path.join(dest_path, date_folder)
            except OSError:
                dest_path = os.path.join(dest_path, "Unknown_Date")

        return os.path.join(dest_path, filename)

    def safe_extract_archive(self, archive_path: str) -> Tuple[bool, str]:
        """Safely extracts an archive with validation.
        
        Args:
            archive_path: Path to the archive file to extract
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not os.path.exists(archive_path):
            return False, f"Archive file not found: {archive_path}"
            
        # Validate archive file size
        try:
            archive_size = os.path.getsize(archive_path)
            if archive_size == 0:
                return False, f"Archive file is empty: {archive_path}"
        except OSError as e:
            return False, f"Cannot access archive file: {e}"
        
        extract_dir = self._get_unique_path(os.path.splitext(archive_path)[0])

        try:
            # Extract archive
            shutil.unpack_archive(archive_path, extract_dir)

            # Validate extraction if safe mode is enabled
            if self.safe_extract_var.get():
                if not os.path.exists(extract_dir):
                    raise Exception("Extraction failed - destination folder was not created")

                if not os.listdir(extract_dir):
                    raise Exception("Extraction failed - destination folder is empty")

                # Check if any files were actually extracted
                extracted_files = []
                for root, dirs, files in os.walk(extract_dir):
                    extracted_files.extend(files)

                if not extracted_files:
                    raise Exception(
                        "Extraction failed - no files found in extracted folder",
                    )
                    
                # Verify total extracted size is reasonable
                total_extracted_size = 0
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_extracted_size += os.path.getsize(file_path)
                        except OSError:
                            continue
                            
                # Check if extracted size is reasonable (should be >= archive size * 0.1)
                if total_extracted_size < archive_size * 0.1:
                    logger.warning(f"Extracted size ({total_extracted_size}) seems small compared to archive size ({archive_size})")

            # Only delete original if extraction was successful
            os.remove(archive_path)
            return (
                True,
                f"Successfully extracted and deleted '{os.path.basename(archive_path)}' ({len(extracted_files) if 'extracted_files' in locals() else 'unknown'} files)",
            )

        except Exception as e:
            # Clean up failed extraction directory
            if os.path.exists(extract_dir):
                try:
                    shutil.rmtree(extract_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup extraction directory: {cleanup_error}")
                    
            return False, f"Failed to extract '{os.path.basename(archive_path)}': {e}"

    def create_backup(self) -> Optional[str]:
        """Creates a backup of source folders before processing."""
        backup_base = os.path.join(
            os.path.dirname(self.source_folders[0]),
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        self.update_status("Creating backup...")
        for i, folder in enumerate(self.source_folders):
            if self.cancel_operation:
                return None

            backup_path = os.path.join(backup_base, os.path.basename(folder))
            shutil.copytree(folder, backup_path)

            progress = (i + 1) / len(self.source_folders) * 20  # 20% for backup
            self.update_progress(
                progress,
                f"Backing up folder {i+1}/{len(self.source_folders)}",
            )

        return backup_base

    def generate_analysis_report(self) -> Optional[str]:
        """Generates a comprehensive analysis report."""
        report = ["=== FOLDER ANALYSIS REPORT ===", f"Generated: {datetime.now()}", ""]

        total_files = 0
        total_size = 0
        file_types = defaultdict(int)
        size_by_type = defaultdict(int)
        largest_files = []

        for folder in self.source_folders:
            report.append(f"Analyzing: {folder}")
            folder_files = 0
            folder_size = 0

            for root, dirs, files in os.walk(folder):
                for file in files:
                    if self.cancel_operation:
                        return None

                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        file_ext = os.path.splitext(file)[1].lower() or "no_extension"

                        total_files += 1
                        folder_files += 1
                        total_size += file_size
                        folder_size += file_size
                        file_types[file_ext] += 1
                        size_by_type[file_ext] += file_size

                        # Track largest files
                        largest_files.append((file_path, file_size))
                        if len(largest_files) > 10:
                            largest_files.sort(key=lambda x: x[1], reverse=True)
                            largest_files = largest_files[:10]

                    except OSError:
                        continue

            report.append(
                f"  Files: {folder_files}, Size: {folder_size/(1024*1024):.1f} MB",
            )

        report.extend(
            [
                "",
                f"TOTAL FILES: {total_files}",
                f"TOTAL SIZE: {total_size/(1024*1024):.1f} MB",
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
            report.append(f"  {os.path.basename(file_path)}: {size_mb:.1f} MB")

        return "\n".join(report)

    # --- Core Application Logic ---
    def run_processing(self) -> None:
        """Main function to start the selected processing workflow."""
        mode = self.operation_mode.get()

        # Analysis mode
        if mode == "analyze":
            if not self.validate_inputs(check_destination=False):
                return
            try:
                self.update_status("Generating analysis report...")
                report = self.generate_analysis_report()
                if report:
                    self.show_text_dialog("Analysis Report", report)
                    messagebox.showinfo(
                        "Analysis Complete",
                        "Analysis report generated successfully!",
                    )
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during analysis: {e}")
            return

        # Handle deduplication mode
        if mode == "deduplicate":
            if not self.validate_inputs(check_destination=False):
                return
            try:
                results_log = self._run_deduplicate_main_op()
                messagebox.showinfo(
                    "Operation Complete",
                    "Deduplication complete.\n\n" + "\n".join(results_log),
                )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"An error occurred during deduplication: {e}",
                )
            return

        # Handle Source -> Destination workflows
        if not self.validate_inputs(check_destination=True):
            return

        # Create backup if requested
        backup_path = None
        if self.backup_before_var.get():
            backup_path = self.create_backup()
            if backup_path is None and self.cancel_operation:
                return

        # Pre-processing
        if self.unzip_var.get():
            try:
                self.update_status("Extracting archives...")
                unzip_log = self._bulk_unzip_enhanced()
                if self.cancel_operation:
                    return
                if not messagebox.askyesno(
                    "Pre-processing Complete",
                    "Bulk Extraction Complete!\n\n"
                    + "\n".join(unzip_log)
                    + "\n\nDo you want to proceed?",
                ):
                    return
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"An error occurred during bulk unzip: {e}",
                )
                return

        # Main Operation
        try:
            self.update_progress(30, "Running main operation...")
            main_op_log = []
            if mode == "combine":
                main_op_log = self._combine_folders_enhanced()
            elif mode == "flatten":
                main_op_log = self._flatten_folders()
            elif mode == "prune":
                main_op_log = self._prune_empty_folders()

            if self.cancel_operation:
                return

            final_summary = "Main Operation Complete!\n\n" + "\n".join(main_op_log)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"An error occurred during the main operation: {e}",
            )
            return

        # Post-processing
        if self.deduplicate_var.get():
            try:
                self.update_progress(70, "Deduplicating files...")
                dedupe_log = self._perform_deduplication(self.dest_folder)
                final_summary += "\n\n--- Deduplication Results ---\n" + "\n".join(
                    dedupe_log,
                )
            except Exception as e:
                final_summary += f"\n\n--- Deduplication FAILED: {e}"

        # Create output ZIP if requested
        if self.zip_output_var.get() and not self.cancel_operation:
            try:
                self.update_progress(85, "Creating ZIP archive...")
                zip_path = self.create_output_zip()
                final_summary += (
                    f"\n\n--- ZIP Archive Created ---\nLocation: {zip_path}"
                )
            except Exception as e:
                final_summary += f"\n\n--- ZIP Creation FAILED: {e}"

        if backup_path and not self.cancel_operation:
            final_summary += f"\n\n--- Backup Created ---\nLocation: {backup_path}"

        self.update_progress(100, "Complete!")

        if not self.cancel_operation:
            messagebox.showinfo("All Operations Complete", final_summary)

    def create_output_zip(self) -> bool:
        """Creates a ZIP archive of the destination folder."""
        if not os.path.exists(self.dest_folder):
            raise Exception("Destination folder does not exist")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(self.dest_folder), zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.dest_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.dest_folder)
                    zipf.write(file_path, arcname)

        return zip_path

    def show_text_dialog(self, title: str, content: str) -> None:
        """Shows a dialog with scrollable text content."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("800x600")

        text_widget = tk.Text(dialog, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")

    def validate_inputs(self, check_destination: bool = True) -> bool:
        """Validate user inputs before processing.
        
        Args:
            check_destination: Whether to validate destination folder selection
            
        Returns:
            True if inputs are valid, False otherwise
        """
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
                    if ext and not ext.startswith('.'):
                        messagebox.showwarning(
                            "Invalid Extension Format",
                            f"Extension '{ext}' should start with a dot (e.g., '.txt')."
                        )
                        return False
            except Exception:
                messagebox.showerror(
                    "Error",
                    "Invalid extension filter format. Use comma-separated values like '.txt,.pdf'."
                )
                return False
                
        return True

    # --- Enhanced Backend Processing Methods ---
    def _bulk_unzip_enhanced(self) -> List[str]:
        """Enhanced bulk extraction with better validation."""
        log = ["Starting enhanced bulk extraction..."]
        extracted_count = 0
        failed_count = 0

        # Find all archives
        archives = []
        for source_folder in self.source_folders:
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    if file.lower().endswith((".zip", ".rar", ".7z")):
                        archives.append(os.path.join(root, file))

        if not archives:
            return ["No archives found to extract."]

        for i, archive_path in enumerate(archives):
            if self.cancel_operation:
                break

            self.update_progress(
                20 + (i / len(archives)) * 10,
                f"Extracting {os.path.basename(archive_path)}...",
            )

            if not os.path.exists(archive_path):
                continue

            success, message = self.safe_extract_archive(archive_path)
            log.append(message)

            if success:
                extracted_count += 1
            else:
                failed_count += 1

        summary = f"Processed {len(archives)} archive(s). "
        summary += f"Successfully extracted: {extracted_count}, Failed: {failed_count}"
        return [summary] + log[1:]

    def _safe_copy_file(self, source_path: str, dest_path: str) -> bool:
        """Safely copy a file with retry logic and error handling.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            True if copy successful, False otherwise
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Ensure destination directory exists
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy file with metadata preservation
                shutil.copy2(source_path, dest_path)
                
                # Verify copy was successful
                if os.path.exists(dest_path):
                    source_size = os.path.getsize(source_path)
                    dest_size = os.path.getsize(dest_path)
                    if source_size == dest_size:
                        return True
                    else:
                        # Size mismatch, remove failed copy and retry
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            continue
                        
            except (OSError, IOError) as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    continue
                else:
                    logger.error(f"Failed to copy {source_path} after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                    
        return False

    def _combine_folders_enhanced(self) -> List[str]:
        """Enhanced combine operation with filtering and organization."""
        log = []
        file_count = 0
        renamed_count = 0
        skipped_count = 0
        failed_count = 0

        os.makedirs(self.dest_folder, exist_ok=True)

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:
            for root, dirs, files in os.walk(src):
                total_files += len(files)

        processed_files = 0

        for src in self.source_folders:
            if self.cancel_operation:
                break

            for root, dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:
                        break

                    source_path = os.path.join(root, file)

                    # Apply filters
                    if not self.validate_file_filters(source_path):
                        skipped_count += 1
                        processed_files += 1
                        continue

                    # Get organized destination path
                    dest_path = self.get_organized_path(source_path, self.dest_folder)
                    dest_dir = os.path.dirname(dest_path)

                    # Create destination directory if needed
                    os.makedirs(dest_dir, exist_ok=True)

                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{os.path.basename(final_dest_path)}'",
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
                    except Exception as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")

                    processed_files += 1
                    if processed_files % 10 == 0:  # Update progress every 10 files
                        progress = 30 + (processed_files / total_files) * 40
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
    def _perform_deduplication(self, target_folder: str) -> List[str]:
        """Core logic to find and delete renamed duplicates in a single target folder."""
        log = []
        deleted_count = 0
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")

        if not self.preview_mode_var.get():
            confirm = messagebox.askyesno(
                "Confirm Deletion",
                f"This will permanently delete duplicate files in:\n{target_folder}\n\n"
                + "It keeps the newest version of files like 'file (1).txt'. "
                + "This cannot be undone. Are you sure?",
            )
            if not confirm:
                return ["Deduplication cancelled by user."]

        log.append(f"Processing folder: {target_folder}")
        for dirpath, _, filenames in os.walk(target_folder):
            if self.cancel_operation:
                break

            files_by_base_name = {}
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
                        f"Duplicate set for '{base_name}': Keeping '{Path(file_to_keep).name}'",
                    )

                    for file_path in files:
                        if file_path != file_to_keep:
                            try:
                                if not self.preview_mode_var.get():
                                    Path(file_path).unlink()
                                log.append(
                                    f"  - {'WOULD DELETE' if self.preview_mode_var.get() else 'DELETED'}: "
                                    f"'{Path(file_path).name}'",
                                )
                                deleted_count += 1
                            except OSError as e:
                                log.append(
                                    f"  - FAILED to delete '{Path(file_path).name}': {e}",
                                )

        summary = [
            f"Deduplication {'preview' if self.preview_mode_var.get() else 'complete'}.",
            f"{'Would delete' if self.preview_mode_var.get() else 'Deleted'} a total of {deleted_count} files.",
            *log[:MAX_LOG_ENTRIES],
        ]

        if len(log) > MAX_LOG_ENTRIES:
            summary.append("... (see log for full details)")

        return summary

    # Keep other existing methods...
    def _run_deduplicate_main_op(self) -> List[str]:
        """Run deduplication as a main, in-place operation on source folders."""
        full_log = []
        for folder in self.source_folders:
            if self.cancel_operation:
                break
            folder_log = self._perform_deduplication(folder)
            full_log.extend(folder_log)
            full_log.append("---")
        return full_log

    def _get_unique_path(self, path: str) -> str:
        """Generate a unique path by appending counter if path exists.
        
        Args:
            path: Original file or directory path
            
        Returns:
            Unique path that doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return path
            
        parent = path_obj.parent
        name = path_obj.name
        is_file = "." in name and not path_obj.is_dir()
        
        if is_file:
            filename = path_obj.stem
            ext = path_obj.suffix
        else:
            filename = name
            ext = ""
            
        counter = 1
        new_path = parent / f"{filename} ({counter}){ext}"
        
        while new_path.exists():
            counter += 1
            new_path = parent / f"{filename} ({counter}){ext}"
            
        return str(new_path)

    def select_source_folders(self) -> None:
        """Open folder selection dialog to add source folders."""
        folder = filedialog.askdirectory(
            mustexist=True,
            title="Select a folder to process",
        )
        if folder and folder not in self.source_folders:
            self.source_folders.append(folder)
            self.source_listbox.insert(tk.END, folder)
            self.update_source_info()

    def remove_selected_source(self) -> None:
        """Remove selected source folders from the list."""
        for i in sorted(self.source_listbox.curselection(), reverse=True):
            self.source_folders.pop(i)
            self.source_listbox.delete(i)
        self.update_source_info()

    def select_dest_folder(self) -> None:
        """Open folder selection dialog to select destination folder."""
        folder = filedialog.askdirectory(
            mustexist=True,
            title="Select the destination folder",
        )
        if folder:
            self.dest_folder = folder
            self.dest_label.config(text=self.dest_folder, foreground="black")

    def _flatten_folders(self) -> List[str]:
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
            for root, dirs, files in os.walk(src):
                total_files += len(files)
        
        processed_files = 0
        
        for src in self.source_folders:
            if self.cancel_operation:
                break
                
            for root, dirs, files in os.walk(src):
                for file in files:
                    if self.cancel_operation:
                        break
                        
                    source_path = os.path.join(root, file)
                    
                    # Apply filters
                    if not self.validate_file_filters(source_path):
                        skipped_count += 1
                        processed_files += 1
                        continue
                    
                    # Get organized destination path (flattened to root)
                    dest_path = self.get_organized_path(source_path, self.dest_folder)
                    dest_dir = os.path.dirname(dest_path)
                    
                    # Create destination directory if needed
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_path)
                    if final_dest_path != dest_path:
                        log.append(
                            f"Renamed: '{file}' to '{os.path.basename(final_dest_path)}'",
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
                    except Exception as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")
                    
                    processed_files += 1
                    if processed_files % 10 == 0:  # Update progress every 10 files
                        progress = 30 + (processed_files / total_files) * 40
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

    def _prune_empty_folders(self) -> List[str]:
        """Copy source folders to destination while preserving structure but skipping empty sub-folders.
        
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
            for root, dirs, files in os.walk(src):
                total_files += len(files)
        
        processed_files = 0
        
        for src in self.source_folders:
            if self.cancel_operation:
                break
                
            src_name = os.path.basename(src)
            dest_src_path = os.path.join(self.dest_folder, src_name)
            
            for root, dirs, files in os.walk(src):
                if self.cancel_operation:
                    break
                    
                # Skip empty folders
                if not files and not any(os.listdir(os.path.join(root, d)) for d in dirs if os.path.exists(os.path.join(root, d))):
                    empty_folders_skipped += 1
                    continue
                
                # Calculate relative path from source root
                rel_path = os.path.relpath(root, src)
                dest_path = os.path.join(dest_src_path, rel_path)
                
                # Create destination directory
                os.makedirs(dest_path, exist_ok=True)
                
                # Copy files in this directory
                for file in files:
                    if self.cancel_operation:
                        break
                        
                    source_file_path = os.path.join(root, file)
                    
                    # Apply filters
                    if not self.validate_file_filters(source_file_path):
                        processed_files += 1
                        continue
                    
                    dest_file_path = os.path.join(dest_path, file)
                    
                    # Handle naming conflicts
                    final_dest_path = self._get_unique_path(dest_file_path)
                    if final_dest_path != dest_file_path:
                        log.append(
                            f"Renamed: '{file}' to '{os.path.basename(final_dest_path)}'",
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
                    except Exception as e:
                        failed_count += 1
                        log.append(f"ERROR copying '{file}': {e}")
                    
                    processed_files += 1
                    if processed_files % 10 == 0:  # Update progress every 10 files
                        progress = 30 + (processed_files / total_files) * 40
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


if __name__ == "__main__":
    root = tk.Tk()
    app = FolderProcessorApp(root)
    root.mainloop()
