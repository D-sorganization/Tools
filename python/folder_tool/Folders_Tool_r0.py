# Standard library imports
import ctypes
import logging
import os
import re
import shutil
import sys
import threading

# Third-party imports
import tkinter as tk
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Final

# Constants for configuration with sources and units
MAX_LOG_ENTRIES: Final[int] = 20  # Maximum number of log entries to display per operation - UI performance limit
PROGRESS_INCREMENT: Final[int] = 10  # Progress bar increment percentage [%] - standard UI update frequency
MAX_FILE_SIZE_MB: Final[int] = 1024  # Maximum file size limit [MB] - Windows FAT32 limit per Microsoft docs
MIN_FILE_SIZE_BYTES: Final[int] = 1  # Minimum file size [bytes] - 1 byte minimum per filesystem standards
DEFAULT_CHUNK_SIZE: Final[int] = 8192  # File copy chunk size [bytes] - optimal for most systems per Python shutil docs
MAX_RETRY_ATTEMPTS: Final[int] = 3  # Maximum retry attempts for file operations - industry standard retry limit
ICON_SIZES: Final[tuple[int, ...]] = (16, 32, 48, 64)  # Standard icon sizes [pixels] per Windows Shell API guidelines
MAX_STATUS_LENGTH: Final[int] = 200  # Maximum status message length [characters] - prevents UI overflow
MAX_UI_UPDATE_FREQUENCY: Final[int] = 10  # Update progress every N files - balances responsiveness with performance
MAX_ARCHIVE_SIZE_RATIO: Final[float] = 0.1  # Minimum extracted size ratio [ratio] - archive size * 0.1 for validation
MAX_DIALOG_WIDTH: Final[int] = 800  # Maximum dialog width [pixels] - prevents dialog overflow
MAX_DIALOG_HEIGHT: Final[int] = 600  # Maximum dialog height [pixels] - prevents dialog overflow
MIN_DIALOG_WIDTH: Final[int] = 400  # Minimum dialog width [pixels] - ensures usability
MIN_DIALOG_HEIGHT: Final[int] = 300  # Minimum dialog height [pixels] - ensures usability

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

        # Validate constants at startup
        self._validate_constants()

        # --- Main Frame with Scrollable Content ---
        self.create_scrollable_interface()

    def _validate_constants(self) -> None:
        """Validates that all constants meet the required constraints.
        
        Raises:
            ValueError: If any constant violates its constraints
        """
        # Validate file size constants
        if MAX_FILE_SIZE_MB <= 0:
            raise ValueError(f"MAX_FILE_SIZE_MB must be positive, got {MAX_FILE_SIZE_MB}")
        if MIN_FILE_SIZE_BYTES < 0:
            raise ValueError(f"MIN_FILE_SIZE_BYTES must be non-negative, got {MIN_FILE_SIZE_BYTES}")
        if MIN_FILE_SIZE_BYTES >= MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"MIN_FILE_SIZE_BYTES must be less than MAX_FILE_SIZE_MB, got {MIN_FILE_SIZE_BYTES}")

        # Validate UI constants
        if MAX_STATUS_LENGTH <= 0:
            raise ValueError(f"MAX_STATUS_LENGTH must be positive, got {MAX_STATUS_LENGTH}")
        if MAX_UI_UPDATE_FREQUENCY <= 0:
            raise ValueError(f"MAX_UI_UPDATE_FREQUENCY must be positive, got {MAX_UI_UPDATE_FREQUENCY}")
        if MAX_DIALOG_WIDTH <= MIN_DIALOG_WIDTH:
            raise ValueError(f"MAX_DIALOG_WIDTH must be greater than MIN_DIALOG_WIDTH, got {MAX_DIALOG_WIDTH} <= {MIN_DIALOG_WIDTH}")
        if MAX_DIALOG_HEIGHT <= MIN_DIALOG_HEIGHT:
            raise ValueError(f"MAX_DIALOG_HEIGHT must be greater than MIN_DIALOG_HEIGHT, got {MAX_DIALOG_HEIGHT} <= {MIN_DIALOG_HEIGHT}")

        # Validate archive constants
        if not 0 < MAX_ARCHIVE_SIZE_RATIO < 1:
            raise ValueError(f"MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1, got {MAX_ARCHIVE_SIZE_RATIO}")

        # Validate retry constants
        if MAX_RETRY_ATTEMPTS <= 0:
            raise ValueError(f"MAX_RETRY_ATTEMPTS must be positive, got {MAX_RETRY_ATTEMPTS}")

        logger.info("All constants validated successfully")

    def get_constants_info(self) -> dict[str, dict[str, str]]:
        """Returns information about all constants for debugging and documentation.
        
        Returns:
            Dictionary mapping constant names to their metadata [dict] - includes value, units, and source
            
        Example:
            {
                'MAX_FILE_SIZE_MB': {
                    'value': '1024',
                    'units': 'MB',
                    'source': 'Windows FAT32 limit per Microsoft docs'
                }
            }
        """
        return {
            "MAX_LOG_ENTRIES": {
                "value": str(MAX_LOG_ENTRIES),
                "units": "entries",
                "source": "UI performance limit"
            },
            "PROGRESS_INCREMENT": {
                "value": str(PROGRESS_INCREMENT),
                "units": "%",
                "source": "Standard UI update frequency"
            },
            "MAX_FILE_SIZE_MB": {
                "value": str(MAX_FILE_SIZE_MB),
                "units": "MB",
                "source": "Windows FAT32 limit per Microsoft docs"
            },
            "MIN_FILE_SIZE_BYTES": {
                "value": str(MIN_FILE_SIZE_BYTES),
                "units": "bytes",
                "source": "1 byte minimum per filesystem standards"
            },
            "DEFAULT_CHUNK_SIZE": {
                "value": str(DEFAULT_CHUNK_SIZE),
                "units": "bytes",
                "source": "Optimal for most systems per Python shutil docs"
            },
            "MAX_RETRY_ATTEMPTS": {
                "value": str(MAX_RETRY_ATTEMPTS),
                "units": "attempts",
                "source": "Industry standard retry limit"
            },
            "ICON_SIZES": {
                "value": str(ICON_SIZES),
                "units": "pixels",
                "source": "Windows Shell API guidelines"
            },
            "MAX_STATUS_LENGTH": {
                "value": str(MAX_STATUS_LENGTH),
                "units": "characters",
                "source": "Prevents UI overflow"
            },
            "MAX_UI_UPDATE_FREQUENCY": {
                "value": str(MAX_UI_UPDATE_FREQUENCY),
                "units": "files",
                "source": "Balances responsiveness with performance"
            },
            "MAX_ARCHIVE_SIZE_RATIO": {
                "value": str(MAX_ARCHIVE_SIZE_RATIO),
                "units": "ratio",
                "source": "Archive size * 0.1 for validation"
            },
            "MAX_DIALOG_WIDTH": {
                "value": str(MAX_DIALOG_WIDTH),
                "units": "pixels",
                "source": "Prevents dialog overflow"
            },
            "MAX_DIALOG_HEIGHT": {
                "value": str(MAX_DIALOG_HEIGHT),
                "units": "pixels",
                "source": "Prevents dialog overflow"
            },
            "MIN_DIALOG_WIDTH": {
                "value": str(MIN_DIALOG_WIDTH),
                "units": "pixels",
                "source": "Ensures usability"
            },
            "MIN_DIALOG_HEIGHT": {
                "value": str(MIN_DIALOG_HEIGHT),
                "units": "pixels",
                "source": "Ensures usability"
            }
        }

    def export_constants_documentation(self, output_path: str) -> bool:
        """Exports constants documentation to a file for reference.
        
        Args:
            output_path: Path to the output file [str] - will be created if it doesn't exist
            
        Returns:
            True if export successful, False otherwise
            
        Raises:
            OSError: If file operations fail
            Exception: If export fails for other reasons
        """
        try:
            constants_info = self.get_constants_info()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Folder Tool Constants Documentation\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write("## Constants Overview\n\n")

                for const_name, info in constants_info.items():
                    f.write(f"### {const_name}\n")
                    f.write(f"- **Value**: {info['value']}\n")
                    f.write(f"- **Units**: {info['units']}\n")
                    f.write(f"- **Source**: {info['source']}\n\n")

            logger.info(f"Constants documentation exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export constants documentation: {e}")
            return False

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
            try:
                if not hasattr(event, "delta"):
                    # Handle different mouse wheel event formats
                    if hasattr(event, "num"):
                        # Linux/Unix mouse wheel
                        delta = 120 if event.num == 4 else -120
                    else:
                        # Unknown format, skip
                        return
                else:
                    delta = event.delta

                # Scroll the canvas
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

            except Exception as e:
                logger.debug(f"Mouse wheel scroll error: {e}")
                # Silently continue - mouse wheel errors shouldn't crash the app

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
        accessible_folders = 0

        for folder in self.source_folders:
            try:
                if not os.path.exists(folder):
                    logger.warning(f"Source folder no longer exists: {folder}")
                    continue

                if not os.access(folder, os.R_OK):
                    logger.warning(f"Cannot access source folder: {folder}")
                    continue

                accessible_folders += 1

                for root, dirs, files in os.walk(folder):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                                file_size = os.path.getsize(file_path)
                                total_size += file_size
                                total_files += 1
                        except (OSError, PermissionError) as e:
                            logger.debug(f"Cannot access file {file_path}: {e}")
                            continue

            except (OSError, PermissionError) as e:
                logger.warning(f"Error accessing folder {folder}: {e}")
                continue

        if accessible_folders == 0:
            self.source_info_label.config(
                text="Warning: No accessible source folders",
                foreground="red"
            )
            return

        size_mb = total_size / (1024 * 1024)
        info_text = f"Total: {total_files} files, {size_mb:.1f} MB ({accessible_folders}/{len(self.source_folders)} folders accessible)"

        # Set color based on accessibility
        if accessible_folders < len(self.source_folders):
            self.source_info_label.config(text=info_text, foreground="orange")
        else:
            self.source_info_label.config(text=info_text, foreground="blue")

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
        """Updates the progress bar and status.
        
        Args:
            value: Progress value (0-100)
            status: Status message to display
        """
        try:
            # Validate progress value
            if not isinstance(value, (int, float)):
                logger.warning(f"Invalid progress value type: {type(value)}")
                return

            # Clamp progress value to valid range
            clamped_value = max(0, min(100, float(value)))
            self.progress_var.set(clamped_value)

            if status:
                self.update_status(status)

            # Update UI
            self.root.update_idletasks()

        except Exception as e:
            logger.error(f"Error updating progress: {e}")

    def update_status(self, status: str) -> None:
        """Updates the status label.
        
        Args:
            status: Status message to display
        """
        try:
            if not isinstance(status, str):
                logger.warning(f"Invalid status type: {type(status)}")
                return

            # Limit status length to prevent UI issues
            max_length = MAX_STATUS_LENGTH
            if len(status) > max_length:
                status = status[:max_length-3] + "..."

            self.status_var.set(status)
            self.root.update_idletasks()

        except Exception as e:
            logger.error(f"Error updating status: {e}")

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
                        "Minimum file size cannot be negative. Setting to 0 MB."
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
        """Returns the organized destination path based on organization options.
        
        Args:
            file_path: Source file path [str] - used to determine file type and modification date
            dest_base: Base destination directory [str] - where organized files will be placed
            
        Returns:
            Organized destination path [str] - includes type/date subdirectories if enabled
            
        Raises:
            OSError: If file system operations fail during path construction
        """
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

    def safe_extract_archive(self, archive_path: str) -> tuple[bool, str]:
        """Safely extracts an archive with validation.
        
        Args:
            archive_path: Path to the archive file to extract [str] - must exist and be readable
            
        Returns:
            Tuple of (success: bool, message: str) - success indicates extraction completed without errors
            
        Raises:
            OSError: If file system operations fail
            ValueError: If archive validation fails
            Exception: If extraction process fails
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

                # Check if extracted size is reasonable (should be >= archive size * MIN_ARCHIVE_SIZE_RATIO)
                if total_extracted_size < archive_size * MAX_ARCHIVE_SIZE_RATIO:
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

    def create_backup(self) -> str | None:
        """Creates a backup of source folders before processing.
        
        Returns:
            Path to backup directory if successful [str], None if failed
            
        Raises:
            OSError: If file system operations fail during backup creation
            PermissionError: If insufficient permissions to create backup
            Exception: If backup process fails for other reasons
        """
        if not self.source_folders:
            logger.error("No source folders to backup")
            return None

        backup_base = os.path.join(
            os.path.dirname(self.source_folders[0]),
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        self.update_status("Creating backup...")

        try:
            # Create backup base directory
            os.makedirs(backup_base, exist_ok=True)

            for i, folder in enumerate(self.source_folders):
                if self.cancel_operation:
                    return None

                if not os.path.exists(folder):
                    logger.warning(f"Source folder no longer exists: {folder}")
                    continue

                backup_path = os.path.join(backup_base, os.path.basename(folder))

                try:
                    shutil.copytree(folder, backup_path)
                    logger.info(f"Backed up folder: {folder} -> {backup_path}")
                except Exception as e:
                    logger.error(f"Failed to backup folder {folder}: {e}")
                    # Continue with other folders
                    continue

                progress = (i + 1) / len(self.source_folders) * 20  # 20% for backup
                self.update_progress(
                    progress,
                    f"Backing up folder {i+1}/{len(self.source_folders)}",
                )

            # Verify backup was created successfully
            if os.path.exists(backup_base) and os.listdir(backup_base):
                logger.info(f"Backup completed successfully: {backup_base}")
                return backup_base
            logger.error("Backup directory is empty or was not created")
            return None

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Cleanup failed backup
            if os.path.exists(backup_base):
                try:
                    shutil.rmtree(backup_base, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup failed backup: {cleanup_error}")
            return None

    def generate_analysis_report(self) -> str | None:
        """Generates a comprehensive analysis report.
        
        Returns:
            Formatted analysis report [str] if successful, None if cancelled or failed
            
        Raises:
            OSError: If file system operations fail during analysis
            Exception: If report generation fails for other reasons
        """
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

    def create_output_zip(self) -> str:
        """Creates a ZIP archive of the destination folder.
        
        Returns:
            Path to the created ZIP file [str] - absolute path to the created archive
            
        Raises:
            FileNotFoundError: If destination folder does not exist
            ValueError: If destination folder is empty
            OSError: If file system operations fail during ZIP creation
            Exception: If ZIP creation fails for other reasons
        """
        if not os.path.exists(self.dest_folder):
            raise Exception("Destination folder does not exist")

        if not os.listdir(self.dest_folder):
            raise Exception("Destination folder is empty - nothing to archive")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(self.dest_folder), zip_filename)

        # Check if ZIP file already exists
        if os.path.exists(zip_path):
            zip_path = self._get_unique_path(zip_path)

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                total_files = 0
                total_size = 0

                for root, dirs, files in os.walk(self.dest_folder):
                    for file in files:
                        if self.cancel_operation:
                            raise Exception("ZIP creation cancelled by user")

                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.dest_folder)

                        try:
                            zipf.write(file_path, arcname)
                            total_files += 1
                            total_size += os.path.getsize(file_path)
                        except Exception as e:
                            logger.warning(f"Failed to add file to ZIP: {file_path} - {e}")
                            continue

                logger.info(f"ZIP archive created: {zip_path} ({total_files} files, {total_size/(1024*1024):.1f} MB)")

        except Exception as e:
            # Cleanup failed ZIP file
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup failed ZIP file: {cleanup_error}")
            raise Exception(f"Failed to create ZIP archive: {e}")

        return zip_path

    def show_text_dialog(self, title: str, content: str) -> None:
        """Shows a dialog with scrollable text content.
        
        Args:
            title: Dialog window title [str] - must not be empty
            content: Text content to display [str] - must not be empty
            
        Raises:
            ValueError: If title or content is empty
            tkinter.TclError: If Tkinter widget creation fails
            Exception: If dialog creation fails for other reasons
        """
        if not title or not content:
            logger.warning("Invalid dialog parameters: title or content is empty")
            return

        try:
            dialog = tk.Toplevel(self.root)
            dialog.title(title)
            dialog.geometry(f"{MAX_DIALOG_WIDTH}x{MAX_DIALOG_HEIGHT}")
            dialog.minsize(MIN_DIALOG_WIDTH, MIN_DIALOG_HEIGHT)

            # Center dialog on screen
            dialog.transient(self.root)
            dialog.grab_set()

            # Create text widget with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Insert content
            text_widget.insert("1.0", content)
            text_widget.config(state="disabled")

            # Add close button
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            ttk.Button(
                button_frame,
                text="Close",
                command=dialog.destroy
            ).pack(side=tk.RIGHT)

            # Set focus and make dialog modal
            dialog.focus_set()
            dialog.wait_window()

        except Exception as e:
            logger.error(f"Failed to show text dialog: {e}")
            # Fallback to simple message box
            messagebox.showinfo(title, content[:500] + "..." if len(content) > 500 else content)

    def validate_inputs(self, check_destination: bool = True) -> bool:
        """Validate user inputs before processing.
        
        Args:
            check_destination: Whether to validate destination folder selection [bool] - defaults to True
            
        Returns:
            True if inputs are valid, False otherwise
            
        Raises:
            ValueError: If file size inputs are invalid
            Exception: If extension filter validation fails
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
                    if ext and not ext.startswith("."):
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

    def validate_application_state(self) -> dict[str, bool]:
        """Validates the current application state and returns validation results.
        
        Returns:
            Dictionary mapping validation checks to their results [dict] - True if valid, False if invalid
            
        Example:
            {
                'source_folders_exist': True,
                'destination_writable': False,
                'constants_valid': True
            }
        """
        validation_results = {}

        # Check source folders
        validation_results["source_folders_exist"] = all(
            os.path.exists(folder) for folder in self.source_folders
        ) if self.source_folders else True

        validation_results["source_folders_readable"] = all(
            os.access(folder, os.R_OK) for folder in self.source_folders
        ) if self.source_folders else True

        # Check destination folder
        if self.dest_folder:
            validation_results["destination_exists"] = os.path.exists(self.dest_folder)
            validation_results["destination_writable"] = os.access(self.dest_folder, os.W_OK)
        else:
            validation_results["destination_exists"] = True  # Not required for all modes
            validation_results["destination_writable"] = True  # Not required for all modes

        # Check file size inputs
        try:
            min_size = float(self.min_file_size.get() or 0)
            max_size = float(self.max_file_size.get() or MAX_FILE_SIZE_MB)
            validation_results["size_inputs_valid"] = (
                0 <= min_size <= MAX_FILE_SIZE_MB and
                0 <= max_size <= MAX_FILE_SIZE_MB and
                min_size <= max_size
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
            except Exception:
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

    # --- Enhanced Backend Processing Methods ---
    def _bulk_unzip_enhanced(self) -> list[str]:
        """Enhanced bulk extraction with better validation."""
        log = ["Starting enhanced bulk extraction..."]
        extracted_count = 0
        failed_count = 0

        # Find all archives
        archives = []
        for source_folder in self.source_folders:
            for root, _dirs, files in os.walk(source_folder):
                for file in files:
                    if file.lower().endswith((".zip", ".rar", ".7z")):
                        archives.append(Path(root) / file)

        if not archives:
            return ["No archives found to extract."]

        for i, archive_path in enumerate(archives):
            if self.cancel_operation:
                break

            self.update_progress(
                20 + (i / len(archives)) * 10,
                f"Extracting {Path(archive_path).name}...",
            )

            if not Path(archive_path).exists():
                continue

            success, message = self.safe_extract_archive(archive_path)
            log.append(message)

            if success:
                extracted_count += 1
            else:
                failed_count += 1

        summary = f"Processed {len(archives)} archive(s). "
        summary += f"Successfully extracted: {extracted_count}, Failed: {failed_count}"
        return [summary, *log[1:]]

    def _safe_copy_file(self, source_path: str, dest_path: str) -> bool:
        """Safely copy a file with retry logic and error handling.

        Args:
            source_path: Source file path [str] - must exist and be readable
            dest_path: Destination file path [str] - parent directory will be created if needed

        Returns:
            True if copy successful, False otherwise

        Raises:
            OSError: If file operations fail after all retry attempts
            IOError: If file I/O operations fail
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Ensure destination directory exists
                dest_dir = Path(dest_path).parent
                Path(dest_dir).mkdir(parents=True, exist_ok=True)

                # Copy file with metadata preservation
                shutil.copy2(source_path, dest_path)

                # Verify copy was successful
                if Path(dest_path).exists():
                    source_size = Path(source_path).stat().st_size
                    dest_size = Path(dest_path).stat().st_size
                    if source_size == dest_size:
                        return True
                    # Size mismatch, remove failed copy and retry
                    if Path(dest_path).exists():
                        Path(dest_path).unlink()
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        continue

            except OSError:
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    continue
                logger.exception(
                    "Failed to copy %s after %d attempts", 
                    source_path, 
                    MAX_RETRY_ATTEMPTS
                )

        return False

    def _combine_folders_enhanced(self) -> list[str]:
        """Enhanced combine operation with filtering and organization."""
        log = []
        file_count = 0
        renamed_count = 0
        skipped_count = 0
        failed_count = 0

        Path(self.dest_folder).mkdir(parents=True, exist_ok=True)

        # Count total files for progress tracking
        total_files = 0
        for src in self.source_folders:
            for _root, _dirs, files in os.walk(src):
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
                    dest_dir = Path(dest_path).parent

                    # Create destination directory if needed
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)

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
                    if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:  # Update progress every N files
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
    def _perform_deduplication(self, target_folder: str) -> list[str]:
        """Core logic to find and delete renamed duplicates in a single target folder."""
        log = []
        deleted_count = 0
        pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")

        if not self.preview_mode_var.get():
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
    def _run_deduplicate_main_op(self) -> list[str]:
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
            path: Original file or directory path [str] - will be analyzed for name and extension
            
        Returns:
            Unique path that doesn't exist [str] - original path or path with counter suffix
            
        Raises:
            OSError: If file system operations fail during path checking
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
        """Open folder selection dialog to add source folders.
        
        Raises:
            OSError: If file system operations fail during folder validation
            PermissionError: If insufficient permissions to access selected folder
            Exception: If folder selection fails for other reasons
        """
        try:
            folder = filedialog.askdirectory(
                mustexist=True,
                title="Select a folder to process",
            )
            if folder:
                # Validate folder exists and is accessible
                if not Path(folder).exists():
                    messagebox.showerror("Error", "Selected folder no longer exists.")
                    return

                if not os.access(folder, os.R_OK):
                    messagebox.showerror("Error", "Cannot access the selected folder. Check permissions.")
                    return

                if folder not in self.source_folders:
                    self.source_folders.append(folder)
                    self.source_listbox.insert(tk.END, folder)
                    self.update_source_info()
                    logger.info("Added source folder: %s", folder)
                else:
                    messagebox.showinfo("Info", "This folder is already in the source list.")
            else:
                logger.debug("Folder selection cancelled by user")

        except Exception as e:
            logger.exception("Error selecting source folder")
            messagebox.showerror("Error", f"Failed to select source folder: {e}")

    def remove_selected_source(self) -> None:
        """Remove selected source folders from the list.
        
        Raises:
            IndexError: If selected indices are invalid
            Exception: If folder removal fails for other reasons
        """
        try:
            selected_indices = list(self.source_listbox.curselection())
            if not selected_indices:
                messagebox.showinfo("Info", "Please select folders to remove.")
                return

            # Confirm removal
            if len(selected_indices) == 1:
                folder_name = os.path.basename(self.source_folders[selected_indices[0]])
                confirm = messagebox.askyesno(
                    "Confirm Removal",
                    f"Remove folder '{folder_name}' from source list?"
                )
            else:
                confirm = messagebox.askyesno(
                    "Confirm Removal",
                    f"Remove {len(selected_indices)} selected folders from source list?"
                )

            if confirm:
                # Remove in reverse order to maintain indices
                for i in sorted(selected_indices, reverse=True):
                    removed_folder = self.source_folders.pop(i)
                    self.source_listbox.delete(i)
                    logger.info("Removed source folder: %s", removed_folder)

                self.update_source_info()

        except Exception as e:
            logger.exception("Error removing source folders")
            messagebox.showerror("Error", f"Failed to remove source folders: {e}")

    def select_dest_folder(self) -> None:
        """Open folder selection dialog to select destination folder.
        
        Raises:
            OSError: If file system operations fail during folder validation
            PermissionError: If insufficient permissions to write to selected folder
            Exception: If folder selection fails for other reasons
        """
        try:
            folder = filedialog.askdirectory(
                mustexist=True,
                title="Select the destination folder",
            )
            if folder:
                # Validate folder exists and is writable
                if not Path(folder).exists():
                    messagebox.showerror("Error", "Selected folder no longer exists.")
                    return

                if not os.access(folder, os.W_OK):
                    messagebox.showerror("Error", "Cannot write to the selected folder. Check permissions.")
                    return

                self.dest_folder = folder
                self.dest_label.config(text=self.dest_folder, foreground="black")
                logger.info("Set destination folder: %s", folder)
            else:
                logger.debug("Destination folder selection cancelled by user")

        except Exception as e:
            logger.exception("Error selecting destination folder")
            messagebox.showerror("Error", f"Failed to select destination folder: {e}")

    def _flatten_folders(self) -> list[str]:
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
                    if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:  # Update progress every N files
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

    def _prune_empty_folders(self) -> list[str]:
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
                    if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:  # Update progress every N files
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
