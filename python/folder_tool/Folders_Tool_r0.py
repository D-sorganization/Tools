# Standard library imports
import ctypes
import logging
import os
import re
import shutil
import sys
import threading
import time

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

# Additional constants for improved maintainability
MAX_TEXT_CONTENT_SIZE: Final[int] = 1000000  # Maximum text content size [characters] - prevents performance issues
MAX_TITLE_LENGTH: Final[int] = 100  # Maximum title length [characters] - prevents window title truncation
MAX_COUNTER_ATTEMPTS: Final[int] = 1000  # Maximum attempts to generate unique filename [attempts] - prevents infinite loops
MAX_FALLBACK_CONTENT_SIZE: Final[int] = 500  # Maximum content size for fallback display [characters] - prevents UI overflow
PROGRESS_BACKUP_PERCENT: Final[int] = 20  # Progress percentage allocated to backup operations [%] - UI progress tracking
PROGRESS_MAIN_OP_PERCENT: Final[int] = 40  # Progress percentage allocated to main operations [%] - UI progress tracking
PROGRESS_ZIP_PERCENT: Final[int] = 10  # Progress percentage allocated to ZIP creation [%] - UI progress tracking
PROGRESS_START_MAIN: Final[int] = 30  # Starting progress percentage for main operations [%] - UI progress tracking
PROGRESS_START_ZIP: Final[int] = 85  # Starting progress percentage for ZIP creation [%] - UI progress tracking

# Dialog layout constants
CHARS_PER_DIALOG_LINE: Final[int] = 80  # Characters per line for dialog width calculation [characters] - standard text width
DIALOG_WIDTH_OFFSET: Final[int] = 100  # Additional width offset for dialog borders [pixels] - accounts for scrollbars and margins
LINE_HEIGHT_PIXELS: Final[int] = 20  # Height per line for dialog height calculation [pixels] - standard line height
MAX_TITLE_PREVIEW_LENGTH: Final[int] = 50  # Maximum title length for preview in logs [characters] - prevents log overflow

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
        
        This method ensures all constants are within valid ranges and follow
        logical relationships. It validates file sizes, UI dimensions, progress
        percentages, and other configuration values.
        
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

        # Validate new constants
        if MAX_TEXT_CONTENT_SIZE <= 0:
            raise ValueError(f"MAX_TEXT_CONTENT_SIZE must be positive, got {MAX_TEXT_CONTENT_SIZE}")
        if MAX_TITLE_LENGTH <= 0:
            raise ValueError(f"MAX_TITLE_LENGTH must be positive, got {MAX_TITLE_LENGTH}")
        if MAX_COUNTER_ATTEMPTS <= 0:
            raise ValueError(f"MAX_COUNTER_ATTEMPTS must be positive, got {MAX_COUNTER_ATTEMPTS}")
        
        # Validate progress constants
        if PROGRESS_BACKUP_PERCENT < 0 or PROGRESS_BACKUP_PERCENT > 100:
            raise ValueError(f"PROGRESS_BACKUP_PERCENT must be between 0 and 100, got {PROGRESS_BACKUP_PERCENT}")
        if PROGRESS_MAIN_OP_PERCENT < 0 or PROGRESS_MAIN_OP_PERCENT > 100:
            raise ValueError(f"PROGRESS_MAIN_OP_PERCENT must be between 0 and 100, got {PROGRESS_MAIN_OP_PERCENT}")
        if PROGRESS_ZIP_PERCENT < 0 or PROGRESS_ZIP_PERCENT > 100:
            raise ValueError(f"PROGRESS_ZIP_PERCENT must be between 0 and 100, got {PROGRESS_ZIP_PERCENT}")
        if PROGRESS_START_MAIN < 0 or PROGRESS_START_MAIN > 100:
            raise ValueError(f"PROGRESS_START_MAIN must be between 0 and 100, got {PROGRESS_START_MAIN}")
        if PROGRESS_START_ZIP < 0 or PROGRESS_START_ZIP > 100:
            raise ValueError(f"PROGRESS_START_ZIP must be between 0 and 100, got {PROGRESS_START_ZIP}")

        # Validate progress flow consistency
        total_progress = PROGRESS_BACKUP_PERCENT + PROGRESS_MAIN_OP_PERCENT + PROGRESS_ZIP_PERCENT
        if total_progress > 100:
            raise ValueError(f"Total progress allocation exceeds 100%: {total_progress}")

        logger.info("All constants validated successfully")

    def get_constants_info(self) -> dict[str, dict[str, str]]:
        """Returns information about all constants for debugging and documentation.
        
        This method provides comprehensive metadata about all constants including
        their values, units, and sources. This is useful for debugging, documentation,
        and system validation.
        
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
            },
            "MAX_TEXT_CONTENT_SIZE": {
                "value": str(MAX_TEXT_CONTENT_SIZE),
                "units": "characters",
                "source": "Prevents performance issues in text dialogs"
            },
            "MAX_TITLE_LENGTH": {
                "value": str(MAX_TITLE_LENGTH),
                "units": "characters",
                "source": "Prevents window title truncation"
            },
            "MAX_COUNTER_ATTEMPTS": {
                "value": str(MAX_COUNTER_ATTEMPTS),
                "units": "attempts",
                "source": "Prevents infinite loops in filename generation"
            },
            "MAX_FALLBACK_CONTENT_SIZE": {
                "value": str(MAX_FALLBACK_CONTENT_SIZE),
                "units": "characters",
                "source": "Prevents UI overflow in fallback dialogs"
            },
            "PROGRESS_BACKUP_PERCENT": {
                "value": str(PROGRESS_BACKUP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for backup operations"
            },
            "PROGRESS_MAIN_OP_PERCENT": {
                "value": str(PROGRESS_MAIN_OP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for main operations"
            },
            "PROGRESS_ZIP_PERCENT": {
                "value": str(PROGRESS_ZIP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for ZIP creation"
            },
            "PROGRESS_START_MAIN": {
                "value": str(PROGRESS_START_MAIN),
                "units": "%",
                "source": "Starting progress for main operations"
            },
            "PROGRESS_START_ZIP": {
                "value": str(PROGRESS_START_ZIP),
                "units": "%",
                "source": "Starting progress for ZIP creation"
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
            ValueError: If archive_path is empty or invalid
            FileNotFoundError: If archive file does not exist
            PermissionError: If insufficient permissions to read archive or write to extract directory
            OSError: If file system operations fail
            Exception: If extraction process fails
        """
        # Input validation
        if not archive_path or not isinstance(archive_path, str):
            raise ValueError(f"Archive path must be non-empty string, got {type(archive_path)}")
        
        archive_path_obj = Path(archive_path)
        
        # Validate archive file exists and is accessible
        if not archive_path_obj.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        if not archive_path_obj.is_file():
            raise ValueError(f"Archive path is not a file: {archive_path}")
        if not os.access(archive_path, os.R_OK):
            raise PermissionError(f"Cannot read archive file: {archive_path}")

        # Validate archive file size
        try:
            archive_size = archive_path_obj.stat().st_size
            if archive_size == 0:
                return False, f"Archive file is empty: {archive_path}"
            if archive_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Archive file exceeds maximum size limit: {archive_path} ({archive_size / (1024*1024):.1f} MB)")
        except OSError as e:
            return False, f"Cannot access archive file: {e}"

        # Validate archive file extension
        archive_ext = archive_path_obj.suffix.lower()
        supported_formats = {'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar'}
        if archive_ext not in supported_formats:
            logger.warning(f"Unsupported archive format: {archive_ext} for {archive_path}")

        # Generate unique extraction directory
        extract_dir = self._get_unique_path(os.path.splitext(archive_path)[0])
        extract_dir_obj = Path(extract_dir)

        try:
            # Create extraction directory
            extract_dir_obj.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created and is writable
            if not extract_dir_obj.exists():
                raise Exception("Failed to create extraction directory")
            if not os.access(extract_dir, os.W_OK):
                raise PermissionError(f"Cannot write to extraction directory: {extract_dir}")

            # Extract archive
            logger.info(f"Extracting archive: {archive_path} -> {extract_dir}")
            shutil.unpack_archive(archive_path, extract_dir)

            # Validate extraction if safe mode is enabled
            if self.safe_extract_var.get():
                if not extract_dir_obj.exists():
                    raise Exception("Extraction failed - destination folder was not created")

                if not os.listdir(extract_dir):
                    raise Exception("Extraction failed - destination folder is empty")

                # Check if any files were actually extracted
                extracted_files = []
                total_extracted_size = 0
                
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            extracted_files.append(file_path)
                            total_extracted_size += file_size
                        except OSError as e:
                            logger.warning(f"Cannot access extracted file size: {file_path} - {e}")

                if not extracted_files:
                    raise Exception("Extraction failed - no files found in extracted folder")

                # Verify total extracted size is reasonable
                if total_extracted_size < archive_size * MAX_ARCHIVE_SIZE_RATIO:
                    logger.warning(
                        f"Extracted size ({total_extracted_size}) seems small compared to archive size ({archive_size})"
                    )

                logger.info(f"Extraction validation passed: {len(extracted_files)} files, {total_extracted_size} bytes")

            # Only delete original if extraction was successful
            try:
                archive_path_obj.unlink()
                logger.info(f"Deleted original archive: {archive_path}")
            except OSError as e:
                logger.warning(f"Failed to delete original archive: {archive_path} - {e}")
                # Don't fail the operation if cleanup fails

            return (
                True,
                f"Successfully extracted and deleted '{os.path.basename(archive_path)}' ({len(extracted_files) if 'extracted_files' in locals() else 'unknown'} files)",
            )

        except Exception as e:
            # Clean up failed extraction directory
            if extract_dir_obj.exists():
                try:
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    logger.info(f"Cleaned up failed extraction directory: {extract_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup extraction directory: {extract_dir} - {cleanup_error}")

            return False, f"Failed to extract '{os.path.basename(archive_path)}': {e}"

    def create_backup(self) -> str | None:
        """Creates a backup of source folders before processing.
        
        Returns:
            Path to backup directory if successful [str], None if failed
            
        Raises:
            ValueError: If source_folders list is empty or invalid
            OSError: If file system operations fail during backup creation
            PermissionError: If insufficient permissions to create backup
            Exception: If backup process fails for other reasons
        """
        # Input validation
        if not self.source_folders:
            raise ValueError("No source folders to backup")
        if not isinstance(self.source_folders, list):
            raise ValueError(f"Source folders must be a list, got {type(self.source_folders)}")
        
        # Validate each source folder
        valid_source_folders = []
        for folder in self.source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
                continue
            if not os.path.exists(folder):
                logger.warning(f"Source folder no longer exists: {folder}")
                continue
            if not os.access(folder, os.R_OK):
                logger.warning(f"Cannot access source folder: {folder}")
                continue
            valid_source_folders.append(folder)
        
        if not valid_source_folders:
            raise ValueError("No valid source folders to backup")

        # Generate backup directory name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_base_name = f"backup_{timestamp}"
        
        # Create backup in parent directory of first source folder
        try:
            first_source_parent = Path(valid_source_folders[0]).parent
            backup_base = first_source_parent / backup_base_name
        except Exception as e:
            raise ValueError(f"Cannot determine backup location: {e}")

        self.update_status("Creating backup...")
        logger.info(f"Creating backup at: {backup_base}")

        try:
            # Create backup base directory
            backup_base.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created and is writable
            if not backup_base.exists():
                raise Exception("Failed to create backup base directory")
            if not os.access(backup_base, os.W_OK):
                raise PermissionError(f"Cannot write to backup directory: {backup_base}")

            total_folders = len(valid_source_folders)
            successful_backups = 0
            failed_backups = 0

            for i, folder in enumerate(valid_source_folders):
                if self.cancel_operation:
                    logger.info("Backup operation cancelled by user")
                    return None

                if not os.path.exists(folder):
                    logger.warning(f"Source folder no longer exists: {folder}")
                    continue

                # Create backup path
                try:
                    folder_name = Path(folder).name
                    backup_path = backup_base / folder_name
                    
                    # Ensure backup path is unique
                    if backup_path.exists():
                        backup_path = self._get_unique_path(str(backup_path))
                        backup_path = Path(backup_path)
                except Exception as e:
                    logger.error(f"Failed to create backup path for {folder}: {e}")
                    failed_backups += 1
                    continue

                try:
                    # Create backup
                    shutil.copytree(folder, backup_path)
                    successful_backups += 1
                    logger.info(f"Backed up folder: {folder} -> {backup_path}")
                    
                    # Verify backup was created successfully
                    if not backup_path.exists():
                        raise Exception("Backup directory was not created")
                    if not os.listdir(backup_path):
                        raise Exception("Backup directory is empty")
                    
                except Exception as e:
                    failed_backups += 1
                    logger.error(f"Failed to backup folder {folder}: {e}")
                    
                    # Clean up failed backup
                    if backup_path.exists():
                        try:
                            shutil.rmtree(backup_path, ignore_errors=True)
                            logger.info(f"Cleaned up failed backup: {backup_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup failed backup: {backup_path} - {cleanup_error}")
                    
                    # Continue with other folders
                    continue

                # Update progress
                progress = (i + 1) / total_folders * PROGRESS_BACKUP_PERCENT  # PROGRESS_BACKUP_PERCENT% for backup
                self.update_progress(
                    progress,
                    f"Backing up folder {i+1}/{total_folders}",
                )

            # Verify overall backup success
            if successful_backups == 0:
                logger.error("No folders were successfully backed up")
                # Clean up empty backup directory
                if backup_base.exists():
                    try:
                        shutil.rmtree(backup_base, ignore_errors=True)
                        logger.info(f"Cleaned up empty backup directory: {backup_base}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup empty backup directory: {backup_base} - {cleanup_error}")
                return None

            # Final verification
            if backup_base.exists() and os.listdir(backup_base):
                logger.info(f"Backup completed successfully: {backup_base}")
                logger.info(f"Backup summary: {successful_backups} successful, {failed_backups} failed")
                return str(backup_base)
            else:
                logger.error("Backup directory is empty or was not created")
                return None

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Cleanup failed backup
            if backup_base.exists():
                try:
                    shutil.rmtree(backup_base, ignore_errors=True)
                    logger.info(f"Cleaned up failed backup: {backup_base}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup failed backup: {backup_base} - {cleanup_error}")
            raise

    def generate_analysis_report(self) -> str | None:
        """Generates a comprehensive analysis report.
        
        Returns:
            Formatted analysis report [str] if successful, None if cancelled or failed
            
        Raises:
            ValueError: If source_folders list is empty or invalid
            OSError: If file system operations fail during analysis
            PermissionError: If insufficient permissions to access source folders
            Exception: If report generation fails for other reasons
        """
        # Input validation
        if not self.source_folders:
            raise ValueError("No source folders to analyze")
        if not isinstance(self.source_folders, list):
            raise ValueError(f"Source folders must be a list, got {type(self.source_folders)}")
        
        # Validate each source folder
        valid_source_folders = []
        for folder in self.source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
                continue
            if not os.path.exists(folder):
                logger.warning(f"Source folder no longer exists: {folder}")
                continue
            if not os.access(folder, os.R_OK):
                logger.warning(f"Cannot access source folder: {folder}")
                continue
            valid_source_folders.append(folder)
        
        if not valid_source_folders:
            raise ValueError("No valid source folders to analyze")

        report = ["=== FOLDER ANALYSIS REPORT ===", f"Generated: {datetime.now()}", ""]
        logger.info(f"Starting analysis of {len(valid_source_folders)} source folders")

        total_files = 0
        total_size = 0
        file_types = defaultdict(int)
        size_by_type = defaultdict(int)
        largest_files = []
        analysis_errors = []

        for folder in valid_source_folders:
            if self.cancel_operation:
                logger.info("Analysis cancelled by user")
                return None

            report.append(f"Analyzing: {folder}")
            folder_files = 0
            folder_size = 0
            folder_errors = 0

            try:
                for root, dirs, files in os.walk(folder):
                    if self.cancel_operation:
                        break

                    for file in files:
                        if self.cancel_operation:
                            break

                        file_path = os.path.join(root, file)
                        try:
                            # Validate file exists and is accessible
                            if not os.path.exists(file_path):
                                folder_errors += 1
                                continue
                            if not os.access(file_path, os.R_OK):
                                folder_errors += 1
                                continue

                            file_size = os.path.getsize(file_path)
                            file_ext = os.path.splitext(file)[1].lower() or "no_extension"

                            # Validate file size
                            if file_size < MIN_FILE_SIZE_BYTES:
                                logger.debug(f"File below minimum size: {file_path} ({file_size} bytes)")
                                continue
                            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                                logger.warning(f"File exceeds maximum size: {file_path} ({file_size / (1024*1024):.1f} MB)")

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

                        except (OSError, PermissionError) as e:
                            folder_errors += 1
                            logger.debug(f"Cannot access file {file_path}: {e}")
                            continue

                # Report folder analysis results
                if folder_errors > 0:
                    report.append(f"  Files: {folder_files}, Size: {folder_size/(1024*1024):.1f} MB, Errors: {folder_errors}")
                    analysis_errors.append(f"Folder {folder}: {folder_errors} access errors")
                else:
                    report.append(f"  Files: {folder_files}, Size: {folder_size/(1024*1024):.1f} MB")

            except (OSError, PermissionError) as e:
                error_msg = f"Error accessing folder {folder}: {e}"
                report.append(f"  ERROR: {error_msg}")
                analysis_errors.append(error_msg)
                logger.error(error_msg)
                continue

        # Add summary statistics
        report.extend([
            "",
            f"TOTAL FILES: {total_files}",
            f"TOTAL SIZE: {total_size/(1024*1024):.1f} MB",
            "",
            "FILE TYPES:",
        ])

        # Sort file types by count
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            size_mb = size_by_type[ext] / (1024 * 1024)
            report.append(f"  {ext}: {count} files, {size_mb:.1f} MB")

        # Add largest files section
        report.extend(["", "LARGEST FILES:"])
        for file_path, size in sorted(largest_files, key=lambda x: x[1], reverse=True):
            size_mb = size / (1024 * 1024)
            report.append(f"  {os.path.basename(file_path)}: {size_mb:.1f} MB")

        # Add error summary if any occurred
        if analysis_errors:
            report.extend(["", "ANALYSIS ERRORS:", *analysis_errors])

        # Add analysis metadata
        report.extend([
            "",
            "ANALYSIS METADATA:",
            f"  Source folders processed: {len(valid_source_folders)}",
            f"  Total folders analyzed: {len(valid_source_folders)}",
            f"  Analysis timestamp: {datetime.now()}",
            f"  File size limits: {MIN_FILE_SIZE_BYTES} bytes - {MAX_FILE_SIZE_MB} MB",
        ])

        logger.info(f"Analysis completed: {total_files} files, {total_size/(1024*1024):.1f} MB")
        if analysis_errors:
            logger.warning(f"Analysis completed with {len(analysis_errors)} errors")

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
            ValueError: If destination folder path is empty or invalid
            FileNotFoundError: If destination folder does not exist
            PermissionError: If insufficient permissions to read destination or write ZIP
            OSError: If file system operations fail during ZIP creation
            Exception: If ZIP creation fails for other reasons
        """
        # Input validation
        if not self.dest_folder:
            raise ValueError("Destination folder not set")
        if not isinstance(self.dest_folder, str):
            raise ValueError(f"Destination folder must be a string, got {type(self.dest_folder)}")
        
        dest_path_obj = Path(self.dest_folder)
        
        # Validate destination folder exists and is accessible
        if not dest_path_obj.exists():
            raise FileNotFoundError(f"Destination folder does not exist: {self.dest_folder}")
        if not dest_path_obj.is_dir():
            raise ValueError(f"Destination path is not a directory: {self.dest_folder}")
        if not os.access(self.dest_folder, os.R_OK):
            raise PermissionError(f"Cannot read destination folder: {self.dest_folder}")

        # Check if destination folder is empty
        try:
            folder_contents = list(dest_path_obj.iterdir())
            if not folder_contents:
                raise ValueError("Destination folder is empty - nothing to archive")
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Cannot access destination folder contents: {self.dest_folder} - {e}")

        # Generate ZIP filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"
        
        # Create ZIP in parent directory of destination
        try:
            zip_path = dest_path_obj.parent / zip_filename
        except Exception as e:
            raise ValueError(f"Cannot determine ZIP location: {e}")

        # Check if ZIP file already exists and generate unique name
        if zip_path.exists():
            zip_path = Path(self._get_unique_path(str(zip_path)))

        logger.info(f"Creating ZIP archive: {zip_path}")

        try:
            # Count total files and size for progress tracking
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(self.dest_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                            total_files += 1
                            total_size += os.path.getsize(file_path)
                    except (OSError, PermissionError):
                        continue
            
            if total_files == 0:
                raise ValueError("No accessible files found in destination folder")

            logger.info(f"ZIP will contain {total_files} files, {total_size/(1024*1024):.1f} MB")

            # Create ZIP archive
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                processed_files = 0
                processed_size = 0
                failed_files = 0

                for root, dirs, files in os.walk(self.dest_folder):
                    for file in files:
                        if self.cancel_operation:
                            raise Exception("ZIP creation cancelled by user")

                        file_path = os.path.join(root, file)
                        
                        # Validate file before adding to ZIP
                        try:
                            if not os.path.exists(file_path):
                                failed_files += 1
                                logger.warning(f"File no longer exists: {file_path}")
                                continue
                            if not os.access(file_path, os.R_OK):
                                failed_files += 1
                                logger.warning(f"Cannot read file: {file_path}")
                                continue
                            
                            # Calculate relative path for archive
                            arcname = os.path.relpath(file_path, self.dest_folder)
                            
                            # Add file to ZIP
                            zipf.write(file_path, arcname)
                            processed_files += 1
                            processed_size += os.path.getsize(file_path)
                            
                            # Update progress every N files
                            if processed_files % MAX_UI_UPDATE_FREQUENCY == 0:
                                progress = PROGRESS_START_ZIP + (processed_files / total_files) * PROGRESS_ZIP_PERCENT
                                self.update_progress(
                                    progress,
                                    f"Added {processed_files}/{total_files} files to ZIP"
                                )
                                
                        except Exception as e:
                            failed_files += 1
                            logger.warning(f"Failed to add file to ZIP: {file_path} - {e}")
                            continue

                # Verify ZIP was created successfully
                if not zip_path.exists():
                    raise Exception("ZIP file was not created")
                
                # Verify ZIP size is reasonable
                try:
                    zip_size = zip_path.stat().st_size
                    if zip_size == 0:
                        raise Exception("ZIP file is empty")
                    logger.info(f"ZIP archive created: {zip_path} ({processed_files} files, {processed_size/(1024*1024):.1f} MB, ZIP size: {zip_size/(1024*1024):.1f} MB)")
                except OSError as e:
                    logger.warning(f"Cannot verify ZIP file size: {e}")

                # Final summary
                if failed_files > 0:
                    logger.warning(f"ZIP creation completed with {failed_files} failed files")
                else:
                    logger.info("ZIP creation completed successfully")

        except Exception as e:
            # Cleanup failed ZIP file
            if zip_path.exists():
                try:
                    zip_path.unlink()
                    logger.info(f"Cleaned up failed ZIP file: {zip_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup failed ZIP file: {zip_path} - {cleanup_error}")
            
            logger.error(f"Failed to create ZIP archive: {e}")
            raise Exception(f"Failed to create ZIP archive: {e}")

        return str(zip_path)

    def show_text_dialog(self, title: str, content: str) -> None:
        """Shows a dialog with scrollable text content.
        
        Args:
            title: Dialog window title [str] - must not be empty
            content: Text content to display [str] - must not be empty
            
        Raises:
            ValueError: If title or content is empty or invalid
            tkinter.TclError: If Tkinter widget creation fails
            Exception: If dialog creation fails for other reasons
        """
        # Input validation
        if not title or not isinstance(title, str):
            raise ValueError(f"Title must be non-empty string, got {type(title)}")
        if not content or not isinstance(content, str):
            raise ValueError(f"Content must be non-empty string, got {type(content)}")
        
        # Validate title and content length
        if len(title.strip()) == 0:
            raise ValueError("Title cannot be empty or whitespace only")
        if len(content.strip()) == 0:
            raise ValueError("Content cannot be empty or whitespace only")
        
        # Validate title length for window title bar
        if len(title) > MAX_TITLE_LENGTH:
            logger.warning(f"Title is very long ({len(title)} chars), may be truncated: {title[:MAX_TITLE_PREVIEW_LENGTH]}...")
        
        # Validate content length for performance
        if len(content) > MAX_TEXT_CONTENT_SIZE:  # MAX_TEXT_CONTENT_SIZE limit for text content
            logger.warning(f"Content is very large ({len(content)} chars), may cause performance issues")
            # Truncate content for display
            content = content[:MAX_TEXT_CONTENT_SIZE] + "\n\n... [Content truncated due to size]"

        logger.info(f"Creating text dialog: '{title}' with {len(content)} characters")

        try:
            # Create dialog window
            dialog = tk.Toplevel(self.root)
            dialog.title(title)
            
            # Set dialog geometry with validation
            dialog_width = min(MAX_DIALOG_WIDTH, max(MIN_DIALOG_WIDTH, len(content) // CHARS_PER_DIALOG_LINE + DIALOG_WIDTH_OFFSET))
            dialog_height = min(MAX_DIALOG_HEIGHT, max(MIN_DIALOG_HEIGHT, len(content.split('\n')) * LINE_HEIGHT_PIXELS + DIALOG_HEIGHT_OFFSET))
            
            dialog.geometry(f"{dialog_width}x{dialog_height}")
            dialog.minsize(MIN_DIALOG_WIDTH, MIN_DIALOG_HEIGHT)

            # Center dialog on screen
            dialog.transient(self.root)
            dialog.grab_set()

            # Create text widget with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Create text widget with appropriate font and settings
            text_widget = tk.Text(
                text_frame, 
                wrap=tk.WORD, 
                font=("Consolas", 10),
                undo=False,  # Disable undo for performance
                maxundo=0,   # No undo history
                selectbackground="lightblue",
                selectforeground="black"
            )
            
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Insert content with error handling
            try:
                text_widget.insert("1.0", content)
                text_widget.config(state="disabled")  # Make read-only
                
                # Set cursor to beginning
                text_widget.mark_set("insert", "1.0")
                text_widget.see("1.0")
                
            except Exception as e:
                logger.error(f"Failed to insert content into text widget: {e}")
                # Fallback: show truncated content
                safe_content = content[:MAX_FALLBACK_CONTENT_SIZE] + "\n\n... [Content truncated due to error]"
                text_widget.insert("1.0", safe_content)
                text_widget.config(state="disabled")

            # Add close button
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            close_button = ttk.Button(
                button_frame,
                text="Close",
                command=dialog.destroy
            )
            close_button.pack(side="right")

            # Add copy button for convenience
            def copy_to_clipboard() -> None:
                """Copy dialog content to clipboard."""
                try:
                    dialog.clipboard_clear()
                    dialog.clipboard_append(content)
                    logger.debug("Dialog content copied to clipboard")
                except Exception as e:
                    logger.warning(f"Failed to copy to clipboard: {e}")

            copy_button = ttk.Button(
                button_frame,
                text="Copy All",
                command=copy_to_clipboard
            )
            copy_button.pack(side="right", padx=(0, 5))

            # Set focus and make dialog modal
            dialog.focus_set()
            close_button.focus_set()  # Focus on close button for better UX
            
            # Bind escape key to close dialog
            def on_escape(event: tk.Event) -> None:
                """Close dialog when escape key is pressed.
                
                Args:
                    event: The key event that triggered this function
                """
                dialog.destroy()
            
            dialog.bind("<Escape>", on_escape)
            
            # Log successful dialog creation
            logger.info(f"Text dialog created successfully: {dialog_width}x{dialog_height}")
            
            # Wait for dialog to close
            dialog.wait_window()

        except tk.TclError as e:
            logger.error(f"Tkinter error creating text dialog: {e}")
            # Fallback to simple message box
            fallback_content = content[:MAX_FALLBACK_CONTENT_SIZE] + "..." if len(content) > MAX_FALLBACK_CONTENT_SIZE else content
            messagebox.showinfo(title, fallback_content)
            raise
            
        except Exception as e:
            logger.error(f"Failed to show text dialog: {e}")
            # Fallback to simple message box
            fallback_content = content[:MAX_FALLBACK_CONTENT_SIZE] + "..." if len(content) > MAX_FALLBACK_CONTENT_SIZE else content
            messagebox.showinfo(title, fallback_content)
            raise

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
            ValueError: If source_path is empty or invalid
            FileNotFoundError: If source file does not exist
            PermissionError: If insufficient permissions to read source or write destination
        """
        # Input validation
        if not source_path or not isinstance(source_path, str):
            raise ValueError(f"Source path must be non-empty string, got {type(source_path)}")
        if not dest_path or not isinstance(dest_path, str):
            raise ValueError(f"Destination path must be non-empty string, got {type(dest_path)}")
        
        source_path_obj = Path(source_path)
        dest_path_obj = Path(dest_path)
        
        # Validate source file exists and is accessible
        if not source_path_obj.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        if not source_path_obj.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")
        if not os.access(source_path, os.R_OK):
            raise PermissionError(f"Cannot read source file: {source_path}")
        
        # Validate source file size
        try:
            source_size = source_path_obj.stat().st_size
            if source_size == 0:
                logger.warning(f"Source file is empty: {source_path}")
            elif source_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Source file exceeds maximum size limit: {source_path} ({source_size / (1024*1024):.1f} MB)")
        except OSError as e:
            logger.warning(f"Cannot access source file size: {source_path} - {e}")
        
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Ensure destination directory exists
                dest_dir = dest_path_obj.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if destination directory is writable
                if not os.access(dest_dir, os.W_OK):
                    raise PermissionError(f"Cannot write to destination directory: {dest_dir}")

                # Copy file with metadata preservation
                shutil.copy2(source_path, dest_path)

                # Verify copy was successful
                if dest_path_obj.exists():
                    try:
                        source_size = source_path_obj.stat().st_size
                        dest_size = dest_path_obj.stat().st_size
                        if source_size == dest_size:
                            logger.debug(f"Successfully copied {source_path} -> {dest_path} ({source_size} bytes)")
                            return True
                        else:
                            logger.warning(f"Size mismatch after copy: source={source_size}, dest={dest_size}")
                            # Size mismatch, remove failed copy and retry
                            if dest_path_obj.exists():
                                dest_path_obj.unlink()
                            if attempt < MAX_RETRY_ATTEMPTS - 1:
                                logger.info(f"Retrying copy due to size mismatch (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                                continue
                    except OSError as e:
                        logger.warning(f"Failed to verify copy sizes: {e}")
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            continue
                else:
                    logger.error(f"Destination file was not created: {dest_path}")
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        continue

            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Copy attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    # Wait before retry (exponential backoff)
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                else:
                    logger.error(f"Failed to copy {source_path} after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                    raise

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
                        progress = PROGRESS_START_MAIN + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
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
            ValueError: If path is empty or invalid
            OSError: If file system operations fail during path checking
            PermissionError: If insufficient permissions to check path existence
        """
        # Input validation
        if not path or not isinstance(path, str):
            raise ValueError(f"Path must be non-empty string, got {type(path)}")
        
        path_obj = Path(path)
        
        # Validate path format
        try:
            # Check if path is absolute or relative
            if path_obj.is_absolute():
                # Ensure drive exists on Windows
                if sys.platform == "win32" and len(path_obj.parts) > 0:
                    drive = path_obj.parts[0]
                    if not os.path.exists(drive):
                        raise ValueError(f"Drive does not exist: {drive}")
        except Exception as e:
            raise ValueError(f"Invalid path format: {path} - {e}")
        
        # Check if path already exists
        try:
            if not path_obj.exists():
                return path
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot check if path exists: {path} - {e}")
            # Assume it doesn't exist and return original path
            return path

        # Path exists, generate unique version
        parent = path_obj.parent
        name = path_obj.name
        
        # Determine if this is a file or directory
        try:
            is_file = path_obj.is_file()
        except (OSError, PermissionError):
            # If we can't determine, assume it's a file if it has an extension
            is_file = "." in name and not name.endswith(".")

        if is_file:
            filename = path_obj.stem
            ext = path_obj.suffix
        else:
            filename = name
            ext = ""

        # Generate unique path with counter
        counter = 1
        
        while counter <= MAX_COUNTER_ATTEMPTS:
            new_name = f"{filename} ({counter}){ext}"
            new_path = parent / new_name
            
            try:
                if not new_path.exists():
                    logger.debug(f"Generated unique path: {path} -> {new_path}")
                    return str(new_path)
            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot check if generated path exists: {new_path} - {e}")
                # If we can't check, assume it's safe to use
                return str(new_path)
            
            counter += 1
        
        # If we've exhausted all reasonable attempts, append timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_name = f"{filename}_{timestamp}{ext}"
        fallback_path = parent / fallback_name
        
        logger.warning(f"Exhausted counter attempts, using timestamp fallback: {fallback_path}")
        return str(fallback_path)

    def select_source_folders(self) -> None:
        """Open folder selection dialog to add source folders.
        
        This method allows users to select folders that will be processed by the application.
        Selected folders are added to the source_folders list and displayed in the UI.
        
        Args:
            None - uses filedialog.askdirectory() for user input
            
        Returns:
            None - updates self.source_folders and UI state
            
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
        
        This method removes user-selected folders from the source_folders list.
        It prompts for confirmation before removal and updates both the internal
        list and the UI display.
        
        Args:
            None - uses self.source_listbox.curselection() for user input
            
        Returns:
            None - updates self.source_folders and UI state
            
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
        
        This method allows users to select the destination folder where processed
        files will be placed. The selected folder is validated for write access
        and stored in self.dest_folder.
        
        Args:
            None - uses filedialog.askdirectory() for user input
            
        Returns:
            None - updates self.dest_folder and UI state
            
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
                        progress = PROGRESS_START_MAIN + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
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
                        progress = PROGRESS_START_MAIN + (processed_files / total_files) * PROGRESS_MAIN_OP_PERCENT
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
