# Standard library imports
import logging

# Third-party imports
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Final

# Constants for configuration with sources and units
MAX_LOG_ENTRIES: Final[int] = (
    20  # Maximum number of log entries to display per operation
    # - UI performance limit
)

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

try:
    from utils.file_utils import safe_write_text
except ImportError:
    # Fallback definition if utils not found
    def safe_write_text(path, content, encoding="utf-8", create_parents=True):
        p = Path(path)
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)


PROGRESS_INCREMENT: Final[int] = (
    10  # Progress bar increment percentage [%]
    # - standard UI update frequency
)
MAX_FILE_SIZE_MB: Final[int] = (
    1024  # Maximum file size limit [MB]
    # - Windows FAT32 limit per Microsoft docs
)
MIN_FILE_SIZE_BYTES: Final[int] = (
    1  # Minimum file size [bytes]
    # - 1 byte minimum per filesystem standards
)
DEFAULT_CHUNK_SIZE: Final[int] = (
    8192  # File copy chunk size [bytes]
    # - optimal for most systems per Python shutil docs
)
MAX_RETRY_ATTEMPTS: Final[int] = (
    3  # Maximum retry attempts for file operations
    # - industry standard retry limit
)
ICON_SIZES: Final[tuple[int, ...]] = (
    16,
    32,
    48,
    64,
)  # Standard icon sizes [pixels] per Windows Shell API guidelines
MAX_STATUS_LENGTH: Final[int] = (
    200  # Maximum status message length [characters]
    # - prevents UI overflow
)
MAX_UI_UPDATE_FREQUENCY: Final[int] = (
    10  # Update progress every N files
    # - balances responsiveness with performance
)
MAX_ARCHIVE_SIZE_RATIO: Final[float] = (
    0.1  # Minimum extracted size ratio [ratio]
    # - archive size * 0.1 for validation
)
MAX_DIALOG_WIDTH: Final[int] = (
    800  # Maximum dialog width [pixels] - prevents dialog overflow
)
MAX_DIALOG_HEIGHT: Final[int] = (
    600  # Maximum dialog height [pixels] - prevents dialog overflow
)
MIN_DIALOG_WIDTH: Final[int] = 400  # Minimum dialog width [pixels] - ensures usability
MIN_DIALOG_HEIGHT: Final[int] = (
    300  # Minimum dialog height [pixels] - ensures usability
)

# Additional constants for improved maintainability
MAX_TEXT_CONTENT_SIZE: Final[int] = (
    1000000  # Maximum text content size [characters]
    # - prevents performance issues
)
MAX_TITLE_LENGTH: Final[int] = (
    100  # Maximum title length [characters]
    # - prevents window title truncation
)
MAX_COUNTER_ATTEMPTS: Final[int] = (
    1000  # Maximum attempts to generate unique filename [attempts]
    # - prevents infinite loops
)
MAX_FALLBACK_CONTENT_SIZE: Final[int] = (
    500  # Maximum content size for fallback display [characters]
    # - prevents UI overflow
)
PROGRESS_BACKUP_PERCENT: Final[int] = (
    20  # Progress percentage allocated to backup operations [%] - UI progress tracking
)
PROGRESS_MAIN_OP_PERCENT: Final[int] = (
    40  # Progress percentage allocated to main operations [%] - UI progress tracking
)
PROGRESS_ZIP_PERCENT: Final[int] = (
    10  # Progress percentage allocated to ZIP creation [%] - UI progress tracking
)
PROGRESS_START_MAIN: Final[int] = (
    30  # Starting progress percentage for main operations [%] - UI progress tracking
)
PROGRESS_START_ZIP: Final[int] = (
    85  # Starting progress percentage for ZIP creation [%] - UI progress tracking
)

# Dialog layout constants
CHARS_PER_DIALOG_LINE: Final[int] = (
    80  # Characters per line for dialog width calculation [characters]
    # - standard text width
)
DIALOG_WIDTH_OFFSET: Final[int] = (
    100  # Additional width offset for dialog borders [pixels]
    # - accounts for scrollbars and margins
)
DIALOG_HEIGHT_OFFSET: Final[int] = (
    100  # Additional height offset for dialog borders [pixels]
    # - accounts for title bar and margins
)
LINE_HEIGHT_PIXELS: Final[int] = (
    20  # Height per line for dialog height calculation [pixels] - standard line height
)
MAX_TITLE_PREVIEW_LENGTH: Final[int] = (
    50  # Maximum title length for preview in logs [characters]
    # - prevents log overflow
)

# Set up logging to capture detailed information
log_filename = "folder_processor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w")],
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Import mixins
from folder_tool_file_ops import FileOperationsMixin  # noqa: E402
from folder_tool_processing import ProcessingMixin  # noqa: E402
from folder_tool_ui import UICreationMixin  # noqa: E402


class FolderProcessorApp(UICreationMixin, FileOperationsMixin, ProcessingMixin):
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
        self.source_folders: list[str] = []
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
        self.cancel_operation: bool = False

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
            raise ValueError(
                f"MAX_FILE_SIZE_MB must be positive, got {MAX_FILE_SIZE_MB}",
            )
        if MIN_FILE_SIZE_BYTES < 0:
            raise ValueError(
                f"MIN_FILE_SIZE_BYTES must be non-negative, got {MIN_FILE_SIZE_BYTES}",
            )
        if MIN_FILE_SIZE_BYTES >= MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"MIN_FILE_SIZE_BYTES must be less than MAX_FILE_SIZE_MB, "
                f"got {MIN_FILE_SIZE_BYTES}",
            )

        # Validate UI constants
        if MAX_STATUS_LENGTH <= 0:
            raise ValueError(
                f"MAX_STATUS_LENGTH must be positive, got {MAX_STATUS_LENGTH}",
            )
        if MAX_UI_UPDATE_FREQUENCY <= 0:
            raise ValueError(
                f"MAX_UI_UPDATE_FREQUENCY must be positive, "
                f"got {MAX_UI_UPDATE_FREQUENCY}",
            )
        if MAX_DIALOG_WIDTH <= MIN_DIALOG_WIDTH:
            raise ValueError(
                f"MAX_DIALOG_WIDTH must be greater than MIN_DIALOG_WIDTH, "
                f"got {MAX_DIALOG_WIDTH} <= {MIN_DIALOG_WIDTH}",
            )
        if MAX_DIALOG_HEIGHT <= MIN_DIALOG_HEIGHT:
            raise ValueError(
                "MAX_DIALOG_HEIGHT must be greater than MIN_DIALOG_HEIGHT, "
                f"got {MAX_DIALOG_HEIGHT} <= {MIN_DIALOG_HEIGHT}",
            )

        # Validate archive constants
        if not 0 < MAX_ARCHIVE_SIZE_RATIO < 1:
            raise ValueError(
                f"MAX_ARCHIVE_SIZE_RATIO must be between 0 and 1, "
                f"got {MAX_ARCHIVE_SIZE_RATIO}",
            )

        # Validate retry constants
        if MAX_RETRY_ATTEMPTS <= 0:
            raise ValueError(
                f"MAX_RETRY_ATTEMPTS must be positive, got {MAX_RETRY_ATTEMPTS}",
            )

        # Validate new constants
        if MAX_TEXT_CONTENT_SIZE <= 0:
            raise ValueError(
                f"MAX_TEXT_CONTENT_SIZE must be positive, got {MAX_TEXT_CONTENT_SIZE}",
            )
        if MAX_TITLE_LENGTH <= 0:
            raise ValueError(
                f"MAX_TITLE_LENGTH must be positive, got {MAX_TITLE_LENGTH}",
            )
        if MAX_COUNTER_ATTEMPTS <= 0:
            raise ValueError(
                f"MAX_COUNTER_ATTEMPTS must be positive, got {MAX_COUNTER_ATTEMPTS}",
            )

        # Validate progress constants
        if PROGRESS_BACKUP_PERCENT < 0 or PROGRESS_BACKUP_PERCENT > 100:
            raise ValueError(
                f"PROGRESS_BACKUP_PERCENT must be between 0 and 100, "
                f"got {PROGRESS_BACKUP_PERCENT}",
            )
        if PROGRESS_MAIN_OP_PERCENT < 0 or PROGRESS_MAIN_OP_PERCENT > 100:
            raise ValueError(
                f"PROGRESS_MAIN_OP_PERCENT must be between 0 and 100, "
                f"got {PROGRESS_MAIN_OP_PERCENT}",
            )
        if PROGRESS_ZIP_PERCENT < 0 or PROGRESS_ZIP_PERCENT > 100:
            raise ValueError(
                f"PROGRESS_ZIP_PERCENT must be between 0 and 100, "
                f"got {PROGRESS_ZIP_PERCENT}",
            )
        if PROGRESS_START_MAIN < 0 or PROGRESS_START_MAIN > 100:
            raise ValueError(
                f"PROGRESS_START_MAIN must be between 0 and 100, "
                f"got {PROGRESS_START_MAIN}",
            )
        if PROGRESS_START_ZIP < 0 or PROGRESS_START_ZIP > 100:
            raise ValueError(
                f"PROGRESS_START_ZIP must be between 0 and 100, "
                f"got {PROGRESS_START_ZIP}",
            )

        # Validate progress flow consistency
        total_progress = (
            PROGRESS_BACKUP_PERCENT + PROGRESS_MAIN_OP_PERCENT + PROGRESS_ZIP_PERCENT
        )
        if total_progress > 100:
            raise ValueError(
                f"Total progress allocation exceeds 100%: {total_progress}",
            )

        logger.info("All constants validated successfully")

    def get_constants_info(self) -> dict[str, dict[str, str]]:
        """Returns information about all constants for debugging and documentation.

        This method provides comprehensive metadata about all constants including
        their values, units, and sources. This is useful for debugging, documentation,
        and system validation.

        Returns:
            Dictionary mapping constant names to their metadata [dict]
                - includes value, units, and source

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
                "source": "UI performance limit",
            },
            "PROGRESS_INCREMENT": {
                "value": str(PROGRESS_INCREMENT),
                "units": "%",
                "source": "Standard UI update frequency",
            },
            "MAX_FILE_SIZE_MB": {
                "value": str(MAX_FILE_SIZE_MB),
                "units": "MB",
                "source": "Windows FAT32 limit per Microsoft docs",
            },
            "MIN_FILE_SIZE_BYTES": {
                "value": str(MIN_FILE_SIZE_BYTES),
                "units": "bytes",
                "source": "1 byte minimum per filesystem standards",
            },
            "DEFAULT_CHUNK_SIZE": {
                "value": str(DEFAULT_CHUNK_SIZE),
                "units": "bytes",
                "source": "Optimal for most systems per Python shutil docs",
            },
            "MAX_RETRY_ATTEMPTS": {
                "value": str(MAX_RETRY_ATTEMPTS),
                "units": "attempts",
                "source": "Industry standard retry limit",
            },
            "ICON_SIZES": {
                "value": str(ICON_SIZES),
                "units": "pixels",
                "source": "Windows Shell API guidelines",
            },
            "MAX_STATUS_LENGTH": {
                "value": str(MAX_STATUS_LENGTH),
                "units": "characters",
                "source": "Prevents UI overflow",
            },
            "MAX_UI_UPDATE_FREQUENCY": {
                "value": str(MAX_UI_UPDATE_FREQUENCY),
                "units": "files",
                "source": "Balances responsiveness with performance",
            },
            "MAX_ARCHIVE_SIZE_RATIO": {
                "value": str(MAX_ARCHIVE_SIZE_RATIO),
                "units": "ratio",
                "source": "Archive size * 0.1 for validation",
            },
            "MAX_DIALOG_WIDTH": {
                "value": str(MAX_DIALOG_WIDTH),
                "units": "pixels",
                "source": "Prevents dialog overflow",
            },
            "MAX_DIALOG_HEIGHT": {
                "value": str(MAX_DIALOG_HEIGHT),
                "units": "pixels",
                "source": "Prevents dialog overflow",
            },
            "MIN_DIALOG_WIDTH": {
                "value": str(MIN_DIALOG_WIDTH),
                "units": "pixels",
                "source": "Ensures usability",
            },
            "MIN_DIALOG_HEIGHT": {
                "value": str(MIN_DIALOG_HEIGHT),
                "units": "pixels",
                "source": "Ensures usability",
            },
            "MAX_TEXT_CONTENT_SIZE": {
                "value": str(MAX_TEXT_CONTENT_SIZE),
                "units": "characters",
                "source": "Prevents performance issues in text dialogs",
            },
            "MAX_TITLE_LENGTH": {
                "value": str(MAX_TITLE_LENGTH),
                "units": "characters",
                "source": "Prevents window title truncation",
            },
            "MAX_COUNTER_ATTEMPTS": {
                "value": str(MAX_COUNTER_ATTEMPTS),
                "units": "attempts",
                "source": "Prevents infinite loops in filename generation",
            },
            "MAX_FALLBACK_CONTENT_SIZE": {
                "value": str(MAX_FALLBACK_CONTENT_SIZE),
                "units": "characters",
                "source": "Prevents UI overflow in fallback dialogs",
            },
            "PROGRESS_BACKUP_PERCENT": {
                "value": str(PROGRESS_BACKUP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for backup operations",
            },
            "PROGRESS_MAIN_OP_PERCENT": {
                "value": str(PROGRESS_MAIN_OP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for main operations",
            },
            "PROGRESS_ZIP_PERCENT": {
                "value": str(PROGRESS_ZIP_PERCENT),
                "units": "%",
                "source": "UI progress tracking for ZIP creation",
            },
            "PROGRESS_START_MAIN": {
                "value": str(PROGRESS_START_MAIN),
                "units": "%",
                "source": "Starting progress for main operations",
            },
            "PROGRESS_START_ZIP": {
                "value": str(PROGRESS_START_ZIP),
                "units": "%",
                "source": "Starting progress for ZIP creation",
            },
        }

    def export_constants_documentation(self, output_path: str) -> bool:
        """Exports constants documentation to a file for reference.

        Args:
            output_path: Path to the output file [str]
                - will be created if it doesn't exist

        Returns:
            True if export successful, False otherwise

        Raises:
            OSError: If file operations fail
            Exception: If export fails for other reasons
        """
        try:
            constants_info = self.get_constants_info()

            content = [
                "# Folder Tool Constants Documentation",
                f"Generated: {datetime.now()}",
                "",
                "## Constants Overview",
                "",
            ]

            for const_name, info in constants_info.items():
                content.append(f"### {const_name}")
                content.append(f"- **Value**: {info['value']}")
                content.append(f"- **Units**: {info['units']}")
                content.append(f"- **Source**: {info['source']}")
                content.append("")

            safe_write_text(output_path, "\n".join(content))

            logger.info(f"Constants documentation exported to: {output_path}")
            return True

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Failed to export constants documentation: {e}")
            return False


if __name__ == "__main__":
    root = tk.Tk()
    app = FolderProcessorApp(root)
    root.mainloop()
