# Standard library imports
import logging
import tkinter as tk

from tkinter import ttk

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

# Import constants from dedicated module (re-export for backward compat)
from folder_tool_constants import (  # noqa: E402, F401
    CHARS_PER_DIALOG_LINE,
    DEFAULT_CHUNK_SIZE,
    DIALOG_HEIGHT_OFFSET,
    DIALOG_WIDTH_OFFSET,
    ICON_SIZES,
    LINE_HEIGHT_PIXELS,
    MAX_ARCHIVE_SIZE_RATIO,
    MAX_COUNTER_ATTEMPTS,
    MAX_DIALOG_HEIGHT,
    MAX_DIALOG_WIDTH,
    MAX_FALLBACK_CONTENT_SIZE,
    MAX_FILE_SIZE_MB,
    MAX_LOG_ENTRIES,
    MAX_RETRY_ATTEMPTS,
    MAX_STATUS_LENGTH,
    MAX_TEXT_CONTENT_SIZE,
    MAX_TITLE_LENGTH,
    MAX_TITLE_PREVIEW_LENGTH,
    MAX_UI_UPDATE_FREQUENCY,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    MIN_FILE_SIZE_BYTES,
    PROGRESS_BACKUP_PERCENT,
    PROGRESS_INCREMENT,
    PROGRESS_MAIN_OP_PERCENT,
    PROGRESS_START_MAIN,
    PROGRESS_START_ZIP,
    PROGRESS_ZIP_PERCENT,
    export_constants_documentation,
    get_constants_info,
    validate_constants,
)

try:
    from utils.file_utils import safe_write_text  # noqa: F401, E402
except ImportError:
    # Fallback definition if utils not found
    from pathlib import Path

    def safe_write_text(
        file_path: Path | str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> bool:
        assert file_path is not None, "file_path must be provided"
        p = Path(file_path)
        if create_parents:
            parent_dir = p.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return True


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
    """An enhanced GUI application for comprehensive folder processing tasks."""

    def __init__(self, root_window: tk.Tk) -> None:
        """Initialize the application's user interface.

        Args:
            root_window: The root Tkinter window
        """
        assert root_window is not None, "root_window must be provided"
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
        validate_constants()

        # --- Main Frame with Scrollable Content ---
        self.create_scrollable_interface()

    # ------------------------------------------------------------------
    # Backward-compatible delegate methods
    # ------------------------------------------------------------------

    def _validate_constants(self) -> None:
        """Validate constants — delegates to module-level function."""
        validate_constants()

    def get_constants_info(self) -> dict[str, dict[str, str]]:
        """Return constants metadata — delegates to module-level function."""
        return get_constants_info()  # type: ignore[no-any-return]

    def export_constants_documentation(self, output_path: str) -> bool:
        """Export constants docs — delegates to module-level function."""
        return export_constants_documentation(output_path)  # type: ignore[no-any-return]


if __name__ == "__main__":
    root = tk.Tk()
    app = FolderProcessorApp(root)
    root.mainloop()
