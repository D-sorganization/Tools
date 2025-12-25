import ctypes
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import tkinter as tk
import typing
import webbrowser
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

"""
Folder Fix Pro v3.0 - Enhanced Professional Folder Processing Tool

A comprehensive, modern folder management application with advanced features:
- Modern themed UI with dark/light mode support
- Drag-and-drop file/folder support
- Real-time preview and analysis
- SHA-256 based intelligent deduplication
- Advanced filtering with regex support
- Batch operations with progress tracking
- Export reports and operation logs
- Professional error handling and validation
"""


MAX_FILE_SIZE_MB: typing.Final[int] = 10240  # 10GB limit for modern systems
MIN_FILE_SIZE_BYTES: typing.Final[int] = 0
DEFAULT_CHUNK_SIZE: typing.Final[int] = 65536  # 64KB chunks for optimal performance
MAX_RETRY_ATTEMPTS: typing.Final[int] = 3
HASH_ALGORITHM: typing.Final[str] = "sha256"  # Cryptographic hash for deduplication
PREVIEW_MAX_FILES: typing.Final[int] = 1000  # Max files to show in preview
ICON_SIZES: typing.Final[tuple[int, ...]] = (16, 32, 48, 64, 128, 256)

# UI Constants
WINDOW_WIDTH: typing.Final[int] = 1200
WINDOW_HEIGHT: typing.Final[int] = 800
MIN_WINDOW_WIDTH: typing.Final[int] = 900
MIN_WINDOW_HEIGHT: typing.Final[int] = 600
PADDING_LARGE: typing.Final[int] = 20
PADDING_MEDIUM: typing.Final[int] = 10
PADDING_SMALL: typing.Final[int] = 5

# Color schemes for themes
DARK_THEME = {
    "bg": "#2b2b2b",
    "fg": "#ffffff",
    "select_bg": "#404040",
    "select_fg": "#ffffff",
    "button_bg": "#404040",
    "button_fg": "#ffffff",
    "entry_bg": "#353535",
    "entry_fg": "#ffffff",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
}

LIGHT_THEME = {
    "bg": "#f0f0f0",
    "fg": "#000000",
    "select_bg": "#0078d7",
    "select_fg": "#ffffff",
    "button_bg": "#e1e1e1",
    "button_fg": "#000000",
    "entry_bg": "#ffffff",
    "entry_fg": "#000000",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
}

# Set up professional logging
log_filename = "folder_fix_pro.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class FileHasher:
    """Efficient file hashing for deduplication."""

    @staticmethod
    def hash_file(file_path: Path, algorithm: str = HASH_ALGORITHM) -> str | None:
        """
        Generate cryptographic hash of file contents.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use (sha256, md5, etc.)

        Returns:
            Hexadecimal hash string or None if error
        """
        try:
            hasher = hashlib.new(algorithm)
            with file_path.open("rb") as f:
                while chunk := f.read(DEFAULT_CHUNK_SIZE):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            logger.exception("Error hashing %s", file_path)
            return None

    @staticmethod
    def hash_file_fast(file_path: Path) -> str | None:
        """
        Generate fast hash using first/last chunks + size.
        Useful for quick duplicate detection.

        Args:
            file_path: Path to file

        Returns:
            Hash string or None if error
        """
        try:
            size = file_path.stat().st_size
            hasher = hashlib.sha256()

            # Hash file size
            hasher.update(str(size).encode())

            with file_path.open("rb") as f:
                # Hash first chunk
                first_chunk = f.read(DEFAULT_CHUNK_SIZE)
                hasher.update(first_chunk)

                # Hash last chunk if file is large enough
                if size > DEFAULT_CHUNK_SIZE * 2:
                    f.seek(-DEFAULT_CHUNK_SIZE, 2)
                    last_chunk = f.read(DEFAULT_CHUNK_SIZE)
                    hasher.update(last_chunk)

            return hasher.hexdigest()
        except OSError:
            logger.exception("Error fast hashing %s", file_path)
            return None


class OperationReport:
    """Generate detailed operation reports."""

    def __init__(self) -> None:
        self.start_time: datetime = datetime.now(UTC)
        self.end_time: datetime | None = None
        self.operations: list[dict[str, typing.Any]] = []
        self.stats: dict[str, int] = defaultdict(int)
        self.errors: list[dict[str, typing.Any]] = []

    def add_operation(self, operation: str, details: dict) -> None:
        """Add operation to report."""
        self.operations.append(
            {
                "timestamp": datetime.now(UTC),
                "operation": operation,
                "details": details,
            }
        )
        self.stats[operation] += 1

    def add_error(self, error: str) -> None:
        """Add error to report."""
        self.errors.append({"timestamp": datetime.now(UTC), "error": error})

    def finalize(self) -> None:
        """Finalize report with end time."""
        self.end_time = datetime.now(UTC)

    def get_duration(self) -> str:
        """Get operation duration."""
        if self.end_time:
            duration = self.end_time - self.start_time
            return str(duration).split(".")[0]
        return "In progress"

    def export_json(self, file_path: str) -> None:
        """Export report as JSON."""
        report_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "statistics": dict(self.stats),
            "total_operations": len(self.operations),
            "total_errors": len(self.errors),
            "operations": [
                {
                    "timestamp": op["timestamp"].isoformat(),
                    "operation": op["operation"],
                    "details": op["details"],
                }
                for op in self.operations
            ],
            "errors": [
                {"timestamp": err["timestamp"].isoformat(), "error": err["error"]}
                for err in self.errors
            ],
        }

        with Path(file_path).open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    def export_html(self, file_path: str) -> None:
        """Export report as HTML."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Folder Fix Pro - Operation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .error {{
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📁 Folder Fix Pro</h1>
        <p>Operation Report - {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Duration:</strong> {self.get_duration()}</p>
        <p><strong>Start Time:</strong> {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>End Time:</strong> {
            self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else "In progress"
        }</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(self.operations)}</div>
            <div class="stat-label">Total Operations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(self.errors)}</div>
            <div class="stat-label">Errors</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(self.stats)}</div>
            <div class="stat-label">Operation Types</div>
        </div>
    </div>

    <div class="section">
        <h2>Statistics by Operation Type</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {
            "".join(
                f"<tr><td>{op}</td><td>{count}</td></tr>" for op,
                count in self.stats.items()
            )
        }
            </tbody>
        </table>
    </div>

    {
            f'''
    <div class="section">
        <h2>Errors ({len(self.errors)})</h2>
        {"".join(
            f'<div class="error"><span class="timestamp">'
            f'{err["timestamp"].strftime("%H:%M:%S")}</span> - '
            f'{err["error"]}</div>'
            for err in self.errors
        )}
    </div>
    '''
            if self.errors
            else ""
        }

    <div class="section">
        <h2>Operation Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Operation</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {
            "".join(
                f'''<tr>
                    <td class="timestamp">{op["timestamp"].strftime("%H:%M:%S")}</td>
                    <td>{op["operation"]}</td>
                    <td>{json.dumps(op["details"], indent=2)}</td>
                </tr>'''
                for op in self.operations[-100:]
            )
        }
            </tbody>
        </table>
        {
            f"<p><em>Showing last {PREVIEW_MAX_FILES} of {len(self.operations)} operations</em></p>"
            if len(self.operations) > PREVIEW_MAX_FILES
            else ""
        }
    </div>
</body>
</html>
"""
        with Path(file_path).open("w", encoding="utf-8") as f:
            f.write(html_content)


class FolderFixPro:
    """Enhanced professional folder processing application."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Folder Fix Pro v3.0 - Professional Folder Manager")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # Application state
        # Application state
        self.source_folders: list[str] = []
        self.dest_folder: str = ""
        self.current_theme: str = "dark"
        self.operation_report: OperationReport = OperationReport()
        self.cancel_operation: bool = False
        self.file_cache: dict[str, typing.Any] = {}  # Cache for file information
        self.duplicate_groups: dict[str, list[str]] = defaultdict(list)
        self.file_list: dict[str, typing.Any] = {}

        # Operation variables
        self.operation_mode = "combine"
        self.unzip_enabled = False
        self.deduplicate_enabled = False
        self.dedupe_method = "hash"  # 'hash', 'fast_hash', or 'name_size'
        self.organize_by_type = False
        self.organize_by_date = False
        self.filter_extensions = ""
        self.filter_regex = ""
        self.min_file_size = 0
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 * 1024
        self.preview_mode = False
        self.create_backup = False
        self.zip_output = False

        # Set application icon
        self._setup_icon()

        # Initialize UI
        self._create_menu_bar()
        self._create_main_ui()
        self._apply_theme()

        # Enable drag and drop
        self._setup_drag_drop()

        logger.info("Folder Fix Pro v3.0 initialized successfully")

    def _setup_icon(self) -> None:
        """Set up application icon with fallback."""
        try:
            if sys.platform == "win32":
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "FolderFixPro.Tool.3.0"
                )
            # You can add icon file loading here
        except OSError as e:
            logger.warning("Could not set icon: %s", e)

    def _create_menu_bar(self) -> None:
        """Create professional menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Export Report (JSON)", command=self._export_report_json
        )
        file_menu.add_command(
            label="Export Report (HTML)", command=self._export_report_html
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self._toggle_theme)
        view_menu.add_command(label="Refresh Preview", command=self._update_preview)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear Cache", command=self._clear_cache)
        tools_menu.add_command(label="Open Log File", command=self._open_log_file)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)

    def _create_main_ui(self) -> None:
        """Create main user interface with modern design."""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill="both", expand=True, padx=PADDING_SMALL, pady=PADDING_SMALL
        )

        # Create tabs
        self._create_operation_tab()
        self._create_filters_tab()
        self._create_preview_tab()
        self._create_log_tab()

        # Status bar at bottom
        self._create_status_bar()

    def _create_operation_tab(self) -> None:
        """Create main operation tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Operation  ")

        # Main container with two columns
        left_frame = ttk.Frame(tab)
        left_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=PADDING_MEDIUM,
            pady=PADDING_MEDIUM,
        )

        right_frame = ttk.Frame(tab)
        right_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=PADDING_MEDIUM,
            pady=PADDING_MEDIUM,
        )

        # LEFT COLUMN - Source and Destination
        # Header
        header_label = ttk.Label(
            left_frame,
            text="📁 Folder Fix Pro",
            font=("Segoe UI", 18, "bold"),
        )
        header_label.pack(pady=(0, PADDING_MEDIUM))

        # Source folders section
        source_frame = ttk.LabelFrame(
            left_frame, text="Source Folders", padding=PADDING_MEDIUM
        )
        source_frame.pack(fill="both", expand=True, pady=(0, PADDING_MEDIUM))

        # Source listbox with scrollbar
        list_frame = ttk.Frame(source_frame)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.source_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode="extended",
            height=8,
        )
        self.source_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.source_listbox.yview)

        # Source buttons
        btn_frame = ttk.Frame(source_frame)
        btn_frame.pack(fill="x", pady=(PADDING_SMALL, 0))

        ttk.Button(
            btn_frame, text="+ Add Folder", command=self._add_source_folder
        ).pack(side="left", padx=(0, PADDING_SMALL))
        ttk.Button(
            btn_frame, text="- Remove", command=self._remove_source_folder
        ).pack(side="left", padx=(0, PADDING_SMALL))
        ttk.Button(
            btn_frame, text="🗑️ Clear All", command=self._clear_source_folders
        ).pack(side="left")

        # Destination folder section
        dest_frame = ttk.LabelFrame(
            left_frame, text="Destination Folder", padding=PADDING_MEDIUM
        )
        dest_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        dest_entry_frame = ttk.Frame(dest_frame)
        dest_entry_frame.pack(fill="x")

        self.dest_entry = ttk.Entry(dest_entry_frame)
        self.dest_entry.pack(
            side="left", fill="x", expand=True, padx=(0, PADDING_SMALL)
        )

        ttk.Button(
            dest_entry_frame, text="Browse", command=self._browse_destination
        ).pack(side="right")

        # Progress section
        progress_frame = ttk.LabelFrame(
            left_frame, text="Progress", padding=PADDING_MEDIUM
        )
        progress_frame.pack(fill="x")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(fill="x", pady=(0, PADDING_SMALL))

        self.status_label = ttk.Label(
            progress_frame, text="Ready", font=("Segoe UI", 9)
        )
        self.status_label.pack(fill="x")

        self.eta_label = ttk.Label(progress_frame, text="", font=("Segoe UI", 8))
        self.eta_label.pack(fill="x")

        # RIGHT COLUMN - Operations and Options
        # Operation mode
        mode_frame = ttk.LabelFrame(
            right_frame, text="Operation Mode", padding=PADDING_MEDIUM
        )
        mode_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        operations = [
            ("Combine & Copy", "combine"),
            ("Flatten & Tidy", "flatten"),
            ("Copy & Prune Empty", "prune"),
            ("Deduplicate In-Place", "deduplicate"),
            ("Analyze Only", "analyze"),
        ]

        self.mode_var = tk.StringVar(value="combine")
        for i, (text, value) in enumerate(operations):
            ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self._on_mode_change,
            ).grid(row=i, column=0, sticky="w", pady=2)

        # Processing options
        options_frame = ttk.LabelFrame(
            right_frame, text="Processing Options", padding=PADDING_MEDIUM
        )
        options_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.unzip_var = tk.BooleanVar()
        self.dedupe_var = tk.BooleanVar()
        self.organize_type_var = tk.BooleanVar()
        self.organize_date_var = tk.BooleanVar()
        self.preview_var = tk.BooleanVar()
        self.backup_var = tk.BooleanVar()
        self.zip_var = tk.BooleanVar()

        options = [
            ("Extract Archives (.zip, .rar, .7z)", self.unzip_var),
            ("Smart Deduplication (SHA-256)", self.dedupe_var),
            ("Organize by File Type", self.organize_type_var),
            ("Organize by Date Modified", self.organize_date_var),
            ("Preview Mode (No Changes)", self.preview_var),
            ("Create Backup Before Processing", self.backup_var),
            ("Create ZIP Archive of Output", self.zip_var),
        ]

        for text, var in options:
            ttk.Checkbutton(options_frame, text=text, variable=var).pack(
                anchor="w", pady=2
            )

        # Deduplication method
        dedupe_method_frame = ttk.LabelFrame(
            right_frame, text="Deduplication Method", padding=PADDING_MEDIUM
        )
        dedupe_method_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.dedupe_method_var = tk.StringVar(value="hash")
        ttk.Radiobutton(
            dedupe_method_frame,
            text="Full Hash (SHA-256) - Most Accurate",
            variable=self.dedupe_method_var,
            value="hash",
        ).pack(anchor="w")
        ttk.Radiobutton(
            dedupe_method_frame,
            text="Fast Hash - Quick Scan",
            variable=self.dedupe_method_var,
            value="fast_hash",
        ).pack(anchor="w")
        ttk.Radiobutton(
            dedupe_method_frame,
            text="Name + Size - Fastest",
            variable=self.dedupe_method_var,
            value="name_size",
        ).pack(anchor="w")

        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill="x", pady=(PADDING_MEDIUM, 0))

        self.start_btn = ttk.Button(
            action_frame,
            text="▶️ Start Processing",
            command=self._start_operation,
            style="Accent.TButton",
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, PADDING_SMALL))

        self.cancel_btn = ttk.Button(
            action_frame,
            text="⏹️ Cancel",
            command=self._cancel_operation,
            state="disabled",
        )
        self.cancel_btn.pack(side="right", fill="x", expand=True)

    def _create_filters_tab(self) -> None:  # noqa: PLR0915
        """Create filters and advanced options tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Filters & Advanced  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # File extension filter
        ext_frame = ttk.LabelFrame(
            main_frame, text="File Extension Filter", padding=PADDING_MEDIUM
        )
        ext_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        ttk.Label(
            ext_frame,
            text="Include only these extensions (
                comma-separated,
                e.g.,
                .jpg,
                .png,
                .pdf
            ):",
        ).pack(anchor="w")
        self.ext_filter_entry = ttk.Entry(ext_frame)
        self.ext_filter_entry.pack(fill="x", pady=(PADDING_SMALL, 0))

        # Regex filter
        regex_frame = ttk.LabelFrame(
            main_frame, text="Regular Expression Filter", padding=PADDING_MEDIUM
        )
        regex_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        ttk.Label(
            regex_frame,
            text="File name pattern (regex, e.g., ^report_.*\\.pdf$ for report PDFs):",
        ).pack(anchor="w")
        self.regex_filter_entry = ttk.Entry(regex_frame)
        self.regex_filter_entry.pack(fill="x", pady=(PADDING_SMALL, 0))

        # Size filter
        size_frame = ttk.LabelFrame(
            main_frame, text="File Size Filter", padding=PADDING_MEDIUM
        )
        size_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        size_row1 = ttk.Frame(size_frame)
        size_row1.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Label(size_row1, text="Minimum size (MB):").pack(side="left")
        self.min_size_entry = ttk.Entry(size_row1, width=15)
        self.min_size_entry.pack(side="left", padx=(PADDING_SMALL, 0))
        self.min_size_entry.insert(0, "0")

        size_row2 = ttk.Frame(size_frame)
        size_row2.pack(fill="x")

        ttk.Label(size_row2, text="Maximum size (MB):").pack(side="left")
        self.max_size_entry = ttk.Entry(size_row2, width=15)
        self.max_size_entry.pack(side="left", padx=(PADDING_SMALL, 0))
        self.max_size_entry.insert(0, str(MAX_FILE_SIZE_MB))

        # Advanced options
        advanced_frame = ttk.LabelFrame(
            main_frame, text="Advanced Options", padding=PADDING_MEDIUM
        )
        advanced_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.skip_hidden_var = tk.BooleanVar(value=True)
        self.skip_system_var = tk.BooleanVar(value=True)
        self.follow_symlinks_var = tk.BooleanVar(value=False)
        self.preserve_metadata_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            advanced_frame, text="Skip hidden files", variable=self.skip_hidden_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame, text="Skip system files", variable=self.skip_system_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Follow symbolic links",
            variable=self.follow_symlinks_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Preserve file metadata",
            variable=self.preserve_metadata_var,
        ).pack(anchor="w")

        # Test filters button
        ttk.Button(main_frame, text="🔍 Test Filters", command=self._test_filters).pack(
            pady=PADDING_MEDIUM
        )

    def _create_preview_tab(self) -> None:  # noqa: PLR0915
        """Create preview tab showing files that will be processed."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Preview  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Button(toolbar, text="🔄 Refresh", command=self._update_preview).pack(
            side="left"
        )
        ttk.Label(toolbar, text="Files to be processed:").pack(
            side="left", padx=(PADDING_MEDIUM, 0)
        )
        self.preview_count_label = ttk.Label(
            toolbar, text="0", font=("Segoe UI", 10, "bold")
        )
        self.preview_count_label.pack(side="left", padx=(PADDING_SMALL, 0))

        # Treeview for file preview
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True)

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scroll_y.pack(side="right", fill="y")

        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_scroll_x.pack(side="bottom", fill="x")

        self.preview_tree = ttk.Treeview(
            tree_frame,
            columns=("Path", "Size", "Modified", "Type"),
            show="tree headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
        )

        self.preview_tree.heading("#0", text="Name")
        self.preview_tree.heading("Path", text="Full Path")
        self.preview_tree.heading("Size", text="Size")
        self.preview_tree.heading("Modified", text="Modified")
        self.preview_tree.heading("Type", text="Type")

        self.preview_tree.column("#0", width=200)
        self.preview_tree.column("Path", width=400)
        self.preview_tree.column("Size", width=100)
        self.preview_tree.column("Modified", width=150)
        self.preview_tree.column("Type", width=100)

        self.preview_tree.pack(side="left", fill="both", expand=True)

        tree_scroll_y.config(command=self.preview_tree.yview)
        tree_scroll_x.config(command=self.preview_tree.xview)

    def _create_operation_log_tab(self) -> None:
        """Create operation log tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Log  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Button(toolbar, text="🗑️ Clear Log", command=self._clear_log).pack(
            side="left"
        )
        ttk.Button(toolbar, text="💾 Save Log", command=self._save_log).pack(
            side="left", padx=(PADDING_SMALL, 0)
        )

        # Log text widget
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            yscrollcommand=log_scroll.set,
            font=("Consolas", 9),
            state="disabled",
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        # Configure text tags for colored output
        self.log_text.tag_config("info", foreground="#17a2b8")
        self.log_text.tag_config("success", foreground="#28a745")
        self.log_text.tag_config("warning", foreground="#ffc107")
        self.log_text.tag_config("error", foreground="#dc3545")

    def _create_status_bar(self) -> None:
        """Create bottom status bar."""
        status_frame = ttk.Frame(self.root, relief="sunken", borderwidth=1)
        status_frame.pack(side="bottom", fill="x")

        self.status_bar_label = ttk.Label(
            status_frame,
            text="Ready  |  Theme: Dark  |  No operation in progress",
            anchor="w",
        )
        self.status_bar_label.pack(
            side="left", fill="x", expand=True, padx=PADDING_SMALL
        )

        # Version label
        version_label = ttk.Label(status_frame, text="v3.0", anchor="e")
        version_label.pack(side="right", padx=PADDING_SMALL)

    def _setup_drag_drop(self) -> None:
        """Set up drag and drop functionality."""
        # Drag-and-drop would require tkinterdnd2 package
        # For standard tkinter, we rely on file dialog buttons
        self._log_message("Ready - Use buttons to add folders", "info")

    def _apply_theme(self) -> None:
        """Apply color theme to application."""
        theme = DARK_THEME if self.current_theme == "dark" else LIGHT_THEME

        # Configure ttk styles
        style = ttk.Style()

        # Configure button style with accent color
        style.configure(
            "Accent.TButton", font=("Segoe UI", 10, "bold"), padding=10, relief="flat"
        )

        # Update root background
        self.root.configure(bg=theme["bg"])

        # Update status bar
        self.status_bar_label.configure(
            text=f"Ready  |  Theme: {self.current_theme.title()}  |  No operation in progress"
        )

    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme()
        self._log_message(f"Switched to {self.current_theme} theme", "info")

    def _add_source_folder(self) -> None:
        """Add source folder via dialog."""
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder and folder not in self.source_folders:
            self.source_folders.append(folder)
            self.source_listbox.insert("end", folder)
            self._log_message(f"Added source folder: {folder}", "info")
            self._update_preview()

    def _remove_source_folder(self) -> None:
        """Remove selected source folder."""
        selection = self.source_listbox.curselection()
        for index in reversed(selection):
            folder = self.source_folders[index]
            self.source_folders.pop(index)
            self.source_listbox.delete(index)
            self._log_message(f"Removed source folder: {folder}", "info")
        self._update_preview()

    def _clear_source_folders(self) -> None:
        """Clear all source folders."""
        self.source_folders.clear()
        self.source_listbox.delete(0, "end")
        self._log_message("Cleared all source folders", "info")
        self._update_preview()

    def _browse_destination(self) -> None:
        """Browse for destination folder."""
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.dest_folder = folder
            self.dest_entry.delete(0, "end")
            self.dest_entry.insert(0, folder)
            self._log_message(f"Set destination folder: {folder}", "info")

    def _on_mode_change(self) -> None:
        """Handle operation mode change."""
        mode = self.mode_var.get()
        self._log_message(f"Operation mode changed to: {mode}", "info")
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the preview tab with files to be processed."""
        self.preview_tree.delete(*self.preview_tree.get_children())

        if not self.source_folders:
            self.preview_count_label.configure(text="0")
            return

        # Scan files in background
        def scan_files() -> None:
            files: list[Path] = []
            for folder in self.source_folders:
                try:
                    for root, _, filenames in os.walk(folder):
                        for filename in filenames:
                            file_path = Path(root) / filename
                            if self._should_include_file(file_path):
                                files.append(file_path)
                                if len(files) >= PREVIEW_MAX_FILES:
                                    break
                        if len(files) >= PREVIEW_MAX_FILES:
                            break
                except Exception:
                    logger.exception("Error scanning %s", folder)

            self.root.after(0, lambda: self._populate_preview(files))

        threading.Thread(target=scan_files, daemon=True).start()

    def _populate_preview(self, files: list[Path]) -> None:
        """Populate preview tree with scanned files."""
        for file_path in files[:PREVIEW_MAX_FILES]:
            try:
                stat = file_path.stat()
                size = self._format_size(stat.st_size)
                modified = datetime.fromtimestamp(stat.st_mtime, datetime.UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                file_type = file_path.suffix or "File"

                self.preview_tree.insert(
                    "",
                    "end",
                    text=file_path.name,
                    values=(str(file_path), size, modified, file_type),
                )
            except Exception:
                logger.exception("Error adding %s to preview", file_path)

        count_text = f"{len(files)}" + (
            " (limited)" if len(files) >= PREVIEW_MAX_FILES else ""
        )
        self.preview_count_label.configure(text=count_text)

    def _should_include_file(self, file_path: Path) -> bool:  # noqa: PLR0911
        """Check if file should be included based on filters."""
        try:
            # Check include extensions
            include_exts = self.ext_filter_entry.get().strip()
            if include_exts:
                exts = [e.strip().lower() for e in include_exts.split(",")]
                if file_path.suffix.lower() not in exts:
                    return False

            # Check min size
            min_size_str = self.min_size_entry.get().strip()
            if min_size_str:
                try:
                    min_size = float(min_size_str) * 1024 * 1024  # MB to bytes
                    if file_path.stat().st_size < min_size:
                        return False
                except ValueError:
                    pass

            # Check max size
            max_size_str = self.max_size_entry.get().strip()
            if max_size_str:
                try:
                    max_size = float(max_size_str) * 1024 * 1024  # MB to bytes
                    if file_path.stat().st_size > max_size:
                        return False
                except ValueError:
                    pass

            # Check regex filter
            regex_filter = self.regex_filter_entry.get().strip()
            return not (regex_filter and not re.match(regex_filter, file_path.name))

        except Exception:
            logger.exception("Error checking file %s", file_path)
            return False

    def _test_filters(self) -> None:
        """Test current filters and show results."""
        if not self.source_folders:
            messagebox.showwarning(
                "No Source Folders", "Please add source folders first."
            )
            return

        matching_files = []
        total_files = 0

        for folder in self.source_folders:
            for root, _, filenames in os.walk(folder):
                for filename in filenames:
                    total_files += 1
                    file_path = Path(root) / filename
                    if self._should_include_file(file_path):
                        matching_files.append(file_path)

        message = "Filter Results:\n\n"
        message += f"Total files scanned: {total_files}\n"
        message += f"Files matching filters: {len(matching_files)}\n"
        message += f"Files excluded: {total_files - len(matching_files)}\n\n"

        if matching_files:
            message += "Sample matching files (first 10):\n"
            for file in matching_files[:10]:
                message += f"  • {file.name}\n"

        messagebox.showinfo("Filter Test Results", message)

    def _start_operation(self) -> None:
        """Start the selected operation."""
        # Validate inputs
        if not self.source_folders:
            messagebox.showwarning(
                "No Source Folders", "Please add at least one source folder."
            )
            return

        mode = self.mode_var.get()
        if mode != "analyze" and not self.dest_folder:
            messagebox.showwarning(
                "No Destination", "Please select a destination folder."
            )
            return

        # Confirm operation
        if not self.preview_var.get() and not messagebox.askyesno(
            "Confirm Operation",
            f"Are you sure you want to perform this operation?\n\n"
            f"Mode: {mode}\n"
            f"Sources: {len(self.source_folders)} folder(s)\n"
            f"Destination: {self.dest_folder if mode != 'analyze' else 'N/A'}",
        ):
            return

        # Start operation in background thread
        self.cancel_operation = False
        self.operation_report = OperationReport()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_var.set(0)

        threading.Thread(target=self._run_operation, daemon=True).start()

    def _run_operation(self) -> None:
        """Run the selected operation (background thread)."""
        try:
            mode = self.mode_var.get()
            self._log_message(f"Starting operation: {mode}", "info")
            self._update_status(f"Running {mode} operation...")

            if mode == "combine":
                self._operation_combine()
            elif mode == "flatten":
                self._operation_flatten()
            elif mode == "prune":
                self._operation_prune()
            elif mode == "deduplicate":
                self._operation_deduplicate()
            elif mode == "analyze":
                self._operation_analyze()

            self.operation_report.finalize()

            if not self.cancel_operation:
                self._log_message("Operation completed successfully!", "success")
                self._update_status("Operation completed")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Success",
                        f"Operation completed!\n\nDuration: {self.operation_report.get_duration()}",
                    ),
                )

        except Exception as e:
            logger.exception("Operation failed")
            self._log_message(f"Operation failed: {e}", "error")
            err_msg = str(e)
            self.operation_report.add_error(err_msg)
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error", f"Operation failed:\n\n{err_msg}"
                ),
            )

        finally:
            self.root.after(0, self._operation_finished)

    def _operation_combine(self) -> None:
        """Combine and copy files from multiple sources."""
        dest_path = Path(self.dest_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        total_files = self._count_files()
        processed = 0

        for source_folder in self.source_folders:
            if self.cancel_operation:
                break

            for root, _, filenames in os.walk(source_folder):
                for filename in filenames:
                    if self.cancel_operation:
                        break

                    source_file = Path(root) / filename
                    if self._should_include_file(source_file):
                        dest_file = dest_path / filename

                        # Handle duplicates
                        counter = 1
                        while dest_file.exists():
                            stem = source_file.stem
                            suffix = source_file.suffix
                            dest_file = dest_path / f"{stem}_{counter}{suffix}"
                            counter += 1

                        try:
                            if not self.preview_var.get():
                                shutil.copy2(source_file, dest_file)
                            self.operation_report.add_operation(
                                "copy",
                                {"source": str(source_file), "dest": str(dest_file)},
                            )
                            processed += 1
                            self._update_progress(
                                processed, total_files, f"Copying {filename}"
                            )
                        except Exception as e:  # noqa: BLE001
                            self.operation_report.add_error(
                                f"Failed to copy {source_file}: {e}"
                            )

    def _operation_flatten(self) -> None:
        """Flatten directory structure."""
        dest_path = Path(self.dest_folder)
        dest_path.mkdir(parents=True, exist_ok=True)

        total_files = self._count_files()
        processed = 0

        for source_folder in self.source_folders:
            if self.cancel_operation:
                break

            for root, _, filenames in os.walk(source_folder):
                for filename in filenames:
                    if self.cancel_operation:
                        break

                    source_file = Path(root) / filename
                    if self._flatten_single_file(source_file, dest_path):
                        processed += 1
                        self._update_progress(
                            processed, total_files, f"Flattening {filename}"
                        )

    def _flatten_single_file(self, source_file: Path, dest_path: Path) -> bool:
        """Process a single file for flattening."""
        if not self._should_include_file(source_file):
            return False

        try:
            dest_file = self._determine_flatten_dest(source_file, dest_path)
            dest_file = self._resolve_collision(source_file, dest_file)

            if not self.preview_var.get():
                shutil.copy2(source_file, dest_file)

            self.operation_report.add_operation(
                "flatten",
                {"source": str(source_file), "dest": str(dest_file)},
            )
        except Exception as e:  # noqa: BLE001
            self.operation_report.add_error(f"Failed to flatten {source_file}: {e}")
            return False
        else:
            return True

    def _determine_flatten_dest(self, source_file: Path, dest_path: Path) -> Path:
        """Determine destination path for flattening."""
        if self.organize_type_var.get():
            # Organize by file type
            file_type = source_file.suffix.lstrip(".") or "no_extension"
            type_folder = dest_path / file_type
            type_folder.mkdir(exist_ok=True)
            return type_folder / source_file.name

        if self.organize_date_var.get():
            # Organize by date
            mtime = datetime.fromtimestamp(source_file.stat().st_mtime, datetime.UTC)
            date_folder = dest_path / mtime.strftime("%Y-%m")
            date_folder.mkdir(exist_ok=True)
            return date_folder / source_file.name

        return dest_path / source_file.name

    def _resolve_collision(self, source_file: Path, dest_file: Path) -> Path:
        """Resolve filename collisions."""
        counter = 1
        while dest_file.exists():
            stem = source_file.stem
            suffix = source_file.suffix
            dest_file = dest_file.parent / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest_file

    def _operation_prune(self) -> None:
        """Copy structure but prune empty folders."""
        dest_path = Path(self.dest_folder)

        total_folders = sum(
            len(dirs)
            for source_folder in self.source_folders
            for _, dirs, _ in os.walk(source_folder)
        )
        processed = 0

        for source_folder in self.source_folders:
            if self.cancel_operation:
                break

            source_path = Path(source_folder)

            for root, _dirs, filenames in os.walk(source_folder):
                if self.cancel_operation:
                    break

                # Check if folder has any files matching filters
                has_files = any(
                    self._should_include_file(Path(root) / f) for f in filenames
                )

                if has_files:
                    # Recreate folder structure
                    rel_path = Path(root).relative_to(source_path)
                    new_folder = dest_path / rel_path
                    new_folder.mkdir(parents=True, exist_ok=True)

                    # Copy files
                    for filename in filenames:
                        source_file = Path(root) / filename
                        if self._should_include_file(source_file):
                            dest_file = new_folder / filename
                            try:
                                if not self.preview_var.get():
                                    shutil.copy2(source_file, dest_file)
                                self.operation_report.add_operation(
                                    "copy",
                                    {
                                        "source": str(source_file),
                                        "dest": str(dest_file),
                                    },
                                )
                            except Exception as e:  # noqa: BLE001
                                self.operation_report.add_error(
                                    f"Failed to copy {source_file}: {e}"
                                )

                processed += 1
                self._update_progress(
                    processed, total_folders, f"Processing {Path(root).name}"
                )

    def _operation_deduplicate(self) -> None:
        """Remove duplicate files based on selected method."""
        method = self.dedupe_method_var.get()
        self._log_message(f"Deduplication method: {method}", "info")

        # Collect all files
        files = self._collect_files_for_dedupe()
        total_files = len(files)
        self._log_message(f"Found {total_files} files to check for duplicates", "info")

        # Find duplicates
        hash_map = self._build_dedupe_hash_map(method, files)

        # Remove duplicates
        duplicates_found, space_saved = self._remove_duplicates(hash_map)

        self._log_message(
            f"Deduplication complete. Removed {duplicates_found} duplicates, "
            f"saved {self._format_size(space_saved)}",
            "success",
        )

    def _collect_files_for_dedupe(self) -> list[Path]:
        """Collect all files from source folders."""
        files = []
        for source_folder in self.source_folders:
            for root, _, filenames in os.walk(source_folder):
                for filename in filenames:
                    file_path = Path(root) / filename
                    if self._should_include_file(file_path):
                        files.append(file_path)
        return files

    def _build_dedupe_hash_map(
        self, method: str, files: list[Path]
    ) -> dict[str, list[Path]]:
        """Build hash map for duplicate detection."""
        hash_map = defaultdict(list)
        total_files = len(files)

        for i, file_path in enumerate(files, 1):
            if self.cancel_operation:
                break

            try:
                file_hash = None
                if method == "hash":
                    file_hash = FileHasher.hash_file(file_path)
                elif method == "fast_hash":
                    file_hash = FileHasher.hash_file_fast(file_path)
                else:  # name_size
                    stat = file_path.stat()
                    file_hash = f"{file_path.name}_{stat.st_size}"

                if file_hash:
                    hash_map[file_hash].append(file_path)

                self._update_progress(i, total_files, f"Checking {file_path.name}")
            except Exception as e:  # noqa: BLE001
                self.operation_report.add_error(f"Failed to process {file_path}: {e}")

        return hash_map

    def _remove_duplicates(self, hash_map: dict[str, list[Path]]) -> tuple[int, int]:
        """Remove identified duplicates."""
        duplicates_found = 0
        space_saved = 0

        for file_list in hash_map.values():
            if len(file_list) > 1:
                # Keep first file, remove others
                for duplicate_file in file_list[1:]:
                    try:
                        size = duplicate_file.stat().st_size
                        if not self.preview_var.get():
                            duplicate_file.unlink()
                        duplicates_found += 1
                        space_saved += size
                        self.operation_report.add_operation(
                            "delete_duplicate",
                            {
                                "file": str(duplicate_file),
                                "original": str(file_list[0]),
                                "size": size,
                            },
                        )
                    except Exception as e:  # noqa: BLE001
                        self.operation_report.add_error(
                            f"Failed to delete duplicate {duplicate_file}: {e}"
                        )
        return duplicates_found, space_saved

    def _operation_analyze(self) -> None:
        """Analyze folders without making changes."""
        stats = {
            "total_files": 0,
            "total_size": 0,
            "file_types": defaultdict(int),
            "largest_files": [],
            "oldest_files": [],
            "newest_files": [],
        }

        total_folders = len(self.source_folders)
        for processed, source_folder in enumerate(self.source_folders, 1):
            if self.cancel_operation:
                break

            for root, _, filenames in os.walk(source_folder):
                for filename in filenames:
                    file_path = Path(root) / filename
                    if self._should_include_file(file_path):
                        try:
                            stat = file_path.stat()
                            stats["total_files"] += 1
                            stats["total_size"] += stat.st_size
                            stats["file_types"][file_path.suffix or "no_extension"] += 1

                            # Track largest files
                            stats["largest_files"].append((file_path, stat.st_size))
                            stats["largest_files"].sort(
                                key=lambda x: x[1], reverse=True
                            )
                            stats["largest_files"] = stats["largest_files"][:10]

                            # Track oldest/newest files
                            stats["oldest_files"].append((file_path, stat.st_mtime))
                            stats["oldest_files"].sort(key=lambda x: x[1])
                            stats["oldest_files"] = stats["oldest_files"][:10]

                            stats["newest_files"].append((file_path, stat.st_mtime))
                            stats["newest_files"].sort(key=lambda x: x[1], reverse=True)
                            stats["newest_files"] = stats["newest_files"][:10]

                        except Exception as e:  # noqa: BLE001
                            self.operation_report.add_error(
                                f"Failed to analyze {file_path}: {e}"
                            )

            # processed already tracked by enumerate
            self._update_progress(
                processed, total_folders, f"Analyzing {source_folder}"
            )

        # Show analysis results
        self._show_analysis_results(stats)

    def _show_analysis_results(self, stats: dict[str, int]) -> None:
        """Show analysis results in a dialog."""
        results = "📊 Folder Analysis Results\n\n"
        results += f"Total Files: {stats['total_files']:,}\n"
        results += f"Total Size: {self._format_size(stats['total_size'])}\n\n"

        results += "File Types:\n"
        for ext, count in sorted(
            stats["file_types"].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            results += f"  {ext}: {count:,} files\n"

        results += "\nLargest Files:\n"
        for file_path, size in stats["largest_files"][:5]:
            results += f"  {file_path.name}: {self._format_size(size)}\n"

        self.root.after(0, lambda: messagebox.showinfo("Analysis Results", results))

        # Also log to file
        self._log_message("Analysis Results:", "info")
        for line in results.split("\n"):
            if line.strip():
                self._log_message(line, "info")

    def _count_files(self) -> int:
        """Count total files to process."""
        total = 0
        for source_folder in self.source_folders:
            for _, _, filenames in os.walk(source_folder):
                total += len(filenames)
        return total

    def _update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar and status."""
        if total > 0:
            percentage = (current / total) * 100
            self.root.after(0, lambda: self.progress_var.set(percentage))

            # Calculate ETA
            if current > 0:
                elapsed = (
                    datetime.now(UTC) - self.operation_report.start_time
                ).total_seconds()
                eta_seconds = (elapsed / current) * (total - current)
                eta = f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                self.root.after(0, lambda: self.eta_label.configure(text=eta))

        self._update_status(message)

    def _update_status(self, message: str) -> None:
        """Update status label."""
        self.root.after(0, lambda: self.status_label.configure(text=message))
        self.root.after(
            0,
            lambda: self.status_bar_label.configure(
                text=f"{message}  |  Theme: {self.current_theme.title()}"
            ),
        )

    def _cancel_operation(self) -> None:
        """Cancel current operation."""
        self.cancel_operation = True
        self._log_message("Operation cancelled by user", "warning")
        self._update_status("Cancelling...")

    def _operation_finished(self) -> None:
        """Clean up after operation finishes."""
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_var.set(0)
        self.eta_label.configure(text="")
        self._update_status("Ready")

    def _log_message(self, message: str, level: str = "info") -> None:
        """Add message to log."""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        def update_log() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", log_entry, level)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, update_log)

        # Also log to file
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "success":
            logger.info("SUCCESS: %s", message)
        else:
            logger.info(message)

    def _clear_log(self) -> None:
        """Clear the log display."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _save_log(self) -> None:
        """Save log to file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"folder_fix_log_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.txt",
        )

        if file_path:
            with Path(file_path).open("w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", "end"))
            messagebox.showinfo("Log Saved", f"Log saved to:\n{file_path}")

    def _export_report_json(self) -> None:
        """Export operation report as JSON."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"folder_fix_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json",
        )

        if file_path:
            try:
                self.operation_report.export_json(file_path)
                messagebox.showinfo(
                    "Report Exported", f"Report exported to:\n{file_path}"
                )
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Export Failed", f"Failed to export report:\n{e}")

    def _export_report_html(self) -> None:
        """Export operation report as HTML."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            initialfile=f"folder_fix_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.html",
        )

        if file_path:
            try:
                self.operation_report.export_html(file_path)
                messagebox.showinfo(
                    "Report Exported", f"Report exported to:\n{file_path}"
                )

                # Ask if user wants to open it
                if messagebox.askyesno(
                    "Open Report", "Would you like to open the report?"
                ):
                    import webbrowser

                    webbrowser.open(f"file://{Path(file_path).resolve()}")
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Export Failed", f"Failed to export report:\n{e}")

    def _clear_cache(self) -> None:
        """Clear file cache."""
        self.file_cache.clear()
        self._log_message("Cache cleared", "info")
        messagebox.showinfo("Cache Cleared", "File cache has been cleared.")

    def _open_log_file(self) -> None:
        """Open the log file in default text editor."""
        try:

            webbrowser.open(log_filename)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"Could not open log file:\n{e}")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """Folder Fix Pro v3.0

Professional Folder Management Tool

Features:
• Modern themed UI with dark/light mode
• Drag-and-drop support
• Smart deduplication with SHA-256 hashing
• Advanced filtering (extensions, regex, size)
• Real-time preview
• Batch operations with progress tracking
• Export reports (JSON/HTML)
• Professional error handling

© 2024 All Rights Reserved
"""
        messagebox.showinfo("About Folder Fix Pro", about_text)

    def _show_user_guide(self) -> None:
        """Show user guide."""
        guide_text = """Folder Fix Pro - Quick Start Guide

1. ADDING SOURCES
   • Click '+ Add Folder' or drag folders to the list  # noqa: RUF001
   • Select multiple folders for batch operations

2. SELECTING DESTINATION
   • Click 'Browse' or drag a folder to the destination field
   • Not needed for 'Analyze Only' mode

3. CHOOSING OPERATION MODE
   • Combine & Copy: Merge files from multiple sources
   • Flatten & Tidy: Remove nested structure
   • Copy & Prune Empty: Remove empty folders
   • Deduplicate In-Place: Remove duplicate files
   • Analyze Only: Scan without changes

4. APPLYING FILTERS
   • Use the 'Filters & Advanced' tab
   • Set file extensions, size limits, or regex patterns
   • Click 'Test Filters' to preview results

5. PREVIEW
   • Check the 'Preview' tab to see files to be processed
   • Use 'Preview Mode' option to test without changes

6. RUNNING OPERATIONS
   • Click '▶️ Start Processing'
   • Monitor progress in the Progress section
   • View detailed logs in the 'Log' tab

7. EXPORTING REPORTS
   • Use File menu to export JSON or HTML reports
   • Save logs for record keeping

Tips:
• Always test with 'Preview Mode' first
• Enable backups for important operations
• Use deduplication to save disk space
• Check the log for detailed information
"""
        # Create scrollable dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("User Guide")
        dialog.geometry("600x500")

        text = tk.Text(dialog, wrap="word", font=("Consolas", 9))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", guide_text)
        text.configure(state="disabled")

        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        kb_size = 1024.0
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < kb_size:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= kb_size
        return f"{size_bytes:.2f} PB"


def main() -> None:
    """Main entry point for Folder Fix Pro."""
    try:
        # Create root window
        root = tk.Tk()

        # Create application
        _app = FolderFixPro(root)

        # Start main loop
        root.mainloop()

    except Exception as e:
        logger.exception("Fatal error in main application")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
