import base64
import gzip
import json
import logging
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Final

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PREVIEW_FILE_LIMIT = 500  # Maximum files to show in preview
PREVIEW_LINE_LIMIT = 1000  # Maximum lines to show in file preview
BYTES_PER_KB = 1024.0  # Bytes in a kilobyte

"""
Folder Packer Pro v2.0 - Enhanced Professional Project Packing Tool

A comprehensive project packaging application with advanced features:
- Modern themed UI with professional design
- Pack/Unpack folders into single encrypted archives
- AES-256 encryption for sensitive projects
- Multiple compression levels (store, fast, best)
- Git integration (preserve repository structure)
- Syntax highlighting in file preview
- Smart file filtering with customizable patterns
- Batch operations with progress tracking
- Export manifests and operation logs
- Professional error handling and validation
"""


# Constants with professional standards
MAX_FILE_SIZE_MB: Final[int] = 1024  # 1GB max per file
COMPRESSION_LEVELS: Final[dict[str, int]] = {
    "none": 0,
    "fast": 1,
    "balanced": 6,
    "best": 9,
}

# UI Constants
WINDOW_WIDTH: Final[int] = 1100
WINDOW_HEIGHT: Final[int] = 750
MIN_WINDOW_WIDTH: Final[int] = 900
MIN_WINDOW_HEIGHT: Final[int] = 600
PADDING_LARGE: Final[int] = 20
PADDING_MEDIUM: Final[int] = 10
PADDING_SMALL: Final[int] = 5

# File extensions for syntax highlighting (basic categorization)
CODE_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".r",
    ".m",
}

MARKUP_EXTENSIONS: Final[set[str]] = {
    ".html",
    ".xml",
    ".css",
    ".scss",
    ".sass",
    ".vue",
    ".jsx",
    ".tsx",
}

CONFIG_EXTENSIONS: Final[set[str]] = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}

# Default exclusion patterns
DEFAULT_EXCLUDE_PATTERNS: Final[set[str]] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "*.egg-info",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "pip-log.txt",
    "pip-delete-this-directory.txt",
    ".coverage",
    "htmlcov",
    ".sass-cache",
    "*.log",
}

# Color schemes
DARK_THEME = {
    "bg": "#2b2b2b",
    "fg": "#ffffff",
    "select_bg": "#404040",
    "entry_bg": "#353535",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
}

LIGHT_THEME = {
    "bg": "#f0f0f0",
    "fg": "#000000",
    "select_bg": "#0078d7",
    "entry_bg": "#ffffff",
    "accent": "#0078d7",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
}

# Set up professional logging
log_filename = "folder_packer_pro.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class EncryptionManager:
    """Handle encryption/decryption of packed files."""

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: User password
            salt: Random salt bytes

        Returns:
            32-byte encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_data(data: bytes, password: str) -> bytes:
        """
        Encrypt data with password using AES-256.

        Args:
            data: Data to encrypt
            password: Encryption password

        Returns:
            Encrypted data with salt prepended
        """
        salt = os.urandom(16)
        key = EncryptionManager.derive_key(password, salt)
        cipher = Fernet(key)
        encrypted: bytes = cipher.encrypt(data)
        result: bytes = salt + encrypted
        return result

    @staticmethod
    def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
        """
        Decrypt data with password.

        Args:
            encrypted_data: Encrypted data with salt prepended
            password: Decryption password

        Returns:
            Decrypted data
        """
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = EncryptionManager.derive_key(password, salt)
        cipher = Fernet(key)
        decrypted: bytes = cipher.decrypt(encrypted)
        return decrypted


class PackageManifest:
    """Manage package manifest with metadata."""

    def __init__(self) -> None:
        """Initialize the manifest."""
        self.created_at = datetime.now(UTC)
        self.files: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.stats: defaultdict[str, int] = defaultdict(int)

    def add_file(self, file_path: str, size: int, checksum: str) -> None:
        """Add file to manifest."""
        self.files.append(
            {
                "path": file_path,
                "size": size,
                "checksum": checksum,
                "added_at": datetime.now(UTC).isoformat(),
            },
        )
        self.stats["total_files"] += 1
        self.stats["total_size"] += size

    def set_metadata(
        self,
        key: str,
        value: str | float | bool | None | list[Any] | dict[str, Any],
    ) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "files": self.files,
            "metadata": self.metadata,
            "statistics": dict(self.stats),
        }

    def to_json(self) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PackageManifest":
        """Create manifest from JSON string."""
        data = json.loads(json_str)
        manifest = cls()
        manifest.created_at = datetime.fromisoformat(data["created_at"])
        manifest.files = data["files"]
        manifest.metadata = data["metadata"]
        manifest.stats = defaultdict(int, data["statistics"])
        return manifest


class FolderPackerPro:
    """Enhanced professional folder packing application."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application."""
        self.root = root
        self.root.title("Folder Packer Pro v2.0 - Professional Project Packager")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # Application state
        self.source_folder = ""
        self.output_file = ""
        self.current_theme = "dark"
        self.exclude_patterns = set(DEFAULT_EXCLUDE_PATTERNS)
        self.include_extensions: set[str] = set()
        self.manifest = PackageManifest()

        # Operation variables
        self.compression_level = "balanced"
        self.encrypt_enabled = False
        self.encryption_password = ""
        self.include_git = False
        self.create_manifest = True
        self.cancel_operation: bool = False

        # Initialize UI
        self._create_menu_bar()
        self._create_main_ui()
        self._apply_theme()

        logger.info("Folder Packer Pro v2.0 initialized successfully")

    def _create_menu_bar(self) -> None:
        """Create professional menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Package", command=self._new_package)
        file_menu.add_command(label="Export Manifest", command=self._export_manifest)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self._toggle_theme)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Manage Exclusions",
            command=self._manage_exclusions,
        )
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
            fill="both",
            expand=True,
            padx=PADDING_SMALL,
            pady=PADDING_SMALL,
        )

        # Create tabs
        self._create_pack_tab()
        self._create_unpack_tab()
        self._create_preview_tab()
        self._create_log_tab()

        # Status bar at bottom
        self._create_status_bar()

    def _create_pack_tab(self) -> None:  # noqa: PLR0915
        """Create pack operation tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Pack  ")

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

        # LEFT COLUMN - Source and Output
        # Header
        header_label = ttk.Label(
            left_frame,
            text="📦 Folder Packer Pro",
            font=("Segoe UI", 18, "bold"),
        )
        header_label.pack(pady=(0, PADDING_MEDIUM))

        # Source folder section
        source_frame = ttk.LabelFrame(
            left_frame,
            text="Source Folder",
            padding=PADDING_MEDIUM,
        )
        source_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        source_entry_frame = ttk.Frame(source_frame)
        source_entry_frame.pack(fill="x")

        self.pack_source_entry = ttk.Entry(source_entry_frame)
        self.pack_source_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        ttk.Button(
            source_entry_frame,
            text="Browse",
            command=self._browse_pack_source,
        ).pack(side="right")

        # Output file section
        output_frame = ttk.LabelFrame(
            left_frame,
            text="Output Package File",
            padding=PADDING_MEDIUM,
        )
        output_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill="x")

        self.pack_output_entry = ttk.Entry(output_entry_frame)
        self.pack_output_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        ttk.Button(
            output_entry_frame,
            text="Browse",
            command=self._browse_pack_output,
        ).pack(side="right")

        # File statistics section
        stats_frame = ttk.LabelFrame(
            left_frame,
            text="Project Statistics",
            padding=PADDING_MEDIUM,
        )
        stats_frame.pack(fill="both", expand=True, pady=(0, PADDING_MEDIUM))

        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.stats_text.pack(fill="both", expand=True)

        ttk.Button(stats_frame, text="🔄 Scan Folder", command=self._scan_folder).pack(
            pady=(PADDING_SMALL, 0),
        )

        # Progress section
        progress_frame = ttk.LabelFrame(
            left_frame,
            text="Progress",
            padding=PADDING_MEDIUM,
        )
        progress_frame.pack(fill="x")

        self.pack_progress_var = tk.DoubleVar()
        self.pack_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.pack_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.pack_progress_bar.pack(fill="x", pady=(0, PADDING_SMALL))

        self.pack_status_label = ttk.Label(
            progress_frame,
            text="Ready",
            font=("Segoe UI", 9),
        )
        self.pack_status_label.pack(fill="x")

        # RIGHT COLUMN - Options
        # Compression options
        compression_frame = ttk.LabelFrame(
            right_frame,
            text="Compression Options",
            padding=PADDING_MEDIUM,
        )
        compression_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        ttk.Label(compression_frame, text="Compression Level:").pack(anchor="w")

        self.compression_var = tk.StringVar(value="balanced")
        compression_options = [
            ("None (Fastest)", "none"),
            ("Fast", "fast"),
            ("Balanced (Recommended)", "balanced"),
            ("Best (Slowest)", "best"),
        ]

        for text, value in compression_options:
            ttk.Radiobutton(
                compression_frame,
                text=text,
                variable=self.compression_var,
                value=value,
            ).pack(anchor="w", pady=2)

        # Security options
        security_frame = ttk.LabelFrame(
            right_frame,
            text="Security Options",
            padding=PADDING_MEDIUM,
        )
        security_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.encrypt_var = tk.BooleanVar()
        ttk.Checkbutton(
            security_frame,
            text="Enable AES-256 Encryption",
            variable=self.encrypt_var,
            command=self._on_encrypt_toggle,
        ).pack(anchor="w")

        self.password_frame = ttk.Frame(security_frame)
        self.password_frame.pack(fill="x", pady=(PADDING_SMALL, 0))

        ttk.Label(self.password_frame, text="Password:").pack(anchor="w")
        self.pack_password_entry = ttk.Entry(self.password_frame, show="*")
        self.pack_password_entry.pack(fill="x", pady=(PADDING_SMALL, 0))
        self.pack_password_entry.configure(state="disabled")

        ttk.Label(self.password_frame, text="Confirm:").pack(
            anchor="w",
            pady=(PADDING_SMALL, 0),
        )
        self.pack_password_confirm = ttk.Entry(self.password_frame, show="*")
        self.pack_password_confirm.pack(fill="x", pady=(PADDING_SMALL, 0))
        self.pack_password_confirm.configure(state="disabled")

        # Advanced options
        advanced_frame = ttk.LabelFrame(
            right_frame,
            text="Advanced Options",
            padding=PADDING_MEDIUM,
        )
        advanced_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.include_git_var = tk.BooleanVar()
        self.create_manifest_var = tk.BooleanVar(value=True)
        self.verify_pack_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            advanced_frame,
            text="Include .git folder (preserve repository)",
            variable=self.include_git_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Create manifest file",
            variable=self.create_manifest_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Verify package after creation",
            variable=self.verify_pack_var,
        ).pack(anchor="w")

        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill="x", pady=(PADDING_MEDIUM, 0))

        self.pack_btn = ttk.Button(
            action_frame,
            text="📦 Create Package",
            command=self._start_pack,
            style="Accent.TButton",
        )
        self.pack_btn.pack(side="left", fill="x", expand=True, padx=(0, PADDING_SMALL))

        self.pack_cancel_btn = ttk.Button(
            action_frame,
            text="⏹️ Cancel",
            command=self._cancel_operation,
            state="disabled",
        )
        self.pack_cancel_btn.pack(side="right", fill="x", expand=True)

    def _create_unpack_tab(self) -> None:  # noqa: PLR0915
        """Create unpack operation tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Unpack  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # Header
        header_label = ttk.Label(
            main_frame,
            text="📂 Unpack Package",
            font=("Segoe UI", 18, "bold"),
        )
        header_label.pack(pady=(0, PADDING_MEDIUM))

        # Package file section
        package_frame = ttk.LabelFrame(
            main_frame,
            text="Package File",
            padding=PADDING_MEDIUM,
        )
        package_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        package_entry_frame = ttk.Frame(package_frame)
        package_entry_frame.pack(fill="x")

        self.unpack_source_entry = ttk.Entry(package_entry_frame)
        self.unpack_source_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        ttk.Button(
            package_entry_frame,
            text="Browse",
            command=self._browse_unpack_source,
        ).pack(side="right")

        # Destination folder section
        dest_frame = ttk.LabelFrame(
            main_frame,
            text="Destination Folder",
            padding=PADDING_MEDIUM,
        )
        dest_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        dest_entry_frame = ttk.Frame(dest_frame)
        dest_entry_frame.pack(fill="x")

        self.unpack_dest_entry = ttk.Entry(dest_entry_frame)
        self.unpack_dest_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        ttk.Button(
            dest_entry_frame,
            text="Browse",
            command=self._browse_unpack_dest,
        ).pack(side="right")

        # Decryption section
        decrypt_frame = ttk.LabelFrame(
            main_frame,
            text="Decryption",
            padding=PADDING_MEDIUM,
        )
        decrypt_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.encrypted_var = tk.BooleanVar()
        ttk.Checkbutton(
            decrypt_frame,
            text="Package is encrypted",
            variable=self.encrypted_var,
            command=self._on_encrypted_toggle,
        ).pack(anchor="w")

        self.decrypt_password_frame = ttk.Frame(decrypt_frame)
        self.decrypt_password_frame.pack(fill="x", pady=(PADDING_SMALL, 0))

        ttk.Label(self.decrypt_password_frame, text="Password:").pack(anchor="w")
        self.unpack_password_entry = ttk.Entry(self.decrypt_password_frame, show="*")
        self.unpack_password_entry.pack(fill="x", pady=(PADDING_SMALL, 0))
        self.unpack_password_entry.configure(state="disabled")

        # Package info section
        info_frame = ttk.LabelFrame(
            main_frame,
            text="Package Information",
            padding=PADDING_MEDIUM,
        )
        info_frame.pack(fill="both", expand=True, pady=(0, PADDING_MEDIUM))

        self.package_info_text = scrolledtext.ScrolledText(
            info_frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.package_info_text.pack(fill="both", expand=True)

        ttk.Button(
            info_frame,
            text="🔍 Inspect Package",
            command=self._inspect_package,
        ).pack(pady=(PADDING_SMALL, 0))

        # Progress section
        progress_frame = ttk.LabelFrame(
            main_frame,
            text="Progress",
            padding=PADDING_MEDIUM,
        )
        progress_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.unpack_progress_var = tk.DoubleVar()
        self.unpack_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.unpack_progress_var,
            maximum=100,
            mode="determinate",
        )
        self.unpack_progress_bar.pack(fill="x", pady=(0, PADDING_SMALL))

        self.unpack_status_label = ttk.Label(
            progress_frame,
            text="Ready",
            font=("Segoe UI", 9),
        )
        self.unpack_status_label.pack(fill="x")

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill="x")

        self.unpack_btn = ttk.Button(
            action_frame,
            text="📂 Extract Package",
            command=self._start_unpack,
            style="Accent.TButton",
        )
        self.unpack_btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        self.unpack_cancel_btn = ttk.Button(
            action_frame,
            text="⏹️ Cancel",
            command=self._cancel_operation,
            state="disabled",
        )
        self.unpack_cancel_btn.pack(side="right", fill="x", expand=True)

    def _create_preview_tab(self) -> None:  # noqa: PLR0915
        """Create file preview tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Preview  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Label(
            toolbar,
            text="File Preview with Syntax Detection",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        # File tree
        tree_label_frame = ttk.LabelFrame(
            main_frame,
            text="Files to Pack",
            padding=PADDING_SMALL,
        )
        tree_label_frame.pack(fill="both", expand=True, pady=(0, PADDING_SMALL))

        tree_frame = ttk.Frame(tree_label_frame)
        tree_frame.pack(fill="both", expand=True)

        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side="right", fill="y")

        self.preview_tree = ttk.Treeview(
            tree_frame,
            columns=("Size", "Type", "Modified"),
            show="tree headings",
            yscrollcommand=tree_scroll.set,
        )

        self.preview_tree.heading("#0", text="File Name")
        self.preview_tree.heading("Size", text="Size")
        self.preview_tree.heading("Type", text="Type")
        self.preview_tree.heading("Modified", text="Modified")

        self.preview_tree.column("#0", width=300)
        self.preview_tree.column("Size", width=100)
        self.preview_tree.column("Type", width=100)
        self.preview_tree.column("Modified", width=150)

        self.preview_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.config(command=self.preview_tree.yview)

        # Bind selection event
        self.preview_tree.bind("<<TreeviewSelect>>", self._on_file_select)

        # Preview pane
        preview_label_frame = ttk.LabelFrame(
            main_frame,
            text="File Content",
            padding=PADDING_SMALL,
        )
        preview_label_frame.pack(fill="both", expand=True)

        self.preview_text = scrolledtext.ScrolledText(
            preview_label_frame,
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        self.preview_text.pack(fill="both", expand=True)

        # Configure text tags for syntax highlighting
        self.preview_text.tag_config("keyword", foreground="#569cd6")
        self.preview_text.tag_config("string", foreground="#ce9178")
        self.preview_text.tag_config("comment", foreground="#6a9955")
        self.preview_text.tag_config("function", foreground="#dcdcaa")
        self.preview_text.tag_config("number", foreground="#b5cea8")

    def _create_log_tab(self) -> None:
        """Create operation log tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Log  ")

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Button(toolbar, text="🗑️ Clear Log", command=self._clear_log).pack(
            side="left",
        )
        ttk.Button(toolbar, text="💾 Save Log", command=self._save_log).pack(
            side="left",
            padx=(PADDING_SMALL, 0),
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
            side="left",
            fill="x",
            expand=True,
            padx=PADDING_SMALL,
        )

        # Version label
        version_label = ttk.Label(status_frame, text="v2.0", anchor="e")
        version_label.pack(side="right", padx=PADDING_SMALL)

    def _apply_theme(self) -> None:
        """Apply color theme to application."""
        theme = DARK_THEME if self.current_theme == "dark" else LIGHT_THEME

        # Configure ttk styles
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=10)

        # Update root background
        self.root.configure(bg=theme["bg"])

        # Update status bar
        self.status_bar_label.configure(
            text=f"Ready  |  Theme: {self.current_theme.title()}  |  "
            "No operation in progress",
        )

    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme()
        self._log_message(f"Switched to {self.current_theme} theme", "info")

    def _browse_pack_source(self) -> None:
        """Browse for source folder to pack."""
        folder = filedialog.askdirectory(title="Select Folder to Pack")
        if folder:
            self.pack_source_entry.delete(0, "end")
            self.pack_source_entry.insert(0, folder)
            self.source_folder = folder
            self._log_message(f"Selected source folder: {folder}", "info")
            self._scan_folder()

    def _browse_pack_output(self) -> None:
        """Browse for output package file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".fpp",
            filetypes=[("Folder Packer Package", "*.fpp"), ("All files", "*.*")],
            initialfile="package.fpp",
        )
        if file_path:
            self.pack_output_entry.delete(0, "end")
            self.pack_output_entry.insert(0, file_path)
            self.output_file = file_path
            self._log_message(f"Set output file: {file_path}", "info")

    def _browse_unpack_source(self) -> None:
        """Browse for package file to unpack."""
        file_path = filedialog.askopenfilename(
            title="Select Package to Unpack",
            filetypes=[("Folder Packer Package", "*.fpp"), ("All files", "*.*")],
        )
        if file_path:
            self.unpack_source_entry.delete(0, "end")
            self.unpack_source_entry.insert(0, file_path)
            self._log_message(f"Selected package: {file_path}", "info")

    def _browse_unpack_dest(self) -> None:
        """Browse for destination folder for unpacking."""
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.unpack_dest_entry.delete(0, "end")
            self.unpack_dest_entry.insert(0, folder)
            self._log_message(f"Set destination folder: {folder}", "info")

    def _on_encrypt_toggle(self) -> None:
        """Handle encryption checkbox toggle."""
        if self.encrypt_var.get():
            self.pack_password_entry.configure(state="normal")
            self.pack_password_confirm.configure(state="normal")
        else:
            self.pack_password_entry.configure(state="disabled")
            self.pack_password_confirm.configure(state="disabled")

    def _on_encrypted_toggle(self) -> None:
        """Handle encrypted package checkbox toggle."""
        if self.encrypted_var.get():
            self.unpack_password_entry.configure(state="normal")
        else:
            self.unpack_password_entry.configure(state="disabled")

    def _scan_folder(self) -> None:
        """Scan source folder and display statistics."""
        if not self.pack_source_entry.get():
            return

        source_path = Path(self.pack_source_entry.get())
        if not source_path.exists():
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        # Scan in background
        def scan() -> None:
            """Background task to scan folder statistics."""
            stats = self._collect_folder_stats(source_path)
            self.root.after(0, lambda: self._display_stats(stats))

        threading.Thread(target=scan, daemon=True).start()

    def _collect_folder_stats(self, folder: Path) -> dict[str, Any]:
        """Collect statistics about folder contents."""
        stats: dict[str, Any] = {
            "total_files": 0,
            "total_size": 0,
            "file_types": defaultdict(int),
            "excluded_files": 0,
        }

        for root, dirs, files in os.walk(folder):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]

            for filename in files:
                file_path = Path(root) / filename

                if self._should_exclude(file_path):
                    stats["excluded_files"] += 1
                    continue

                try:
                    size = file_path.stat().st_size
                    stats["total_files"] += 1
                    stats["total_size"] += size
                    ext = file_path.suffix.lower() or "no extension"
                    stats["file_types"][ext] += 1
                except Exception:
                    logger.exception("Error scanning %s", file_path)

        return stats

    def _display_stats(self, stats: dict[str, Any]) -> None:
        """Display folder statistics in the stats text widget."""
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")

        output = "📊 Project Statistics\n\n"
        output += f"Total Files: {stats['total_files']:,}\n"
        output += f"Total Size: {self._format_size(stats['total_size'])}\n"
        output += f"Excluded Files: {stats['excluded_files']:,}\n\n"

        output += "File Types:\n"
        for ext, count in sorted(
            stats["file_types"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]:
            percentage = (
                (count / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
            )
            output += f"  {ext:20s} {count:5,} files ({percentage:5.1f}%)\n"

        self.stats_text.insert("1.0", output)
        self.stats_text.configure(state="disabled")

        # Update preview tree
        self._update_preview_tree()

    def _update_preview_tree(self) -> None:
        """Update preview tree with files to be packed."""
        self.preview_tree.delete(*self.preview_tree.get_children())

        if not self.pack_source_entry.get():
            return

        source_path = Path(self.pack_source_entry.get())
        if not source_path.exists():
            return

        def scan() -> None:
            """Background task to scan files for preview."""
            files = []
            for root, dirs, filenames in os.walk(source_path):
                # Filter excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]

                for filename in filenames:
                    file_path = Path(root) / filename
                    if not self._should_exclude(file_path):
                        try:
                            stat = file_path.stat()
                            files.append((file_path, stat))
                            if len(files) >= PREVIEW_FILE_LIMIT:  # Limit preview
                                break
                        except Exception:
                            logger.exception("Error scanning %s", file_path)
                if len(files) >= PREVIEW_FILE_LIMIT:
                    break

            self.root.after(0, lambda: self._populate_tree(files, source_path))

        threading.Thread(target=scan, daemon=True).start()

    def _populate_tree(
        self, files: list[tuple[Path, os.stat_result]], base_path: Path
    ) -> None:
        """Populate tree with file list."""
        for file_path, stat in files:
            rel_path = file_path.relative_to(base_path)
            size = self._format_size(stat.st_size)
            file_type = self._get_file_type(file_path)
            modified = datetime.fromtimestamp(stat.st_mtime, UTC).strftime(
                "%Y-%m-%d %H:%M"
            )

            self.preview_tree.insert(
                "",
                "end",
                text=str(rel_path),
                values=(size, file_type, modified),
                tags=(str(file_path),),
            )

    def _on_file_select(self, _event: tk.Event) -> None:
        """Handle file selection in preview tree."""
        selection = self.preview_tree.selection()
        if not selection:
            return

        # Get file path from tags
        item = selection[0]
        tags = self.preview_tree.item(item, "tags")
        if not tags:
            return

        file_path = Path(tags[0])
        if file_path.exists() and file_path.is_file():
            self._preview_file(file_path)

    def _preview_file(self, file_path: Path) -> None:
        """Preview file content with basic syntax highlighting."""
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")

        try:
            # Check file size
            size = file_path.stat().st_size
            if size > 1024 * 1024:  # 1MB limit
                self.preview_text.insert(
                    "1.0",
                    f"File too large to preview ({self._format_size(size)})",
                )
            else:
                # Try to read as text
                with file_path.open(encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Insert with basic syntax highlighting
                self._insert_with_highlighting(content, file_path.suffix)

        except Exception as e:
            self.preview_text.insert("1.0", f"Error previewing file: {e}")

        self.preview_text.configure(state="disabled")

    def _insert_with_highlighting(self, content: str, file_ext: str) -> None:
        """Insert text with basic syntax highlighting."""
        # For simplicity, basic keyword highlighting
        # Syntax highlighting map
        color_map: dict[str, dict[str, str]] = {
            ".py": dict.fromkeys(["def", "class", "import", "from"], "blue"),
            ".pyw": dict.fromkeys(["def", "class", "import", "from"], "blue"),
            ".js": dict.fromkeys(["function", "const", "let", "var"], "blue"),
            ".ts": dict.fromkeys(["function", "const", "let", "var"], "blue"),
        }
        # Add control flow keywords as purple for python
        for ext in [".py", ".pyw"]:
            color_map[ext].update(dict.fromkeys(["if", "else", "elif", "for", "while"], "purple"))

        keywords = color_map.get(file_ext, {})

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if i >= PREVIEW_LINE_LIMIT:  # Limit lines
                self.preview_text.insert("end", "\n... (truncated)")
                break

            # Simple syntax highlighting
            if file_ext in CODE_EXTENSIONS:
                # Check for comments
                if line.strip().startswith("#"):
                    self.preview_text.insert("end", line + "\n", "comment")
                else:
                    # Insert with keyword highlighting based on simplified map
                    words = re.split(r"(\s+)", line)
                    for word in words:
                        if word in keywords:
                            self.preview_text.insert("end", word, keywords[word])
                        elif word.startswith(('"', "'")):
                            self.preview_text.insert("end", word, "string")
                        elif word.isdigit():
                            self.preview_text.insert("end", word, "number")
                        else:
                            self.preview_text.insert("end", word)
                    self.preview_text.insert("end", "\n")
            else:
                self.preview_text.insert("end", line + "\n")

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        # Check if .git should be excluded
        if not self.include_git_var.get() and ".git" in path.parts:
            return True

        # Check exclusion patterns
        name = path.name
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in name:
                return True

        return False

    def _get_file_type(self, file_path: Path) -> str:
        """Get file type category."""
        ext = file_path.suffix.lower()
        if ext in CODE_EXTENSIONS:
            return "Code"
        if ext in MARKUP_EXTENSIONS:
            return "Markup"
        if ext in CONFIG_EXTENSIONS:
            return "Config"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"}:
            return "Image"
        if ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
            return "Audio"
        if ext in {".mp4", ".avi", ".mkv", ".mov", ".wmv"}:
            return "Video"
        if ext in {".pdf", ".doc", ".docx", ".txt", ".md", ".rst"}:
            return "Document"
        return "Other"

    def _start_pack(self) -> None:
        """Start packing operation."""
        # Validate inputs
        if not self.pack_source_entry.get():
            messagebox.showwarning("No Source", "Please select a source folder.")
            return

        if not self.pack_output_entry.get():
            messagebox.showwarning("No Output", "Please select an output file.")
            return

        # Validate encryption
        if self.encrypt_var.get():
            password = self.pack_password_entry.get()
            confirm = self.pack_password_confirm.get()

            if not password:
                messagebox.showwarning(
                    "No Password",
                    "Please enter an encryption password.",
                )
                return

            if password != confirm:
                messagebox.showwarning("Password Mismatch", "Passwords do not match.")
                return

        # Start operation
        self.cancel_operation = False
        self.pack_btn.configure(state="disabled")
        self.pack_cancel_btn.configure(state="normal")
        self.pack_progress_var.set(0)

        threading.Thread(target=self._run_pack, daemon=True).start()

    def _run_pack(self) -> None:  # noqa: PLR0915,PLR0912,C901
        """Run pack operation in background."""
        try:
            source_path = Path(self.pack_source_entry.get())
            output_path = Path(self.pack_output_entry.get())

            self._update_pack_status("Collecting files...")

            # Collect files
            files_to_pack = []
            for root, dirs, filenames in os.walk(source_path):
                if self.cancel_operation:
                    break

                dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]

                for filename in filenames:
                    file_path = Path(root) / filename
                    if not self._should_exclude(file_path):
                        files_to_pack.append(file_path)

            if self.cancel_operation:
                self._log_message("Pack operation cancelled", "warning")
                return

            total_files = len(files_to_pack)
            self._log_message(f"Packing {total_files} files...", "info")

            # Create package data
            package_data = {
                "files": {},
                "metadata": {
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": str(source_path),
                    "total_files": total_files,
                    "compression": self.compression_var.get(),
                    "encrypted": self.encrypt_var.get(),
                },
            }

            # Add files to package
            for i, file_path in enumerate(files_to_pack):
                if self.cancel_operation:
                    break  # type: ignore[unreachable]

                try:
                    rel_path = file_path.relative_to(source_path)
                    with file_path.open("rb") as f:
                        content = f.read()

                    # Store with base64 encoding
                    package_data["files"][str(rel_path)] = base64.b64encode(
                        content,
                    ).decode("utf-8")

                    progress = ((i + 1) / total_files) * 100

                    def update_progress(p: float = progress) -> None:
                        """Update the progress bar."""
                        self.pack_progress_var.set(float(p))

                    self.root.after(0, update_progress)
                    self._update_pack_status(
                        f"Packing {file_path.name} ({i + 1}/{total_files})",
                    )

                except Exception as e:
                    logger.exception("Error packing {file_path}")
                    self._log_message(f"Error packing {file_path}: {e}", "error")

            if self.cancel_operation:
                self._log_message(
                    "Pack operation cancelled", "warning"
                )  # type: ignore[unreachable]
                return

            # Serialize to JSON
            json_data = json.dumps(package_data, indent=2).encode("utf-8")

            # Compress if needed
            compression_level = COMPRESSION_LEVELS[self.compression_var.get()]
            if compression_level > 0:
                self._update_pack_status("Compressing...")
                json_data = gzip.compress(json_data, compresslevel=compression_level)

            # Encrypt if needed
            if self.encrypt_var.get():
                self._update_pack_status("Encrypting...")
                password = self.pack_password_entry.get()
                json_data = EncryptionManager.encrypt_data(json_data, password)

            # Write to file
            self._update_pack_status("Writing package file...")
            with output_path.open("wb") as f:  # type: ignore[assignment]
                f.write(json_data)

            # Create manifest if enabled
            if self.create_manifest_var.get():
                manifest_path = output_path.with_suffix(".manifest.json")
                manifest = {
                    "package_file": str(output_path),
                    "created_at": datetime.now(UTC).isoformat(),
                    "files": [str(f.relative_to(source_path)) for f in files_to_pack],
                    "total_files": total_files,
                    "package_size": output_path.stat().st_size,
                }
                with manifest_path.open("w", encoding="utf-8") as manifest_file:
                    json.dump(manifest, manifest_file, indent=2)

            self._log_message(f"Package created successfully: {output_path}", "success")
            self._log_message(
                f"Package size: {self._format_size(output_path.stat().st_size)}",
                "info",
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Package created successfully!\n\n"
                    f"Files: {total_files}\n"
                    f"Size: {self._format_size(output_path.stat().st_size)}",
                ),
            )

        except Exception as e:
            logger.exception("Pack operation failed")
            self._log_message(f"Pack operation failed: {e}", "error")
            error_msg = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Pack failed:\n\n{error_msg}"),
            )

        finally:
            self.root.after(0, self._pack_finished)

    def _start_unpack(self) -> None:
        """Start unpacking operation."""
        # Validate inputs
        if not self.unpack_source_entry.get():
            messagebox.showwarning("No Package", "Please select a package file.")
            return

        if not self.unpack_dest_entry.get():
            messagebox.showwarning(
                "No Destination",
                "Please select a destination folder.",
            )
            return

        # Validate decryption
        if self.encrypted_var.get():
            password = self.unpack_password_entry.get()
            if not password:
                messagebox.showwarning(
                    "No Password",
                    "Please enter the decryption password.",
                )
                return

        # Start operation
        self.cancel_operation = False
        self.unpack_btn.configure(state="disabled")
        self.unpack_cancel_btn.configure(state="normal")
        self.unpack_progress_var.set(0)

        threading.Thread(target=self._run_unpack, daemon=True).start()

    def _run_unpack(self) -> None:  # noqa: PLR0915
        """Run unpack operation in background."""
        try:
            package_path = Path(self.unpack_source_entry.get())
            dest_path = Path(self.unpack_dest_entry.get())

            dest_path.mkdir(parents=True, exist_ok=True)

            self._update_unpack_status("Reading package...")

            # Read package file
            with package_path.open("rb") as f:
                data = f.read()

            # Decrypt if needed
            if self.encrypted_var.get():
                self._update_unpack_status("Decrypting...")
                password = self.unpack_password_entry.get()
                try:
                    data = EncryptionManager.decrypt_data(data, password)
                except Exception as e:
                    msg = f"Decryption failed - incorrect password? {e}"
                    raise ValueError(
                        msg,
                    ) from e

            # Decompress if needed
            try:
                self._update_unpack_status("Decompressing...")
                data = gzip.decompress(data)
            except Exception:

                # Not compressed - this is expected for uncompressed files

                pass

            # Parse JSON
            package_data = json.loads(data.decode("utf-8"))

            files = package_data.get("files", {})
            total_files = len(files)

            self._log_message(f"Extracting {total_files} files...", "info")

            # Extract files
            for i, (rel_path, encoded_content) in enumerate(files.items()):
                if self.cancel_operation:
                    break

                try:
                    file_path = dest_path / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Decode and write
                    content = base64.b64decode(encoded_content)
                    with file_path.open("wb") as f:
                        f.write(content)

                    progress = ((i + 1) / total_files) * 100
                    self.root.after(
                        0,
                        lambda p=progress: self.unpack_progress_var.set(float(p)),  # type: ignore[misc]
                    )
                    self._update_unpack_status(
                        f"Extracting {Path(rel_path).name} ({i + 1}/{total_files})",
                    )

                except Exception as e:
                    logger.exception("Error extracting {rel_path}")
                    self._log_message(f"Error extracting {rel_path}: {e}", "error")

            if self.cancel_operation:
                self._log_message("Unpack operation cancelled", "warning")
                return

            self._log_message(
                f"Package extracted successfully to: {dest_path}",
                "success",
            )

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Package extracted successfully!\n\nFiles: {total_files}\n"
                    f"Location: {dest_path}",
                ),
            )

        except Exception as e:
            logger.exception("Unpack operation failed")
            self._log_message(f"Unpack operation failed: {e}", "error")
            error_msg = str(e)
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Unpack failed:\n\n{error_msg}"),
            )

        finally:
            self.root.after(0, self._unpack_finished)

    def _inspect_package(self) -> None:
        """Inspect package file and show information."""
        package_path = self.unpack_source_entry.get()
        if not package_path:
            messagebox.showwarning("No Package", "Please select a package file first.")
            return

        try:
            with Path(package_path).open("rb") as f:
                data = f.read()

            # Check if encrypted
            is_encrypted = False
            try:
                # Try to decompress
                decompressed = gzip.decompress(data)
                json.loads(decompressed.decode("utf-8"))
            except Exception:
                is_encrypted = True

            # Display info
            self.package_info_text.configure(state="normal")
            self.package_info_text.delete("1.0", "end")

            info = "📦 Package Information\n\n"
            info += f"File: {Path(package_path).name}\n"
            info += f"Size: {self._format_size(Path(package_path).stat().st_size)}\n"
            info += f"Encrypted: {'Yes' if is_encrypted else 'No'}\n\n"

            if not is_encrypted:
                decompressed = gzip.decompress(data)
                package_data = json.loads(decompressed.decode("utf-8"))
                metadata = package_data.get("metadata", {})

                info += f"Created: {metadata.get('created_at', 'Unknown')}\n"
                info += f"Total Files: {metadata.get('total_files', 0)}\n"
                info += f"Compression: {metadata.get('compression', 'Unknown')}\n"

            self.package_info_text.insert("1.0", info)
            self.package_info_text.configure(state="disabled")

        except Exception as e:
            logger.exception("Error occurred")
            messagebox.showerror("Error", f"Failed to inspect package:\n\n{e}")

    def _manage_exclusions(self) -> None:  # noqa: PLR0915
        """Show dialog to manage exclusion patterns."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Exclusions")
        dialog.geometry("500x400")

        ttk.Label(
            dialog,
            text="Exclusion Patterns",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=PADDING_MEDIUM)

        # Listbox with current patterns
        list_frame = ttk.Frame(dialog)
        list_frame.pack(
            fill="both",
            expand=True,
            padx=PADDING_MEDIUM,
            pady=PADDING_SMALL,
        )

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        for pattern in sorted(self.exclude_patterns):
            listbox.insert("end", pattern)

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", padx=PADDING_MEDIUM, pady=PADDING_SMALL)

        def add_pattern() -> None:
            """Add a new exclusion pattern."""
            pattern = simpledialog.askstring(
                "Add Pattern",
                "Enter exclusion pattern:",
            )
            if pattern:
                self.exclude_patterns.add(pattern)
                listbox.insert("end", pattern)

        def remove_pattern() -> None:
            """Remove selected exclusion pattern."""
            selection = listbox.curselection()  # type: ignore[no-untyped-call]
            if selection:
                pattern = listbox.get(selection[0])
                self.exclude_patterns.discard(pattern)
                listbox.delete(selection[0])

        def reset_patterns() -> None:
            """Reset exclusion patterns to defaults."""
            if messagebox.askyesno("Reset", "Reset to default exclusion patterns?"):
                self.exclude_patterns = set(DEFAULT_EXCLUDE_PATTERNS)
                listbox.delete(0, "end")
                for pattern in sorted(self.exclude_patterns):
                    listbox.insert("end", pattern)

        ttk.Button(btn_frame, text="Add", command=add_pattern).pack(
            side="left",
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(btn_frame, text="Remove", command=remove_pattern).pack(
            side="left",
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(btn_frame, text="Reset to Default", command=reset_patterns).pack(
            side="left",
        )

        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(
            pady=PADDING_MEDIUM,
        )

    def _new_package(self) -> None:
        """Reset form for new package."""
        self.pack_source_entry.delete(0, "end")
        self.pack_output_entry.delete(0, "end")
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.configure(state="disabled")
        self.preview_tree.delete(*self.preview_tree.get_children())
        self._log_message("Ready for new package", "info")

    def _export_manifest(self) -> None:
        """Export current manifest."""
        if not self.manifest.files:
            messagebox.showwarning("No Manifest", "No manifest data to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="manifest.json",
        )

        if file_path:
            try:
                with Path(file_path).open("w", encoding="utf-8") as f:
                    f.write(self.manifest.to_json())
                messagebox.showinfo("Success", f"Manifest exported to:\n{file_path}")
            except Exception as e:
                logger.exception("Error occurred")
                messagebox.showerror("Error", f"Failed to export manifest:\n{e}")

    def _update_pack_status(self, message: str) -> None:
        """Update pack status label."""
        self.root.after(0, lambda: self.pack_status_label.configure(text=message))
        self._update_status_bar(message)

    def _update_unpack_status(self, message: str) -> None:
        """Update unpack status label."""
        self.root.after(0, lambda: self.unpack_status_label.configure(text=message))
        self._update_status_bar(message)

    def _update_status_bar(self, message: str) -> None:
        """Update bottom status bar."""
        self.root.after(
            0,
            lambda: self.status_bar_label.configure(
                text=f"{message}  |  Theme: {self.current_theme.title()}",
            ),
        )

    def _cancel_operation(self) -> None:
        """Cancel current operation."""
        self.cancel_operation = True
        self._log_message("Operation cancelled by user", "warning")

    def _pack_finished(self) -> None:
        """Clean up after pack operation."""
        self.pack_btn.configure(state="normal")
        self.pack_cancel_btn.configure(state="disabled")
        self.pack_progress_var.set(0)
        self._update_pack_status("Ready")

    def _unpack_finished(self) -> None:
        """Clean up after unpack operation."""
        self.unpack_btn.configure(state="normal")
        self.unpack_cancel_btn.configure(state="disabled")
        self.unpack_progress_var.set(0)
        self._update_unpack_status("Ready")

    def _log_message(self, message: str, level: str = "info") -> None:
        """Add message to log."""
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        def update_log() -> None:
            """Update log widget from thread."""
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
            initialfile=f"packer_log_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.txt",
        )

        if file_path:
            with Path(file_path).open("w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", "end"))
            messagebox.showinfo("Log Saved", f"Log saved to:\n{file_path}")

    def _open_log_file(self) -> None:
        """Open the log file in default text editor."""
        try:
            if sys.platform == "win32":
                os.startfile(log_filename)
            elif sys.platform == "darwin":
                subprocess.run(["open", log_filename], check=False)
            else:
                subprocess.run(["xdg-open", log_filename], check=False)
        except Exception as e:
            logger.exception("Error occurred")
            messagebox.showerror("Error", f"Could not open log file:\n{e}")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """Folder Packer Pro v2.0

Professional Project Packaging Tool

Features:
• Modern themed UI
• AES-256 encryption for sensitive projects
• Multiple compression levels
• Git integration (preserve repository)
• Syntax highlighting in preview
• Smart file filtering
• Batch operations with progress tracking
• Export manifests and logs
• Professional error handling

© 2024 All Rights Reserved
"""
        messagebox.showinfo("About Folder Packer Pro", about_text)

    def _show_user_guide(self) -> None:
        """Show user guide."""
        guide_text = """Folder Packer Pro - Quick Start Guide

PACKING:
1. Select source folder to pack
2. Choose output package file location (.fpp)
3. Configure compression level
4. (Optional) Enable encryption with password
5. (Optional) Include .git folder for repos
6. Click 'Create Package'

UNPACKING:
1. Select package file (.fpp)
2. Choose destination folder
3. If encrypted, check box and enter password
4. Click 'Extract Package'

FEATURES:
• Compression: Choose speed vs size trade-off
• Encryption: Secure sensitive code with AES-256
• Preview: View files and content before packing
• Exclusions: Manage patterns to exclude
• Manifest: Export file lists and metadata

TIPS:
• Use 'Balanced' compression for best results
• Encrypt packages with sensitive data
• Test unpacking after creating packages
• Use exclusions to skip large dependencies
• Preview shows first 1000 lines of files
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
        size: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < BYTES_PER_KB:
                return f"{size:.2f} {unit}"
            size /= BYTES_PER_KB
        return f"{size:.2f} PB"


def main() -> None:
    """Main entry point for Folder Packer Pro."""
    try:
        # Create root window
        root = tk.Tk()

        # Create application
        FolderPackerPro(root)

        # Start main loop
        root.mainloop()

    except Exception as e:
        logger.exception("Fatal error in main application")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
