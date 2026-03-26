"""Main application for Folder Packer Pro.

Slim orchestrator that composes UI tab mixins, dialog mixins,
operation mixins, and delegates core operations to the pack engine.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from utils.file_utils import safe_write_text

from .constants import (
    DARK_THEME,
    DEFAULT_EXCLUDE_PATTERNS,
    LIGHT_THEME,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    PADDING_SMALL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from .dialogs import DialogsMixin
from .file_ops import format_size
from .manifest import PackageManifest
from .operations import PackOperationMixin, ScanPreviewMixin, UnpackOperationMixin
from .ui_tabs import LogTabMixin, PackTabMixin, PreviewTabMixin, UnpackTabMixin

logger = logging.getLogger(__name__)


class FolderPackerPro(
    PackTabMixin,
    UnpackTabMixin,
    PreviewTabMixin,
    LogTabMixin,
    DialogsMixin,
    ScanPreviewMixin,
    PackOperationMixin,
    UnpackOperationMixin,
):
    """Enhanced professional folder packing application.

    Composes UI, dialog, and operation mixins with pack engine delegation.
    """

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application.

        Args:
            root: The tkinter root window.
        """
        if not (root is not None):
            raise ValueError("root must be provided")
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

    # -- Menu & Layout ---------------------------------------------------------

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
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill="both",
            expand=True,
            padx=PADDING_SMALL,
            pady=PADDING_SMALL,
        )

        # Create tabs (delegated to mixins)
        self._create_pack_tab()
        self._create_unpack_tab()
        self._create_preview_tab()
        self._create_log_tab()

        # Status bar at bottom
        self._create_status_bar()

    def _create_status_bar(self) -> None:
        """Create bottom status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", side="bottom")

        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            font=("Segoe UI", 8),
            padding=(PADDING_SMALL, 2),
        )
        self.status_label.pack(side="left")

        self.status_right = ttk.Label(
            status_frame,
            text="Folder Packer Pro v2.0",
            font=("Segoe UI", 8),
            padding=(PADDING_SMALL, 2),
        )
        self.status_right.pack(side="right")

    # -- Theme -----------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply color theme to application."""
        theme = DARK_THEME if self.current_theme == "dark" else LIGHT_THEME

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=theme["bg"], foreground=theme["fg"])
        style.configure("TFrame", background=theme["bg"])
        style.configure("TLabel", background=theme["bg"], foreground=theme["fg"])
        style.configure("TLabelframe", background=theme["bg"], foreground=theme["fg"])
        style.configure("TLabelframe.Label", background=theme["bg"], foreground=theme["fg"])
        style.configure(
            "Accent.TButton",
            background=theme["accent"],
            foreground="#ffffff",
        )

    def _toggle_theme(self) -> None:
        """Toggle between dark and light themes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme()

    # -- Browse Dialogs --------------------------------------------------------

    def _browse_pack_source(self) -> None:
        """Browse for source folder to pack."""
        folder = filedialog.askdirectory(title="Select Source Folder")
        if folder:
            self.pack_source_entry.delete(0, "end")
            self.pack_source_entry.insert(0, folder)
            self.source_folder = folder
            self._scan_folder()

    def _browse_pack_output(self) -> None:
        """Browse for output package file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Package As",
            defaultextension=".fpp",
            filetypes=[("FPP Package", "*.fpp"), ("All Files", "*.*")],
        )
        if file_path:
            self.pack_output_entry.delete(0, "end")
            self.pack_output_entry.insert(0, file_path)

    def _browse_unpack_source(self) -> None:
        """Browse for package file to unpack."""
        file_path = filedialog.askopenfilename(
            title="Select Package File",
            filetypes=[("FPP Package", "*.fpp"), ("All Files", "*.*")],
        )
        if file_path:
            self.unpack_source_entry.delete(0, "end")
            self.unpack_source_entry.insert(0, file_path)

    def _browse_unpack_dest(self) -> None:
        """Browse for destination folder for unpacking."""
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.unpack_dest_entry.delete(0, "end")
            self.unpack_dest_entry.insert(0, folder)

    # -- UI Toggles ------------------------------------------------------------

    def _on_encrypt_toggle(self) -> None:
        """Handle encryption checkbox toggle."""
        state = "normal" if self.encrypt_var.get() else "disabled"
        self.pack_password_entry.configure(state=state)
        self.pack_password_confirm.configure(state=state)
        if not self.encrypt_var.get():
            self.pack_password_entry.delete(0, "end")
            self.pack_password_confirm.delete(0, "end")

    def _on_encrypted_toggle(self) -> None:
        """Handle encrypted package checkbox toggle."""
        state = "normal" if self.encrypted_var.get() else "disabled"
        self.unpack_password_entry.configure(state=state)

    # -- File Selection --------------------------------------------------------

    def _on_file_select(self, event: tk.Event) -> None:
        """Handle file selection in preview tree.

        Args:
            event: The tkinter event.
        """
        if not (event is not None):
            raise ValueError("event must be provided")
        selection = self.preview_tree.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.preview_tree.item(item, "tags")
        if not tags:
            return

        file_path = Path(tags[0])
        if file_path.exists() and file_path.is_file():
            self._preview_file(file_path)

    def _preview_file(self, file_path: Path) -> None:
        """Preview file content with basic syntax highlighting.

        Args:
            file_path: Path to the file to preview.
        """
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")

        try:
            size = file_path.stat().st_size
            if size > 1024 * 1024:  # 1MB limit
                self.preview_text.insert(
                    "1.0",
                    f"File too large to preview ({format_size(size)})",
                )
            else:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                self._insert_with_highlighting(content, file_path.suffix)

        except (OSError, UnicodeDecodeError) as e:
            self.preview_text.insert("1.0", f"Error previewing file: {e}")

        self.preview_text.configure(state="disabled")

    # -- Status Updates --------------------------------------------------------

    def _update_pack_status(self, message: str) -> None:
        """Update pack status label.

        Args:
            message: Status message to display.
        """
        self.root.after(0, lambda: self.pack_status_label.configure(text=message))

    def _update_unpack_status(self, message: str) -> None:
        """Update unpack status label.

        Args:
            message: Status message to display.
        """
        self.root.after(0, lambda: self.unpack_status_label.configure(text=message))

    def _update_status_bar(self, message: str) -> None:
        """Update bottom status bar.

        Args:
            message: Status message to display.
        """
        if not (message is not None):
            raise ValueError("message must be provided")
        self.status_label.configure(text=message)
        self.root.update_idletasks()

    # -- Operation Lifecycle ---------------------------------------------------

    def _cancel_operation(self) -> None:
        """Cancel current operation."""
        self.cancel_operation = True

    def _pack_finished(self) -> None:
        """Clean up after pack operation."""
        self.pack_btn.configure(state="normal")
        self.pack_cancel_btn.configure(state="disabled")
        self._update_pack_status("Ready")

    def _unpack_finished(self) -> None:
        """Clean up after unpack operation."""
        self.unpack_btn.configure(state="normal")
        self.unpack_cancel_btn.configure(state="disabled")
        self._update_unpack_status("Ready")

    # -- Log Operations --------------------------------------------------------

    def _log_message(self, message: str, level: str = "info") -> None:
        """Add message to log.

        Args:
            message: Log message text.
            level: Log level ("info", "success", "warning", "error").
        """
        if not (message is not None):
            raise ValueError("message must be provided")
        timestamp = datetime.now().strftime("%H:%M:%S")

        def update_log() -> None:
            """Update log widget from thread."""
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{timestamp}] {message}\n", level)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, update_log)

        # Also log via standard logging
        getattr(logger, level if level != "success" else "info")(message)

    def _clear_log(self) -> None:
        """Clear the log display."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _save_log(self) -> None:
        """Save log to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt")],
        )
        if file_path:
            content = self.log_text.get("1.0", "end")
            safe_write_text(Path(file_path), content)
            self._log_message(f"Log saved to: {file_path}")

    def _open_log_file(self) -> None:
        """Open the log file in default text editor."""
        from .constants import LOG_FILENAME

        log_path = Path(LOG_FILENAME)
        if log_path.exists():
            try:
                if sys.platform == "win32":
                    os.startfile(log_path)  # type: ignore[attr-defined]  # noqa: S606
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(log_path)], check=False)  # noqa: S603, S607
                else:
                    subprocess.run(["xdg-open", str(log_path)], check=False)  # noqa: S603, S607
            except (OSError, subprocess.SubprocessError) as e:
                messagebox.showerror("Error", f"Could not open log file: {e}")
        else:
            messagebox.showinfo("No Log", "No log file exists yet.")

    # -- File Menu Actions -----------------------------------------------------

    def _new_package(self) -> None:
        """Reset form for new package."""
        self.pack_source_entry.delete(0, "end")
        self.pack_output_entry.delete(0, "end")
        self.pack_progress_var.set(0)
        self._update_pack_status("Ready")
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.configure(state="disabled")

    def _export_manifest(self) -> None:
        """Export current manifest."""
        file_path = filedialog.asksaveasfilename(
            title="Export Manifest",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
        )
        if file_path:
            try:
                manifest_json = self.manifest.to_json()
                safe_write_text(Path(file_path), manifest_json)
                self._log_message(f"Manifest exported to: {file_path}")
            except (OSError, ValueError) as e:
                messagebox.showerror("Error", f"Failed to export manifest: {e}")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes.

        Returns:
            Human-readable size string.
        """
        return format_size(size_bytes)
