"""UI tab creation for Folder Packer Pro.

Contains the tab creation methods for the pack, unpack, preview,
and log tabs. Extracted from the monolithic FolderPackerPro class
for maintainability.
"""

from __future__ import annotations

import logging
import re
import tkinter as tk
from tkinter import scrolledtext, ttk

from .constants import CODE_EXTENSIONS, PADDING_MEDIUM, PADDING_SMALL

logger = logging.getLogger(__name__)


class PackTabMixin:
    """Mixin providing the Pack tab creation."""

    def _create_pack_tab(self) -> None:
        """Create pack operation tab."""
        tab = ttk.Frame(self.notebook)  # type: ignore[attr-defined]
        self.notebook.add(tab, text="  Pack  ")  # type: ignore[attr-defined]

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

        self._create_pack_left_column(left_frame)
        self._create_pack_options(right_frame)

    def _create_pack_left_column(self, parent: ttk.Frame) -> None:
        """Create the left column of the pack tab (source, output, stats, progress)."""
        assert parent is not None, "parent must be provided"
        header_label = ttk.Label(
            parent,
            text="📦 Folder Packer Pro",
            font=("Segoe UI", 18, "bold"),
        )
        header_label.pack(pady=(0, PADDING_MEDIUM))

        self._create_pack_source_section(parent)
        self._create_pack_output_section(parent)
        self._create_pack_stats_section(parent)
        self._create_pack_progress_section(parent)

    def _create_pack_source_section(self, parent: ttk.Frame) -> None:
        """Create the source folder input section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(parent, text="Source Folder", padding=PADDING_MEDIUM)
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill="x")

        self.pack_source_entry = ttk.Entry(entry_frame)  # type: ignore
        self.pack_source_entry.pack(  # type: ignore
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(
            entry_frame,
            text="Browse",
            command=self._browse_pack_source,  # type: ignore[attr-defined]
        ).pack(side="right")

    def _create_pack_output_section(self, parent: ttk.Frame) -> None:
        """Create the output package file input section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(
            parent, text="Output Package File", padding=PADDING_MEDIUM
        )
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill="x")

        self.pack_output_entry = ttk.Entry(entry_frame)  # type: ignore
        self.pack_output_entry.pack(  # type: ignore
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(
            entry_frame,
            text="Browse",
            command=self._browse_pack_output,  # type: ignore[attr-defined]
        ).pack(side="right")

    def _create_pack_stats_section(self, parent: ttk.Frame) -> None:
        """Create the project statistics display section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(
            parent, text="Project Statistics", padding=PADDING_MEDIUM
        )
        frame.pack(fill="both", expand=True, pady=(0, PADDING_MEDIUM))

        self.stats_text = scrolledtext.ScrolledText(  # type: ignore
            frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.stats_text.pack(fill="both", expand=True)  # type: ignore

        ttk.Button(
            frame,
            text="🔄 Scan Folder",
            command=self._scan_folder,  # type: ignore[attr-defined]
        ).pack(pady=(PADDING_SMALL, 0))

    def _create_pack_progress_section(self, parent: ttk.Frame) -> None:
        """Create the progress bar and status label section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(parent, text="Progress", padding=PADDING_MEDIUM)
        frame.pack(fill="x")

        self.pack_progress_var = tk.DoubleVar()  # type: ignore
        self.pack_progress_bar = ttk.Progressbar(  # type: ignore
            frame,
            variable=self.pack_progress_var,  # type: ignore
            maximum=100,
            mode="determinate",
        )
        self.pack_progress_bar.pack(fill="x", pady=(0, PADDING_SMALL))  # type: ignore

        self.pack_status_label = ttk.Label(  # type: ignore
            frame,
            text="Ready",
            font=("Segoe UI", 9),
        )
        self.pack_status_label.pack(fill="x")  # type: ignore

    def _create_pack_options(self, right_frame: ttk.Frame) -> None:  # noqa: PLR0915
        """Create pack options in the right column.

        Args:
            right_frame: Parent frame for options.
        """
        # Compression options
        assert right_frame is not None, "right_frame must be provided"
        compression_frame = ttk.LabelFrame(
            right_frame,
            text="Compression Options",
            padding=PADDING_MEDIUM,
        )
        compression_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        ttk.Label(compression_frame, text="Compression Level:").pack(anchor="w")

        self.compression_var = tk.StringVar(value="balanced")  # type: ignore
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
                variable=self.compression_var,  # type: ignore
                value=value,
            ).pack(anchor="w", pady=2)

        # Security options
        security_frame = ttk.LabelFrame(
            right_frame,
            text="Security Options",
            padding=PADDING_MEDIUM,
        )
        security_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.encrypt_var = tk.BooleanVar()  # type: ignore
        ttk.Checkbutton(
            security_frame,
            text="Enable AES-256 Encryption",
            variable=self.encrypt_var,  # type: ignore
            command=self._on_encrypt_toggle,  # type: ignore[attr-defined]
        ).pack(anchor="w")

        self.password_frame = ttk.Frame(security_frame)  # type: ignore
        self.password_frame.pack(fill="x", pady=(PADDING_SMALL, 0))  # type: ignore

        ttk.Label(self.password_frame, text="Password:").pack(anchor="w")  # type: ignore
        self.pack_password_entry = ttk.Entry(self.password_frame, show="*")  # type: ignore
        self.pack_password_entry.pack(fill="x", pady=(PADDING_SMALL, 0))  # type: ignore
        self.pack_password_entry.configure(state="disabled")  # type: ignore

        ttk.Label(self.password_frame, text="Confirm:").pack(  # type: ignore
            anchor="w",
            pady=(PADDING_SMALL, 0),
        )
        self.pack_password_confirm = ttk.Entry(self.password_frame, show="*")  # type: ignore
        self.pack_password_confirm.pack(fill="x", pady=(PADDING_SMALL, 0))  # type: ignore
        self.pack_password_confirm.configure(state="disabled")  # type: ignore

        # Advanced options
        advanced_frame = ttk.LabelFrame(
            right_frame,
            text="Advanced Options",
            padding=PADDING_MEDIUM,
        )
        advanced_frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.include_git_var = tk.BooleanVar()  # type: ignore
        self.create_manifest_var = tk.BooleanVar(value=True)  # type: ignore
        self.verify_pack_var = tk.BooleanVar(value=True)  # type: ignore

        ttk.Checkbutton(
            advanced_frame,
            text="Include .git folder (preserve repository)",
            variable=self.include_git_var,  # type: ignore
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Create manifest file",
            variable=self.create_manifest_var,  # type: ignore
        ).pack(anchor="w")
        ttk.Checkbutton(
            advanced_frame,
            text="Verify package after creation",
            variable=self.verify_pack_var,  # type: ignore
        ).pack(anchor="w")

        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill="x", pady=(PADDING_MEDIUM, 0))

        self.pack_btn = ttk.Button(  # type: ignore
            action_frame,
            text="📦 Create Package",
            command=self._start_pack,  # type: ignore[attr-defined]
            style="Accent.TButton",
        )
        self.pack_btn.pack(  # type: ignore
            side="left", fill="x", expand=True, padx=(0, PADDING_SMALL)
        )

        self.pack_cancel_btn = ttk.Button(  # type: ignore
            action_frame,
            text="⏹️ Cancel",
            command=self._cancel_operation,  # type: ignore[attr-defined]
            state="disabled",
        )
        self.pack_cancel_btn.pack(side="right", fill="x", expand=True)  # type: ignore


class UnpackTabMixin:
    """Mixin providing the Unpack tab creation."""

    def _create_unpack_tab(self) -> None:
        """Create unpack operation tab."""
        tab = ttk.Frame(self.notebook)  # type: ignore[attr-defined]
        self.notebook.add(tab, text="  Unpack  ")  # type: ignore[attr-defined]

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        header_label = ttk.Label(
            main_frame,
            text="📂 Unpack Package",
            font=("Segoe UI", 18, "bold"),
        )
        header_label.pack(pady=(0, PADDING_MEDIUM))

        self._create_unpack_source_section(main_frame)
        self._create_unpack_dest_section(main_frame)
        self._create_unpack_decrypt_section(main_frame)
        self._create_unpack_info_section(main_frame)
        self._create_unpack_progress_section(main_frame)
        self._create_unpack_action_buttons(main_frame)

    def _create_unpack_source_section(self, parent: ttk.Frame) -> None:
        """Create the package file source input section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(parent, text="Package File", padding=PADDING_MEDIUM)
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill="x")

        self.unpack_source_entry = ttk.Entry(entry_frame)  # type: ignore
        self.unpack_source_entry.pack(  # type: ignore
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(
            entry_frame,
            text="Browse",
            command=self._browse_unpack_source,  # type: ignore[attr-defined]
        ).pack(side="right")

    def _create_unpack_dest_section(self, parent: ttk.Frame) -> None:
        """Create the destination folder input section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(
            parent, text="Destination Folder", padding=PADDING_MEDIUM
        )
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill="x")

        self.unpack_dest_entry = ttk.Entry(entry_frame)  # type: ignore
        self.unpack_dest_entry.pack(  # type: ignore
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )
        ttk.Button(
            entry_frame,
            text="Browse",
            command=self._browse_unpack_dest,  # type: ignore[attr-defined]
        ).pack(side="right")

    def _create_unpack_decrypt_section(self, parent: ttk.Frame) -> None:
        """Create the decryption controls section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(parent, text="Decryption", padding=PADDING_MEDIUM)
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.encrypted_var = tk.BooleanVar()  # type: ignore
        ttk.Checkbutton(
            frame,
            text="Package is encrypted",
            variable=self.encrypted_var,  # type: ignore
            command=self._on_encrypted_toggle,  # type: ignore[attr-defined]
        ).pack(anchor="w")

        self.decrypt_password_frame = ttk.Frame(frame)  # type: ignore
        self.decrypt_password_frame.pack(fill="x", pady=(PADDING_SMALL, 0))  # type: ignore

        ttk.Label(self.decrypt_password_frame, text="Password:").pack(anchor="w")  # type: ignore
        self.unpack_password_entry = ttk.Entry(  # type: ignore
            self.decrypt_password_frame,
            show="*",  # type: ignore
        )
        self.unpack_password_entry.pack(fill="x", pady=(PADDING_SMALL, 0))  # type: ignore
        self.unpack_password_entry.configure(state="disabled")  # type: ignore

    def _create_unpack_info_section(self, parent: ttk.Frame) -> None:
        """Create the package information display section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(
            parent, text="Package Information", padding=PADDING_MEDIUM
        )
        frame.pack(fill="both", expand=True, pady=(0, PADDING_MEDIUM))

        self.package_info_text = scrolledtext.ScrolledText(  # type: ignore
            frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.package_info_text.pack(fill="both", expand=True)  # type: ignore

        ttk.Button(
            frame,
            text="🔍 Inspect Package",
            command=self._inspect_package,  # type: ignore[attr-defined]
        ).pack(pady=(PADDING_SMALL, 0))

    def _create_unpack_progress_section(self, parent: ttk.Frame) -> None:
        """Create the progress bar and status label section."""
        assert parent is not None, "parent must be provided"
        frame = ttk.LabelFrame(parent, text="Progress", padding=PADDING_MEDIUM)
        frame.pack(fill="x", pady=(0, PADDING_MEDIUM))

        self.unpack_progress_var = tk.DoubleVar()  # type: ignore
        self.unpack_progress_bar = ttk.Progressbar(  # type: ignore
            frame,
            variable=self.unpack_progress_var,  # type: ignore
            maximum=100,
            mode="determinate",
        )
        self.unpack_progress_bar.pack(fill="x", pady=(0, PADDING_SMALL))  # type: ignore

        self.unpack_status_label = ttk.Label(  # type: ignore
            frame,
            text="Ready",
            font=("Segoe UI", 9),
        )
        self.unpack_status_label.pack(fill="x")  # type: ignore

    def _create_unpack_action_buttons(self, parent: ttk.Frame) -> None:
        """Create the extract and cancel action buttons."""
        assert parent is not None, "parent must be provided"
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill="x")

        self.unpack_btn = ttk.Button(  # type: ignore
            action_frame,
            text="📂 Extract Package",
            command=self._start_unpack,  # type: ignore[attr-defined]
            style="Accent.TButton",
        )
        self.unpack_btn.pack(  # type: ignore
            side="left",
            fill="x",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        self.unpack_cancel_btn = ttk.Button(  # type: ignore
            action_frame,
            text="⏹️ Cancel",
            command=self._cancel_operation,  # type: ignore[attr-defined]
            state="disabled",
        )
        self.unpack_cancel_btn.pack(side="right", fill="x", expand=True)  # type: ignore


class PreviewTabMixin:
    """Mixin providing the Preview tab creation."""

    def _create_preview_tab(self) -> None:  # noqa: PLR0915
        """Create file preview tab."""
        tab = ttk.Frame(self.notebook)  # type: ignore[attr-defined]
        self.notebook.add(tab, text="  Preview  ")  # type: ignore[attr-defined]

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
        tree_label_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, PADDING_SMALL),
        )

        tree_frame = ttk.Frame(tree_label_frame)
        tree_frame.pack(fill="both", expand=True)

        self.preview_tree = ttk.Treeview(  # type: ignore
            tree_frame,
            columns=("size", "type", "modified"),
            selectmode="browse",
        )
        self.preview_tree.heading("#0", text="File Path", anchor="w")  # type: ignore
        self.preview_tree.heading("size", text="Size", anchor="w")  # type: ignore
        self.preview_tree.heading("type", text="Type", anchor="w")  # type: ignore
        self.preview_tree.heading("modified", text="Modified", anchor="w")  # type: ignore

        self.preview_tree.column("#0", width=300)  # type: ignore
        self.preview_tree.column("size", width=80)  # type: ignore
        self.preview_tree.column("type", width=80)  # type: ignore
        self.preview_tree.column("modified", width=120)  # type: ignore

        tree_scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.preview_tree.yview,  # type: ignore
        )
        self.preview_tree.configure(yscrollcommand=tree_scroll.set)  # type: ignore

        self.preview_tree.pack(side="left", fill="both", expand=True)  # type: ignore
        tree_scroll.pack(side="right", fill="y")

        self.preview_tree.bind(  # type: ignore
            "<<TreeviewSelect>>",
            self._on_file_select,  # type: ignore[attr-defined]
        )

        # Preview pane
        preview_label_frame = ttk.LabelFrame(
            main_frame,
            text="File Content",
            padding=PADDING_SMALL,
        )
        preview_label_frame.pack(side="right", fill="both", expand=True)

        self.preview_text = scrolledtext.ScrolledText(  # type: ignore
            preview_label_frame,
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
        )
        self.preview_text.pack(fill="both", expand=True)  # type: ignore

        # Configure syntax highlighting tags
        self.preview_text.tag_configure("keyword", foreground="#569cd6")  # type: ignore
        self.preview_text.tag_configure("string", foreground="#ce9178")  # type: ignore
        self.preview_text.tag_configure("comment", foreground="#6a9955")  # type: ignore
        self.preview_text.tag_configure("number", foreground="#b5cea8")  # type: ignore

    def _insert_with_highlighting(self, content: str, file_ext: str) -> None:
        """Insert text with basic syntax highlighting.

        Args:
            content: File content to display.
            file_ext: File extension for syntax detection.
        """
        assert content is not None, "content must be provided"
        keywords = {
            "def",
            "class",
            "import",
            "from",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "return",
            "try",
            "except",
            "with",
            "as",
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
            "in",
            "is",
        }

        lines = content.splitlines()
        for i, line in enumerate(lines):
            if i >= 1000:
                self.preview_text.insert("end", "\n... (truncated)")  # type: ignore
                break

            if file_ext in CODE_EXTENSIONS:
                if line.strip().startswith("#"):
                    self.preview_text.insert("end", line + "\n", "comment")  # type: ignore
                else:
                    words = re.split(r"(\s+)", line)
                    for word in words:
                        if word in keywords:
                            self.preview_text.insert("end", word, "keyword")  # type: ignore
                        elif word.startswith('"') or word.startswith("'"):
                            self.preview_text.insert("end", word, "string")  # type: ignore
                        elif word.isdigit():
                            self.preview_text.insert("end", word, "number")  # type: ignore
                        else:
                            self.preview_text.insert("end", word)  # type: ignore
                    self.preview_text.insert("end", "\n")  # type: ignore
            else:
                self.preview_text.insert("end", line + "\n")  # type: ignore


class LogTabMixin:
    """Mixin providing the Log tab creation."""

    def _create_log_tab(self) -> None:
        """Create operation log tab."""
        tab = ttk.Frame(self.notebook)  # type: ignore[attr-defined]
        self.notebook.add(tab, text="  Log  ")  # type: ignore[attr-defined]

        main_frame = ttk.Frame(tab, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill="x", pady=(0, PADDING_SMALL))

        ttk.Label(
            toolbar,
            text="Operation Log",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        ttk.Button(
            toolbar,
            text="Clear",
            command=self._clear_log,  # type: ignore[attr-defined]
        ).pack(side="right", padx=(PADDING_SMALL, 0))

        ttk.Button(
            toolbar,
            text="Save Log",
            command=self._save_log,  # type: ignore[attr-defined]
        ).pack(side="right")

        self.log_text = scrolledtext.ScrolledText(  # type: ignore
            main_frame,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)  # type: ignore

        # Configure log level tags
        self.log_text.tag_configure("info", foreground="#ffffff")  # type: ignore
        self.log_text.tag_configure("success", foreground="#28a745")  # type: ignore
        self.log_text.tag_configure("warning", foreground="#ffc107")  # type: ignore
        self.log_text.tag_configure("error", foreground="#dc3545")  # type: ignore
