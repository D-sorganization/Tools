"""Dialog windows for Folder Packer Pro.

Contains the exclusion manager, about dialog, and user guide dialog.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from .constants import DEFAULT_EXCLUDE_PATTERNS, PADDING_MEDIUM, PADDING_SMALL

logger = logging.getLogger(__name__)


class DialogsMixin:
    """Mixin providing dialog windows for the application."""

    def _manage_exclusions(self) -> None:
        """Show dialog to manage exclusion patterns."""
        dialog = tk.Toplevel(self.root)  # type: ignore[attr-defined]
        dialog.title("Manage Exclusions")
        dialog.geometry("500x400")
        dialog.transient(self.root)  # type: ignore[attr-defined]
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding=PADDING_MEDIUM)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(
            main_frame,
            text="Exclusion Patterns",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(0, PADDING_MEDIUM))

        # Listbox with current patterns
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill="both", expand=True)

        listbox = tk.Listbox(list_frame, selectmode="single")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)

        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Populate with current patterns
        for pattern in sorted(self.exclude_patterns):  # type: ignore[attr-defined]
            listbox.insert("end", pattern)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(PADDING_MEDIUM, 0))

        def add_pattern() -> None:
            """Add a new exclusion pattern."""
            pattern = simpledialog.askstring(
                "Add Pattern",
                "Enter exclusion pattern:",
                parent=dialog,
            )
            if pattern:
                self.exclude_patterns.add(pattern)  # type: ignore[attr-defined]
                listbox.insert("end", pattern)

        def remove_pattern() -> None:
            """Remove selected exclusion pattern."""
            selection = listbox.curselection()
            if selection:
                pattern = listbox.get(selection[0])
                self.exclude_patterns.discard(pattern)  # type: ignore[attr-defined]
                listbox.delete(selection[0])

        def reset_patterns() -> None:
            """Reset exclusion patterns to defaults."""
            self.exclude_patterns = set(DEFAULT_EXCLUDE_PATTERNS)  # type: ignore[attr-defined]
            listbox.delete(0, "end")
            for pattern in sorted(self.exclude_patterns):  # type: ignore[attr-defined]
                listbox.insert("end", pattern)

        ttk.Button(button_frame, text="Add", command=add_pattern).pack(
            side="left", padx=(0, PADDING_SMALL)
        )
        ttk.Button(button_frame, text="Remove", command=remove_pattern).pack(
            side="left", padx=(0, PADDING_SMALL)
        )
        ttk.Button(button_frame, text="Reset", command=reset_patterns).pack(
            side="left", padx=(0, PADDING_SMALL)
        )
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side="right")

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About Folder Packer Pro",
            "Folder Packer Pro v2.0\n\n"
            "Professional Project Packaging Tool\n\n"
            "Features:\n"
            "• Pack/Unpack folders into single archives\n"
            "• AES-256 encryption\n"
            "• Multiple compression levels\n"
            "• Smart file filtering\n"
            "• Syntax highlighting preview\n"
            "• Operation logging\n\n"
            "Built with Python and Tkinter",
        )

    def _show_user_guide(self) -> None:
        """Show user guide."""
        guide = tk.Toplevel(self.root)  # type: ignore[attr-defined]
        guide.title("User Guide")
        guide.geometry("600x500")
        guide.transient(self.root)  # type: ignore[attr-defined]

        text = tk.Text(guide, wrap="word", font=("Segoe UI", 10), padx=20, pady=20)
        text.pack(fill="both", expand=True)

        guide_content = """
📦 Folder Packer Pro - User Guide

═══════════════════════════════════

1. PACKING A FOLDER
   a. Select source folder using Browse
   b. Choose output file location
   c. Set compression level
   d. Enable encryption if needed
   e. Click "Create Package"

2. UNPACKING A PACKAGE
   a. Select package file using Browse
   b. Choose destination folder
   c. Check "Package is encrypted" if applicable
   d. Enter password if encrypted
   e. Click "Extract Package"

3. FILE PREVIEW
   After scanning a folder, switch to the Preview
   tab to browse and preview files with syntax
   highlighting.

4. COMPRESSION OPTIONS
   • None: No compression (fastest)
   • Fast: Quick compression
   • Balanced: Good compression/speed ratio
   • Best: Maximum compression (slowest)

5. ENCRYPTION
   Uses AES-256 encryption with PBKDF2 key
   derivation. Make sure to remember your password!

6. EXCLUSION PATTERNS
   Use Tools > Manage Exclusions to customize
   which files and folders are excluded from packing.

7. OPERATION LOG
   All operations are logged in the Log tab and
   saved to folder_packer_pro.log.
"""
        text.insert("1.0", guide_content)
        text.configure(state="disabled")

        ttk.Button(guide, text="Close", command=guide.destroy).pack(
            pady=PADDING_MEDIUM,
        )
