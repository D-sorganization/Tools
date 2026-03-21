"""UIProcessingMixin -- UI dialogs, progress, status, threading for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from Folders_Tool_r0 import (
    CHARS_PER_DIALOG_LINE,
    DIALOG_HEIGHT_OFFSET,
    DIALOG_WIDTH_OFFSET,
    LINE_HEIGHT_PIXELS,
    MAX_DIALOG_HEIGHT,
    MAX_DIALOG_WIDTH,
    MAX_FALLBACK_CONTENT_SIZE,
    MAX_STATUS_LENGTH,
    MAX_TEXT_CONTENT_SIZE,
    MAX_TITLE_LENGTH,
    MAX_TITLE_PREVIEW_LENGTH,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
)

logger = logging.getLogger(__name__)


class UIProcessingMixin:
    """UI dialog, progress, and threading methods for FolderProcessorApp.

    Expects the host class to provide:
    - self.root: tk.Tk
    - self.source_folders: list[str]
    - self.cancel_operation: bool
    - self.run_button: tk.Button
    - self.cancel_button: tk.Button
    - self.progress_var: tk.DoubleVar
    - self.status_var: tk.StringVar
    - self.source_info_label: ttk.Label
    - self.run_processing() -> None
    """

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
        assert title is not None, "title must be provided"
        content = self._validate_dialog_inputs(title, content)
        logger.info(f"Creating text dialog: '{title}' with {len(content)} characters")

        try:
            dialog, dialog_width, dialog_height = self._create_dialog_window(
                title, content
            )
            self._create_text_area(dialog, content)
            self._create_dialog_buttons(dialog, content)
            self._finalize_dialog(dialog, dialog_width, dialog_height)
        except tk.TclError as e:
            logger.error(f"Tkinter error creating text dialog: {e}")
            self._show_fallback_messagebox(title, content)
            raise
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Failed to show text dialog: {e}")
            self._show_fallback_messagebox(title, content)
            raise

    @staticmethod
    def _validate_dialog_inputs(title: str, content: str) -> str:
        """Validate and sanitize dialog title and content.

        Returns:
            Possibly-truncated content string.
        """
        if not title or not isinstance(title, str):
            raise ValueError(f"Title must be non-empty string, got {type(title)}")
        if not content or not isinstance(content, str):
            raise ValueError(f"Content must be non-empty string, got {type(content)}")
        if len(title.strip()) == 0:
            raise ValueError("Title cannot be empty or whitespace only")
        if len(content.strip()) == 0:
            raise ValueError("Content cannot be empty or whitespace only")
        if len(title) > MAX_TITLE_LENGTH:
            logger.warning(
                f"Title is very long ({len(title)} chars), may be truncated: "
                f"{title[:MAX_TITLE_PREVIEW_LENGTH]}...",
            )
        if len(content) > MAX_TEXT_CONTENT_SIZE:
            logger.warning(
                f"Content is very large ({len(content)} chars), may cause "
                "performance issues",
            )
            content = (
                content[:MAX_TEXT_CONTENT_SIZE]
                + "\n\n... [Content truncated due to size]"
            )
        return content

    def _create_dialog_window(
        self, title: str, content: str
    ) -> tuple[tk.Toplevel, int, int]:
        """Create and configure the dialog window, returning it with dimensions."""
        assert title is not None, "title must be provided"
        dialog = tk.Toplevel(self.root)  # type: ignore[attr-defined]
        dialog.title(title)

        dialog_width = min(
            MAX_DIALOG_WIDTH,
            max(
                MIN_DIALOG_WIDTH,
                len(content) // CHARS_PER_DIALOG_LINE + DIALOG_WIDTH_OFFSET,
            ),
        )
        dialog_height = min(
            MAX_DIALOG_HEIGHT,
            max(
                MIN_DIALOG_HEIGHT,
                len(content.split("\n")) * LINE_HEIGHT_PIXELS + DIALOG_HEIGHT_OFFSET,
            ),
        )

        dialog.geometry(f"{dialog_width}x{dialog_height}")
        dialog.minsize(MIN_DIALOG_WIDTH, MIN_DIALOG_HEIGHT)
        dialog.transient(self.root)  # type: ignore[attr-defined]
        dialog.grab_set()

        return dialog, dialog_width, dialog_height

    @staticmethod
    def _create_text_area(dialog: tk.Toplevel, content: str) -> tk.Text:
        """Create the scrollable text area and insert content."""
        assert dialog is not None, "dialog must be provided"
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            undo=False,
            maxundo=0,
            selectbackground="lightblue",
            selectforeground="black",
        )
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=text_widget.yview
        )
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        try:
            text_widget.insert("1.0", content)
            text_widget.config(state="disabled")
            text_widget.mark_set("insert", "1.0")
            text_widget.see("1.0")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to insert content into text widget: {e}")
            safe_content = (
                content[:MAX_FALLBACK_CONTENT_SIZE]
                + "\n\n... [Content truncated due to error]"
            )
            text_widget.insert("1.0", safe_content)
            text_widget.config(state="disabled")

        return text_widget

    @staticmethod
    def _create_dialog_buttons(dialog: tk.Toplevel, content: str) -> ttk.Button:
        """Create Close and Copy All buttons. Returns close_button for focus."""
        assert dialog is not None, "dialog must be provided"
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        close_button = ttk.Button(button_frame, text="Close", command=dialog.destroy)
        close_button.pack(side="right")

        def copy_to_clipboard() -> None:
            try:
                dialog.clipboard_clear()
                dialog.clipboard_append(content)
                logger.debug("Dialog content copied to clipboard")
            except (RuntimeError, OSError) as e:
                logger.warning(f"Failed to copy to clipboard: {e}")

        copy_button = ttk.Button(
            button_frame, text="Copy All", command=copy_to_clipboard
        )
        copy_button.pack(side="right", padx=(0, 5))

        return close_button

    @staticmethod
    def _finalize_dialog(
        dialog: tk.Toplevel, dialog_width: int, dialog_height: int
    ) -> None:
        """Set focus, bind keys, and wait for dialog to close."""
        assert dialog is not None, "dialog must be provided"
        dialog.focus_set()
        dialog.bind("<Escape>", lambda event: dialog.destroy())

        logger.info(
            f"Text dialog created successfully: {dialog_width}x{dialog_height}",
        )
        dialog.wait_window()

    @staticmethod
    def _show_fallback_messagebox(title: str, content: str) -> None:
        """Show a simple message box as fallback when dialog creation fails."""
        assert title is not None, "title must be provided"
        fallback_content = (
            content[:MAX_FALLBACK_CONTENT_SIZE] + "..."
            if len(content) > MAX_FALLBACK_CONTENT_SIZE
            else content
        )
        messagebox.showinfo(title, fallback_content)

    def update_source_info(self) -> None:
        """Updates the source folder information display."""
        if not self.source_folders:  # type: ignore[attr-defined]
            self.source_info_label.config(text="")  # type: ignore[attr-defined]
            return

        total_size = 0
        total_files = 0
        accessible_folders = 0

        for folder in self.source_folders:  # type: ignore[attr-defined]
            try:
                if not Path(folder).exists():
                    logger.warning(f"Source folder no longer exists: {folder}")
                    continue

                if not os.access(folder, os.R_OK):
                    logger.warning(f"Cannot access source folder: {folder}")
                    continue

                accessible_folders += 1

                for root, _dirs, files in os.walk(folder):
                    for file in files:
                        try:
                            file_path = Path(root) / file
                            if Path(file_path).exists() and os.access(
                                file_path,
                                os.R_OK,
                            ):
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
            self.source_info_label.config(  # type: ignore[attr-defined]
                text="Warning: No accessible source folders",
                foreground="red",
            )
            return

        size_mb = total_size / (1024 * 1024)
        info_text = (
            f"Total: {total_files} files, {size_mb:.1f} MB "
            f"({accessible_folders}/{len(self.source_folders)} folders accessible)"  # type: ignore[attr-defined]
        )

        # Set color based on accessibility
        if accessible_folders < len(self.source_folders):  # type: ignore[attr-defined]
            self.source_info_label.config(text=info_text, foreground="orange")  # type: ignore[attr-defined]
        else:
            self.source_info_label.config(text=info_text, foreground="blue")  # type: ignore[attr-defined]

    def run_processing_threaded(self) -> None:
        """Runs the processing in a separate thread to keep UI responsive."""
        self.cancel_operation = False
        self.run_button.config(state=tk.DISABLED)  # type: ignore[attr-defined]
        self.cancel_button.config(state=tk.NORMAL)  # type: ignore[attr-defined]

        def processing_thread() -> None:
            """Run the processing operation in a separate thread."""
            try:
                self.run_processing()  # type: ignore[attr-defined]
            finally:
                self.root.after(0, self.processing_complete)  # type: ignore[attr-defined]

        thread = threading.Thread(target=processing_thread, daemon=True)
        thread.start()

    def cancel_processing(self) -> None:
        """Cancels the current operation."""
        self.cancel_operation = True
        self.update_status("Cancelling operation...")

    def processing_complete(self) -> None:
        """Called when processing is complete to reset UI state."""
        self.run_button.config(state=tk.NORMAL)  # type: ignore[attr-defined]
        self.cancel_button.config(state=tk.DISABLED)  # type: ignore[attr-defined]
        self.progress_var.set(0)  # type: ignore[attr-defined]
        self.update_status("Ready")

    def update_progress(self, value: float, status: str = "") -> None:
        """Updates the progress bar and status.

        Args:
            value: Progress value (0-100)
            status: Status message to display
        """
        try:
            # Validate progress value
            if not isinstance(value, int | float):
                logger.warning(f"Invalid progress value type: {type(value)}")  # type: ignore[unreachable]
                return
            # Clamp progress value to valid range
            clamped_value = max(0, min(100, float(value)))
            self.progress_var.set(clamped_value)  # type: ignore[attr-defined]

            if status:
                self.update_status(status)

            # Update UI
            self.root.update_idletasks()  # type: ignore[attr-defined]

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            logger.exception("Error updating progress")

    def update_status(self, status: str) -> None:
        """Updates the status label.

        Args:
            status: Status message to display
        """
        try:
            # Limit status length to prevent UI issues
            max_length = MAX_STATUS_LENGTH
            if len(status) > max_length:
                status = status[: max_length - 3] + "..."

            self.status_var.set(status)  # type: ignore[attr-defined]
            self.root.update_idletasks()  # type: ignore[attr-defined]

        except (RuntimeError, AttributeError):
            logger.exception("Error updating status")
