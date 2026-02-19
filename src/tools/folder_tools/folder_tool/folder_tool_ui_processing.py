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
            logger.warning(
                f"Title is very long ({len(title)} chars), may be truncated: "
                f"{title[:MAX_TITLE_PREVIEW_LENGTH]}...",
            )

        # Validate content length for performance
        if (
            len(content) > MAX_TEXT_CONTENT_SIZE
        ):  # MAX_TEXT_CONTENT_SIZE limit for text content
            logger.warning(
                f"Content is very large ({len(content)} chars), may cause "
                "performance issues",
            )
            # Truncate content for display
            content = (
                content[:MAX_TEXT_CONTENT_SIZE]
                + "\n\n... [Content truncated due to size]"
            )

        logger.info(f"Creating text dialog: '{title}' with {len(content)} characters")

        try:
            # Create dialog window
            dialog = tk.Toplevel(self.root)
            dialog.title(title)

            # Set dialog geometry with validation
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
                    len(content.split("\n")) * LINE_HEIGHT_PIXELS
                    + DIALOG_HEIGHT_OFFSET,
                ),
            )

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
                maxundo=0,  # No undo history
                selectbackground="lightblue",
                selectforeground="black",
            )

            scrollbar = ttk.Scrollbar(
                text_frame,
                orient="vertical",
                command=text_widget.yview,
            )
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

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to insert content into text widget: {e}")
                # Fallback: show truncated content
                safe_content = (
                    content[:MAX_FALLBACK_CONTENT_SIZE]
                    + "\n\n... [Content truncated due to error]"
                )
                text_widget.insert("1.0", safe_content)
                text_widget.config(state="disabled")

            # Add close button
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            close_button = ttk.Button(
                button_frame,
                text="Close",
                command=dialog.destroy,
            )
            close_button.pack(side="right")

            # Add copy button for convenience
            def copy_to_clipboard() -> None:
                """Copy dialog content to clipboard."""
                try:
                    dialog.clipboard_clear()
                    dialog.clipboard_append(content)
                    logger.debug("Dialog content copied to clipboard")
                except (RuntimeError, OSError) as e:
                    logger.warning(f"Failed to copy to clipboard: {e}")

            copy_button = ttk.Button(
                button_frame,
                text="Copy All",
                command=copy_to_clipboard,
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
            logger.info(
                f"Text dialog created successfully: {dialog_width}x{dialog_height}",
            )

            # Wait for dialog to close
            dialog.wait_window()

        except tk.TclError as e:
            logger.error(f"Tkinter error creating text dialog: {e}")
            # Fallback to simple message box
            fallback_content = (
                content[:MAX_FALLBACK_CONTENT_SIZE] + "..."
                if len(content) > MAX_FALLBACK_CONTENT_SIZE
                else content
            )
            messagebox.showinfo(title, fallback_content)
            raise

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Failed to show text dialog: {e}")
            # Fallback to simple message box
            fallback_content = (
                content[:MAX_FALLBACK_CONTENT_SIZE] + "..."
                if len(content) > MAX_FALLBACK_CONTENT_SIZE
                else content
            )
            messagebox.showinfo(title, fallback_content)
            raise

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
            self.source_info_label.config(
                text="Warning: No accessible source folders",
                foreground="red",
            )
            return

        size_mb = total_size / (1024 * 1024)
        info_text = (
            f"Total: {total_files} files, {size_mb:.1f} MB "
            f"({accessible_folders}/{len(self.source_folders)} folders accessible)"
        )

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

    def update_progress(self, value: float, status: str = "") -> None:
        """Updates the progress bar and status.

        Args:
            value: Progress value (0-100)
            status: Status message to display
        """
        try:
            # Validate progress value
            if not isinstance(value, int | float):
                logger.warning(f"Invalid progress value type: {type(value)}")
                return
            # Clamp progress value to valid range
            clamped_value = max(0, min(100, float(value)))
            self.progress_var.set(clamped_value)

            if status:
                self.update_status(status)

            # Update UI
            self.root.update_idletasks()

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

            self.status_var.set(status)
            self.root.update_idletasks()

        except (RuntimeError, AttributeError):
            logger.exception("Error updating status")
