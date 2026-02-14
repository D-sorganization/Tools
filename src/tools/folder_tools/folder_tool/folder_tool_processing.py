"""ProcessingMixin -- Processing and reporting methods for FolderProcessorApp."""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
import zipfile
from collections import defaultdict
from datetime import datetime
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
    MAX_FILE_SIZE_MB,
    MAX_STATUS_LENGTH,
    MAX_TEXT_CONTENT_SIZE,
    MAX_TITLE_LENGTH,
    MAX_TITLE_PREVIEW_LENGTH,
    MAX_UI_UPDATE_FREQUENCY,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    MIN_FILE_SIZE_BYTES,
    PROGRESS_START_ZIP,
    PROGRESS_ZIP_PERCENT,
)

logger = logging.getLogger(__name__)


class ProcessingMixin:
    """Processing and reporting methods for FolderProcessorApp."""

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
            raise ValueError(
                f"Source folders must be a list, got {type(self.source_folders)}",
            )

        # Validate each source folder
        valid_source_folders = []
        for folder in self.source_folders:
            if not folder or not isinstance(folder, str):
                logger.warning(f"Invalid source folder: {folder}")
                continue
            if not Path(folder).exists():
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
        file_types: dict[str, int] = defaultdict(int)
        size_by_type: dict[str, int] = defaultdict(int)
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
                for root, _dirs, files in os.walk(folder):
                    if self.cancel_operation:
                        break  # type: ignore[unreachable]

                    for file in files:
                        if self.cancel_operation:
                            break  # type: ignore[unreachable]

                        file_path = Path(root) / file
                        try:
                            # Validate file exists and is accessible
                            if not Path(file_path).exists():
                                folder_errors += 1
                                continue
                            if not os.access(file_path, os.R_OK):
                                folder_errors += 1
                                continue

                            file_size = os.path.getsize(file_path)
                            file_ext = Path(file).suffix.lower() or "no_extension"

                            # Validate file size
                            if file_size < MIN_FILE_SIZE_BYTES:
                                logger.debug(
                                    f"File below minimum size: {file_path} "
                                    f"({file_size} bytes)",
                                )
                                continue
                            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                                logger.warning(
                                    f"File exceeds maximum size: {file_path} "
                                    f"({file_size / (1024 * 1024):.1f} MB)",
                                )

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
                    report.append(
                        f"  Files: {folder_files}, "
                        f"Size: {folder_size / (1024 * 1024):.1f} MB, "
                        f"Errors: {folder_errors}",
                    )
                    analysis_errors.append(
                        f"Folder {folder}: {folder_errors} access errors",
                    )
                else:
                    report.append(
                        f"  Files: {folder_files}, "
                        f"Size: {folder_size / (1024 * 1024):.1f} MB",
                    )

            except (OSError, PermissionError) as e:
                error_msg = f"Error accessing folder {folder}: {e}"
                report.append(f"  ERROR: {error_msg}")
                analysis_errors.append(error_msg)
                logger.error(error_msg)
                continue

        # Add summary statistics
        report.extend(
            [
                "",
                f"TOTAL FILES: {total_files}",
                f"TOTAL SIZE: {total_size / (1024 * 1024):.1f} MB",
                "",
                "FILE TYPES:",
            ],
        )

        # Sort file types by count
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            size_mb = size_by_type[ext] / (1024 * 1024)
            report.append(f"  {ext}: {count} files, {size_mb:.1f} MB")

        # Add largest files section
        report.extend(["", "LARGEST FILES:"])
        for file_path, size in sorted(largest_files, key=lambda x: x[1], reverse=True):
            size_mb = size / (1024 * 1024)
            report.append(f"  {Path(file_path).name}: {size_mb:.1f} MB")

        # Add error summary if any occurred
        if analysis_errors:
            report.extend(["", "ANALYSIS ERRORS:", *analysis_errors])

        # Add analysis metadata
        report.extend(
            [
                "",
                "ANALYSIS METADATA:",
                f"  Source folders processed: {len(valid_source_folders)}",
                f"  Total folders analyzed: {len(valid_source_folders)}",
                f"  Analysis timestamp: {datetime.now()}",
                f"  File size limits: {MIN_FILE_SIZE_BYTES} bytes - "
                f"{MAX_FILE_SIZE_MB} MB",
            ],
        )

        logger.info(
            f"Analysis completed: {total_files} files, "
            f"{total_size / (1024 * 1024):.1f} MB",
        )
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
            except OSError as e:
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
            except OSError as e:
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
            except OSError as e:
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
        except OSError as e:
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
            except OSError as e:
                final_summary += f"\n\n--- Deduplication FAILED: {e}"

        # Create output ZIP if requested
        if self.zip_output_var.get() and not self.cancel_operation:
            try:
                self.update_progress(85, "Creating ZIP archive...")
                zip_path = self.create_output_zip()
                final_summary += (
                    f"\n\n--- ZIP Archive Created ---\nLocation: {zip_path}"
                )
            except OSError as e:
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
            PermissionError: If insufficient permissions to read destination or
                write ZIP
            OSError: If file system operations fail during ZIP creation
            Exception: If ZIP creation fails for other reasons
        """
        # Input validation
        if not self.dest_folder:
            raise ValueError("Destination folder not set")
        if not isinstance(self.dest_folder, str):
            raise ValueError(
                f"Destination folder must be a string, got {type(self.dest_folder)}",
            )

        dest_path_obj = Path(self.dest_folder)

        # Validate destination folder exists and is accessible
        if not dest_path_obj.exists():
            raise FileNotFoundError(
                f"Destination folder does not exist: {self.dest_folder}",
            )
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
            raise PermissionError(
                f"Cannot access destination folder contents: {self.dest_folder} - {e}",
            ) from e

        # Generate ZIP filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"processed_files_{timestamp}.zip"

        # Create ZIP in parent directory of destination
        try:
            zip_path = dest_path_obj.parent / zip_filename
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            raise ValueError(f"Cannot determine ZIP location: {e}") from e

        # Check if ZIP file already exists and generate unique name
        if zip_path.exists():
            zip_path = Path(self._get_unique_path(str(zip_path)))

        logger.info(f"Creating ZIP archive: {zip_path}")

        try:
            # Count total files and size for progress tracking
            total_files = 0
            total_size = 0

            for root, _dirs, files in os.walk(self.dest_folder):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        if Path(file_path).exists() and os.access(file_path, os.R_OK):
                            total_files += 1
                            total_size += os.path.getsize(file_path)
                    except (OSError, PermissionError):
                        continue

            if total_files == 0:
                raise ValueError("No accessible files found in destination folder")

            logger.info(
                f"ZIP will contain {total_files} files, "
                f"{total_size / (1024 * 1024):.1f} MB",
            )

            # Create ZIP archive
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                processed_files = 0
                processed_size = 0
                failed_files = 0

                for root, _dirs, files in os.walk(self.dest_folder):
                    for file in files:
                        if self.cancel_operation:
                            raise Exception("ZIP creation cancelled by user")

                        file_path = Path(root) / file

                        # Validate file before adding to ZIP
                        try:
                            if not Path(file_path).exists():
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
                                progress = (
                                    PROGRESS_START_ZIP
                                    + (processed_files / total_files)
                                    * PROGRESS_ZIP_PERCENT
                                )
                                self.update_progress(
                                    progress,
                                    f"Added {processed_files}/{total_files} files "
                                    "to ZIP",
                                )

                        except (IOError, PermissionError, OSError) as e:
                            failed_files += 1
                            logger.warning(
                                f"Failed to add file to ZIP: {file_path} - {e}",
                            )
                            continue

                # Verify ZIP was created successfully
                if not zip_path.exists():
                    raise Exception("ZIP file was not created")

                # Verify ZIP size is reasonable
                try:
                    zip_size = zip_path.stat().st_size
                    if zip_size == 0:
                        raise Exception("ZIP file is empty")
                    logger.info(
                        f"ZIP archive created: {zip_path} ({processed_files} files, "
                        f"{processed_size / (1024 * 1024):.1f} MB, "
                        f"ZIP size: {zip_size / (1024 * 1024):.1f} MB)",
                    )
                except OSError as e:
                    logger.warning(f"Cannot verify ZIP file size: {e}")

                # Final summary
                if failed_files > 0:
                    logger.warning(
                        f"ZIP creation completed with {failed_files} failed files",
                    )
                else:
                    logger.info("ZIP creation completed successfully")

        except (IOError, PermissionError, OSError) as e:
            # Cleanup failed ZIP file
            if zip_path.exists():
                try:
                    zip_path.unlink()
                    logger.info(f"Cleaned up failed ZIP file: {zip_path}")
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup failed ZIP file: {zip_path} - "
                        f"{cleanup_error}",
                    )

            logger.error(f"Failed to create ZIP archive: {e}")
            raise Exception(f"Failed to create ZIP archive: {e}") from e

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

    def validate_inputs(self, check_destination: bool = True) -> bool:
        """Validate user inputs before processing.

        Args:
            check_destination: Whether to validate destination folder selection [bool]
                - defaults to True

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
                            f"Extension '{ext}' should start with a dot "
                            "(e.g., '.txt').",
                        )
                        return False
            except (KeyError, ValueError, TypeError):
                messagebox.showerror(
                    "Error",
                    "Invalid extension filter format. Use comma-separated values "
                    "like '.txt,.pdf'.",
                )
                return False

        return True

    def validate_application_state(self) -> dict[str, bool]:
        """Validates the current application state and returns validation results.

        Returns:
            Dictionary mapping validation checks to their results [dict]
                - True if valid, False if invalid

        Example:
            {
                'source_folders_exist': True,
                'destination_writable': False,
                'constants_valid': True
            }
        """
        validation_results = {}

        # Check source folders
        validation_results["source_folders_exist"] = (
            all(Path(folder).exists() for folder in self.source_folders)
            if self.source_folders
            else True
        )

        validation_results["source_folders_readable"] = (
            all(os.access(folder, os.R_OK) for folder in self.source_folders)
            if self.source_folders
            else True
        )

        # Check destination folder
        if self.dest_folder:
            validation_results["destination_exists"] = Path(self.dest_folder).exists()
            validation_results["destination_writable"] = os.access(
                self.dest_folder,
                os.W_OK,
            )
        else:
            validation_results["destination_exists"] = (
                True  # Not required for all modes
            )
            validation_results["destination_writable"] = (
                True  # Not required for all modes
            )

        # Check file size inputs
        try:
            min_size = float(self.min_file_size.get() or 0)
            max_size = float(self.max_file_size.get() or MAX_FILE_SIZE_MB)
            validation_results["size_inputs_valid"] = (
                0 <= min_size <= MAX_FILE_SIZE_MB
                and 0 <= max_size <= MAX_FILE_SIZE_MB
                and min_size <= max_size
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
            except (KeyError, ValueError, TypeError):
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
