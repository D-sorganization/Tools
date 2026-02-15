"""ProcessingMixin -- Processing and reporting methods for FolderProcessorApp.

This facade re-exports all mixin classes from decomposed submodules for
backward compatibility. New code should import directly from:
- folder_tool_analysis: AnalysisMixin (report generation, validation)
- folder_tool_archive: ArchiveMixin (ZIP creation)
- folder_tool_ui_processing: UIProcessingMixin (dialogs, progress, threading)
"""

from __future__ import annotations

import logging
from tkinter import messagebox

from folder_tool_analysis import AnalysisMixin  # noqa: F401
from folder_tool_archive import ArchiveMixin  # noqa: F401
from folder_tool_ui_processing import UIProcessingMixin  # noqa: F401

logger = logging.getLogger(__name__)


class ProcessingMixin(AnalysisMixin, ArchiveMixin, UIProcessingMixin):
    """Processing and reporting methods for FolderProcessorApp.

    Composes:
    - AnalysisMixin: Report generation and input validation
    - ArchiveMixin: ZIP archive creation
    - UIProcessingMixin: Text dialogs, progress, status, threading
    """

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
