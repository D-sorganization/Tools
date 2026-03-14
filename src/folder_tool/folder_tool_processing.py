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

        if mode == "analyze":
            self._run_analyze_mode()
            return
        if mode == "deduplicate":
            self._run_deduplicate_mode()
            return

        self._run_destination_workflow(mode)

    def _run_analyze_mode(self) -> None:
        """Run analysis-only mode."""
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

    def _run_deduplicate_mode(self) -> None:
        """Run deduplication-only mode."""
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

    def _run_destination_workflow(self, mode: str) -> None:
        """Run source-to-destination workflow (combine, flatten, prune)."""
        assert mode is not None, "mode must be provided"
        if not self.validate_inputs(check_destination=True):
            return

        backup_path = None
        if self.backup_before_var.get():
            backup_path = self.create_backup()
            if backup_path is None and self.cancel_operation:
                return

        if not self._run_pre_processing():
            return

        final_summary = self._run_main_operation(mode)
        if final_summary is None:
            return

        final_summary = self._run_post_processing(final_summary, backup_path)

        self.update_progress(100, "Complete!")
        if not self.cancel_operation:
            messagebox.showinfo("All Operations Complete", final_summary)

    def _run_pre_processing(self) -> bool:
        """Run pre-processing steps (archive extraction). Returns False to abort."""
        if not self.unzip_var.get():
            return True
        try:
            self.update_status("Extracting archives...")
            unzip_log = self._bulk_unzip_enhanced()
            if self.cancel_operation:
                return False
            if not messagebox.askyesno(
                "Pre-processing Complete",
                "Bulk Extraction Complete!\n\n"
                + "\n".join(unzip_log)
                + "\n\nDo you want to proceed?",
            ):
                return False
        except OSError as e:
            messagebox.showerror(
                "Error",
                f"An error occurred during bulk unzip: {e}",
            )
            return False
        return True

    def _run_main_operation(self, mode: str) -> str | None:
        """Run the main folder operation. Returns summary or None on failure."""
        try:
            self.update_progress(30, "Running main operation...")
            main_op_log: list[str] = []
            if mode == "combine":
                main_op_log = self._combine_folders_enhanced()
            elif mode == "flatten":
                main_op_log = self._flatten_folders()
            elif mode == "prune":
                main_op_log = self._prune_empty_folders()

            if self.cancel_operation:
                return None
            return "Main Operation Complete!\n\n" + "\n".join(main_op_log)
        except OSError as e:
            messagebox.showerror(
                "Error",
                f"An error occurred during the main operation: {e}",
            )
            return None

    def _run_post_processing(self, final_summary: str, backup_path: str | None) -> str:
        """Run post-processing steps (dedup, zip, backup note)."""
        assert final_summary is not None, "final_summary must be provided"
        if self.deduplicate_var.get():
            try:
                self.update_progress(70, "Deduplicating files...")
                dedupe_log = self._perform_deduplication(self.dest_folder)
                final_summary += "\n\n--- Deduplication Results ---\n" + "\n".join(
                    dedupe_log,
                )
            except OSError as e:
                final_summary += f"\n\n--- Deduplication FAILED: {e}"

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

        return final_summary
