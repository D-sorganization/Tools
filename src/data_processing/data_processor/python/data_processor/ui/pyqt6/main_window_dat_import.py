"""DAT file import mixin for DataProcessorMainWindow.

Contains browse, preview, import, and convert-to-CSV for DAT files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)


class DatImportMixin:
    """Mixin providing DAT file import operations for DataProcessorMainWindow."""

    def _browse_dat_file(self) -> None:
        """Browse for DAT file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,  # type: ignore[arg-type]
            "Open DAT File",
            "",
            "DAT Files (*.dat);;DBF Files (*.dbf);;All Files (*)",
        )
        if filename:
            self.dat_file_edit.setText(filename)  # type: ignore[attr-defined]

    def _preview_dat_file(self) -> None:
        """Preview DAT file contents."""
        filename = self.dat_file_edit.text()  # type: ignore[attr-defined]
        if not filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")  # type: ignore[arg-type]
            return

        try:
            with open(filename, encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(20)]

            preview_text = "".join(lines)
            self.dat_preview_text.setText(preview_text)  # type: ignore[attr-defined]

        except (PermissionError, OSError) as e:
            logger.error("DAT preview error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to preview file:\n{e}")  # type: ignore[arg-type]

    def _import_dat_file(self) -> None:
        """Import DAT file as data."""
        filename = self.dat_file_edit.text()  # type: ignore[attr-defined]
        if not filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")  # type: ignore[arg-type]
            return

        try:
            delimiters = {"Tab": "\t", "Comma": ",", "Semicolon": ";", "Space": " "}
            delimiter = delimiters.get(
                self.dat_delimiter_combo.currentText(),
                "\t",  # type: ignore[attr-defined]
            )

            self.status_bar.set_status("Importing DAT file...")  # type: ignore[attr-defined]

            from data_processor.core.dat_importer import read_dat_file

            self.current_data = read_dat_file(filename, delimiter=delimiter)  # type: ignore[attr-defined]

            if self.current_data is not None:  # type: ignore[attr-defined]
                self.available_signals = self.data_loader.get_numeric_signals(  # type: ignore[attr-defined]
                    self.current_data  # type: ignore[attr-defined]
                )

                self._update_data_info()  # type: ignore[attr-defined]
                self.signal_list.set_signals(self.available_signals)  # type: ignore[attr-defined]
                self.preview_widget.update_preview(self.current_data)  # type: ignore[attr-defined]
                self.analysis_panel.set_dataframe(self.current_data)  # type: ignore[attr-defined]

                self.status_bar.set_status(  # type: ignore[attr-defined]
                    f"Imported DAT file: {len(self.current_data)} rows"  # type: ignore[attr-defined]
                )
                QMessageBox.information(
                    self,  # type: ignore[arg-type]
                    "Success",
                    f"Imported DAT file\n"
                    f"Rows: {len(self.current_data)}\n"  # type: ignore[attr-defined]
                    f"Columns: {len(self.current_data.columns)}",  # type: ignore[attr-defined]
                )

        except (RuntimeError, AttributeError) as e:
            logger.error("DAT import error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to import DAT file:\n{e}")  # type: ignore[arg-type]

    def _convert_dat_to_csv(self) -> None:
        """Convert DAT file to CSV."""
        dat_filename = self.dat_file_edit.text()  # type: ignore[attr-defined]
        if not dat_filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")  # type: ignore[arg-type]
            return

        default_name = Path(dat_filename).stem + ".csv"
        output_filename, _ = QFileDialog.getSaveFileName(
            self,  # type: ignore[arg-type]
            "Save CSV File",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not output_filename:
            return

        try:
            self.status_bar.set_status("Converting DAT to CSV...")  # type: ignore[attr-defined]

            from data_processor.core.dat_importer import export_dat_to_csv

            output_path = export_dat_to_csv(dat_filename, output_filename)

            self.status_bar.set_status("DAT converted to CSV")  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"DAT file converted to CSV:\n{output_path}",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error("DAT conversion error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to convert DAT file:\n{e}")  # type: ignore[arg-type]
