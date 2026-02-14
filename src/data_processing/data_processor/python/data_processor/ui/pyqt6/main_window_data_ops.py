"""DataOperationsMixin -- data processing methods for DataProcessorMainWindow.

Handles filter application, signal integration/differentiation,
formula application, statistics, data export, and file management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from data_processor.core.dataset_naming import generate_dataset_name
from data_processor.core.signal_processing import (
    apply_custom_variable,
    differentiate_signals,
    integrate_signals,
)
from data_processor.models.processing_config import FilterConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataOperationsMixin:
    """Mixin providing data processing and export methods."""

    def _apply_filter(self) -> None:
        """Apply filter to current data."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            self.status_bar.set_status("Applying filter...")
            self.status_bar.show_progress()
            QApplication.processEvents()

            filter_type = self.filter_config.get_filter_type()
            params = self.filter_config.get_parameters()

            config = FilterConfig(filter_type=filter_type, parameters=params)
            self.current_data = self.signal_processor.apply_filter(
                self.current_data, config
            )

            self.preview_widget.update_preview(self.current_data)
            self.status_bar.hide_progress()
            self.status_bar.set_status(f"Applied {filter_type}")

            QMessageBox.information(
                self, "Success", f"{filter_type} applied successfully"
            )

        except (RuntimeError, AttributeError) as e:
            self.status_bar.hide_progress()
            self.status_bar.set_status("Filter failed")
            logger.error(f"Filter error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Filter failed:\n{e}")

    def _integrate_signals(self) -> None:
        """Integrate selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals[:10]  # Limit to first 10 if none selected

        if not self.time_column:
            QMessageBox.warning(self, "No Time Column", "Time column not detected.")
            return

        try:
            self.status_bar.set_status("Integrating...")
            method = self.int_method_combo.currentText()

            # Use core library function
            self.current_data = integrate_signals(
                self.current_data,
                time_col=self.time_column,
                signals=signals,
                method=method,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)
            self.analysis_panel.set_dataframe(self.current_data)

            self.status_bar.set_status(f"Integration complete ({method})")
            QMessageBox.information(
                self,
                "Success",
                f"Integration complete\n"
                f"Method: {method}\n"
                f"Signals: {len(signals)}",
            )
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Integration error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Integration failed:\n{e}")

    def _differentiate_signals(self) -> None:
        """Differentiate selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals[:10]  # Limit to first 10 if none selected

        if not self.time_column:
            QMessageBox.warning(self, "No Time Column", "Time column not detected.")
            return

        try:
            self.status_bar.set_status("Differentiating...")
            method = self.diff_method_combo.currentText()
            order = self.diff_order_spin.value()
            window_size = self.diff_window_spin.value()
            poly_order = self.diff_poly_order_spin.value()

            # Ensure window size is odd
            if window_size % 2 == 0:
                window_size += 1

            # Use core library function
            self.current_data = differentiate_signals(
                self.current_data,
                time_col=self.time_column,
                signals=signals,
                method=method,
                orders=[order],
                window_size=window_size,
                poly_order=poly_order,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)
            self.analysis_panel.set_dataframe(self.current_data)

            self.status_bar.set_status(f"Differentiation complete ({method})")
            QMessageBox.information(
                self,
                "Success",
                f"Differentiation complete\n"
                f"Method: {method}\n"
                f"Order: {order}\n"
                f"Signals: {len(signals)}",
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Differentiation error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Differentiation failed:\n{e}")

    def _apply_formula(self) -> None:
        """Apply custom formula."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        name = self.formula_name_edit.text().strip()
        formula = self.formula_edit.text().strip()

        if not name or not formula:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter both name and formula."
            )
            return

        try:
            self.status_bar.set_status(f"Applying formula: {name}...")

            # Use core library function
            self.current_data = apply_custom_variable(
                self.current_data,
                name=name,
                formula=formula,
                time_col=self.time_column,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)
            self.analysis_panel.set_dataframe(self.current_data)
            self._update_column_combos()

            self.status_bar.set_status(f"Created signal: {name}")
            QMessageBox.information(
                self,
                "Success",
                f"Signal '{name}' created successfully\n" f"Formula: {formula}",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Formula error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Formula failed:\n{e}")

    def _calculate_statistics(self) -> None:
        """Calculate statistics for signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals[:20]  # Limit to 20

        self.stats_widget.update_statistics(self.current_data, signals)
        self.status_bar.set_status("Statistics calculated")

    def _export_data(self) -> None:
        """Export data to file."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        format_type = self.export_format_combo.currentText()
        extensions = {
            "csv": ("CSV Files (*.csv)", ".csv"),
            "excel": ("Excel Files (*.xlsx)", ".xlsx"),
            "parquet": ("Parquet Files (*.parquet)", ".parquet"),
            "hdf5": ("HDF5 Files (*.h5)", ".h5"),
            "feather": ("Feather Files (*.feather)", ".feather"),
        }

        filter_str, ext = extensions.get(format_type, ("All Files (*)", ""))

        # Get default filename from dataset name or generate one
        default_name = self.dataset_name_edit.text().strip()
        if not default_name:
            default_name = "processed_data"

        # Use output directory as starting point
        default_path = str(Path(self.output_directory) / (default_name + ext))

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            default_path,
            filter_str,
        )

        if not filename:
            return

        try:
            self.status_bar.set_status("Exporting...")

            # Prepare data for export
            export_data = self.current_data

            # If export selected only, filter columns
            if self.export_selected_only_check.isChecked():
                selected_signals = self.signal_list.get_selected_signals()
                if selected_signals:
                    # Keep time column and selected signals
                    columns_to_keep = []
                    if self.time_column:
                        columns_to_keep.append(self.time_column)
                    columns_to_keep.extend(
                        [s for s in selected_signals if s in export_data.columns]
                    )
                    export_data = export_data[columns_to_keep]

            success = self.data_loader.save_dataframe(
                export_data, filename, format_type=format_type
            )

            if success:
                self.status_bar.set_status(f"Exported to {Path(filename).name}")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Data exported to:\n{filename}\n\n"
                    f"Rows: {len(export_data)}\n"
                    f"Columns: {len(export_data.columns)}",
                )
            else:
                QMessageBox.warning(self, "Error", "Export failed")

        except (PermissionError, OSError) as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def _browse_output_folder(self) -> None:
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.output_directory,
        )
        if folder:
            self.output_directory = folder
            self.output_folder_edit.setText(folder)
            self.status_bar.set_status(f"Output folder: {folder}")

    def _auto_generate_name(self) -> None:
        """Auto-generate dataset name."""
        base_name = "processed_data"
        if self.selected_files:
            # Use first file name as base
            base_name = Path(self.selected_files[0]).stem

        include_timestamp = self.include_timestamp_check.isChecked()
        include_filter = self.include_filter_check.isChecked()

        # Get current filter type if any
        filter_type = None
        if include_filter:
            try:
                filter_type = self.filter_config.get_filter_type()
            except (KeyError, ValueError, TypeError):
                pass

        name = generate_dataset_name(
            base_name=base_name,
            include_timestamp=include_timestamp,
            include_filter=include_filter,
            filter_type=filter_type,
        )

        self.dataset_name_edit.setText(name)
        self.status_bar.set_status(f"Generated name: {name}")

    def _clear_data(self) -> None:
        """Clear all data."""
        self.current_data = None
        self.filtered_data = None
