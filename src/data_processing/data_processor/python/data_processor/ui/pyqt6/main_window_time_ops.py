# ruff: noqa: E501
"""Time operations mixin for DataProcessorMainWindow.

Contains resample, time range info/trim, and trendline calculation.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class TimeOpsMixin:
    """Mixin providing time-based data operations for DataProcessorMainWindow."""

    def _apply_resample(self) -> None:
        """Apply time resampling to data."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")  # type: ignore[arg-type]
            return

        try:
            time_col = self.time_col_combo.currentText()  # type: ignore[attr-defined]
            if not time_col:
                time_col = self.time_column  # type: ignore[attr-defined]
            if not time_col:
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "No Time Column",
                    "Please select a time column.",
                )
                return

            rule = self.resample_rule_combo.currentText()  # type: ignore[attr-defined]
            method = self.resample_method_combo.currentText()  # type: ignore[attr-defined]
            interpolate = self.interpolate_check.isChecked()  # type: ignore[attr-defined]

            self.status_bar.set_status(f"Resampling to {rule}...")  # type: ignore[attr-defined]

            from data_processor.core.signal_processing import resample_data

            self.current_data = resample_data(
                self.current_data,
                time_col=time_col,
                rule=rule,
                method=method,
                interpolate=interpolate,
            )

            self.preview_widget.update_preview(self.current_data)  # type: ignore[attr-defined]
            self._update_data_info()  # type: ignore[attr-defined]

            self.status_bar.set_status(f"Resampled to {rule}")  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Data resampled to {rule}\n"
                f"Method: {method}\n"
                f"Rows: {len(self.current_data)}",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error("Resample error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Resampling failed:\n{e}")  # type: ignore[arg-type]

    def _update_time_range_info(self) -> None:
        """Update time range information display."""
        if self.current_data is None or not self.time_column:  # type: ignore[attr-defined]
            self.data_start_label.setText("-")  # type: ignore[attr-defined]
            self.data_end_label.setText("-")  # type: ignore[attr-defined]
            self.data_duration_label.setText("-")  # type: ignore[attr-defined]
            return

        try:
            time_data = self.current_data[self.time_column]  # type: ignore[attr-defined]
            start = time_data.min()
            end = time_data.max()

            self.data_start_label.setText(str(start))  # type: ignore[attr-defined]
            self.data_end_label.setText(str(end))  # type: ignore[attr-defined]

            try:
                duration = end - start
                self.data_duration_label.setText(str(duration))  # type: ignore[attr-defined]
            except (RuntimeError, AttributeError):
                self.data_duration_label.setText("-")  # type: ignore[attr-defined]

        except (RuntimeError, AttributeError) as e:
            logger.error("Time range info error: %s", e)

    def _trim_time_range(self) -> None:
        """Trim data to specified time range."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")  # type: ignore[arg-type]
            return

        try:
            time_col = self.time_column  # type: ignore[attr-defined]
            if not time_col:
                QMessageBox.warning(self, "No Time Column", "Time column not detected.")  # type: ignore[arg-type]
                return

            start_str = self.start_time_edit.text().strip()  # type: ignore[attr-defined]
            end_str = self.end_time_edit.text().strip()  # type: ignore[attr-defined]
            date_str = self.date_filter_edit.text().strip()  # type: ignore[attr-defined]

            start_time = float(start_str) if start_str else None
            end_time = float(end_str) if end_str else None

            self.status_bar.set_status("Trimming time range...")  # type: ignore[attr-defined]

            from data_processor.core.signal_processing import trim_time_range

            self.current_data = trim_time_range(
                self.current_data,
                time_col=time_col,
                start_time=start_time,
                end_time=end_time,
                date=date_str if date_str else None,
            )

            self.preview_widget.update_preview(self.current_data)  # type: ignore[attr-defined]
            self._update_data_info()  # type: ignore[attr-defined]
            self._update_time_range_info()

            self.status_bar.set_status("Time range trimmed")  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Data trimmed to time range\nRows: {len(self.current_data)}",
            )

        except ValueError:
            QMessageBox.warning(
                self,  # type: ignore[arg-type]
                "Invalid Input",
                "Please enter valid numeric time values.",
            )
        except (ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error("Trim time range error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Time range trim failed:\n{e}")  # type: ignore[arg-type]

    def _copy_time_range_to_preview(self) -> None:
        """Copy current time range to preview filter."""
        if self.current_data is None or not self.time_column:  # type: ignore[attr-defined]
            return

        try:
            time_data = self.current_data[self.time_column]  # type: ignore[attr-defined]
            self.start_time_edit.setText(str(time_data.min()))  # type: ignore[attr-defined]
            self.end_time_edit.setText(str(time_data.max()))  # type: ignore[attr-defined]
            self.status_bar.set_status("Time range copied")  # type: ignore[attr-defined]
        except (RuntimeError, AttributeError) as e:
            logger.error("Copy time range error: %s", e)

    def _calculate_trendline(self) -> None:
        """Calculate trendline for selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")  # type: ignore[arg-type]
            return

        try:
            x_col = self.x_axis_combo.currentText()  # type: ignore[attr-defined]
            if not x_col:
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "No X-Axis",
                    "Please select an X-axis signal.",
                )
                return

            selected = self.signal_list.get_selected_signals()  # type: ignore[attr-defined]
            if not selected:
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "No Y-Signals",
                    "Please select Y-axis signals.",
                )
                return

            trend_type = self.trendline_type_combo.currentText()  # type: ignore[attr-defined]
            if trend_type == "None":
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "No Trendline",
                    "Please select a trendline type.",
                )
                return

            degree = self.poly_degree_spin.value()  # type: ignore[attr-defined]

            x_min = None
            x_max = None
            x_min_text = self.trend_x_min_edit.text().strip()  # type: ignore[attr-defined]
            x_max_text = self.trend_x_max_edit.text().strip()  # type: ignore[attr-defined]
            if x_min_text:
                try:
                    x_min = float(x_min_text)
                except ValueError:
                    QMessageBox.warning(
                        self,  # type: ignore[arg-type]
                        "Invalid X Range",
                        f"'{x_min_text}' is not a valid number for the X minimum.",
                    )
                    return
            if x_max_text:
                try:
                    x_max = float(x_max_text)
                except ValueError:
                    QMessageBox.warning(
                        self,  # type: ignore[arg-type]
                        "Invalid X Range",
                        f"'{x_max_text}' is not a valid number for the X maximum.",
                    )
                    return

            y_col = selected[0]

            from data_processor.core.signal_processing import calculate_trendline

            result = calculate_trendline(
                self.current_data,
                x_col=x_col,
                y_col=y_col,
                trend_type=trend_type,
                degree=degree,
                x_min=x_min,
                x_max=x_max,
            )

            result_text = f"Trendline: {trend_type}\n"
            result_text += f"Equation: {result.get('equation', 'N/A')}\n"
            result_text += f"R² = {result.get('r_squared', 0):.6f}\n"
            if "coefficients" in result:
                result_text += f"Coefficients: {result['coefficients']}\n"

            self.trendline_results.setText(result_text)  # type: ignore[attr-defined]
            self.status_bar.set_status("Trendline calculated")  # type: ignore[attr-defined]

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error("Trendline error: %s", e, exc_info=True)
            QMessageBox.critical(
                self,  # type: ignore[arg-type]
                "Error",
                f"Trendline calculation failed:\n{e}",
            )
