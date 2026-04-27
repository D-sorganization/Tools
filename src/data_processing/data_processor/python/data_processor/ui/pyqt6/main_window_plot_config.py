"""Plot configuration management mixin for DataProcessorMainWindow.

Contains save/load/delete/refresh for plot configurations.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class PlotConfigMixin:
    """Mixin providing plot configuration CRUD for DataProcessorMainWindow."""

    def _save_plot_config(self) -> None:
        """Save current plot configuration."""
        name = self.plot_name_edit.text().strip()  # type: ignore[attr-defined]
        if not name:
            QMessageBox.warning(self, "No Name", "Please enter a configuration name.")  # type: ignore[arg-type]
            return

        try:
            selected_signals = self.signal_list.get_selected_signals()  # type: ignore[attr-defined]
            x_axis = self.x_axis_combo.currentText()  # type: ignore[attr-defined]

            config = {
                "name": name,
                "x_axis": x_axis,
                "y_signals": selected_signals,
                "trendline_type": self.trendline_type_combo.currentText(),  # type: ignore[attr-defined]
                "poly_degree": self.poly_degree_spin.value(),  # type: ignore[attr-defined]
                "trend_x_min": self.trend_x_min_edit.text(),  # type: ignore[attr-defined]
                "trend_x_max": self.trend_x_max_edit.text(),  # type: ignore[attr-defined]
            }

            self.plot_config_manager.save_plot_config(name, config)  # type: ignore[attr-defined]
            self._refresh_saved_plots_list()

            self.status_bar.set_status(f"Saved plot config: {name}")  # type: ignore[attr-defined]

        except (RuntimeError, AttributeError) as e:
            logger.error("Save plot config error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save plot config:\n{e}")  # type: ignore[arg-type]

    def _load_plot_config(self) -> None:
        """Load selected plot configuration."""
        item = self.saved_plots_list.currentItem()  # type: ignore[attr-defined]
        if not item:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a configuration to load.",
            )
            return

        try:
            name = item.text()
            config = self.plot_config_manager.load_plot_config(name)  # type: ignore[attr-defined]

            if config:
                self.plot_name_edit.setText(config.get("name", ""))  # type: ignore[attr-defined]

                x_axis = config.get("x_axis", "")
                idx = self.x_axis_combo.findText(x_axis)  # type: ignore[attr-defined]
                if idx >= 0:
                    self.x_axis_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

                trend_type = config.get("trendline_type", "None")
                idx = self.trendline_type_combo.findText(trend_type)  # type: ignore[attr-defined]
                if idx >= 0:
                    self.trendline_type_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

                self.poly_degree_spin.setValue(config.get("poly_degree", 2))  # type: ignore[attr-defined]

                self.trend_x_min_edit.setText(config.get("trend_x_min", ""))  # type: ignore[attr-defined]
                self.trend_x_max_edit.setText(config.get("trend_x_max", ""))  # type: ignore[attr-defined]

                y_signals = config.get("y_signals", [])
                self.signal_list.select_signals(y_signals)  # type: ignore[attr-defined]

                self.status_bar.set_status(f"Loaded plot config: {name}")  # type: ignore[attr-defined]

        except (RuntimeError, AttributeError) as e:
            logger.error("Load plot config error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load plot config:\n{e}")  # type: ignore[arg-type]

    def _delete_plot_config(self) -> None:
        """Delete selected plot configuration."""
        item = self.saved_plots_list.currentItem()  # type: ignore[attr-defined]
        if not item:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a configuration to delete.",
            )
            return

        name = item.text()
        reply = QMessageBox.question(
            self,  # type: ignore[arg-type]
            "Confirm Delete",
            f"Delete plot configuration '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.plot_config_manager.delete_plot_config(name)  # type: ignore[attr-defined]
                self._refresh_saved_plots_list()
                self.status_bar.set_status(f"Deleted plot config: {name}")  # type: ignore[attr-defined]
            except (RuntimeError, AttributeError) as e:
                logger.error("Delete plot config error: %s", e, exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete:\n{e}")  # type: ignore[arg-type]

    def _refresh_saved_plots_list(self) -> None:
        """Refresh the saved plot configurations list."""
        self.saved_plots_list.clear()  # type: ignore[attr-defined]
        try:
            configs = self.plot_config_manager.list_plot_configs()  # type: ignore[attr-defined]
            for name in configs:
                self.saved_plots_list.addItem(name)  # type: ignore[attr-defined]
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Refresh plots list error: %s", e)
