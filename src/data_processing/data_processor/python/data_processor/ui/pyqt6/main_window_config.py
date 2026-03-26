"""Configuration management mixin for DataProcessorMainWindow.

Contains signal set load/save and app configuration load/save handlers.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QLineEdit, QMessageBox

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ConfigMixin:
    """Mixin providing signal set and app configuration management."""

    def _load_signal_set(self) -> None:
        """Load a signal set from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,  # type: ignore[arg-type]
            "Load Signal Set",
            "",
            "Signal Set Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        try:
            with open(filename, encoding="utf-8") as f:
                signal_set = json.load(f)

            if not isinstance(signal_set, dict):
                raise ValueError("Invalid signal set format")

            selected_signals = signal_set.get("selected_signals", [])
            if not isinstance(selected_signals, list):
                raise ValueError("Invalid selected_signals format")

            self.signal_list.select_signals(selected_signals)  # type: ignore[attr-defined]
            self.status_bar.set_status(  # type: ignore[attr-defined]
                f"Loaded signal set: {len(selected_signals)} signals"
            )

            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Loaded signal set from:\n{filename}\n\n"
                f"Selected {len(selected_signals)} signals",
            )

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            QMessageBox.critical(self, "Error", f"Invalid JSON file:\n{e}")  # type: ignore[arg-type]
        except (PermissionError, OSError) as e:
            logger.error("Load signal set error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load signal set:\n{e}")  # type: ignore[arg-type]

    def _save_signal_set(self) -> None:
        """Save current signal selection to JSON file."""
        selected_signals = self.signal_list.get_selected_signals()  # type: ignore[attr-defined]

        if not selected_signals:
            QMessageBox.warning(self, "No Selection", "Please select signals to save.")  # type: ignore[arg-type]
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,  # type: ignore[arg-type]
            "Save Signal Set",
            "",
            "Signal Set Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        if not filename.endswith(".json"):
            filename += ".json"

        try:
            signal_set = {
                "selected_signals": selected_signals,
                "total_available": len(self.available_signals),  # type: ignore[attr-defined]
                "source_files": [str(f) for f in self.selected_files],  # type: ignore[attr-defined]
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(signal_set, f, indent=2)

            self.status_bar.set_status(  # type: ignore[attr-defined]
                f"Saved signal set: {len(selected_signals)} signals"
            )

            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Signal set saved to:\n{filename}\n\n" f"Saved {len(selected_signals)} signals",
            )

        except (PermissionError, OSError) as e:
            logger.error("Save signal set error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save signal set:\n{e}")  # type: ignore[arg-type]

    def _save_app_config(self) -> None:
        """Save current app configuration."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,  # type: ignore[arg-type]
            "Save Configuration",
            "Configuration name:",
            QLineEdit.EchoMode.Normal,
            "my_config",
        )

        if not ok or not name:
            return

        try:
            config = {
                "output_directory": self.output_directory,  # type: ignore[attr-defined]
                "export_format": self.export_format_combo.currentText(),  # type: ignore[attr-defined]
                "include_timestamp": self.include_timestamp_check.isChecked(),  # type: ignore[attr-defined]
                "include_filter": self.include_filter_check.isChecked(),  # type: ignore[attr-defined]
                "export_selected_only": self.export_selected_only_check.isChecked(),  # type: ignore[attr-defined]
                "resample_rule": self.resample_rule_combo.currentText(),  # type: ignore[attr-defined]
                "resample_method": self.resample_method_combo.currentText(),  # type: ignore[attr-defined]
                "interpolate": self.interpolate_check.isChecked(),  # type: ignore[attr-defined]
                "integration_method": self.int_method_combo.currentText(),  # type: ignore[attr-defined]
                "differentiation_method": self.diff_method_combo.currentText(),  # type: ignore[attr-defined]
                "diff_order": self.diff_order_spin.value(),  # type: ignore[attr-defined]
                "diff_window_size": self.diff_window_spin.value(),  # type: ignore[attr-defined]
                "diff_poly_order": self.diff_poly_order_spin.value(),  # type: ignore[attr-defined]
            }

            self.config_manager.save_config(name, config)  # type: ignore[attr-defined]
            self.status_bar.set_status(f"Saved configuration: {name}")  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Configuration '{name}' saved successfully.",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error("Save config error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")  # type: ignore[arg-type]

    def _load_app_config(self) -> None:
        """Load app configuration."""
        try:
            configs = self.config_manager.list_configs()  # type: ignore[attr-defined]
            if not configs:
                QMessageBox.information(
                    self,  # type: ignore[arg-type]
                    "No Configurations",
                    "No saved configurations found.",
                )
                return

            from PyQt6.QtWidgets import QInputDialog

            name, ok = QInputDialog.getItem(
                self,  # type: ignore[arg-type]
                "Load Configuration",
                "Select configuration:",
                configs,
                0,
                False,
            )

            if not ok or not name:
                return

            config = self.config_manager.load_config(name)  # type: ignore[attr-defined]
            if not config:
                QMessageBox.warning(self, "Error", f"Configuration '{name}' not found.")  # type: ignore[arg-type]
                return

            # Apply configuration
            if "output_directory" in config:
                self.output_directory = config["output_directory"]  # type: ignore[attr-defined]
                self.output_folder_edit.setText(self.output_directory)  # type: ignore[attr-defined]

            if "export_format" in config:
                idx = self.export_format_combo.findText(config["export_format"])  # type: ignore[attr-defined]
                if idx >= 0:
                    self.export_format_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

            if "include_timestamp" in config:
                self.include_timestamp_check.setChecked(config["include_timestamp"])  # type: ignore[attr-defined]

            if "include_filter" in config:
                self.include_filter_check.setChecked(config["include_filter"])  # type: ignore[attr-defined]

            if "export_selected_only" in config:
                self.export_selected_only_check.setChecked(  # type: ignore[attr-defined]
                    config["export_selected_only"]
                )

            if "resample_rule" in config:
                idx = self.resample_rule_combo.findText(config["resample_rule"])  # type: ignore[attr-defined]
                if idx >= 0:
                    self.resample_rule_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

            if "resample_method" in config:
                idx = self.resample_method_combo.findText(config["resample_method"])  # type: ignore[attr-defined]
                if idx >= 0:
                    self.resample_method_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

            if "interpolate" in config:
                self.interpolate_check.setChecked(config["interpolate"])  # type: ignore[attr-defined]

            if "integration_method" in config:
                idx = self.int_method_combo.findText(config["integration_method"])  # type: ignore[attr-defined]
                if idx >= 0:
                    self.int_method_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

            if "differentiation_method" in config:
                idx = self.diff_method_combo.findText(config["differentiation_method"])  # type: ignore[attr-defined]
                if idx >= 0:
                    self.diff_method_combo.setCurrentIndex(idx)  # type: ignore[attr-defined]

            if "diff_order" in config:
                self.diff_order_spin.setValue(config["diff_order"])  # type: ignore[attr-defined]

            if "diff_window_size" in config:
                self.diff_window_spin.setValue(config["diff_window_size"])  # type: ignore[attr-defined]

            if "diff_poly_order" in config:
                self.diff_poly_order_spin.setValue(config["diff_poly_order"])  # type: ignore[attr-defined]

            self.status_bar.set_status(f"Loaded configuration: {name}")  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Success",
                f"Configuration '{name}' loaded successfully.",
            )

        except ImportError as e:
            logger.error("Load config error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")  # type: ignore[arg-type]
