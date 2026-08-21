"""Persistence and restricted export actions for performance analytics."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from rate_of_closure.application.atomic_text_files import write_utf8_text_atomic
from rate_of_closure.launch_monitor_workspace import dataset_reference_for_frame
from rate_of_closure.ui.pyqt6.launch_monitor_performance_files import (
    load_performance_settings_versioned,
    performance_document,
)


class PerformancePersistenceMixin:
    """Keep workspace persistence bounded and separate from analytics."""

    def _document(self: Any) -> dict[str, object]:
        reference = dataset_reference_for_frame(self._frame, self._source_name)
        settings = {
            "carry": self.carry_combo.currentText(),
            "lateral": self.lateral_combo.currentText(),
            "carry_unit": self.carry_unit.currentText(),
            "lateral_unit": self.lateral_unit.currentText(),
            "target_yards": self.target_distance.value(),
            "before": self.before_combo.currentText(),
            "after": self.after_combo.currentText(),
            "baseline": self.baseline_url.text(),
            "player": self.player_combo.currentText(),
            "session": self.session_combo.currentText(),
            "order": self.order_combo.currentText(),
            "metric": self.metric_combo.currentText(),
            "player_attested": self.player_attest.isChecked(),
            "session_attested": self.session_attest.isChecked(),
        }
        document: dict[str, object] = performance_document(
            reference, settings, self._dispersion, self._proxy, self._trend
        )
        return document

    def save_dialog(self: Any) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Performance Analysis",
            "performance.lmanalysis.json",
            "JSON (*.json)",
        )
        if selected:
            write_utf8_text_atomic(
                json.dumps(self._document(), indent=2),
                selected,
                document_name="performance analysis",
            )

    def load_dialog(self: Any) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load Performance Analysis", "", "JSON (*.json)"
        )
        if not selected:
            return
        try:
            reference = dataset_reference_for_frame(self._frame, self._source_name)
            settings, imported_from = load_performance_settings_versioned(
                selected, reference.sha256
            )
            self._install_settings(settings, imported_from)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Analysis Not Loaded", str(error))

    def _install_settings(
        self: Any, settings: dict[str, object], imported_from: str = "v3"
    ) -> None:
        controls = (
            ("carry", self.carry_combo),
            ("lateral", self.lateral_combo),
            ("carry_unit", self.carry_unit),
            ("lateral_unit", self.lateral_unit),
            ("before", self.before_combo),
            ("after", self.after_combo),
            ("player", self.player_combo),
            ("session", self.session_combo),
            ("order", self.order_combo),
            ("metric", self.metric_combo),
        )
        for key, control in controls:
            value = settings.get(key)
            if isinstance(value, str):
                control.setCurrentText(value)
        target = settings["target_yards"]
        if isinstance(target, bool) or not isinstance(target, (int, float, str)):
            raise TypeError("saved target distance must be numeric")
        self.target_distance.setValue(float(target))
        if isinstance(settings.get("baseline"), str):
            self.baseline_url.setText(settings["baseline"])
        self.player_attest.setChecked(settings.get("player_attested") is True)
        self.session_attest.setChecked(settings.get("session_attested") is True)
        self.dispersion_status.setText(
            f"{imported_from} settings restored; rerun to regenerate "
            "row-aligned results."
        )

    def export_plot_dialog(self: Any) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Unit-Labeled Plot",
            "dispersion.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if selected:
            self.canvas.figure.savefig(selected)

    def export_data_dialog(self: Any) -> None:
        approval = QMessageBox.question(
            self,
            "Export Restricted Backing Rows?",
            "This export contains retained source rows. Export only to an approved "
            "restricted-data location. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if approval != QMessageBox.StandardButton.Yes:
            self.dispersion_status.setText(
                "Backing-data export unavailable: explicit restricted approval "
                "was not granted."
            )
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Backing Data",
            "performance-backing.csv",
            "CSV (*.csv);;JSON (*.json)",
        )
        if selected and selected.lower().endswith(".json"):
            self._frame.to_json(selected, orient="records", indent=2)
        elif selected:
            self._frame.to_csv(selected, index=False)


__all__ = ["PerformancePersistenceMixin"]
