"""Persist and restore player-covariation presentation selections."""

from __future__ import annotations

from typing import Any

from rate_of_closure.ui.pyqt6.launch_monitor_player_controls import (
    LaunchMonitorPlayerControls,
)


def covariation_project_selections(
    controls: LaunchMonitorPlayerControls,
) -> dict[str, object]:
    """Return inspectable covariation settings for an analysis project."""

    return {
        "covariation_x": controls.covariation_x_combo.currentText(),
        "covariation_y": controls.covariation_y_combo.currentText(),
        "covariation_method": controls.covariation_method_combo.currentText(),
        "covariation_min_samples": controls.covariation_min_samples_spin.value(),
        "covariation_confidence": controls.covariation_confidence_spin.value(),
    }


def restore_covariation_project_selections(
    controls: LaunchMonitorPlayerControls, selections: dict[str, Any]
) -> None:
    """Validate and restore covariation settings after data is loaded."""

    minimum = selections.get("covariation_min_samples", 8)
    confidence = selections.get("covariation_confidence", 0.95)
    if not isinstance(minimum, int) or minimum < 4:
        raise ValueError("saved covariation minimum sample count is invalid")
    if not isinstance(confidence, (int, float)) or not 0.51 <= confidence <= 0.999:
        raise ValueError("saved covariation confidence is invalid")
    controls.covariation_x_combo.setCurrentText(
        str(selections.get("covariation_x", controls.covariation_x_combo.currentText()))
    )
    controls.covariation_y_combo.setCurrentText(
        str(selections.get("covariation_y", controls.covariation_y_combo.currentText()))
    )
    controls.covariation_method_combo.setCurrentText(
        str(selections.get("covariation_method", "Pearson"))
    )
    controls.covariation_min_samples_spin.setValue(minimum)
    controls.covariation_confidence_spin.setValue(float(confidence))


__all__ = ["covariation_project_selections", "restore_covariation_project_selections"]
