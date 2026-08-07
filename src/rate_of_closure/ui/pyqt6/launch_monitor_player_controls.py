"""Focused PyQt controls for dispersion, strokes gained, and session trends."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QWidget,
)


class LaunchMonitorPlayerControls(QGroupBox):
    """Role mappings and assumptions shared by player-oriented plot modes."""

    def __init__(self) -> None:
        super().__init__("Player Analytics")
        self._build_controls()
        layout = QFormLayout(self)
        for control, label, help_text in self._control_rows():
            control.setAccessibleName(label)
            control.setToolTip(help_text)
            layout.addRow(f"{label}:", control)

    def _build_controls(self) -> None:
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(
            [
                "Relationship",
                "Directional Dispersion",
                "Strokes Gained",
                "Session Trend",
                "Within-Player Covariation",
                "Covariation Pair Scan",
            ]
        )
        self.lateral_combo = QComboBox()
        self.carry_combo = QComboBox()
        self.session_combo = QComboBox()
        self.player_combo = QComboBox()
        self.time_combo = QComboBox()
        self.covariation_x_combo = QComboBox()
        self.covariation_y_combo = QComboBox()
        self.covariation_method_combo = QComboBox()
        self.covariation_method_combo.addItems(["Pearson", "Spearman"])
        self.covariation_min_samples_spin = QSpinBox()
        self.covariation_min_samples_spin.setRange(4, 1_000_000)
        self.covariation_min_samples_spin.setValue(8)
        self.covariation_confidence_spin = QDoubleSpinBox()
        self.covariation_confidence_spin.setRange(0.51, 0.999)
        self.covariation_confidence_spin.setSingleStep(0.01)
        self.covariation_confidence_spin.setValue(0.95)
        self.target_distance_spin = QDoubleSpinBox()
        self.target_distance_spin.setRange(10.0, 600.0)
        self.target_distance_spin.setSuffix(" yd")
        self.target_distance_spin.setValue(240.0)
        self.start_lie_combo = QComboBox()
        self.start_lie_combo.addItems(["tee", "fairway", "rough", "sand", "recovery"])
        self.end_lie_combo = QComboBox()
        self.end_lie_combo.addItems(["fairway", "rough", "sand", "recovery"])

    def _control_rows(self) -> tuple[tuple[QWidget, str, str], ...]:
        return self._base_control_rows() + self._covariation_control_rows()

    def _base_control_rows(self) -> tuple[tuple[QWidget, str, str], ...]:
        return (
            (
                self.plot_mode_combo,
                "Plot Mode",
                "Choose relationship, signed dispersion, strokes gained, or "
                "session trend",
            ),
            (
                self.lateral_combo,
                "Lateral Outcome",
                "Map signed lateral distance; negative is left and positive is right",
            ),
            (
                self.carry_combo,
                "Carry / Downrange",
                "Map carry or downrange distance in metres or yards",
            ),
            (
                self.session_combo,
                "Session Identifier",
                "Choose the column that identifies each practice session",
            ),
            (
                self.player_combo,
                "Player Identifier",
                "Optionally separate longitudinal summaries by player",
            ),
            (
                self.time_combo,
                "Session Time",
                "Optionally order sessions by timestamp and report change per day",
            ),
            (
                self.target_distance_spin,
                "Target Distance",
                "Set the intended target distance for the range-shot SG proxy",
            ),
            (
                self.start_lie_combo,
                "Starting Lie",
                "Choose the Broadie benchmark lie before the shot",
            ),
            (
                self.end_lie_combo,
                "Assumed Ending Lie",
                "Assume the post-shot lie; range data does not observe this",
            ),
        )

    def _covariation_control_rows(self) -> tuple[tuple[QWidget, str, str], ...]:
        return (
            (
                self.covariation_x_combo,
                "Covariation X Variable",
                "Choose any numeric explanatory variable; selection does not "
                "imply causality",
            ),
            (
                self.covariation_y_combo,
                "Covariation Y Variable",
                "Choose any different numeric response variable; axes retain "
                "source units",
            ),
            (
                self.covariation_method_combo,
                "Covariation Method",
                "Pearson measures linear association; Spearman measures "
                "monotonic rank association",
            ),
            (
                self.covariation_min_samples_spin,
                "Minimum Shots per Player",
                "Players below this complete-pair count remain visible as insufficient",
            ),
            (
                self.covariation_confidence_spin,
                "Covariation Confidence",
                "Set Fisher-z interval coverage for Pearson correlations",
            ),
        )

    @staticmethod
    def _select_preferred(combo: QComboBox, preferred: tuple[str, ...]) -> None:
        for candidate in preferred:
            if combo.findText(candidate) >= 0:
                combo.setCurrentText(candidate)
                return

    def refresh_columns(self, frame: pd.DataFrame, numeric: list[str]) -> None:
        """Refresh role mappings while preferring campaign canonical fields."""

        for combo in (self.lateral_combo, self.carry_combo):
            combo.clear()
            combo.addItems(numeric)
        self._select_preferred(
            self.lateral_combo,
            (
                "observed_lateral_m",
                "predicted_lateral_m",
                "carry_flat_side_yd",
                "lateral_distance",
            ),
        )
        self._select_preferred(
            self.carry_combo,
            (
                "observed_carry_m",
                "predicted_carry_m",
                "carry_distance",
                "carry_yd",
            ),
        )
        columns = [str(column) for column in frame.columns]
        self.session_combo.clear()
        self.session_combo.addItem("(unavailable)")
        self.session_combo.addItems(columns)
        self._select_preferred(self.session_combo, ("session_id", "session", "model"))
        self.player_combo.clear()
        self.player_combo.addItem("(all players)")
        self.player_combo.addItems(columns)
        self._select_preferred(self.player_combo, ("player_id", "player", "golfer_id"))
        self.time_combo.clear()
        self.time_combo.addItem("(row order)")
        self.time_combo.addItems(columns)
        self._select_preferred(
            self.time_combo,
            ("recorded_at", "timestamp", "shot_time", "session_date", "date"),
        )
        self._refresh_covariation_columns(numeric)

    def _refresh_covariation_columns(self, numeric: list[str]) -> None:
        for combo in (self.covariation_x_combo, self.covariation_y_combo):
            combo.clear()
            combo.addItems(numeric)
        self._select_preferred(
            self.covariation_x_combo,
            ("club_path_deg", "club_path", "attack_angle_deg", "attack_angle"),
        )
        self._select_preferred(
            self.covariation_y_combo,
            ("face_angle_deg", "face_angle", "launch_direction_deg"),
        )
        if (
            self.covariation_y_combo.currentText()
            == self.covariation_x_combo.currentText()
            and len(numeric) > 1
        ):
            self.covariation_y_combo.setCurrentIndex(1)


__all__ = ["LaunchMonitorPlayerControls"]
