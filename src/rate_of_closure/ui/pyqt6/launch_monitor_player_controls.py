"""Focused PyQt controls for dispersion, strokes gained, and session trends."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox


class LaunchMonitorPlayerControls(QGroupBox):
    """Role mappings and assumptions shared by player-oriented plot modes."""

    def __init__(self) -> None:
        super().__init__("Player Analytics")
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(
            [
                "Relationship",
                "Directional Dispersion",
                "Strokes Gained",
                "Session Trend",
            ]
        )
        self.lateral_combo = QComboBox()
        self.carry_combo = QComboBox()
        self.session_combo = QComboBox()
        self.player_combo = QComboBox()
        self.time_combo = QComboBox()
        self.target_distance_spin = QDoubleSpinBox()
        self.target_distance_spin.setRange(10.0, 600.0)
        self.target_distance_spin.setSuffix(" yd")
        self.target_distance_spin.setValue(240.0)
        self.start_lie_combo = QComboBox()
        self.start_lie_combo.addItems(["tee", "fairway", "rough", "sand", "recovery"])
        self.end_lie_combo = QComboBox()
        self.end_lie_combo.addItems(["fairway", "rough", "sand", "recovery"])
        controls = (
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
        layout = QFormLayout(self)
        for control, label, help_text in controls:
            control.setAccessibleName(label)
            control.setToolTip(help_text)
            layout.addRow(f"{label}:", control)

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


__all__ = ["LaunchMonitorPlayerControls"]
