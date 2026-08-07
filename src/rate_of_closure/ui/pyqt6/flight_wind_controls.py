"""Focused PyQt controls for paired no-wind and selected-wind flight."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from rate_of_closure.simulation import WindComparison
from shared.python.swing_sim.flight import WindScenario

_DELTA_ROWS = (
    ("carry_m", "Carry", "m"),
    ("max_height_m", "Apex", "m"),
    ("flight_time_s", "Flight Time", "s"),
    ("landing_angle_deg", "Landing Angle", "deg"),
    ("lateral_m", "Lateral", "m"),
)


def _wind_spin(high: float, default: float, tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setRange(-high, high)
    spin.setDecimals(1)
    spin.setValue(default)
    spin.setToolTip(tooltip)
    return spin


class FlightWindControls(QGroupBox):
    """Meteorological steady-wind inputs and comparison delta display."""

    def __init__(self) -> None:
        super().__init__("Wind Comparison")
        layout = QVBoxLayout(self)
        self.enabled_check = QCheckBox("Compare No Wind and Selected Wind")
        self.enabled_check.setChecked(False)
        self.enabled_check.setToolTip(
            "Run identical launch conditions with no wind and selected wind. "
            "Source: canonical wind-scenario/v1 paired-comparison contract."
        )
        layout.addWidget(self.enabled_check)
        form = QFormLayout()
        self.speed_spin = _wind_spin(
            150.0,
            10.0,
            "Horizontal wind speed; negative values are not allowed. "
            "Source: canonical wind-scenario/v1 meteorological adapter.",
        )
        self.speed_spin.setRange(0.0, 150.0)
        self.speed_spin.setSuffix(" mph")
        form.addRow("Wind Speed", self.speed_spin)
        self.bearing_spin = _wind_spin(
            3600.0,
            0.0,
            "Bearing the wind comes from: 0 headwind, 90 from right, 180 tailwind. "
            "Source: canonical wind-scenario/v1 meteorological adapter.",
        )
        self.bearing_spin.setSuffix("° from")
        form.addRow("Wind From Bearing", self.bearing_spin)
        layout.addLayout(form)
        self.direction_label = QLabel()
        self.direction_label.setToolTip(
            "Meteorological from-bearing; the arrow shows the wind-to direction. "
            "Source: canonical wind-scenario/v1 meteorological adapter."
        )
        layout.addWidget(self.direction_label)
        self._delta_labels: dict[str, QLabel] = {}
        delta_form = QFormLayout()
        for key, label, _unit in _DELTA_ROWS:
            value = QLabel("—")
            value.setToolTip("Selected-wind result minus the no-wind result.")
            self._delta_labels[key] = value
            delta_form.addRow(f"Δ {label}", value)
        layout.addLayout(delta_form)
        self.speed_spin.valueChanged.connect(self._refresh_direction)
        self.bearing_spin.valueChanged.connect(self._refresh_direction)
        self._refresh_direction()

    def scenario(self) -> WindScenario:
        """Return the declared wind as flight-frame wind-to velocity."""
        return WindScenario.from_meteorological(
            self.speed_spin.value() / 2.236936292054402,
            self.bearing_spin.value(),
        )

    def comparison_enabled(self) -> bool:
        """Return whether paired wind comparison is requested."""
        return bool(self.enabled_check.isChecked())

    def optional_scenario(self) -> WindScenario | None:
        """Return the selected wind only when paired comparison is enabled."""
        return self.scenario() if self.comparison_enabled() else None

    def set_comparison(self, comparison: WindComparison | None) -> None:
        """Populate selected-wind-minus-calm deltas or clear them."""
        for key, _label, unit in _DELTA_ROWS:
            text = (
                "—" if comparison is None else f"{comparison.deltas[key]:+.2f} {unit}"
            )
            self._delta_labels[key].setText(text)

    def delta_text(self, key: str) -> str:
        """Return one formatted delta value for GUI integration tests."""
        if key not in self._delta_labels:
            raise KeyError(key)
        return str(self._delta_labels[key].text())

    def _refresh_direction(self) -> None:
        to_bearing = (self.bearing_spin.value() + 180.0) % 360.0
        self.direction_label.setText(
            f"{self.speed_spin.value():.1f} mph from "
            f"{self.bearing_spin.value():.1f}°; toward {to_bearing:.1f}°."
        )


__all__ = ["FlightWindControls"]
