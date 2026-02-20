#!/usr/bin/env python3
"""Thermal Profile Predictor PyQt6 Main Window.

A PyQt6 GUI for predicting temperature profiles in heated vessels.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Catppuccin Mocha color palette
CATPPUCCIN_MOCHA = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QWidget {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    font-family: "Segoe UI", "Arial", sans-serif;
}}

QScrollArea {{
    border: none;
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QGroupBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {CATPPUCCIN_MOCHA["mauve"]};
}}

QLabel {{
    color: {CATPPUCCIN_MOCHA["text"]};
    background-color: transparent;
}}

QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {CATPPUCCIN_MOCHA["surface2"]};
}}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 150px;
}}

QComboBox:hover {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {CATPPUCCIN_MOCHA["text"]};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    selection-background-color: {CATPPUCCIN_MOCHA["surface2"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
}}

QTextEdit {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
}}

QPushButton {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    color: {CATPPUCCIN_MOCHA["crust"]};
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {CATPPUCCIN_MOCHA["sapphire"]};
}}

QPushButton:pressed {{
    background-color: {CATPPUCCIN_MOCHA["lavender"]};
}}

QPushButton:disabled {{
    background-color: {CATPPUCCIN_MOCHA["surface2"]};
    color: {CATPPUCCIN_MOCHA["overlay0"]};
}}
"""


class ThermalProfilePredictorWindow(QMainWindow):
    """Main window for Thermal Profile Predictor application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._notes_dock: Any | None = None
        self._setup_ui()

    # -- Notes integration (shared workspace) --
    def _toggle_notes(self) -> None:
        """Show/hide the shared notes dock widget."""
        try:
            from pathlib import Path

            from notes.integration import attach_notes_dock
        except ImportError:
            return
        if self._notes_dock is None:
            project_dir = Path(__file__).resolve().parents[4]
            self._notes_dock = attach_notes_dock(self, project_dir=project_dir)
        self._notes_dock.setVisible(not self._notes_dock.isVisible())

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Thermal Profile Predictor")
        self.setMinimumSize(650, 700)
        self.setStyleSheet(STYLESHEET)

        # Menu bar with Notes toggle
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("&View")
        notes_action = view_menu.addAction("Toggle &Notes")
        notes_action.triggered.connect(self._toggle_notes)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("Thermal Profile Predictor")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Input section
        main_layout.addWidget(self._create_thermal_params_group())
        main_layout.addWidget(self._create_time_params_group())
        main_layout.addWidget(self._create_power_params_group())

        # Calculate button
        self.calculate_btn = QPushButton("Predict Temperature Profile")
        self.calculate_btn.clicked.connect(self._calculate)
        main_layout.addWidget(self.calculate_btn)

        # Results section
        main_layout.addWidget(self._create_results_group())

        main_layout.addStretch()

    def _create_thermal_params_group(self) -> QGroupBox:
        """Create the thermal parameters group."""
        group = QGroupBox("Thermal Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Initial temperature
        layout.addWidget(QLabel("Initial Temperature (°C):"), 0, 0)
        self.initial_temp_input = QDoubleSpinBox()
        self.initial_temp_input.setRange(-273.15, 2000)
        self.initial_temp_input.setDecimals(1)
        self.initial_temp_input.setValue(25.0)
        self.initial_temp_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.initial_temp_input, 0, 1)

        # Ambient temperature
        layout.addWidget(QLabel("Ambient Temperature (°C):"), 1, 0)
        self.ambient_temp_input = QDoubleSpinBox()
        self.ambient_temp_input.setRange(-273.15, 500)
        self.ambient_temp_input.setDecimals(1)
        self.ambient_temp_input.setValue(25.0)
        self.ambient_temp_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.ambient_temp_input, 1, 1)

        # Thermal mass
        layout.addWidget(QLabel("Thermal Mass (J/K):"), 2, 0)
        self.thermal_mass_input = QDoubleSpinBox()
        self.thermal_mass_input.setRange(1, 1e9)
        self.thermal_mass_input.setDecimals(0)
        self.thermal_mass_input.setValue(50000)
        self.thermal_mass_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.thermal_mass_input, 2, 1)

        # Heat loss coefficient
        layout.addWidget(QLabel("Heat Loss Coeff (W/K):"), 3, 0)
        self.heat_loss_input = QDoubleSpinBox()
        self.heat_loss_input.setRange(0, 10000)
        self.heat_loss_input.setDecimals(1)
        self.heat_loss_input.setValue(50.0)
        self.heat_loss_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.heat_loss_input, 3, 1)

        return group

    def _create_time_params_group(self) -> QGroupBox:
        """Create the time parameters group."""
        group = QGroupBox("Time Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Start time
        layout.addWidget(QLabel("Start Time (s):"), 0, 0)
        self.start_time_input = QDoubleSpinBox()
        self.start_time_input.setRange(0, 1e6)
        self.start_time_input.setDecimals(0)
        self.start_time_input.setValue(0)
        layout.addWidget(self.start_time_input, 0, 1)

        # End time
        layout.addWidget(QLabel("End Time (s):"), 1, 0)
        self.end_time_input = QDoubleSpinBox()
        self.end_time_input.setRange(1, 1e6)
        self.end_time_input.setDecimals(0)
        self.end_time_input.setValue(3600)
        layout.addWidget(self.end_time_input, 1, 1)

        # Number of points
        layout.addWidget(QLabel("Data Points:"), 2, 0)
        self.num_points_input = QSpinBox()
        self.num_points_input.setRange(10, 10000)
        self.num_points_input.setValue(100)
        layout.addWidget(self.num_points_input, 2, 1)

        return group

    def _create_power_params_group(self) -> QGroupBox:
        """Create the power input parameters group."""
        group = QGroupBox("Power Input")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Power profile type
        layout.addWidget(QLabel("Power Profile:"), 0, 0)
        self.power_profile_combo = QComboBox()
        self.power_profile_combo.addItems(["Constant", "Linear Ramp", "Step Function"])
        self.power_profile_combo.currentTextChanged.connect(
            self._on_power_profile_changed
        )
        layout.addWidget(self.power_profile_combo, 0, 1)

        # Power value (constant)
        layout.addWidget(QLabel("Power (W):"), 1, 0)
        self.power_input = QDoubleSpinBox()
        self.power_input.setRange(0, 1e6)
        self.power_input.setDecimals(0)
        self.power_input.setValue(5000)
        layout.addWidget(self.power_input, 1, 1)

        # Ramp rate (for linear ramp)
        layout.addWidget(QLabel("Ramp Rate (W/s):"), 2, 0)
        self.ramp_rate_input = QDoubleSpinBox()
        self.ramp_rate_input.setRange(0, 10000)
        self.ramp_rate_input.setDecimals(1)
        self.ramp_rate_input.setValue(1.0)
        self.ramp_rate_input.setEnabled(False)
        layout.addWidget(self.ramp_rate_input, 2, 1)

        # Step time (for step function)
        layout.addWidget(QLabel("Step Time (s):"), 3, 0)
        self.step_time_input = QDoubleSpinBox()
        self.step_time_input.setRange(0, 1e6)
        self.step_time_input.setDecimals(0)
        self.step_time_input.setValue(1800)
        self.step_time_input.setEnabled(False)
        layout.addWidget(self.step_time_input, 3, 1)

        return group

    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Temperature Profile Results")
        layout = QVBoxLayout(group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText(
            "Click 'Predict Temperature Profile' to see results..."
        )
        layout.addWidget(self.results_text)

        return group

    def _on_power_profile_changed(self, profile: str) -> None:
        """Handle power profile selection change."""
        self.ramp_rate_input.setEnabled(profile == "Linear Ramp")
        self.step_time_input.setEnabled(profile == "Step Function")

    def _get_power_function(self) -> Callable[[float], float]:
        """Create power function based on selected profile."""
        profile = self.power_profile_combo.currentText()
        power = self.power_input.value()

        if profile == "Constant":
            return lambda t: power
        elif profile == "Linear Ramp":
            ramp_rate = self.ramp_rate_input.value()
            return lambda t: power + ramp_rate * t
        else:  # Step Function
            step_time = self.step_time_input.value()
            return lambda t: power if t < step_time else 0

    def _calculate(self) -> None:
        """Perform the temperature profile prediction."""
        try:
            from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
                predict_temperature_profile,
            )

            # Get inputs
            initial_temp = self.initial_temp_input.value()
            ambient_temp = self.ambient_temp_input.value()
            thermal_mass = self.thermal_mass_input.value()
            heat_loss = self.heat_loss_input.value()

            t_start = self.start_time_input.value()
            t_end = self.end_time_input.value()
            num_points = self.num_points_input.value()

            # Create time array
            t_eval = np.linspace(t_start, t_end, num_points)

            # Get power function
            power_func = self._get_power_function()

            # Run prediction
            times, temps = predict_temperature_profile(
                t_span=(t_start, t_end),
                t_eval=t_eval,
                initial_temp=initial_temp,
                thermal_mass=thermal_mass,
                heat_loss_coeff=heat_loss,
                ambient_temp=ambient_temp,
                power_func=power_func,
            )

            # Format results
            results = []
            results.append("Temperature Profile Prediction Results")
            results.append("=" * 40)
            results.append("\nInput Parameters:")
            results.append(f"  Initial Temperature: {initial_temp:.1f} °C")
            results.append(f"  Ambient Temperature: {ambient_temp:.1f} °C")
            results.append(f"  Thermal Mass: {thermal_mass:.0f} J/K")
            results.append(f"  Heat Loss Coeff: {heat_loss:.1f} W/K")
            results.append(f"  Power Profile: {self.power_profile_combo.currentText()}")
            results.append(f"  Power: {self.power_input.value():.0f} W")

            results.append("\nKey Results:")
            results.append(f"  Final Temperature: {temps[-1]:.1f} °C")
            results.append(f"  Max Temperature: {np.max(temps):.1f} °C")
            results.append(f"  Min Temperature: {np.min(temps):.1f} °C")
            results.append(f"  Temperature Change: {temps[-1] - temps[0]:.1f} °C")

            results.append("\nSample Data Points (Time, Temperature):")
            results.append("-" * 30)

            # Show every 10th point
            step = max(1, len(times) // 10)
            for i in range(0, len(times), step):
                results.append(f"  {times[i]:8.0f} s  |  {temps[i]:8.1f} °C")

            self.results_text.setPlainText("\n".join(results))
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")

        except ImportError as e:
            self.results_text.setPlainText(f"Error: {e}")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")


def main() -> int:
    """Run the Thermal Profile Predictor application."""
    app = QApplication(sys.argv)
    window = ThermalProfilePredictorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
