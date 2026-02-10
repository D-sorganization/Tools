#!/usr/bin/env python3
"""Syngas Water Calculator PyQt6 Main Window.

A PyQt6 GUI for calculating water content and dew point in syngas systems.
"""

from __future__ import annotations

import sys

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


class SyngasWaterCalculatorWindow(QMainWindow):
    """Main window for Syngas Water Calculator application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.result_labels: dict[str, QLabel] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Syngas Water Calculator")
        self.setMinimumSize(650, 700)
        self.setStyleSheet(STYLESHEET)

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
        title_label = QLabel("Syngas Water Calculator")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Input section
        main_layout.addWidget(self._create_input_group())

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Water Content")
        self.calculate_btn.clicked.connect(self._calculate)
        main_layout.addWidget(self.calculate_btn)

        # Results section
        main_layout.addWidget(self._create_results_group())

        # Risk assessment section
        main_layout.addWidget(self._create_risk_group())

        main_layout.addStretch()

    def _create_input_group(self) -> QGroupBox:
        """Create the input parameters group."""
        group = QGroupBox("Input Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Temperature
        layout.addWidget(QLabel("Temperature (°C):"), 0, 0)
        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-50, 400)
        self.temp_input.setDecimals(1)
        self.temp_input.setValue(40.0)
        self.temp_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.temp_input, 0, 1)

        # Pressure
        layout.addWidget(QLabel("Pressure (bar):"), 1, 0)
        self.pressure_input = QDoubleSpinBox()
        self.pressure_input.setRange(0.1, 500)
        self.pressure_input.setDecimals(2)
        self.pressure_input.setValue(30.0)
        self.pressure_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.pressure_input, 1, 1)

        # Gas composition preset
        layout.addWidget(QLabel("Gas Composition:"), 2, 0)
        self.composition_combo = QComboBox()
        self.composition_combo.addItems(
            [
                "Typical Syngas",
                "Biomass Gasification",
                "Coal Gasification",
                "Natural Gas Reforming",
            ]
        )
        layout.addWidget(self.composition_combo, 2, 1)

        # Vapor pressure method
        layout.addWidget(QLabel("Calculation Method:"), 3, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["Auto (Recommended)", "Antoine", "Buck", "IAPWS-IF97", "Magnus"]
        )
        layout.addWidget(self.method_combo, 3, 1)

        return group

    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Water Content Results")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        result_fields = [
            ("Mole Fraction:", "mole_fraction"),
            ("Water Content (mg/Nm³):", "mg_nm3"),
            ("Water Content (ppmv):", "ppmv"),
            ("Water Content (g/m³):", "g_m3"),
            ("Water Content (lb/MMscf):", "lb_mmscf"),
            ("Vapor Pressure (bar):", "vapor_pressure"),
            ("Dew Point (°C):", "dew_point"),
        ]

        for row, (label_text, key) in enumerate(result_fields):
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['subtext1']};")
            layout.addWidget(label, row, 0)

            value_label = QLabel("--")
            value_label.setStyleSheet(
                f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
            )
            layout.addWidget(value_label, row, 1)
            self.result_labels[key] = value_label

        return group

    def _create_risk_group(self) -> QGroupBox:
        """Create the condensation risk assessment group."""
        group = QGroupBox("Condensation Risk Assessment")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Dew point margin
        layout.addWidget(QLabel("Temperature Margin (°C):"), 0, 0)
        self.margin_label = QLabel("--")
        self.margin_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
        )
        layout.addWidget(self.margin_label, 0, 1)

        # Risk level
        layout.addWidget(QLabel("Risk Level:"), 1, 0)
        self.risk_label = QLabel("--")
        self.risk_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
        )
        layout.addWidget(self.risk_label, 1, 1)

        # Recommended temperature
        layout.addWidget(QLabel("Recommended Min Temp (°C):"), 2, 0)
        self.recommended_temp_label = QLabel("--")
        self.recommended_temp_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['yellow']}; font-weight: bold;"
        )
        layout.addWidget(self.recommended_temp_label, 2, 1)

        return group

    def _calculate(self) -> None:
        """Perform the water content calculation."""
        try:
            from upstream_drift_tools.process_calculators.syngas_water_calculator import (
                SyngasWaterCalculator,
                estimate_condensation_risk,
            )

            # Get inputs
            temperature_c = self.temp_input.value()
            pressure_bar = self.pressure_input.value()

            # Map composition selection to preset key
            composition_map = {
                "Typical Syngas": "typical_syngas",
                "Biomass Gasification": "biomass_syngas",
                "Coal Gasification": "coal_syngas",
                "Natural Gas Reforming": "natural_gas_reforming",
            }
            composition_key = composition_map.get(
                self.composition_combo.currentText(), "typical_syngas"
            )

            # Map method selection
            method_map = {
                "Auto (Recommended)": "auto",
                "Antoine": "antoine",
                "Buck": "buck",
                "IAPWS-IF97": "iapws",
                "Magnus": "magnus",
            }
            method = method_map.get(self.method_combo.currentText(), "auto")

            # Calculate water content
            calculator = SyngasWaterCalculator()
            result = calculator.calculate_water_content(
                temperature_c, pressure_bar, composition_key, method
            )

            # Update result labels
            self.result_labels["mole_fraction"].setText(
                f"{result.mole_fraction_water:.6f}"
            )
            self.result_labels["mg_nm3"].setText(
                f"{result.water_content_mg_per_nm3:,.2f}"
            )
            self.result_labels["ppmv"].setText(f"{result.water_content_ppmv:,.1f}")
            self.result_labels["g_m3"].setText(f"{result.water_content_g_per_m3:,.4f}")
            self.result_labels["lb_mmscf"].setText(
                f"{result.water_content_lb_per_mmscf:,.2f}"
            )
            self.result_labels["vapor_pressure"].setText(
                f"{result.vapor_pressure_bar:.4f}"
            )
            self.result_labels["dew_point"].setText(f"{result.dew_point_c:.1f}")

            # Update risk assessment
            risk = estimate_condensation_risk(temperature_c, pressure_bar)

            margin = risk["temperature_margin_c"]
            self.margin_label.setText(f"{margin:.1f}")

            risk_level = str(risk["condensation_risk"])
            self.risk_label.setText(risk_level)

            # Color code risk level
            if "Critical" in risk_level:
                self.risk_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['red']}; font-weight: bold;"
                )
                self.margin_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['red']}; font-weight: bold;"
                )
            elif risk_level == "High":
                self.risk_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['peach']}; font-weight: bold;"
                )
                self.margin_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['peach']}; font-weight: bold;"
                )
            elif risk_level == "Medium":
                self.risk_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['yellow']}; font-weight: bold;"
                )
                self.margin_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['yellow']}; font-weight: bold;"
                )
            else:
                self.risk_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
                )
                self.margin_label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
                )

            recommended_temp = risk["recommended_temperature_c"]
            self.recommended_temp_label.setText(f"{recommended_temp:.1f}")

            # Set result colors to green for success
            for key in self.result_labels:
                self.result_labels[key].setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
                )

        except ImportError as e:
            # Show error in results
            error_msg = f"Error: {e}"
            for label in self.result_labels.values():
                label.setText("--")
                label.setStyleSheet(
                    f"color: {CATPPUCCIN_MOCHA['red']}; font-weight: bold;"
                )
            self.margin_label.setText(error_msg[:30])
            self.margin_label.setStyleSheet(
                f"color: {CATPPUCCIN_MOCHA['red']}; font-weight: bold;"
            )
            self.risk_label.setText("--")
            self.recommended_temp_label.setText("--")


def main() -> int:
    """Run the Syngas Water Calculator application."""
    app = QApplication(sys.argv)
    window = SyngasWaterCalculatorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
