#!/usr/bin/env python3
"""Flow Rate Converter PyQt6 Main Window.

A PyQt6 GUI for converting between mass, molar, and volumetric flow rate units.
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
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.ui.widgets.base_calculator_widget import BaseCalculatorWindow

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

QTabWidget::pane {{
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    background-color: {CATPPUCCIN_MOCHA["mantle"]};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["subtext1"]};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["blue"]};
}}

QTabBar::tab:hover {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
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

QLabel[class="result-label"] {{
    color: {CATPPUCCIN_MOCHA["green"]};
    font-weight: bold;
}}

QLabel[class="unit-label"] {{
    color: {CATPPUCCIN_MOCHA["subtext0"]};
}}

QLabel[class="header-label"] {{
    color: {CATPPUCCIN_MOCHA["blue"]};
    font-size: 14px;
    font-weight: bold;
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
    min-width: 120px;
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
    padding: 8px 20px;
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

QFrame[class="separator"] {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
}}
"""


class FlowRateConverterWindow(BaseCalculatorWindow):
    """Main window for Flow Rate Converter application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__(
            calculator_name="FlowRateConverter",
            window_title="Flow Rate Converter",
            min_size=(600, 500),
        )
        self.result_labels: dict[str, QLabel] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setStyleSheet(STYLESHEET)

        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        main_layout = self.main_layout

        # Title
        title_label = QLabel("Flow Rate Converter")
        title_label.setProperty("class", "header-label")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Tab widget for different conversion types
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_mass_tab(), "Mass Flow")
        self.tab_widget.addTab(self._create_molar_tab(), "Molar Flow")
        self.tab_widget.addTab(self._create_volumetric_tab(), "Volumetric Flow")

    def _create_mass_tab(self) -> QWidget:
        """Create the mass flow conversion tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(10)

        input_layout.addWidget(QLabel("Value:"), 0, 0)
        self.mass_value_input = QDoubleSpinBox()
        self.mass_value_input.setRange(0, 1e12)
        self.mass_value_input.setDecimals(6)
        self.mass_value_input.setValue(1000.0)
        self.mass_value_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        input_layout.addWidget(self.mass_value_input, 0, 1)

        input_layout.addWidget(QLabel("From Unit:"), 1, 0)
        self.mass_from_unit = QComboBox()
        self.mass_from_unit.addItems(
            ["kg/s", "kg/h", "kg/min", "g/s", "g/h", "lb/s", "lb/h", "lb/min", "ton/h"]
        )
        self.mass_from_unit.setCurrentText("kg/h")
        input_layout.addWidget(self.mass_from_unit, 1, 1)

        input_layout.addWidget(QLabel("To Unit:"), 2, 0)
        self.mass_to_unit = QComboBox()
        self.mass_to_unit.addItems(
            ["kg/s", "kg/h", "kg/min", "g/s", "g/h", "lb/s", "lb/h", "lb/min", "ton/h"]
        )
        self.mass_to_unit.setCurrentText("lb/h")
        input_layout.addWidget(self.mass_to_unit, 2, 1)

        layout.addWidget(input_group)

        # Convert button
        self.mass_convert_btn = QPushButton("Convert")
        self.mass_convert_btn.clicked.connect(self._convert_mass)
        layout.addWidget(self.mass_convert_btn)

        # Results group
        results_group = QGroupBox("Result")
        results_layout = QVBoxLayout(results_group)

        self.mass_result_label = QLabel("--")
        self.mass_result_label.setProperty("class", "result-label")
        result_font = QFont()
        result_font.setPointSize(16)
        self.mass_result_label.setFont(result_font)
        self.mass_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.mass_result_label)

        layout.addWidget(results_group)
        layout.addStretch()

        return tab

    def _create_molar_tab(self) -> QWidget:
        """Create the molar flow conversion tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(10)

        input_layout.addWidget(QLabel("Value:"), 0, 0)
        self.molar_value_input = QDoubleSpinBox()
        self.molar_value_input.setRange(0, 1e12)
        self.molar_value_input.setDecimals(6)
        self.molar_value_input.setValue(100.0)
        self.molar_value_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        input_layout.addWidget(self.molar_value_input, 0, 1)

        input_layout.addWidget(QLabel("From Unit:"), 1, 0)
        self.molar_from_unit = QComboBox()
        self.molar_from_unit.addItems(
            [
                "mol/s",
                "mol/h",
                "mol/min",
                "kmol/s",
                "kmol/h",
                "kmol/min",
                "lbmol/s",
                "lbmol/h",
                "lbmol/min",
            ]
        )
        self.molar_from_unit.setCurrentText("kmol/h")
        input_layout.addWidget(self.molar_from_unit, 1, 1)

        input_layout.addWidget(QLabel("To Unit:"), 2, 0)
        self.molar_to_unit = QComboBox()
        self.molar_to_unit.addItems(
            [
                "mol/s",
                "mol/h",
                "mol/min",
                "kmol/s",
                "kmol/h",
                "kmol/min",
                "lbmol/s",
                "lbmol/h",
                "lbmol/min",
            ]
        )
        self.molar_to_unit.setCurrentText("lbmol/h")
        input_layout.addWidget(self.molar_to_unit, 2, 1)

        layout.addWidget(input_group)

        # Convert button
        self.molar_convert_btn = QPushButton("Convert")
        self.molar_convert_btn.clicked.connect(self._convert_molar)
        layout.addWidget(self.molar_convert_btn)

        # Results group
        results_group = QGroupBox("Result")
        results_layout = QVBoxLayout(results_group)

        self.molar_result_label = QLabel("--")
        self.molar_result_label.setProperty("class", "result-label")
        result_font = QFont()
        result_font.setPointSize(16)
        self.molar_result_label.setFont(result_font)
        self.molar_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.molar_result_label)

        layout.addWidget(results_group)
        layout.addStretch()

        return tab

    def _create_volumetric_tab(self) -> QWidget:
        """Create the volumetric flow conversion tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Input group
        input_group = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(10)

        input_layout.addWidget(QLabel("Value:"), 0, 0)
        self.vol_value_input = QDoubleSpinBox()
        self.vol_value_input.setRange(0, 1e12)
        self.vol_value_input.setDecimals(6)
        self.vol_value_input.setValue(1000.0)
        self.vol_value_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        input_layout.addWidget(self.vol_value_input, 0, 1)

        input_layout.addWidget(QLabel("From Unit:"), 1, 0)
        self.vol_from_unit = QComboBox()
        self.vol_from_unit.addItems(
            [
                "m3/s",
                "m3/h",
                "m3/min",
                "L/s",
                "L/min",
                "L/h",
                "ft3/s",
                "ft3/min",
                "ft3/h",
                "CFM",
                "GPM",
            ]
        )
        self.vol_from_unit.setCurrentText("m3/h")
        input_layout.addWidget(self.vol_from_unit, 1, 1)

        input_layout.addWidget(QLabel("To Unit:"), 2, 0)
        self.vol_to_unit = QComboBox()
        self.vol_to_unit.addItems(
            [
                "m3/s",
                "m3/h",
                "m3/min",
                "L/s",
                "L/min",
                "L/h",
                "ft3/s",
                "ft3/min",
                "ft3/h",
                "CFM",
                "GPM",
            ]
        )
        self.vol_to_unit.setCurrentText("CFM")
        input_layout.addWidget(self.vol_to_unit, 2, 1)

        layout.addWidget(input_group)

        # Convert button
        self.vol_convert_btn = QPushButton("Convert")
        self.vol_convert_btn.clicked.connect(self._convert_volumetric)
        layout.addWidget(self.vol_convert_btn)

        # Results group
        results_group = QGroupBox("Result")
        results_layout = QVBoxLayout(results_group)

        self.vol_result_label = QLabel("--")
        self.vol_result_label.setProperty("class", "result-label")
        result_font = QFont()
        result_font.setPointSize(16)
        self.vol_result_label.setFont(result_font)
        self.vol_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.vol_result_label)

        layout.addWidget(results_group)
        layout.addStretch()

        return tab

    def _convert_mass(self) -> None:
        """Perform mass flow rate conversion."""
        try:
            from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
                mass_to_mass,
            )

            value = self.mass_value_input.value()
            from_unit = self.mass_from_unit.currentText()
            to_unit = self.mass_to_unit.currentText()

            result = mass_to_mass(value, from_unit, to_unit)
            self.mass_result_label.setText(f"{result:,.6g} {to_unit}")
            self.mass_result_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")
        except ImportError as e:
            self.mass_result_label.setText(f"Error: {e}")
            self.mass_result_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _convert_molar(self) -> None:
        """Perform molar flow rate conversion."""
        try:
            from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
                molar_to_molar,
            )

            value = self.molar_value_input.value()
            from_unit = self.molar_from_unit.currentText()
            to_unit = self.molar_to_unit.currentText()

            result = molar_to_molar(value, from_unit, to_unit)
            self.molar_result_label.setText(f"{result:,.6g} {to_unit}")
            self.molar_result_label.setStyleSheet(
                f"color: {CATPPUCCIN_MOCHA['green']};"
            )
        except ImportError as e:
            self.molar_result_label.setText(f"Error: {e}")
            self.molar_result_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _convert_volumetric(self) -> None:
        """Perform volumetric flow rate conversion."""
        try:
            from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
                VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
            )

            value = self.vol_value_input.value()
            from_unit = self.vol_from_unit.currentText()
            to_unit = self.vol_to_unit.currentText()

            # Convert to m3/s, then to target unit
            m3_per_s = value * VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[from_unit]
            result = m3_per_s / VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[to_unit]

            self.vol_result_label.setText(f"{result:,.6g} {to_unit}")
            self.vol_result_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")
        except ImportError as e:
            self.vol_result_label.setText(f"Error: {e}")
            self.vol_result_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")


def main() -> int:
    """Run the Flow Rate Converter application."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    window = FlowRateConverterWindow()
    setup_themed_app(app, window, settings_app="FlowRateConverter")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
