"""Flare Calculator Main Window - PyQt6 GUI.

Provides a standalone interface for flare sizing and safety analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.ui.widgets.base_calculator_widget import BaseCalculatorWindow

# Catppuccin Mocha theme colors
CATPPUCCIN_MOCHA: Final[dict[str, str]] = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "surface0": "#313244",
    "surface1": "#45475a",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
}

# Gas properties database
GAS_PROPERTIES: Final[dict[str, dict[str, float]]] = {
    "H2": {"mw": 2.016, "hv": 119930, "cp": 14.3},
    "CO": {"mw": 28.01, "hv": 10100, "cp": 1.04},
    "CH4": {"mw": 16.04, "hv": 50010, "cp": 2.22},
    "C2H6": {"mw": 30.07, "hv": 47520, "cp": 1.75},
    "C3H8": {"mw": 44.10, "hv": 46360, "cp": 1.67},
    "C4H10": {"mw": 58.12, "hv": 45720, "cp": 1.66},
    "H2S": {"mw": 34.08, "hv": 16500, "cp": 1.05},
    "N2": {"mw": 28.01, "hv": 0, "cp": 1.04},
    "CO2": {"mw": 44.01, "hv": 0, "cp": 0.84},
    "H2O": {"mw": 18.02, "hv": 0, "cp": 1.87},
}

# Universal gas constant [J/(mol·K)]
R_UNIVERSAL: Final[float] = 8.314
HOUR_TO_SECOND: Final[float] = 3600.0


@dataclass
class FlareDesign:
    """Flare design parameters."""

    height: float  # m
    diameter: float  # m
    exit_velocity: float  # m/s
    heat_release: float  # kW
    radiation_intensity: float  # kW/m²


class FlareCalculatorEngine:
    """Core flare calculation engine (standalone version)."""

    def __init__(self) -> None:
        """Initialize the flare calculator."""
        self.gas_properties = GAS_PROPERTIES

    def calculate_flare_size(
        self,
        total_flow: float,
        gas_composition: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> FlareDesign:
        """Calculate flare size based on flow conditions."""
        total_comp = sum(gas_composition.values())
        if total_comp == 0:
            comp_fractions = dict.fromkeys(gas_composition, 0.0)
        else:
            comp_fractions = {k: v / total_comp for k, v in gas_composition.items()}

        mix_mw = sum(
            comp_fractions.get(gas, 0) * self.gas_properties[gas]["mw"]
            for gas in self.gas_properties
            if gas in comp_fractions
        )
        mix_hv = sum(
            comp_fractions.get(gas, 0) * self.gas_properties[gas]["hv"]
            for gas in self.gas_properties
            if gas in comp_fractions
        )

        heat_release = total_flow * mix_hv / HOUR_TO_SECOND

        pressure_pa = pressure * 100000.0
        mix_mw_kg = mix_mw / 1000.0

        if temperature > 0 and mix_mw_kg > 0:
            gas_density = pressure_pa / ((R_UNIVERSAL / mix_mw_kg) * temperature)
        else:
            gas_density = 1.0

        target_velocity = 170.0
        mass_flow_kg_s = total_flow / HOUR_TO_SECOND

        if gas_density > 0 and target_velocity > 0:
            area = mass_flow_kg_s / (gas_density * target_velocity)
            diameter = math.sqrt(4 * area / math.pi)
        else:
            diameter = 0.0

        target_radiation = 1.6
        emissivity = 0.3

        if target_radiation > 0 and heat_release > 0:
            height = math.sqrt(
                emissivity * heat_release / (4 * math.pi * target_radiation)
            )
        else:
            height = 0.0

        height = max(height, 10.0)

        return FlareDesign(
            height=height,
            diameter=diameter,
            exit_velocity=target_velocity,
            heat_release=heat_release,
            radiation_intensity=target_radiation,
        )

    def calculate_radiation_zones(self, flare_design: FlareDesign) -> dict[str, float]:
        """Calculate radiation zones around the flare."""
        zones = {"lethal": 0.0, "damage": 0.0, "safe": 0.0, "comfort": 0.0}
        emissivity = 0.3
        heat_release = flare_design.heat_release
        radiation_levels = {"lethal": 37.5, "damage": 12.5, "safe": 1.6, "comfort": 0.5}

        for zone, level in radiation_levels.items():
            if level > 0 and heat_release > 0:
                distance = math.sqrt(emissivity * heat_release / (4 * math.pi * level))
                zones[zone] = distance

        return zones


class FlareCalculatorMainWindow(BaseCalculatorWindow):
    """Main window for the Flare Calculator application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__(
            calculator_name="FlareCalculator",
            window_title="Flare Calculator",
            min_size=(1200, 800),
        )
        self.engine = FlareCalculatorEngine()
        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QHBoxLayout()
        self.main_layout.addLayout(main_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self._create_input_panel(left_layout)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self._create_results_panel(right_layout)
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])

    def _create_input_panel(self, layout: QVBoxLayout) -> None:
        """Create input panel with operating conditions and gas composition."""
        # Operating Conditions
        conditions_group = QGroupBox("Operating Conditions")
        conditions_layout = QFormLayout()

        self.flow_rate = QDoubleSpinBox()
        self.flow_rate.setRange(0.1, 100000)
        self.flow_rate.setValue(1000)
        self.flow_rate.setSuffix(" kg/hr")
        conditions_layout.addRow("Total Flow Rate:", self.flow_rate)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(200, 1500)
        self.temperature.setValue(473)
        self.temperature.setSuffix(" K")
        conditions_layout.addRow("Temperature:", self.temperature)

        self.pressure = QDoubleSpinBox()
        self.pressure.setRange(0.1, 100)
        self.pressure.setValue(1.5)
        self.pressure.setSuffix(" bar")
        conditions_layout.addRow("Pressure:", self.pressure)

        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)

        # Gas Composition
        composition_group = QGroupBox("Gas Composition (mol%)")
        composition_layout = QFormLayout()

        self.gas_inputs: dict[str, QDoubleSpinBox] = {}
        for gas in ["H2", "CO", "CH4", "CO2", "N2", "H2O", "H2S"]:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 100)
            spinbox.setValue(0)
            spinbox.setSuffix(" %")
            if gas == "H2":
                spinbox.setValue(35)
            elif gas == "CO":
                spinbox.setValue(30)
            elif gas == "CH4":
                spinbox.setValue(5)
            elif gas == "CO2":
                spinbox.setValue(15)
            elif gas == "N2":
                spinbox.setValue(5)
            elif gas == "H2O":
                spinbox.setValue(10)
            self.gas_inputs[gas] = spinbox
            composition_layout.addRow(f"{gas}:", spinbox)

        composition_group.setLayout(composition_layout)
        layout.addWidget(composition_group)

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Flare Design")
        self.calculate_btn.clicked.connect(self._calculate)
        layout.addWidget(self.calculate_btn)

        layout.addStretch()

    def _create_results_panel(self, layout: QVBoxLayout) -> None:
        """Create results panel with design parameters and safety zones."""
        # Flare Design Results
        design_group = QGroupBox("Flare Design Parameters")
        design_layout = QVBoxLayout()

        self.design_table = QTableWidget(5, 2)
        self.design_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        design_header = self.design_table.horizontalHeader()
        if design_header is not None:
            design_header.setStretchLastSection(True)

        design_params = [
            "Flare Height (m)",
            "Flare Diameter (m)",
            "Exit Velocity (m/s)",
            "Heat Release (kW)",
            "Design Radiation (kW/m²)",
        ]
        for i, param in enumerate(design_params):
            self.design_table.setItem(i, 0, QTableWidgetItem(param))
            self.design_table.setItem(i, 1, QTableWidgetItem("-"))

        design_layout.addWidget(self.design_table)
        design_group.setLayout(design_layout)
        layout.addWidget(design_group)

        # Safety Zones
        zones_group = QGroupBox("Radiation Safety Zones")
        zones_layout = QVBoxLayout()

        self.zones_table = QTableWidget(4, 3)
        self.zones_table.setHorizontalHeaderLabels(
            ["Zone", "Radiation (kW/m²)", "Distance (m)"]
        )
        zones_header = self.zones_table.horizontalHeader()
        if zones_header is not None:
            zones_header.setStretchLastSection(True)

        zone_names = [
            ("Lethal", "37.5"),
            ("Damage", "12.5"),
            ("Safe Access", "1.6"),
            ("Comfort", "0.5"),
        ]
        for i, (name, radiation) in enumerate(zone_names):
            self.zones_table.setItem(i, 0, QTableWidgetItem(name))
            self.zones_table.setItem(i, 1, QTableWidgetItem(radiation))
            self.zones_table.setItem(i, 2, QTableWidgetItem("-"))

        zones_layout.addWidget(self.zones_table)
        zones_group.setLayout(zones_layout)
        layout.addWidget(zones_group)

        # Gas Properties Summary
        props_group = QGroupBox("Gas Mixture Properties")
        props_layout = QVBoxLayout()

        self.props_table = QTableWidget(3, 2)
        self.props_table.setHorizontalHeaderLabels(["Property", "Value"])
        props_header = self.props_table.horizontalHeader()
        if props_header is not None:
            props_header.setStretchLastSection(True)

        props = ["Mixture MW (g/mol)", "Heating Value (kJ/kg)", "Gas Density (kg/m³)"]
        for i, prop in enumerate(props):
            self.props_table.setItem(i, 0, QTableWidgetItem(prop))
            self.props_table.setItem(i, 1, QTableWidgetItem("-"))

        props_layout.addWidget(self.props_table)
        props_group.setLayout(props_layout)
        layout.addWidget(props_group)

        layout.addStretch()

    def _calculate(self) -> None:
        """Run flare design calculations."""
        composition = {gas: spin.value() for gas, spin in self.gas_inputs.items()}
        total_flow = self.flow_rate.value()
        temperature = self.temperature.value()
        pressure = self.pressure.value()

        # Calculate flare design
        design = self.engine.calculate_flare_size(
            total_flow, composition, temperature, pressure
        )

        # Update design table
        design_values = [
            f"{design.height:.2f}",
            f"{design.diameter:.4f}",
            f"{design.exit_velocity:.1f}",
            f"{design.heat_release:.1f}",
            f"{design.radiation_intensity:.2f}",
        ]
        for i, val in enumerate(design_values):
            self.design_table.setItem(i, 1, QTableWidgetItem(val))

        # Calculate and update safety zones
        zones = self.engine.calculate_radiation_zones(design)
        zone_keys = ["lethal", "damage", "safe", "comfort"]
        for i, key in enumerate(zone_keys):
            self.zones_table.setItem(i, 2, QTableWidgetItem(f"{zones[key]:.2f}"))

        # Calculate mixture properties
        total_comp = sum(composition.values())
        if total_comp > 0:
            fractions = {k: v / total_comp for k, v in composition.items()}
            mix_mw = sum(
                fractions.get(gas, 0) * GAS_PROPERTIES[gas]["mw"]
                for gas in GAS_PROPERTIES
                if gas in fractions
            )
            mix_hv = sum(
                fractions.get(gas, 0) * GAS_PROPERTIES[gas]["hv"]
                for gas in GAS_PROPERTIES
                if gas in fractions
            )

            pressure_pa = pressure * 100000.0
            mix_mw_kg = mix_mw / 1000.0
            if temperature > 0 and mix_mw_kg > 0:
                gas_density = pressure_pa / ((R_UNIVERSAL / mix_mw_kg) * temperature)
            else:
                gas_density = 0.0

            self.props_table.setItem(0, 1, QTableWidgetItem(f"{mix_mw:.2f}"))
            self.props_table.setItem(1, 1, QTableWidgetItem(f"{mix_hv:.1f}"))
            self.props_table.setItem(2, 1, QTableWidgetItem(f"{gas_density:.3f}"))

    def _apply_theme(self) -> None:
        """Apply Catppuccin Mocha dark theme."""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {CATPPUCCIN_MOCHA["base"]};
                color: {CATPPUCCIN_MOCHA["text"]};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {CATPPUCCIN_MOCHA["blue"]};
            }}
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: {CATPPUCCIN_MOCHA["surface0"]};
                border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
                border-radius: 4px;
                padding: 4px;
                color: {CATPPUCCIN_MOCHA["text"]};
            }}
            QPushButton {{
                background-color: {CATPPUCCIN_MOCHA["blue"]};
                color: {CATPPUCCIN_MOCHA["base"]};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {CATPPUCCIN_MOCHA["mauve"]};
            }}
            QTableWidget {{
                background-color: {CATPPUCCIN_MOCHA["surface0"]};
                border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
                gridline-color: {CATPPUCCIN_MOCHA["surface1"]};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QHeaderView::section {{
                background-color: {CATPPUCCIN_MOCHA["surface1"]};
                color: {CATPPUCCIN_MOCHA["text"]};
                padding: 4px;
                border: none;
            }}
        """)


def main() -> None:
    """Run the Flare Calculator application."""
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = FlareCalculatorMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
