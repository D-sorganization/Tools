"""Baghouse Calculator PyQt6 Main Window.

Provides a GUI for baghouse filter performance calculations
using the Catppuccin Mocha dark theme.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Catppuccin Mocha colors
COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "surface0": "#313244",
    "surface1": "#45475a",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "lavender": "#b4befe",
}


def get_stylesheet() -> str:
    """Generate Catppuccin Mocha stylesheet."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {COLORS["base"]};
            color: {COLORS["text"]};
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {COLORS["surface1"]};
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
            background-color: {COLORS["mantle"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {COLORS["lavender"]};
        }}
        QLabel {{ color: {COLORS["text"]}; }}
        QDoubleSpinBox {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            padding: 4px 8px;
            min-width: 100px;
        }}
        QDoubleSpinBox:focus {{ border-color: {COLORS["blue"]}; }}
        QPushButton {{
            background-color: {COLORS["blue"]};
            color: {COLORS["base"]};
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {COLORS["lavender"]}; }}
        QScrollArea {{ border: none; background-color: transparent; }}
    """


@dataclass
class BaghouseDesign:
    """Results container for display."""

    carbon_removed: float = 0.0
    ash_removed: float = 0.0
    total_solids: float = 0.0
    drum_fill_hours: float = 0.0
    drum_fill_days: float = 0.0
    flow_acfm: float = 0.0
    air_to_cloth: float = 0.0
    outlet_temp_c: float = 0.0


class BaghouseCalculatorEngine:
    """Wrapper for the baghouse calculator engine."""

    def __init__(self) -> None:
        """Initialize the engine."""
        from upstream_drift_tools.process_calculators.baghouse_calculator import (
            BaghouseCalculator,
        )

        self._calculator = BaghouseCalculator()

    def calculate(
        self,
        gas_flow: float,
        inlet_temp: float,
        pressure: float,
        carbon_in: float,
        ash_in: float,
        carbon_eff: float,
        ash_eff: float,
        heat_loss: float,
        drum_volume: float,
        solid_density: float,
        bag_area: float,
    ) -> BaghouseDesign:
        """Run baghouse calculation."""
        result = self._calculator.calculate(
            gas_flow_kg_s=gas_flow,
            inlet_temp_k=inlet_temp + 273.15,
            pressure_pa=pressure * 1000,  # kPa to Pa
            composition={"H2": 0.35, "CO": 0.30, "CO2": 0.15, "N2": 0.10, "H2O": 0.10},
            solid_carbon_in_kg_hr=carbon_in,
            ash_in_kg_hr=ash_in,
            carbon_removal_efficiency=carbon_eff / 100,
            ash_removal_efficiency=ash_eff / 100,
            heat_loss_w=heat_loss * 1000,  # kW to W
            drum_volume_m3=drum_volume,
            solid_density_kg_m3=solid_density,
            bag_area_ft2=bag_area,
        )

        return BaghouseDesign(
            carbon_removed=result.carbon_removed_rate,
            ash_removed=result.ash_removed_rate,
            total_solids=result.total_solids_removed_rate,
            drum_fill_hours=result.drum_fill_time_hours,
            drum_fill_days=result.drum_fill_time_days,
            flow_acfm=result.flow_acfm,
            air_to_cloth=result.air_to_cloth_ratio,
            outlet_temp_c=result.outlet_temperature_c,
        )


class BaghouseCalculatorMainWindow(QMainWindow):
    """Main window for Baghouse Calculator application."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.setWindowTitle("Baghouse Calculator")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(get_stylesheet())

        self.engine = BaghouseCalculatorEngine()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - inputs
        left_panel = self._create_input_panel()
        layout.addWidget(left_panel, stretch=1)

        # Right panel - results
        right_panel = self._create_results_panel()
        layout.addWidget(right_panel, stretch=1)

    def _create_input_panel(self) -> QWidget:
        """Create the input panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Title
        title = QLabel("Baghouse Filter Calculator")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['blue']};")
        layout.addWidget(title)

        # Gas Flow Group
        gas_group = QGroupBox("Gas Stream")
        gas_layout = QFormLayout(gas_group)

        self.gas_flow_input = QDoubleSpinBox()
        self.gas_flow_input.setRange(0, 1000)
        self.gas_flow_input.setValue(10)
        self.gas_flow_input.setSuffix(" kg/s")
        gas_layout.addRow("Gas Flow Rate:", self.gas_flow_input)

        self.inlet_temp_input = QDoubleSpinBox()
        self.inlet_temp_input.setRange(0, 1000)
        self.inlet_temp_input.setValue(200)
        self.inlet_temp_input.setSuffix(" °C")
        gas_layout.addRow("Inlet Temperature:", self.inlet_temp_input)

        self.pressure_input = QDoubleSpinBox()
        self.pressure_input.setRange(50, 500)
        self.pressure_input.setValue(101.325)
        self.pressure_input.setSuffix(" kPa")
        gas_layout.addRow("Pressure:", self.pressure_input)

        layout.addWidget(gas_group)

        # Solids Input Group
        solids_group = QGroupBox("Solids Input")
        solids_layout = QFormLayout(solids_group)

        self.carbon_input = QDoubleSpinBox()
        self.carbon_input.setRange(0, 1000)
        self.carbon_input.setValue(50)
        self.carbon_input.setSuffix(" kg/hr")
        solids_layout.addRow("Carbon Rate:", self.carbon_input)

        self.ash_input = QDoubleSpinBox()
        self.ash_input.setRange(0, 500)
        self.ash_input.setValue(20)
        self.ash_input.setSuffix(" kg/hr")
        solids_layout.addRow("Ash Rate:", self.ash_input)

        layout.addWidget(solids_group)

        # Efficiency Group
        eff_group = QGroupBox("Removal Efficiency")
        eff_layout = QFormLayout(eff_group)

        self.carbon_eff_input = QDoubleSpinBox()
        self.carbon_eff_input.setRange(0, 100)
        self.carbon_eff_input.setValue(99)
        self.carbon_eff_input.setSuffix(" %")
        eff_layout.addRow("Carbon Removal:", self.carbon_eff_input)

        self.ash_eff_input = QDoubleSpinBox()
        self.ash_eff_input.setRange(0, 100)
        self.ash_eff_input.setValue(99)
        self.ash_eff_input.setSuffix(" %")
        eff_layout.addRow("Ash Removal:", self.ash_eff_input)

        layout.addWidget(eff_group)

        # Equipment Group
        equip_group = QGroupBox("Equipment Parameters")
        equip_layout = QFormLayout(equip_group)

        self.heat_loss_input = QDoubleSpinBox()
        self.heat_loss_input.setRange(0, 100)
        self.heat_loss_input.setValue(5)
        self.heat_loss_input.setSuffix(" kW")
        equip_layout.addRow("Heat Loss:", self.heat_loss_input)

        self.drum_volume_input = QDoubleSpinBox()
        self.drum_volume_input.setRange(0.1, 10)
        self.drum_volume_input.setValue(0.5)
        self.drum_volume_input.setSuffix(" m³")
        equip_layout.addRow("Drum Volume:", self.drum_volume_input)

        self.solid_density_input = QDoubleSpinBox()
        self.solid_density_input.setRange(100, 2000)
        self.solid_density_input.setValue(500)
        self.solid_density_input.setSuffix(" kg/m³")
        equip_layout.addRow("Solid Density:", self.solid_density_input)

        self.bag_area_input = QDoubleSpinBox()
        self.bag_area_input.setRange(100, 10000)
        self.bag_area_input.setValue(1000)
        self.bag_area_input.setSuffix(" ft²")
        equip_layout.addRow("Bag Filter Area:", self.bag_area_input)

        layout.addWidget(equip_group)

        # Calculate Button
        self.calculate_btn = QPushButton("Calculate Baghouse Performance")
        self.calculate_btn.setMinimumHeight(50)
        self.calculate_btn.clicked.connect(self._on_calculate)
        layout.addWidget(self.calculate_btn)

        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_results_panel(self) -> QWidget:
        """Create the results panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)

        # Results title
        title = QLabel("Calculation Results")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['green']};")
        layout.addWidget(title)

        # Key metrics grid
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(
            f"background-color: {COLORS['surface0']}; "
            f"border-radius: 8px; padding: 15px;"
        )
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setSpacing(20)

        self.metric_labels = {}
        metrics = [
            ("carbon", "Carbon Removed", "0 kg/hr", COLORS["green"]),
            ("ash", "Ash Removed", "0 kg/hr", COLORS["yellow"]),
            ("total", "Total Solids", "0 kg/hr", COLORS["blue"]),
            ("fill_hours", "Drum Fill Time", "0 hours", COLORS["peach"]),
            ("fill_days", "Drum Fill Time", "0 days", COLORS["mauve"]),
            ("acfm", "Gas Flow", "0 ACFM", COLORS["teal"]),
            ("atc", "Air-to-Cloth", "0 ft/min", COLORS["lavender"]),
            ("outlet_temp", "Outlet Temp", "0 °C", COLORS["red"]),
        ]

        for i, (key, label, default, color) in enumerate(metrics):
            row, col = divmod(i, 2)

            name_label = QLabel(label)
            name_label.setStyleSheet(f"color: {COLORS['subtext0']};")

            value_label = QLabel(default)
            value_label.setFont(QFont("", 14, QFont.Weight.Bold))
            value_label.setStyleSheet(f"color: {color};")
            self.metric_labels[key] = value_label

            cell = QVBoxLayout()
            cell.addWidget(name_label)
            cell.addWidget(value_label)
            metrics_layout.addLayout(cell, row, col)

        layout.addWidget(metrics_frame)
        layout.addStretch()

        return container

    def _on_calculate(self) -> None:
        """Handle calculate button click."""
        results = self.engine.calculate(
            gas_flow=self.gas_flow_input.value(),
            inlet_temp=self.inlet_temp_input.value(),
            pressure=self.pressure_input.value(),
            carbon_in=self.carbon_input.value(),
            ash_in=self.ash_input.value(),
            carbon_eff=self.carbon_eff_input.value(),
            ash_eff=self.ash_eff_input.value(),
            heat_loss=self.heat_loss_input.value(),
            drum_volume=self.drum_volume_input.value(),
            solid_density=self.solid_density_input.value(),
            bag_area=self.bag_area_input.value(),
        )

        self._update_results(results)

    def _update_results(self, results: BaghouseDesign) -> None:
        """Update results display."""
        self.metric_labels["carbon"].setText(f"{results.carbon_removed:.1f} kg/hr")
        self.metric_labels["ash"].setText(f"{results.ash_removed:.1f} kg/hr")
        self.metric_labels["total"].setText(f"{results.total_solids:.1f} kg/hr")
        self.metric_labels["fill_hours"].setText(f"{results.drum_fill_hours:.1f} hours")
        self.metric_labels["fill_days"].setText(f"{results.drum_fill_days:.2f} days")
        self.metric_labels["acfm"].setText(f"{results.flow_acfm:,.0f} ACFM")
        self.metric_labels["atc"].setText(f"{results.air_to_cloth:.2f} ft/min")
        self.metric_labels["outlet_temp"].setText(f"{results.outlet_temp_c:.1f} °C")
