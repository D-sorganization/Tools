"""Main window for WGS Reactor Calculator PyQt6 application."""

from __future__ import annotations

import math
import sys
from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.ui.catppuccin_theme import COLORS
from upstream_drift_tools.ui.catppuccin_theme import get_stylesheet as _base_stylesheet


def get_stylesheet() -> str:
    """Get the Catppuccin Mocha stylesheet with ResultCard extension."""
    return str(_base_stylesheet() + f"""
        QFrame#resultCard {{
            background-color: {COLORS["surface0"]};
            border-radius: 8px;
            padding: 10px;
        }}
    """)


class ResultCard(QFrame):
    """A card widget for displaying a single result."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {COLORS['subtext0']}; font-size: 11px;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 18px; font-weight: bold;"
        )
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        """Set the displayed value."""
        self.value_label.setText(value)

    def set_color(self, color: str) -> None:
        """Set the value label color."""
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: bold;"
        )


class WGSReactorEngine:
    """Core engine for WGS reactor calculations (standalone version)."""

    def __init__(self) -> None:
        """Initialize the engine."""
        self.R = 8.314  # [J/(mol·K)] Universal gas constant

        # Standard formation enthalpies at 298.15 K [kJ/mol]
        self._formation_enthalpies = {
            "CO": -110.525,
            "CO2": -393.509,
            "H2": 0.0,
            "H2O": -241.826,
        }

        # Standard entropies at 298.15 K [J/(mol·K)]
        self._standard_entropies = {
            "CO": 197.66,
            "CO2": 213.74,
            "H2": 130.68,
            "H2O": 188.83,
        }

    def calculate_equilibrium_constant(self, temperature: float) -> float:
        """Calculate WGS equilibrium constant using Van't Hoff equation.

        CO + H2O <-> CO2 + H2
        dH = -41.2 kJ/mol, dS = -42.1 J/(mol·K)
        """
        # DbC precondition
        assert temperature > 0, f"Temperature must be positive (K), got {temperature}"

        delta_H = -41200  # J/mol
        delta_S = -42.1  # J/(mol·K)

        ln_K = -delta_H / (self.R * temperature) + delta_S / self.R
        return math.exp(ln_K)

    def calculate_equilibrium_composition(
        self,
        inlet_composition: dict[str, float],
        temperature: float,
        pressure: float,
        steam_ratio: float = 2.0,
    ) -> dict[str, Any]:
        """Calculate equilibrium composition for WGS reaction."""
        # DbC preconditions
        assert temperature > 0, f"Temperature must be positive (K), got {temperature}"
        assert pressure > 0, f"Pressure must be positive (bar), got {pressure}"
        assert steam_ratio >= 0, f"Steam ratio must be non-negative, got {steam_ratio}"

        # Initial moles (normalize to 100 basis)
        n_CO_0 = inlet_composition.get("CO", 0)
        n_H2O_0 = inlet_composition.get("H2O", 0) + n_CO_0 * steam_ratio
        n_CO2_0 = inlet_composition.get("CO2", 0)
        n_H2_0 = inlet_composition.get("H2", 0)

        n_total_0 = n_CO_0 + n_H2O_0 + n_CO2_0 + n_H2_0

        if n_total_0 == 0:
            return {
                "conversion": 0.0,
                "composition": {"CO": 0.0, "H2O": 0.0, "CO2": 0.0, "H2": 0.0},
                "h2_co_ratio": 0.0,
                "equilibrium_constant": self.calculate_equilibrium_constant(
                    temperature
                ),
                "heat_released": 0.0,
            }

        K_eq = self.calculate_equilibrium_constant(temperature)

        # Solve for extent of reaction using equilibrium constant
        # K = (n_CO2 * n_H2) / (n_CO * n_H2O)
        # At equilibrium: K = ((n_CO2_0 + x) * (n_H2_0 + x)) / ((n_CO_0 - x) * (n_H2O_0 - x))

        # Solve quadratic: (K-1)x^2 + (K*(n_CO_0 + n_H2O_0) + n_CO2_0 + n_H2_0)x
        #                  + K*n_CO_0*n_H2O_0 - n_CO2_0*n_H2_0 = 0

        a = K_eq - 1
        b = K_eq * (n_CO_0 + n_H2O_0) + n_CO2_0 + n_H2_0
        c = K_eq * n_CO_0 * n_H2O_0 - n_CO2_0 * n_H2_0

        if abs(a) < 1e-10:
            # Linear case (K ≈ 1)
            x_eq = -c / b if abs(b) > 1e-10 else 0
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                x_eq = 0
            else:
                # Take the root that gives valid (positive) compositions
                x1 = (-b + math.sqrt(discriminant)) / (2 * a)
                x2 = (-b - math.sqrt(discriminant)) / (2 * a)

                # Choose the valid extent
                max_extent = min(n_CO_0, n_H2O_0)
                if 0 <= x1 <= max_extent:
                    x_eq = x1
                elif 0 <= x2 <= max_extent:
                    x_eq = x2
                else:
                    x_eq = max(0, min(x1, max_extent))

        # Equilibrium composition
        n_CO_eq = n_CO_0 - x_eq
        n_H2O_eq = n_H2O_0 - x_eq
        n_CO2_eq = n_CO2_0 + x_eq
        n_H2_eq = n_H2_0 + x_eq
        n_total_eq = n_CO_eq + n_H2O_eq + n_CO2_eq + n_H2_eq

        composition_out = {
            "CO": (n_CO_eq / n_total_eq) * 100 if n_total_eq > 0 else 0,
            "H2O": (n_H2O_eq / n_total_eq) * 100 if n_total_eq > 0 else 0,
            "CO2": (n_CO2_eq / n_total_eq) * 100 if n_total_eq > 0 else 0,
            "H2": (n_H2_eq / n_total_eq) * 100 if n_total_eq > 0 else 0,
        }

        conversion = (x_eq / n_CO_0) * 100 if n_CO_0 > 0 else 0
        h2_co_ratio = (
            composition_out["H2"] / composition_out["CO"]
            if composition_out["CO"] > 0
            else float("inf")
        )
        heat_released = x_eq * 41.2  # kJ per mol CO converted

        return {
            "conversion": conversion,
            "composition": composition_out,
            "h2_co_ratio": h2_co_ratio,
            "equilibrium_constant": K_eq,
            "heat_released": heat_released,
        }

    def size_reactor(
        self,
        feed_rate: float,
        conversion: float,
        temperature: float,
    ) -> dict[str, float]:
        """Size WGS reactor based on throughput and conversion."""
        # Space velocity (GHSV) - typical for WGS
        ghsv = 3000  # h^-1

        # Reactor volume
        reactor_volume = feed_rate / ghsv if ghsv > 0 else 0

        # Catalyst volume (80% of reactor)
        catalyst_volume = reactor_volume * 0.8

        # Reactor dimensions (L/D = 3)
        ld_ratio = 3.0
        diameter = (4 * reactor_volume / (math.pi * ld_ratio)) ** (1 / 3)
        length = diameter * ld_ratio

        # Heat duty (kW)
        heat_duty = feed_rate * conversion / 100 * 41.2 / 3.6

        return {
            "reactor_volume": reactor_volume,
            "catalyst_volume": catalyst_volume,
            "diameter": diameter,
            "length": length,
            "heat_duty": heat_duty,
            "ghsv": ghsv,
        }


class WGSReactorWindow(QMainWindow):
    """Main window for WGS Reactor Calculator application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Water-Gas Shift Reactor Calculator")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_stylesheet())

        self.engine = WGSReactorEngine()
        self.results: dict[str, Any] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - Inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # Reactor Configuration Group
        reactor_group = self._create_reactor_group()
        left_layout.addWidget(reactor_group)

        # Feed Composition Group
        feed_group = self._create_feed_group()
        left_layout.addWidget(feed_group)

        # Shift Type Selection
        shift_group = self._create_shift_type_group()
        left_layout.addWidget(shift_group)

        # Calculate Button
        calc_button = QPushButton("Calculate WGS Performance")
        calc_button.setFont(QFont("", 12, QFont.Weight.Bold))
        calc_button.clicked.connect(self._calculate)
        left_layout.addWidget(calc_button)

        left_layout.addStretch()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # Summary Cards
        summary_layout = self._create_summary_cards()
        right_layout.addLayout(summary_layout)

        # Composition Table
        comp_group = self._create_composition_table()
        right_layout.addWidget(comp_group)

        # Reactor Sizing Table
        sizing_group = self._create_sizing_table()
        right_layout.addWidget(sizing_group)

        main_layout.addWidget(right_panel, 2)

    def _create_reactor_group(self) -> QGroupBox:
        """Create the reactor configuration input group."""
        group = QGroupBox("Reactor Configuration")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Temperature
        layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 800)
        self.temp_spin.setValue(400)
        self.temp_spin.setSuffix(" °C")
        layout.addWidget(self.temp_spin, 0, 1)

        # Pressure
        layout.addWidget(QLabel("Pressure:"), 1, 0)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(1, 100)
        self.pressure_spin.setValue(25)
        self.pressure_spin.setSuffix(" bar")
        layout.addWidget(self.pressure_spin, 1, 1)

        # Steam/CO Ratio
        layout.addWidget(QLabel("Steam/CO Ratio:"), 2, 0)
        self.steam_ratio_spin = QDoubleSpinBox()
        self.steam_ratio_spin.setRange(0.5, 10)
        self.steam_ratio_spin.setValue(2.0)
        self.steam_ratio_spin.setDecimals(1)
        layout.addWidget(self.steam_ratio_spin, 2, 1)

        # Feed Rate
        layout.addWidget(QLabel("Feed Rate:"), 3, 0)
        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(1, 100000)
        self.feed_rate_spin.setValue(100)
        self.feed_rate_spin.setSuffix(" kmol/h")
        self.feed_rate_spin.setDecimals(0)
        layout.addWidget(self.feed_rate_spin, 3, 1)

        return group

    def _create_feed_group(self) -> QGroupBox:
        """Create the feed composition input group."""
        group = QGroupBox("Feed Composition (mol%)")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # CO
        layout.addWidget(QLabel("CO:"), 0, 0)
        self.co_spin = QDoubleSpinBox()
        self.co_spin.setRange(0, 100)
        self.co_spin.setValue(25)
        self.co_spin.setSuffix(" %")
        layout.addWidget(self.co_spin, 0, 1)

        # H2
        layout.addWidget(QLabel("H2:"), 1, 0)
        self.h2_spin = QDoubleSpinBox()
        self.h2_spin.setRange(0, 100)
        self.h2_spin.setValue(20)
        self.h2_spin.setSuffix(" %")
        layout.addWidget(self.h2_spin, 1, 1)

        # CO2
        layout.addWidget(QLabel("CO2:"), 2, 0)
        self.co2_spin = QDoubleSpinBox()
        self.co2_spin.setRange(0, 100)
        self.co2_spin.setValue(10)
        self.co2_spin.setSuffix(" %")
        layout.addWidget(self.co2_spin, 2, 1)

        # H2O
        layout.addWidget(QLabel("H2O:"), 3, 0)
        self.h2o_spin = QDoubleSpinBox()
        self.h2o_spin.setRange(0, 100)
        self.h2o_spin.setValue(5)
        self.h2o_spin.setSuffix(" %")
        layout.addWidget(self.h2o_spin, 3, 1)

        # N2 (inert)
        layout.addWidget(QLabel("N2 (inert):"), 4, 0)
        self.n2_spin = QDoubleSpinBox()
        self.n2_spin.setRange(0, 100)
        self.n2_spin.setValue(40)
        self.n2_spin.setSuffix(" %")
        layout.addWidget(self.n2_spin, 4, 1)

        return group

    def _create_shift_type_group(self) -> QGroupBox:
        """Create the shift type selection group."""
        group = QGroupBox("Shift Configuration")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Shift Type:"), 0, 0)
        self.shift_combo = QComboBox()
        self.shift_combo.addItems(
            [
                "High Temperature Shift (HTS)",
                "Low Temperature Shift (LTS)",
                "Two-Stage (HTS + LTS)",
            ]
        )
        self.shift_combo.currentIndexChanged.connect(self._on_shift_type_changed)
        layout.addWidget(self.shift_combo, 0, 1)

        # Catalyst info label
        self.catalyst_info = QLabel("Catalyst: Fe-Cr (350-450°C)")
        self.catalyst_info.setStyleSheet(f"color: {COLORS['subtext0']};")
        layout.addWidget(self.catalyst_info, 1, 0, 1, 2)

        return group

    def _on_shift_type_changed(self, index: int) -> None:
        """Handle shift type selection change."""
        if index == 0:  # HTS
            self.temp_spin.setValue(400)
            self.catalyst_info.setText("Catalyst: Fe-Cr (350-450°C)")
        elif index == 1:  # LTS
            self.temp_spin.setValue(220)
            self.catalyst_info.setText("Catalyst: Cu-Zn-Al (180-250°C)")
        else:  # Two-Stage
            self.temp_spin.setValue(400)
            self.catalyst_info.setText("Stage 1: Fe-Cr, Stage 2: Cu-Zn-Al")

    def _create_summary_cards(self) -> QHBoxLayout:
        """Create summary result cards."""
        layout = QHBoxLayout()
        layout.setSpacing(10)

        self.conversion_card = ResultCard("CO Conversion")
        layout.addWidget(self.conversion_card)

        self.h2_co_card = ResultCard("H2/CO Ratio")
        layout.addWidget(self.h2_co_card)

        self.heat_duty_card = ResultCard("Heat Duty")
        layout.addWidget(self.heat_duty_card)

        self.keq_card = ResultCard("Equilibrium K")
        layout.addWidget(self.keq_card)

        return layout

    def _create_composition_table(self) -> QGroupBox:
        """Create the composition comparison table."""
        group = QGroupBox("Composition Comparison")
        layout = QVBoxLayout(group)

        self.comp_table = QTableWidget()
        self.comp_table.setColumnCount(3)
        self.comp_table.setHorizontalHeaderLabels(
            ["Species", "Inlet (mol%)", "Outlet (mol%)"]
        )
        comp_header = self.comp_table.horizontalHeader()
        if comp_header is not None:
            comp_header.setStretchLastSection(True)
            comp_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.comp_table.setAlternatingRowColors(True)
        self.comp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.comp_table)

        return group

    def _create_sizing_table(self) -> QGroupBox:
        """Create the reactor sizing table."""
        group = QGroupBox("Reactor Sizing")
        layout = QVBoxLayout(group)

        self.sizing_table = QTableWidget()
        self.sizing_table.setColumnCount(2)
        self.sizing_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        sizing_header = self.sizing_table.horizontalHeader()
        if sizing_header is not None:
            sizing_header.setStretchLastSection(True)
            sizing_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sizing_table.setAlternatingRowColors(True)
        self.sizing_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.sizing_table)

        return group

    def _calculate(self) -> None:
        """Perform the WGS reactor calculations."""
        # Get inputs
        temp_c = self.temp_spin.value()
        temp_k = temp_c + 273.15
        pressure = self.pressure_spin.value()
        steam_ratio = self.steam_ratio_spin.value()
        feed_rate = self.feed_rate_spin.value()

        inlet_composition = {
            "CO": self.co_spin.value(),
            "H2": self.h2_spin.value(),
            "CO2": self.co2_spin.value(),
            "H2O": self.h2o_spin.value(),
        }

        # Calculate equilibrium
        equilibrium = self.engine.calculate_equilibrium_composition(
            inlet_composition, temp_k, pressure, steam_ratio
        )

        # Size reactor
        sizing = self.engine.size_reactor(feed_rate, equilibrium["conversion"], temp_k)

        # Store results
        self.results = {
            "equilibrium": equilibrium,
            "sizing": sizing,
            "inlet": inlet_composition,
        }

        # Update UI
        self._update_results_display()

    def _update_results_display(self) -> None:
        """Update the results display with calculated values."""
        eq = self.results["equilibrium"]
        sizing = self.results["sizing"]
        inlet = self.results["inlet"]

        # Update summary cards
        self.conversion_card.set_value(f"{eq['conversion']:.1f}%")
        if eq["conversion"] > 80:
            self.conversion_card.set_color(COLORS["green"])
        elif eq["conversion"] > 50:
            self.conversion_card.set_color(COLORS["yellow"])
        else:
            self.conversion_card.set_color(COLORS["red"])

        h2_co = eq["h2_co_ratio"]
        if h2_co == float("inf"):
            self.h2_co_card.set_value("∞")
        else:
            self.h2_co_card.set_value(f"{h2_co:.2f}")

        self.heat_duty_card.set_value(f"{sizing['heat_duty']:.1f} kW")
        self.keq_card.set_value(f"{eq['equilibrium_constant']:.2f}")

        # Update composition table
        species = ["CO", "H2", "CO2", "H2O"]
        self.comp_table.setRowCount(len(species))
        for i, sp in enumerate(species):
            self.comp_table.setItem(i, 0, QTableWidgetItem(sp))
            self.comp_table.setItem(i, 1, QTableWidgetItem(f"{inlet.get(sp, 0):.2f}"))
            self.comp_table.setItem(
                i, 2, QTableWidgetItem(f"{eq['composition'].get(sp, 0):.2f}")
            )

        # Update sizing table
        sizing_data = [
            ("Reactor Volume", f"{sizing['reactor_volume']:.2f} m³"),
            ("Catalyst Volume", f"{sizing['catalyst_volume']:.2f} m³"),
            ("Diameter", f"{sizing['diameter']:.2f} m"),
            ("Length", f"{sizing['length']:.2f} m"),
            ("GHSV", f"{sizing['ghsv']:.0f} h⁻¹"),
            ("Heat Released", f"{eq['heat_released']:.1f} kJ/mol CO"),
        ]

        self.sizing_table.setRowCount(len(sizing_data))
        for i, (param, value) in enumerate(sizing_data):
            self.sizing_table.setItem(i, 0, QTableWidgetItem(param))
            self.sizing_table.setItem(i, 1, QTableWidgetItem(value))


def main() -> None:
    """Run the WGS Reactor Calculator application."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = WGSReactorWindow()
    setup_themed_app(app, window, settings_app="WGSReactorCalculator")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
