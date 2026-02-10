#!/usr/bin/env python3
"""Water-Gas Shift Reactor Calculator
====================================

Comprehensive WGS reactor analysis tool for:
- High-temperature shift (HTS) and low-temperature shift (LTS)
- Equilibrium composition at various temperatures
- H2/CO ratio adjustment for downstream processes
- Reactor sizing and heat integration
- Catalyst selection guidance
- Multi-stage reactor design

Author: AI Assistant
Version: 1.0
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from scipy.optimize import minimize


def safe_read_json(file_path: Path | str, default: Any = None) -> Any:
    """Read JSON from a file, returning a default on failure."""
    path = Path(file_path)
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt6.QtCore import QTimer, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QScrollArea,
        QSplitter,
        QTableWidget,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True

else:
    try:
        from PyQt6.QtCore import QTimer, pyqtSignal
        from PyQt6.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QLabel,
            QPushButton,
            QScrollArea,
            QSplitter,
            QTableWidget,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False
        # Mock/dummy classes to prevent NameError in type hints or unused imports if needed
        QWidget = object
        QTimer = object

        def pyqtSignal(*args):
            return None

    try:
        HEADLESS = os.environ.get("HEADLESS", "false").lower() == "true"
        if PYQT_AVAILABLE and not HEADLESS:
            mpl.use("QtAgg")
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
            from matplotlib.figure import Figure
        else:
            # Fallback for headless environments
            mpl.use("Agg")
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure
    except ImportError:
        pass

# Try to import species database; provide fallback for standalone use
try:
    from integrated_process_simulator.calculators.thermodynamic_properties.species_database import (
        DEFAULT_DATABASE_PATH,
        get_species_database,
    )

    HAS_SPECIES_DB = True
except ImportError:
    HAS_SPECIES_DB = False
    DEFAULT_DATABASE_PATH = None

    class _MinimalSpeciesDB:
        """Minimal species database fallback for standalone use."""

        # Standard formation enthalpies at 298.15 K [kJ/mol]
        _FORMATION_ENTHALPIES = {
            "CO": -110.525,
            "CO2": -393.509,
            "H2": 0.0,
            "H2O": -241.826,
            "CH4": -74.81,
            "N2": 0.0,
            "O2": 0.0,
        }

        # Standard entropies at 298.15 K [J/(mol·K)]
        _STANDARD_ENTROPIES = {
            "CO": 197.66,
            "CO2": 213.74,
            "H2": 130.68,
            "H2O": 188.83,
            "CH4": 186.26,
            "N2": 191.61,
            "O2": 205.14,
        }

        def get_formation_enthalpy(self, species: str) -> float | None:
            return self._FORMATION_ENTHALPIES.get(species)

        def get_standard_entropy(self, species: str) -> float | None:
            return self._STANDARD_ENTROPIES.get(species)

        def get_molecular_weight(self, species: str) -> float | None:
            mw = {
                "CO": 0.028,
                "CO2": 0.044,
                "H2": 0.002,
                "H2O": 0.018,
                "CH4": 0.016,
                "N2": 0.028,
                "O2": 0.032,
            }
            return mw.get(species)

    _minimal_db = _MinimalSpeciesDB()

    def get_species_database():
        return _minimal_db


try:
    from integrated_process_simulator.ui.mixins.base_calculator_widget import (
        BaseCalculatorWidget,
    )

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class WGSReactorEngine:
    """Core engine for WGS reactor calculations"""

    def __init__(self, data_file: str | None = None) -> None:
        """Initialize the engine"""
        self.R = 8.314  # [J/(mol·K)] Universal gas constant, NIST CODATA 2018
        self.catalysts: dict[str, Any] = {}
        self.species_db = get_species_database()
        if data_file:
            self._load_data(data_file)
        else:
            self._load_data(DEFAULT_DATABASE_PATH)

    def _load_data(self, data_file: str | None) -> None:
        """Load catalyst and other relevant data from a JSON file."""
        if data_file is None:
            logger.debug("No data file specified, using empty catalyst data")
            return
        data = safe_read_json(data_file, default={})
        self.catalysts = data.get("catalysts", {})

    def calculate_equilibrium_constant(self, temperature: float) -> float:
        """Calculate WGS equilibrium constant using Van't Hoff equation"""
        # CO + H2O ⇌ CO2 + H2
        # ΔH° = -41.2 kJ/mol, ΔS° = -42.1 J/(mol·K)

        delta_H = -41200  # J/mol
        delta_S = -42.1  # J/(mol·K)

        # Van't Hoff equation
        ln_K = -delta_H / (self.R * temperature) + delta_S / self.R
        K_eq = math.exp(ln_K)

        return K_eq

    def calculate_equilibrium_composition(
        self,
        inlet_composition: dict[str, float],
        temperature: float,
        pressure: float,
        steam_ratio: float = 2.0,
    ) -> dict[str, Any]:
        """Calculate equilibrium composition for WGS reaction
        using Gibbs free energy minimization."""

        # Initial moles
        n_CO_0 = inlet_composition.get("CO", 0)
        n_H2O_0 = inlet_composition.get("H2O", 0) + n_CO_0 * steam_ratio
        n_CO2_0 = inlet_composition.get("CO2", 0)
        n_H2_0 = inlet_composition.get("H2", 0)

        # Total initial moles
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

        # Gibbs free energy of formation at standard state (298.15 K) in J/mol
        def get_g_f(species_name):
            species = self.species_db.get_species(species_name)
            if not species:
                return 0
            # G = H - TS
            # Using actual reaction temperature for Gibbs free energy calculation
            return (
                species.formation_enthalpy * 1000
                - temperature * species.formation_entropy
            )

        g_f_CO = get_g_f("CO")
        g_f_H2O = get_g_f("H2O_g")
        g_f_CO2 = get_g_f("CO2")
        g_f_H2 = get_g_f("H2")

        # The objective function to minimize is the total Gibbs free energy of the mixture
        def total_gibbs_energy(x):
            # x is the extent of reaction
            # Scipy minimize passes x as a numpy array
            extent = x[0] if hasattr(x, "__len__") else x

            n_CO = n_CO_0 - extent
            n_H2O = n_H2O_0 - extent
            n_CO2 = n_CO2_0 + extent
            n_H2 = n_H2_0 + extent
            n_total = n_total_0

            # Mole fractions
            # Mole fractions - Handle potential division by zero if n_total is near zero
            if n_total > 1e-10:
                y_CO = n_CO / n_total
                y_H2O = n_H2O / n_total
                y_CO2 = n_CO2 / n_total
                y_H2 = n_H2 / n_total
            else:
                y_CO = 0.0
                y_H2O = 0.0
                y_CO2 = 0.0
                y_H2 = 0.0

            # Partial pressures
            # Input pressure is in bar, convert to Pa for standard state comparison
            pressure_pa = pressure * 100000.0
            p_CO = y_CO * pressure_pa
            p_H2O = y_H2O * pressure_pa
            p_CO2 = y_CO2 * pressure_pa
            p_H2 = y_H2 * pressure_pa

            # Gibbs free energy of each component at reaction conditions
            # Standard pressure (1 bar = 100,000 Pa) used for reference state
            P_std = 100000.0

            # Gibbs free energy of each component at reaction conditions
            # Use activity (p_i / P_std) for log term to ensure dimensionless argument
            g_CO = (
                g_f_CO + self.R * temperature * math.log(p_CO / P_std)
                if p_CO > 0
                else 0
            )
            g_H2O = (
                g_f_H2O + self.R * temperature * math.log(p_H2O / P_std)
                if p_H2O > 0
                else 0
            )
            g_CO2 = (
                g_f_CO2 + self.R * temperature * math.log(p_CO2 / P_std)
                if p_CO2 > 0
                else 0
            )
            g_H2 = (
                g_f_H2 + self.R * temperature * math.log(p_H2 / P_std)
                if p_H2 > 0
                else 0
            )

            # Total Gibbs energy of the mixture
            return n_CO * g_CO + n_H2O * g_H2O + n_CO2 * g_CO2 + n_H2 * g_H2

        # Initial guess for the extent of reaction
        x_initial = 0.5 * min(n_CO_0, n_H2O_0)

        # Bounds for the extent of reaction
        bounds = [(0, min(n_CO_0, n_H2O_0))]

        # Minimize the total Gibbs free energy
        result = minimize(total_gibbs_energy, x_initial, bounds=bounds)
        x_eq = result.x[0]

        # Equilibrium composition
        n_CO_eq = n_CO_0 - x_eq
        n_H2O_eq = n_H2O_0 - x_eq
        n_CO2_eq = n_CO2_0 + x_eq
        n_H2_eq = n_H2_0 + x_eq
        n_total_eq = n_CO_eq + n_H2O_eq + n_CO2_eq + n_H2_eq

        composition_out = {
            "CO": (n_CO_eq / n_total_eq) * 100,
            "H2O": (n_H2O_eq / n_total_eq) * 100,
            "CO2": (n_CO2_eq / n_total_eq) * 100,
            "H2": (n_H2_eq / n_total_eq) * 100,
        }

        conversion = (x_eq / n_CO_0) * 100 if n_CO_0 > 0 else 0
        h2_co_ratio = (
            composition_out["H2"] / composition_out["CO"]
            if composition_out["CO"] > 0
            else float("inf")
        )
        heat_released = x_eq * 41.2  # kJ/mol CO in

        return {
            "conversion": conversion,
            "composition": composition_out,
            "h2_co_ratio": h2_co_ratio,
            "equilibrium_constant": self.calculate_equilibrium_constant(temperature),
            "heat_released": heat_released,
        }

    def size_wgs_reactor(
        self,
        feed_rate: float,
        conversion: float,
        temperature: float,
        catalyst_type: str,
    ) -> dict[str, Any]:
        """Size WGS reactor based on throughput and conversion"""
        # Space velocity (GHSV)
        ghsv = 3000  # h^-1 (typical for WGS)

        # Reactor volume
        reactor_volume = feed_rate / ghsv  # m3

        # Catalyst volume (80% of reactor)
        catalyst_volume = reactor_volume * 0.8

        # Reactor dimensions (L/D = 3)
        ld_ratio = 3.0
        diameter = (4 * reactor_volume / (math.pi * ld_ratio)) ** (1 / 3)
        length = diameter * ld_ratio

        # Heat duty
        heat_duty = feed_rate * conversion / 100 * 41.2 / 3.6  # kW

        return {
            "reactor_volume": reactor_volume,
            "catalyst_volume": catalyst_volume,
            "diameter": diameter,
            "length": length,
            "heat_duty": heat_duty,
            "ghsv": ghsv,
        }


if BASE_CALCULATOR_AVAILABLE:

    class WGSReactorCalculatorWidget(BaseCalculatorWidget):
        """Main WGS reactor calculator widget"""

        calculation_finished = pyqtSignal(dict)

        def __init__(self, parent=None) -> None:
            """Initialize calculator"""
            super().__init__(calculator_name="WGSReactor", parent=parent)
            self.engine = WGSReactorEngine()
            self.init_ui()
            self.set_default_values()
            QTimer.singleShot(0, self.setup_state_management)

        def setup_state_management(self) -> None:
            """Set up state management"""
            for splitter in self.findChildren(QSplitter):
                self.register_splitter(splitter, "main_splitter")
            for table in self.findChildren(QTableWidget):
                self.register_copyable_widget(table, "table")
            for text_edit in self.findChildren(QTextEdit):
                self.register_copyable_widget(text_edit, "text")

        def closeEvent(self, event) -> None:
            """Handle close event"""
            self.save_state()
            super().closeEvent(event)

        def init_ui(self) -> None:
            """Initialize UI"""
            layout = QVBoxLayout()

            title = QLabel("Water-Gas Shift Reactor Calculator")
            title.setStyleSheet("font-size: 16pt; font-weight: bold;")
            layout.addWidget(title)

            self.tab_widget = QTabWidget()
            self.create_input_tab()
            self.create_results_tab()
            self.create_plots_tab()

            layout.addWidget(self.tab_widget)
            self.setLayout(layout)

        def create_input_tab(self) -> None:
            """Create input tab"""
            input_widget = QWidget()
            scroll = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout()

            # Catalyst selection
            catalyst_group = QGroupBox("Reactor Configuration")
            catalyst_layout = QFormLayout()

            self.catalyst_combo = QComboBox()
            self.catalyst_combo.addItems(list(self.engine.catalysts.keys()))
            catalyst_layout.addRow("Catalyst Type:", self.catalyst_combo)

            self.temperature = QDoubleSpinBox()
            self.temperature.setRange(200, 800)
            self.temperature.setSuffix(" °C")
            catalyst_layout.addRow("Temperature:", self.temperature)

            self.pressure = QDoubleSpinBox()
            self.pressure.setRange(1, 100)
            self.pressure.setSuffix(" bar")
            catalyst_layout.addRow("Pressure:", self.pressure)

            self.steam_ratio = QDoubleSpinBox()
            self.steam_ratio.setRange(0.5, 10)
            self.steam_ratio.setDecimals(1)
            catalyst_layout.addRow("Steam/CO Ratio:", self.steam_ratio)

            catalyst_group.setLayout(catalyst_layout)
            scroll_layout.addWidget(catalyst_group)

            # Feed composition
            feed_group = QGroupBox("Feed Composition (mol%)")
            feed_layout = QFormLayout()

            self.feed_rate = QDoubleSpinBox()
            self.feed_rate.setRange(0, 100000)
            self.feed_rate.setSuffix(" kmol/h")
            feed_layout.addRow("Feed Rate:", self.feed_rate)

            self.co_inlet = QDoubleSpinBox()
            self.co_inlet.setRange(0, 100)
            self.co_inlet.setSuffix(" %")
            feed_layout.addRow("CO:", self.co_inlet)

            self.h2_inlet = QDoubleSpinBox()
            self.h2_inlet.setRange(0, 100)
            self.h2_inlet.setSuffix(" %")
            feed_layout.addRow("H2:", self.h2_inlet)

            self.co2_inlet = QDoubleSpinBox()
            self.co2_inlet.setRange(0, 100)
            self.co2_inlet.setSuffix(" %")
            feed_layout.addRow("CO2:", self.co2_inlet)

            self.h2o_inlet = QDoubleSpinBox()
            self.h2o_inlet.setRange(0, 100)
            self.h2o_inlet.setSuffix(" %")
            feed_layout.addRow("H2O:", self.h2o_inlet)

            feed_group.setLayout(feed_layout)
            scroll_layout.addWidget(feed_group)

            calc_btn = QPushButton("Calculate WGS Performance")
            calc_btn.clicked.connect(self.calculate)
            scroll_layout.addWidget(calc_btn)

            scroll_widget.setLayout(scroll_layout)
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)

            input_widget.setLayout(QVBoxLayout())
            input_layout = input_widget.layout()
            if input_layout:
                input_layout.addWidget(scroll)
            self.tab_widget.addTab(input_widget, "Reactor Parameters")

        def create_results_tab(self) -> None:
            """Create results tab"""
            results_widget = QWidget()
            layout = QVBoxLayout()

            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

            results_widget.setLayout(layout)
            self.tab_widget.addTab(results_widget, "WGS Results")

        def create_plots_tab(self) -> None:
            """Create plots tab"""
            plots_widget = QWidget()
            layout = QVBoxLayout()

            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)

            # Check if canvas is a valid widget before adding to layout
            if isinstance(self.canvas, QWidget):
                layout.addWidget(self.canvas)
            else:
                layout.addWidget(QLabel("Plot not available in headless mode"))

            plots_widget.setLayout(layout)
            self.tab_widget.addTab(plots_widget, "Composition Profiles")

        def set_default_values(self) -> None:
            """Set default values"""
            self.temperature.setValue(400.0)
            self.pressure.setValue(25.0)
            self.steam_ratio.setValue(2.0)
            self.feed_rate.setValue(100.0)
            self.co_inlet.setValue(25.0)
            self.h2_inlet.setValue(20.0)
            self.co2_inlet.setValue(10.0)
            self.h2o_inlet.setValue(5.0)

        def calculate(self) -> None:
            """Perform WGS calculations"""
            try:
                inlet_comp = {
                    "CO": self.co_inlet.value(),
                    "H2": self.h2_inlet.value(),
                    "CO2": self.co2_inlet.value(),
                    "H2O": self.h2o_inlet.value(),
                }

                temp_k = self.temperature.value() + 273.15

                # Calculate equilibrium
                equilibrium = self.engine.calculate_equilibrium_composition(
                    inlet_comp,
                    temp_k,
                    self.pressure.value(),
                    self.steam_ratio.value(),
                )

                # Size reactor
                sizing = self.engine.size_wgs_reactor(
                    self.feed_rate.value(),
                    equilibrium["conversion"],
                    temp_k,
                    self.catalyst_combo.currentText(),
                )

                # Display results using list join for O(n) instead of O(n²)
                output_parts = [
                    "WATER-GAS SHIFT REACTOR RESULTS\n",
                    "=" * 60 + "\n\n",
                    f"Catalyst: {self.catalyst_combo.currentText()}\n",
                    f"Temperature: {self.temperature.value():.0f} °C\n",
                    f"Pressure: {self.pressure.value():.1f} bar\n\n",
                    "Performance:\n",
                    f"  CO Conversion: {equilibrium['conversion']:.2f}%\n",
                    f"  Equilibrium Constant: {equilibrium['equilibrium_constant']:.2f}\n",
                    f"  H2/CO Ratio: {equilibrium['h2_co_ratio']:.2f}\n",
                    f"  Heat Released: {equilibrium['heat_released']:.2f} kJ/mol\n\n",
                    "Product Composition:\n",
                ]

                for species, content in equilibrium["composition"].items():
                    output_parts.append(f"  {species}: {content:.2f} mol%\n")

                output_parts.extend(
                    [
                        "\n",
                        "Reactor Sizing:\n",
                        f"  Reactor Volume: {sizing['reactor_volume']:.2f} m³\n",
                        f"  Catalyst Volume: {sizing['catalyst_volume']:.2f} m³\n",
                        f"  Diameter: {sizing['diameter']:.2f} m\n",
                        f"  Length: {sizing['length']:.2f} m\n",
                        f"  Heat Duty: {sizing['heat_duty']:.1f} kW\n",
                    ]
                )

                output = "".join(output_parts)
                self.results_text.setText(output)
                self.create_plots(inlet_comp, equilibrium["composition"])

                results = {"equilibrium": equilibrium, "sizing": sizing}
                self.calculation_finished.emit(results)

            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.critical(self, "Calculation Error", str(e))

        def create_plots(
            self, inlet: dict[str, float], outlet: dict[str, float]
        ) -> None:
            """Create composition comparison plot"""
            self.figure.clear()

            ax = self.figure.add_subplot(111)

            species = list(inlet.keys())
            inlet_values = [inlet[sp] for sp in species]
            outlet_values = [outlet[sp] for sp in species]

            x = np.arange(len(species))
            width = 0.35

            ax.bar(x - width / 2, inlet_values, width, label="Inlet", alpha=0.7)
            ax.bar(x + width / 2, outlet_values, width, label="Outlet", alpha=0.7)

            ax.set_ylabel("Composition (mol%)")
            ax.set_title("WGS Reactor: Inlet vs Outlet Composition")
            ax.set_xticks(x)
            ax.set_xticklabels(species)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            self.figure.tight_layout()
            self.canvas.draw()


def create_wgs_reactor_calculator(parent=None) -> QWidget:
    """Factory function"""
    return WGSReactorCalculatorWidget(parent=parent)
