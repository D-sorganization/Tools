"""Main window for Scrubber Calculator PyQt6 application."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

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
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

# Import the scrubber calculator engine
try:
    from upstream_drift_tools.process_calculators.scrubber_calculator import (
        PACKING_DATABASE,
        calculate_caustic_requirement,
        calculate_column_diameter,
        calculate_cooling_water_requirement,
        calculate_flooding_velocity,
        calculate_gas_density,
        calculate_heat_transfer_duty,
        calculate_htu,
        calculate_ntu_removal,
        calculate_pressure_drop,
        calculate_required_packed_height,
    )

    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    PACKING_DATABASE = {}


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


class ScrubberCalculatorWindow(QMainWindow):
    """Main window for Scrubber Calculator application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Packed Bed Scrubber Calculator")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_stylesheet())

        # Store results for display
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

        # Gas Conditions Group
        gas_group = self._create_gas_conditions_group()
        left_layout.addWidget(gas_group)

        # Packing Selection Group
        packing_group = self._create_packing_group()
        left_layout.addWidget(packing_group)

        # Acid Gas Composition Group
        acid_gas_group = self._create_acid_gas_group()
        left_layout.addWidget(acid_gas_group)

        # Liquid Conditions Group
        liquid_group = self._create_liquid_group()
        left_layout.addWidget(liquid_group)

        # Calculate Button
        calc_button = QPushButton("Calculate Design")
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

        # Results Table
        results_group = self._create_results_table()
        right_layout.addWidget(results_group)

        # Detailed Results Table
        details_group = self._create_details_table()
        right_layout.addWidget(details_group)

        main_layout.addWidget(right_panel, 2)

    def _create_gas_conditions_group(self) -> QGroupBox:
        """Create the gas conditions input group."""
        group = QGroupBox("Gas Conditions")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Gas flow rate
        layout.addWidget(QLabel("Gas Flow Rate:"), 0, 0)
        self.gas_flow_spin = QDoubleSpinBox()
        self.gas_flow_spin.setRange(100, 1000000)
        self.gas_flow_spin.setValue(10000)
        self.gas_flow_spin.setSuffix(" kg/hr")
        self.gas_flow_spin.setDecimals(0)
        layout.addWidget(self.gas_flow_spin, 0, 1)

        # Temperature
        layout.addWidget(QLabel("Inlet Temperature:"), 1, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0, 500)
        self.temp_spin.setValue(200)
        self.temp_spin.setSuffix(" °C")
        layout.addWidget(self.temp_spin, 1, 1)

        # Pressure
        layout.addWidget(QLabel("Pressure:"), 2, 0)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.1, 100)
        self.pressure_spin.setValue(1.5)
        self.pressure_spin.setSuffix(" bar")
        self.pressure_spin.setDecimals(2)
        layout.addWidget(self.pressure_spin, 2, 1)

        # Molecular Weight
        layout.addWidget(QLabel("Avg. Molecular Weight:"), 3, 0)
        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setRange(2, 100)
        self.mw_spin.setValue(22)
        self.mw_spin.setSuffix(" kg/kmol")
        layout.addWidget(self.mw_spin, 3, 1)

        # Outlet Temperature
        layout.addWidget(QLabel("Target Outlet Temp:"), 4, 0)
        self.outlet_temp_spin = QDoubleSpinBox()
        self.outlet_temp_spin.setRange(10, 200)
        self.outlet_temp_spin.setValue(38)
        self.outlet_temp_spin.setSuffix(" °C")
        layout.addWidget(self.outlet_temp_spin, 4, 1)

        return group

    def _create_packing_group(self) -> QGroupBox:
        """Create the packing selection group."""
        group = QGroupBox("Packing Selection")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Packing Type
        layout.addWidget(QLabel("Packing Type:"), 0, 0)
        self.packing_combo = QComboBox()
        if ENGINE_AVAILABLE:
            self.packing_combo.addItems(list(PACKING_DATABASE.keys()))
        else:
            self.packing_combo.addItems(
                [
                    "Ceramic Raschig Rings",
                    "Metal Pall Rings",
                    "Plastic Cascade Rings",
                    "Structured Packing",
                ]
            )
        layout.addWidget(self.packing_combo, 0, 1)

        # Percent of Flood
        layout.addWidget(QLabel("% of Flooding:"), 1, 0)
        self.flood_percent_spin = QDoubleSpinBox()
        self.flood_percent_spin.setRange(50, 90)
        self.flood_percent_spin.setValue(70)
        self.flood_percent_spin.setSuffix(" %")
        layout.addWidget(self.flood_percent_spin, 1, 1)

        # Safety Factor
        layout.addWidget(QLabel("Height Safety Factor:"), 2, 0)
        self.safety_factor_spin = QDoubleSpinBox()
        self.safety_factor_spin.setRange(1.0, 2.0)
        self.safety_factor_spin.setValue(1.2)
        self.safety_factor_spin.setDecimals(2)
        layout.addWidget(self.safety_factor_spin, 2, 1)

        return group

    def _create_acid_gas_group(self) -> QGroupBox:
        """Create the acid gas composition group."""
        group = QGroupBox("Acid Gas Composition (ppmv inlet)")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # HCl
        layout.addWidget(QLabel("HCl:"), 0, 0)
        self.hcl_spin = QDoubleSpinBox()
        self.hcl_spin.setRange(0, 10000)
        self.hcl_spin.setValue(500)
        self.hcl_spin.setSuffix(" ppmv")
        self.hcl_spin.setDecimals(0)
        layout.addWidget(self.hcl_spin, 0, 1)

        layout.addWidget(QLabel("Removal:"), 0, 2)
        self.hcl_removal_spin = QDoubleSpinBox()
        self.hcl_removal_spin.setRange(0, 99.99)
        self.hcl_removal_spin.setValue(99.0)
        self.hcl_removal_spin.setSuffix(" %")
        layout.addWidget(self.hcl_removal_spin, 0, 3)

        # SO2
        layout.addWidget(QLabel("SO2:"), 1, 0)
        self.so2_spin = QDoubleSpinBox()
        self.so2_spin.setRange(0, 10000)
        self.so2_spin.setValue(200)
        self.so2_spin.setSuffix(" ppmv")
        self.so2_spin.setDecimals(0)
        layout.addWidget(self.so2_spin, 1, 1)

        layout.addWidget(QLabel("Removal:"), 1, 2)
        self.so2_removal_spin = QDoubleSpinBox()
        self.so2_removal_spin.setRange(0, 99.99)
        self.so2_removal_spin.setValue(95.0)
        self.so2_removal_spin.setSuffix(" %")
        layout.addWidget(self.so2_removal_spin, 1, 3)

        # H2S
        layout.addWidget(QLabel("H2S:"), 2, 0)
        self.h2s_spin = QDoubleSpinBox()
        self.h2s_spin.setRange(0, 50000)
        self.h2s_spin.setValue(1000)
        self.h2s_spin.setSuffix(" ppmv")
        self.h2s_spin.setDecimals(0)
        layout.addWidget(self.h2s_spin, 2, 1)

        layout.addWidget(QLabel("Removal:"), 2, 2)
        self.h2s_removal_spin = QDoubleSpinBox()
        self.h2s_removal_spin.setRange(0, 99.99)
        self.h2s_removal_spin.setValue(90.0)
        self.h2s_removal_spin.setSuffix(" %")
        layout.addWidget(self.h2s_removal_spin, 2, 3)

        # HF
        layout.addWidget(QLabel("HF:"), 3, 0)
        self.hf_spin = QDoubleSpinBox()
        self.hf_spin.setRange(0, 10000)
        self.hf_spin.setValue(100)
        self.hf_spin.setSuffix(" ppmv")
        self.hf_spin.setDecimals(0)
        layout.addWidget(self.hf_spin, 3, 1)

        layout.addWidget(QLabel("Removal:"), 3, 2)
        self.hf_removal_spin = QDoubleSpinBox()
        self.hf_removal_spin.setRange(0, 99.99)
        self.hf_removal_spin.setValue(99.0)
        self.hf_removal_spin.setSuffix(" %")
        layout.addWidget(self.hf_removal_spin, 3, 3)

        return group

    def _create_liquid_group(self) -> QGroupBox:
        """Create the liquid conditions group."""
        group = QGroupBox("Liquid/Caustic Conditions")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # L/G Ratio
        layout.addWidget(QLabel("L/G Ratio:"), 0, 0)
        self.lg_ratio_spin = QDoubleSpinBox()
        self.lg_ratio_spin.setRange(0.5, 20)
        self.lg_ratio_spin.setValue(3.0)
        self.lg_ratio_spin.setSuffix(" kg/kg")
        self.lg_ratio_spin.setDecimals(1)
        layout.addWidget(self.lg_ratio_spin, 0, 1)

        # Caustic Concentration
        layout.addWidget(QLabel("NaOH Concentration:"), 1, 0)
        self.caustic_conc_spin = QDoubleSpinBox()
        self.caustic_conc_spin.setRange(1, 50)
        self.caustic_conc_spin.setValue(20)
        self.caustic_conc_spin.setSuffix(" wt%")
        layout.addWidget(self.caustic_conc_spin, 1, 1)

        # Cooling Water Inlet Temp
        layout.addWidget(QLabel("Cooling Water Inlet:"), 2, 0)
        self.cw_inlet_spin = QDoubleSpinBox()
        self.cw_inlet_spin.setRange(5, 40)
        self.cw_inlet_spin.setValue(25)
        self.cw_inlet_spin.setSuffix(" °C")
        layout.addWidget(self.cw_inlet_spin, 2, 1)

        # KLa (mass transfer coefficient)
        layout.addWidget(QLabel("KLa:"), 3, 0)
        self.kla_spin = QDoubleSpinBox()
        self.kla_spin.setRange(10, 1000)
        self.kla_spin.setValue(200)
        self.kla_spin.setSuffix(" 1/hr")
        layout.addWidget(self.kla_spin, 3, 1)

        return group

    def _create_summary_cards(self) -> QHBoxLayout:
        """Create summary result cards."""
        layout = QHBoxLayout()
        layout.setSpacing(10)

        self.diameter_card = ResultCard("Column Diameter")
        layout.addWidget(self.diameter_card)

        self.height_card = ResultCard("Packed Height")
        layout.addWidget(self.height_card)

        self.pressure_drop_card = ResultCard("Pressure Drop")
        layout.addWidget(self.pressure_drop_card)

        self.caustic_card = ResultCard("NaOH Requirement")
        layout.addWidget(self.caustic_card)

        return layout

    def _create_results_table(self) -> QGroupBox:
        """Create the main results table."""
        group = QGroupBox("Design Results")
        layout = QVBoxLayout(group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        results_header = self.results_table.horizontalHeader()
        if results_header is not None:
            results_header.setStretchLastSection(True)
            results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.results_table)

        return group

    def _create_details_table(self) -> QGroupBox:
        """Create the acid gas details table."""
        group = QGroupBox("Acid Gas Removal Details")
        layout = QVBoxLayout(group)

        self.details_table = QTableWidget()
        self.details_table.setColumnCount(5)
        self.details_table.setHorizontalHeaderLabels(
            ["Component", "Inlet (ppmv)", "Outlet (ppmv)", "Removed (kg/hr)", "NTU"]
        )
        header = self.details_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.details_table)

        return group

    def _calculate(self) -> None:
        """Perform the scrubber design calculations."""
        if not ENGINE_AVAILABLE:
            self._show_error("Engine not available. Cannot perform calculations.")
            return

        # Get inputs
        gas_flow_kg_hr = self.gas_flow_spin.value()
        inlet_temp_c = self.temp_spin.value()
        pressure_bar = self.pressure_spin.value()
        mw = self.mw_spin.value()
        outlet_temp_c = self.outlet_temp_spin.value()

        packing_name = self.packing_combo.currentText()
        packing = PACKING_DATABASE.get(packing_name)
        if packing is None:
            self._show_error(f"Unknown packing type: {packing_name}")
            return

        percent_flood = self.flood_percent_spin.value()
        safety_factor = self.safety_factor_spin.value()

        lg_ratio = self.lg_ratio_spin.value()
        caustic_conc = self.caustic_conc_spin.value()
        cw_inlet_temp = self.cw_inlet_spin.value()
        kla = self.kla_spin.value()

        # Acid gas inputs
        acid_gases = {
            "HCl": (self.hcl_spin.value(), self.hcl_removal_spin.value()),
            "SO2": (self.so2_spin.value(), self.so2_removal_spin.value()),
            "H2S": (self.h2s_spin.value(), self.h2s_removal_spin.value()),
            "HF": (self.hf_spin.value(), self.hf_removal_spin.value()),
        }

        # Calculate gas properties
        temp_k = inlet_temp_c + 273.15
        pressure_pa = pressure_bar * 1e5

        gas_density = calculate_gas_density(temp_k, pressure_pa, mw)

        # Calculate liquid mass flux
        gas_flow_kg_s = gas_flow_kg_hr / 3600.0
        liquid_flow_kg_hr = gas_flow_kg_hr * lg_ratio
        liquid_density = 1000.0 + 10.8 * caustic_conc  # NaOH solution density

        # Calculate flooding velocity
        # Estimate cross-section area first (iterate)
        estimated_area = 2.0  # m² initial guess
        liquid_mass_flux = (liquid_flow_kg_hr / 3600.0) / estimated_area

        flooding_velocity = calculate_flooding_velocity(
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            liquid_density=liquid_density,
            packing=packing,
        )

        # Calculate column diameter
        column_sizing = calculate_column_diameter(
            gas_flow_kg_hr=gas_flow_kg_hr,
            gas_density=gas_density,
            flooding_velocity=flooding_velocity,
            percent_of_flood=percent_flood,
        )

        # Recalculate liquid mass flux with actual area
        actual_area = column_sizing["cross_section_m2"]
        if isinstance(actual_area, float) and actual_area > 0:
            liquid_mass_flux = (liquid_flow_kg_hr / 3600.0) / actual_area
            gas_mass_flux = gas_flow_kg_s / actual_area
        else:
            liquid_mass_flux = 0.0
            gas_mass_flux = 0.0

        # Calculate NTU and HTU for each acid gas
        acid_gas_details = []
        acid_gas_removed: dict[str, float] = {}
        max_ntu = 0.0

        # Molecular weights for conversion
        mw_gases = {"HCl": 36.458, "SO2": 64.06, "H2S": 34.08, "HF": 20.01}

        for gas_name, (inlet_ppmv, removal_pct) in acid_gases.items():
            if inlet_ppmv > 0 and removal_pct > 0:
                inlet_frac = inlet_ppmv / 1e6
                outlet_ppmv = inlet_ppmv * (1 - removal_pct / 100.0)
                outlet_frac = outlet_ppmv / 1e6

                ntu = calculate_ntu_removal(inlet_frac, outlet_frac)
                max_ntu = max(max_ntu, ntu)

                # Calculate mass removed
                mw_gas = mw_gases.get(gas_name, 30.0)
                gas_molar_flow = gas_flow_kg_hr / mw  # kmol/hr
                removed_kmol_hr = gas_molar_flow * (inlet_frac - outlet_frac)
                removed_kg_hr = removed_kmol_hr * mw_gas

                acid_gas_details.append(
                    {
                        "name": gas_name,
                        "inlet_ppmv": inlet_ppmv,
                        "outlet_ppmv": outlet_ppmv,
                        "removed_kg_hr": removed_kg_hr,
                        "ntu": ntu,
                    }
                )
                acid_gas_removed[gas_name] = removed_kg_hr

        # Calculate HTU
        htu = calculate_htu(
            gas_mass_flux=gas_mass_flux,
            liquid_mass_flux=liquid_mass_flux,
            gas_density=gas_density,
            packing=packing,
            kla=kla,
        )

        # Calculate required packed height
        packed_height = calculate_required_packed_height(
            ntu=max_ntu, htu=htu, safety_factor=safety_factor
        )

        # Calculate pressure drop
        design_velocity = column_sizing.get("design_velocity_m_s", 0.0)
        if isinstance(design_velocity, float) and design_velocity > 0:
            pressure_drop = calculate_pressure_drop(
                gas_velocity=design_velocity,
                gas_density=gas_density,
                liquid_mass_flux=liquid_mass_flux,
                liquid_density=liquid_density,
                packing=packing,
                packed_height=packed_height,
            )
        else:
            pressure_drop = 0.0

        # Calculate caustic requirement
        caustic_req = calculate_caustic_requirement(
            acid_gas_removed=acid_gas_removed, caustic_concentration=caustic_conc
        )

        # Calculate heat transfer duty
        water_condensed = (
            gas_flow_kg_hr * 0.15 * (inlet_temp_c - outlet_temp_c) / 100.0
        )  # Approximate
        heat_duty = calculate_heat_transfer_duty(
            gas_flow_kg_hr=gas_flow_kg_hr,
            inlet_temp_c=inlet_temp_c,
            outlet_temp_c=outlet_temp_c,
            water_condensed_kg_hr=water_condensed,
        )

        # Calculate cooling water requirement
        cooling_water = calculate_cooling_water_requirement(
            heat_duty_kw=heat_duty["total_heat_kw"],
            water_inlet_temp_c=cw_inlet_temp,
            outlet_gas_temp_c=outlet_temp_c,
        )

        # Store results
        self.results = {
            "column_sizing": column_sizing,
            "packed_height": packed_height,
            "pressure_drop": pressure_drop,
            "caustic_req": caustic_req,
            "heat_duty": heat_duty,
            "cooling_water": cooling_water,
            "acid_gas_details": acid_gas_details,
            "gas_density": gas_density,
            "flooding_velocity": flooding_velocity,
            "htu": htu,
            "max_ntu": max_ntu,
        }

        # Update UI
        self._update_results_display()

    def _update_results_display(self) -> None:
        """Update the results display with calculated values."""
        r = self.results

        # Update summary cards
        diameter_m = r["column_sizing"].get("diameter_m", 0.0)
        if isinstance(diameter_m, float):
            self.diameter_card.set_value(f"{diameter_m:.2f} m")
        else:
            self.diameter_card.set_value("--")

        self.height_card.set_value(f"{r['packed_height']:.2f} m")

        pressure_drop_kpa = r["pressure_drop"] / 1000.0
        self.pressure_drop_card.set_value(f"{pressure_drop_kpa:.2f} kPa")

        naoh_kg_hr = r["caustic_req"].get("naoh_pure_kg_hr", 0.0)
        self.caustic_card.set_value(f"{naoh_kg_hr:.1f} kg/hr")

        # Update results table
        results_data = [
            ("Gas Density", f"{r['gas_density']:.3f} kg/m³"),
            ("Flooding Velocity", f"{r['flooding_velocity']:.2f} m/s"),
            (
                "Design Velocity",
                f"{r['column_sizing'].get('design_velocity_m_s', 0.0):.2f} m/s",
            ),
            (
                "Column Cross-Section",
                f"{r['column_sizing'].get('cross_section_m2', 0.0):.2f} m²",
            ),
            ("Height of Transfer Unit", f"{r['htu']:.2f} m"),
            ("Number of Transfer Units", f"{r['max_ntu']:.2f}"),
            ("Total Heat Duty", f"{r['heat_duty']['total_heat_kw']:.1f} kW"),
            ("Sensible Heat", f"{r['heat_duty']['sensible_heat_kw']:.1f} kW"),
            ("Latent Heat", f"{r['heat_duty']['latent_heat_kw']:.1f} kW"),
            (
                "Cooling Water Flow",
                f"{r['cooling_water'].get('water_flow_L_min', 0.0):.1f} L/min",
            ),
            (
                "NaOH Solution Flow",
                f"{r['caustic_req'].get('naoh_solution_L_hr', 0.0):.1f} L/hr",
            ),
            (
                "Salt Produced",
                f"{r['caustic_req'].get('salt_produced_kg_hr', 0.0):.2f} kg/hr",
            ),
        ]

        self.results_table.setRowCount(len(results_data))
        for i, (param, value) in enumerate(results_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Update details table
        acid_gas_details = r.get("acid_gas_details", [])
        self.details_table.setRowCount(len(acid_gas_details))
        for i, detail in enumerate(acid_gas_details):
            self.details_table.setItem(i, 0, QTableWidgetItem(detail["name"]))
            self.details_table.setItem(
                i, 1, QTableWidgetItem(f"{detail['inlet_ppmv']:.0f}")
            )
            self.details_table.setItem(
                i, 2, QTableWidgetItem(f"{detail['outlet_ppmv']:.1f}")
            )
            self.details_table.setItem(
                i, 3, QTableWidgetItem(f"{detail['removed_kg_hr']:.3f}")
            )
            self.details_table.setItem(i, 4, QTableWidgetItem(f"{detail['ntu']:.2f}"))

    def _show_error(self, message: str) -> None:
        """Display an error message in the UI."""
        self.diameter_card.set_value("Error")
        self.diameter_card.set_color(COLORS["red"])
        self.height_card.set_value(message[:20])


def main() -> None:
    """Run the Scrubber Calculator application."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ScrubberCalculatorWindow()
    setup_themed_app(app, window, settings_app="ScrubberCalculator")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
