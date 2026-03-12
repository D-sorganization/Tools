"""Main window for Scrubber Calculator PyQt6 application."""

from __future__ import annotations

import logging
import sys

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
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.process_calculators.scrubber import (
    ScrubberEngine,
    ScrubberInputs,
    ScrubberResults,
)
from upstream_drift_tools.process_calculators.scrubber_calculator import (
    PACKING_DATABASE,
)
from upstream_drift_tools.ui.catppuccin_theme import COLORS
from upstream_drift_tools.ui.catppuccin_theme import get_stylesheet as _base_stylesheet
from upstream_drift_tools.ui.widgets import BaseCalculatorWidget

logger = logging.getLogger(__name__)


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


class ScrubberCalculatorWindow(BaseCalculatorWidget):
    """Main window for Scrubber Calculator application."""

    def __init__(self) -> None:
        super().__init__(
            calculator_name="ScrubberCalculator",
        )
        # Note: Set window title and min size on the parent QMainWindow instead
        self.setStyleSheet(get_stylesheet())

        # Main layout for widgets
        self.main_layout = QVBoxLayout(self)

        # Store results for display
        self.last_results: ScrubberResults | None = None

        self._setup_ui()

        # Load last state
        self.load_calculator_state()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(content_widget)

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

        # State Buttons
        state_layout = QHBoxLayout()
        save_btn, load_btn = self.create_save_load_buttons()
        state_layout.addWidget(save_btn)
        state_layout.addWidget(load_btn)
        left_layout.addLayout(state_layout)

        # Copy Results Button
        left_layout.addWidget(self.create_copy_button())

        left_layout.addStretch()
        content_layout.addWidget(left_panel, 1)

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

        content_layout.addWidget(right_panel, 2)

        # Register components for state management
        self.auto_register_widgets()

    def _create_gas_conditions_group(self) -> QGroupBox:
        """Create the gas conditions input group."""
        group = QGroupBox("Gas Conditions")
        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Gas flow rate
        layout.addWidget(QLabel("Gas Flow Rate:"), 0, 0)
        self.gas_flow_spin = QDoubleSpinBox()
        self.gas_flow_spin.setObjectName("gas_flow_spin")
        self.gas_flow_spin.setRange(100, 1000000)
        self.gas_flow_spin.setValue(10000)
        self.gas_flow_spin.setSuffix(" kg/hr")
        self.gas_flow_spin.setDecimals(0)
        layout.addWidget(self.gas_flow_spin, 0, 1)

        # Temperature
        layout.addWidget(QLabel("Inlet Temperature:"), 1, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setObjectName("temp_spin")
        self.temp_spin.setRange(0, 500)
        self.temp_spin.setValue(200)
        self.temp_spin.setSuffix(" °C")
        layout.addWidget(self.temp_spin, 1, 1)

        # Pressure
        layout.addWidget(QLabel("Pressure:"), 2, 0)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setObjectName("pressure_spin")
        self.pressure_spin.setRange(0.1, 100)
        self.pressure_spin.setValue(1.5)
        self.pressure_spin.setSuffix(" bar")
        self.pressure_spin.setDecimals(2)
        layout.addWidget(self.pressure_spin, 2, 1)

        # Molecular Weight
        layout.addWidget(QLabel("Avg. Molecular Weight:"), 3, 0)
        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setObjectName("mw_spin")
        self.mw_spin.setRange(2, 100)
        self.mw_spin.setValue(22)
        self.mw_spin.setSuffix(" kg/kmol")
        layout.addWidget(self.mw_spin, 3, 1)

        # Outlet Temperature
        layout.addWidget(QLabel("Target Outlet Temp:"), 4, 0)
        self.outlet_temp_spin = QDoubleSpinBox()
        self.outlet_temp_spin.setObjectName("outlet_temp_spin")
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
        self.packing_combo.setObjectName("packing_combo")
        self.packing_combo.addItems(list(PACKING_DATABASE.keys()))
        layout.addWidget(self.packing_combo, 0, 1)

        # Percent of Flood
        layout.addWidget(QLabel("% of Flooding:"), 1, 0)
        self.flood_percent_spin = QDoubleSpinBox()
        self.flood_percent_spin.setObjectName("flood_percent_spin")
        self.flood_percent_spin.setRange(50, 90)
        self.flood_percent_spin.setValue(70)
        self.flood_percent_spin.setSuffix(" %")
        layout.addWidget(self.flood_percent_spin, 1, 1)

        # Safety Factor
        layout.addWidget(QLabel("Height Safety Factor:"), 2, 0)
        self.safety_factor_spin = QDoubleSpinBox()
        self.safety_factor_spin.setObjectName("safety_factor_spin")
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
        self.hcl_spin.setObjectName("hcl_spin")
        self.hcl_spin.setRange(0, 10000)
        self.hcl_spin.setValue(500)
        self.hcl_spin.setSuffix(" ppmv")
        self.hcl_spin.setDecimals(0)
        layout.addWidget(self.hcl_spin, 0, 1)

        layout.addWidget(QLabel("Removal:"), 0, 2)
        self.hcl_removal_spin = QDoubleSpinBox()
        self.hcl_removal_spin.setObjectName("hcl_removal_spin")
        self.hcl_removal_spin.setRange(0, 99.99)
        self.hcl_removal_spin.setValue(99.0)
        self.hcl_removal_spin.setSuffix(" %")
        layout.addWidget(self.hcl_removal_spin, 0, 3)

        # SO2
        layout.addWidget(QLabel("SO2:"), 1, 0)
        self.so2_spin = QDoubleSpinBox()
        self.so2_spin.setObjectName("so2_spin")
        self.so2_spin.setRange(0, 10000)
        self.so2_spin.setValue(200)
        self.so2_spin.setSuffix(" ppmv")
        self.so2_spin.setDecimals(0)
        layout.addWidget(self.so2_spin, 1, 1)

        layout.addWidget(QLabel("Removal:"), 1, 2)
        self.so2_removal_spin = QDoubleSpinBox()
        self.so2_removal_spin.setObjectName("so2_removal_spin")
        self.so2_removal_spin.setRange(0, 99.99)
        self.so2_removal_spin.setValue(95.0)
        self.so2_removal_spin.setSuffix(" %")
        layout.addWidget(self.so2_removal_spin, 1, 3)

        # H2S
        layout.addWidget(QLabel("H2S:"), 2, 0)
        self.h2s_spin = QDoubleSpinBox()
        self.h2s_spin.setObjectName("h2s_spin")
        self.h2s_spin.setRange(0, 50000)
        self.h2s_spin.setValue(1000)
        self.h2s_spin.setSuffix(" ppmv")
        self.h2s_spin.setDecimals(0)
        layout.addWidget(self.h2s_spin, 2, 1)

        layout.addWidget(QLabel("Removal:"), 2, 2)
        self.h2s_removal_spin = QDoubleSpinBox()
        self.h2s_removal_spin.setObjectName("h2s_removal_spin")
        self.h2s_removal_spin.setRange(0, 99.99)
        self.h2s_removal_spin.setValue(90.0)
        self.h2s_removal_spin.setSuffix(" %")
        layout.addWidget(self.h2s_removal_spin, 2, 3)

        # HF
        layout.addWidget(QLabel("HF:"), 3, 0)
        self.hf_spin = QDoubleSpinBox()
        self.hf_spin.setObjectName("hf_spin")
        self.hf_spin.setRange(0, 10000)
        self.hf_spin.setValue(100)
        self.hf_spin.setSuffix(" ppmv")
        self.hf_spin.setDecimals(0)
        layout.addWidget(self.hf_spin, 3, 1)

        layout.addWidget(QLabel("Removal:"), 3, 2)
        self.hf_removal_spin = QDoubleSpinBox()
        self.hf_removal_spin.setObjectName("hf_removal_spin")
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
        self.lg_ratio_spin.setObjectName("lg_ratio_spin")
        self.lg_ratio_spin.setRange(0.5, 20)
        self.lg_ratio_spin.setValue(3.0)
        self.lg_ratio_spin.setSuffix(" kg/kg")
        self.lg_ratio_spin.setDecimals(1)
        layout.addWidget(self.lg_ratio_spin, 0, 1)

        # Caustic Concentration
        layout.addWidget(QLabel("NaOH Concentration:"), 1, 0)
        self.caustic_conc_spin = QDoubleSpinBox()
        self.caustic_conc_spin.setObjectName("caustic_conc_spin")
        self.caustic_conc_spin.setRange(1, 50)
        self.caustic_conc_spin.setValue(20)
        self.caustic_conc_spin.setSuffix(" wt%")
        layout.addWidget(self.caustic_conc_spin, 1, 1)

        # Cooling Water Inlet Temp
        layout.addWidget(QLabel("Cooling Water Inlet:"), 2, 0)
        self.cw_inlet_spin = QDoubleSpinBox()
        self.cw_inlet_spin.setObjectName("cw_inlet_spin")
        self.cw_inlet_spin.setRange(5, 40)
        self.cw_inlet_spin.setValue(25)
        self.cw_inlet_spin.setSuffix(" °C")
        layout.addWidget(self.cw_inlet_spin, 2, 1)

        # KLa (mass transfer coefficient)
        layout.addWidget(QLabel("KLa:"), 3, 0)
        self.kla_spin = QDoubleSpinBox()
        self.kla_spin.setObjectName("kla_spin")
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
        self.results_table.setObjectName("results_table")
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
        self.details_table.setObjectName("details_table")
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
        try:
            # Prepare inputs
            inputs = ScrubberInputs(
                gas_flow_kg_hr=self.gas_flow_spin.value(),
                inlet_temp_c=self.temp_spin.value(),
                pressure_bar=self.pressure_spin.value(),
                molecular_weight=self.mw_spin.value(),
                target_outlet_temp_c=self.outlet_temp_spin.value(),
                packing_name=self.packing_combo.currentText(),
                percent_of_flood=self.flood_percent_spin.value(),
                height_safety_factor=self.safety_factor_spin.value(),
                lg_ratio=self.lg_ratio_spin.value(),
                caustic_concentration_wt_pct=self.caustic_conc_spin.value(),
                cooling_water_inlet_temp_c=self.cw_inlet_spin.value(),
                kla_hr=self.kla_spin.value(),
                acid_gas_composition_ppmv={
                    "HCl": self.hcl_spin.value(),
                    "SO2": self.so2_spin.value(),
                    "H2S": self.h2s_spin.value(),
                    "HF": self.hf_spin.value(),
                },
                acid_gas_removal_pct={
                    "HCl": self.hcl_removal_spin.value(),
                    "SO2": self.so2_removal_spin.value(),
                    "H2S": self.h2s_removal_spin.value(),
                    "HF": self.hf_removal_spin.value(),
                },
            )

            # Perform calculation using Engine
            self.last_results = ScrubberEngine.calculate(inputs)

            # Update UI
            self._update_results_display()
            self.mark_changed()

        except (ValueError, TypeError, ArithmeticError, KeyError) as e:
            logger.exception("Calculation failed")
            self.show_error("Calculation Error", str(e))

    def _update_results_display(self) -> None:
        """Update the results display with calculated values."""
        if not self.last_results:
            return

        r = self.last_results

        # Update summary cards
        self.diameter_card.set_value(f"{r.column_diameter_m:.2f} m")
        self.height_card.set_value(f"{r.packed_height_m:.2f} m")
        self.pressure_drop_card.set_value(f"{r.pressure_drop_kpa:.2f} kPa")
        self.caustic_card.set_value(f"{r.naoh_pure_kg_hr:.1f} kg/hr")

        # Update results table
        results_data = [
            ("Gas Density", f"{r.gas_density_kg_m3:.3f} kg/m³"),
            ("Flooding Velocity", f"{r.flooding_velocity_m_s:.2f} m/s"),
            (
                "Design Velocity",
                f"{r.flooding_velocity_m_s * 0.7:.2f} m/s",
            ),  # Approximate
            ("Height of Transfer Unit", f"{r.htu_m:.2f} m"),
            ("Number of Transfer Units", f"{r.max_ntu:.2f}"),
            ("Total Heat Duty", f"{r.total_heat_duty_kw:.1f} kW"),
            ("Cooling Water Flow", f"{r.cooling_water_flow_L_min:.1f} L/min"),
            ("NaOH Solution Flow", f"{r.naoh_solution_L_hr:.1f} L/hr"),
        ]

        self.results_table.setRowCount(len(results_data))
        for i, (param, value) in enumerate(results_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Update details table
        details = r.acid_gas_details
        self.details_table.setRowCount(len(details))
        for i, detail in enumerate(details):
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


def main() -> None:
    """Run the Scrubber Calculator application."""
    from shared.python.theme import setup_themed_app
    from PyQt6.QtWidgets import QMainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Wrap the widget in a QMainWindow for setup_themed_app
    main_window = QMainWindow()
    widget = ScrubberCalculatorWindow()
    main_window.setCentralWidget(widget)
    main_window.setWindowTitle("Packed Bed Scrubber Calculator")
    main_window.setMinimumSize(1200, 800)

    setup_themed_app(app, main_window, settings_app="ScrubberCalculator")
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
