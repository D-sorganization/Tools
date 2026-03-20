"""Pressure Drop Calculator PyQt6 Main Window.

A comprehensive GUI for calculating pipe pressure drops using the
PressureDropCalculationEngine from the shared process calculators.
"""

from __future__ import annotations

import os
from typing import Any

if os.environ.get("HEADLESS", "false").lower() == "true":
    import matplotlib

    matplotlib.use("Agg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PressureDropCalculatorWidget(QWidget):
    """Main widget for the Pressure Drop Calculator application."""

    PIPE_SCHEDULES = [
        "5",
        "10",
        "20",
        "30",
        "40",
        "60",
        "80",
        "100",
        "120",
        "140",
        "160",
        "STD",
        "XS",
        "XXS",
    ]
    PIPE_SIZES = [
        "0.5",
        "0.75",
        "1",
        "1.25",
        "1.5",
        "2",
        "2.5",
        "3",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
        "20",
        "24",
    ]
    FLOW_UNITS = ["kg/h", "kg/s", "lb/hr", "m³/h", "SCFM", "Nm³/h"]
    FRICTION_METHODS = ["colebrook", "swamee-jain", "churchill", "haaland"]
    MATERIALS = ["Carbon Steel", "Stainless Steel", "Copper", "PVC", "HDPE", "Concrete"]
    ROUGHNESS_VALUES = {
        "Carbon Steel": 0.000046,
        "Stainless Steel": 0.000015,
        "Copper": 0.0000015,
        "PVC": 0.0000015,
        "HDPE": 0.000007,
        "Concrete": 0.0003,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the Pressure Drop Calculator widget."""
        super().__init__(parent)
        self.results: dict[str, Any] | None = None
        self._init_ui()
        self._apply_styling()
        self._connect_signals()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_input_panel())
        splitter.addWidget(self._create_results_panel())
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _create_input_panel(self) -> QWidget:
        """Create the left input panel with pipe, flow, and gas composition controls."""
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        input_layout.addWidget(self._create_pipe_group())
        input_layout.addWidget(self._create_flow_group())
        input_layout.addWidget(self._create_gas_composition_group())

        self.calculate_btn = QPushButton("Calculate Pressure Drop")
        self.calculate_btn.setMinimumHeight(40)
        input_layout.addWidget(self.calculate_btn)

        input_layout.addStretch()
        return input_widget

    def _create_pipe_group(self) -> QGroupBox:
        """Create the pipe parameters input group."""
        pipe_group = QGroupBox("Pipe Parameters")
        pipe_layout = QFormLayout(pipe_group)

        self.pipe_size_combo = QComboBox()
        self.pipe_size_combo.addItems(self.PIPE_SIZES)
        self.pipe_size_combo.setCurrentText("4")
        pipe_layout.addRow("Nominal Size (in):", self.pipe_size_combo)

        self.pipe_schedule_combo = QComboBox()
        self.pipe_schedule_combo.addItems(self.PIPE_SCHEDULES)
        self.pipe_schedule_combo.setCurrentText("40")
        pipe_layout.addRow("Schedule:", self.pipe_schedule_combo)

        self.pipe_length_spin = QDoubleSpinBox()
        self.pipe_length_spin.setRange(0.1, 100000)
        self.pipe_length_spin.setValue(100)
        self.pipe_length_spin.setSuffix(" m")
        pipe_layout.addRow("Length:", self.pipe_length_spin)

        self.material_combo = QComboBox()
        self.material_combo.addItems(self.MATERIALS)
        pipe_layout.addRow("Material:", self.material_combo)

        self.elevation_spin = QDoubleSpinBox()
        self.elevation_spin.setRange(-1000, 1000)
        self.elevation_spin.setValue(0)
        self.elevation_spin.setSuffix(" m")
        pipe_layout.addRow("Elevation Change:", self.elevation_spin)

        return pipe_group

    def _create_flow_group(self) -> QGroupBox:
        """Create the flow conditions input group."""
        flow_group = QGroupBox("Flow Conditions")
        flow_layout = QFormLayout(flow_group)

        self.flow_rate_spin = QDoubleSpinBox()
        self.flow_rate_spin.setRange(0.001, 1000000)
        self.flow_rate_spin.setValue(1000)
        self.flow_rate_spin.setDecimals(2)
        flow_layout.addRow("Flow Rate:", self.flow_rate_spin)

        self.flow_unit_combo = QComboBox()
        self.flow_unit_combo.addItems(self.FLOW_UNITS)
        flow_layout.addRow("Flow Unit:", self.flow_unit_combo)

        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.1, 1000)
        self.pressure_spin.setValue(10)
        self.pressure_spin.setSuffix(" bar")
        self.pressure_spin.setDecimals(2)
        flow_layout.addRow("Inlet Pressure:", self.pressure_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(200, 1500)
        self.temperature_spin.setValue(300)
        self.temperature_spin.setSuffix(" K")
        flow_layout.addRow("Temperature:", self.temperature_spin)

        self.friction_method_combo = QComboBox()
        self.friction_method_combo.addItems(self.FRICTION_METHODS)
        flow_layout.addRow("Friction Method:", self.friction_method_combo)

        return flow_group

    def _create_gas_composition_group(self) -> QGroupBox:
        """Create the gas composition input group."""
        gas_group = QGroupBox("Gas Composition (mol %)")
        gas_layout = QGridLayout(gas_group)

        self.gas_spins: dict[str, QDoubleSpinBox] = {}
        gas_components = ["N2", "O2", "CO2", "H2O", "H2", "CO", "CH4", "Ar"]
        default_air = {"N2": 78, "O2": 21, "Ar": 1}

        for i, comp in enumerate(gas_components):
            label = QLabel(f"{comp}:")
            spin = QDoubleSpinBox()
            spin.setRange(0, 100)
            spin.setValue(default_air.get(comp, 0))
            spin.setDecimals(1)
            self.gas_spins[comp] = spin
            gas_layout.addWidget(label, i // 2, (i % 2) * 2)
            gas_layout.addWidget(spin, i // 2, (i % 2) * 2 + 1)

        return gas_group

    def _create_results_panel(self) -> QWidget:
        """Create the right results panel with tabs for results, flow, warnings, chart."""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        self.tabs = QTabWidget()

        self.tabs.addTab(self._create_results_tab(), "Results")
        self.tabs.addTab(self._create_flow_tab(), "Flow Properties")
        self.tabs.addTab(self._create_warnings_tab(), "Warnings")
        self.tabs.addTab(self._create_chart_tab(), "Pressure Profile")

        results_layout.addWidget(self.tabs)
        return results_widget

    def _create_results_tab(self) -> QWidget:
        """Create the main results table tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.results_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        layout.addWidget(self.results_table)

        return tab

    def _create_flow_tab(self) -> QWidget:
        """Create the flow properties table tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.flow_table = QTableWidget()
        self.flow_table.setColumnCount(2)
        self.flow_table.setHorizontalHeaderLabels(["Property", "Value"])
        header = self.flow_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        layout.addWidget(self.flow_table)

        return tab

    def _create_warnings_tab(self) -> QWidget:
        """Create the warnings text tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        layout.addWidget(self.warnings_text)

        return tab

    def _create_chart_tab(self) -> QWidget:
        """Create the pressure profile chart tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.figure = Figure(figsize=(8, 5), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        return tab

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.calculate_btn.clicked.connect(self._calculate)
        self.material_combo.currentTextChanged.connect(self._update_roughness)

    def _update_roughness(self) -> None:
        """Update roughness based on material selection."""
        pass  # Roughness is looked up during calculation

    def _calculate(self) -> None:
        """Perform the pressure drop calculation."""
        try:
            from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
                calculate_pressure_drop,
            )

            # Get gas composition
            gas_comp: dict[str, float] = {}
            total: float = 0.0
            for comp, spin in self.gas_spins.items():
                val = spin.value()
                if val > 0:
                    gas_comp[comp] = val / 100  # Convert to fraction
                    total += val

            if abs(total - 100) > 1:
                QMessageBox.warning(
                    self,
                    "Invalid Composition",
                    f"Gas composition must sum to 100% (current: {total:.1f}%)",
                )
                return

            # Normalize if needed
            if total > 0:
                gas_comp = {k: v / (total / 100) for k, v in gas_comp.items()}

            material = self.material_combo.currentText()
            roughness = self.ROUGHNESS_VALUES.get(material, 0.000046)

            result = calculate_pressure_drop(
                pipe_size=self.pipe_size_combo.currentText(),
                pipe_schedule=self.pipe_schedule_combo.currentText(),
                pipe_length=self.pipe_length_spin.value(),
                flow_rate=self.flow_rate_spin.value(),
                flow_unit=self.flow_unit_combo.currentText(),
                pressure=self.pressure_spin.value(),
                temperature=self.temperature_spin.value(),
                gas_composition=gas_comp,
                elevation_change=self.elevation_spin.value(),
                roughness=roughness,
                friction_method=self.friction_method_combo.currentText(),
            )

            self.results = result
            self._update_results_display()

        except ImportError as e:
            QMessageBox.critical(self, "Calculation Error", str(e))

    def _update_results_display(self) -> None:
        """Update the results display with calculation results."""
        if self.results is None:
            return

        r = self.results

        # Main results table
        results_data = [
            ("Total Pressure Drop", f"{r.get('total_pressure_drop', 0):.2f} Pa"),
            ("Outlet Pressure", f"{r.get('outlet_pressure', 0) / 1e5:.4f} bar"),
            ("Friction Pressure Drop", f"{r.get('friction_pressure_drop', 0):.2f} Pa"),
            ("Fitting Pressure Drop", f"{r.get('fitting_pressure_drop', 0):.2f} Pa"),
            (
                "Elevation Pressure Drop",
                f"{r.get('elevation_pressure_drop', 0):.2f} Pa",
            ),
            ("Friction Factor", f"{r.get('friction_factor', 0):.6f}"),
            (
                "Pressure Drop per 100ft",
                f"{r.get('pressure_drop_per_100ft', 0):.2f} Pa/100ft",
            ),
            ("Flow Regime", r.get("flow_regime", "Unknown")),
        ]

        self.results_table.setRowCount(len(results_data))
        for i, (param, value) in enumerate(results_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Flow properties table
        flow_props = r.get("flow_properties", {})
        if isinstance(flow_props, dict):
            flow_data = [
                ("Velocity", f"{flow_props.get('velocity', 0):.3f} m/s"),
                ("Reynolds Number", f"{flow_props.get('reynolds_number', 0):.0f}"),
                ("Mach Number", f"{flow_props.get('mach_number', 0):.4f}"),
                ("Density", f"{flow_props.get('density', 0):.3f} kg/m³"),
                ("Viscosity", f"{flow_props.get('viscosity', 0) * 1e6:.3f} µPa·s"),
                ("Erosional Velocity", f"{r.get('erosional_velocity', 0):.2f} m/s"),
                ("Erosion Ratio", f"{r.get('erosion_ratio', 0):.3f}"),
            ]

            self.flow_table.setRowCount(len(flow_data))
            for i, (param, value) in enumerate(flow_data):
                self.flow_table.setItem(i, 0, QTableWidgetItem(param))
                self.flow_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Warnings
        warnings = r.get("warnings", [])
        if warnings:
            self.warnings_text.setText("\n".join(f"⚠ {w}" for w in warnings))
        else:
            self.warnings_text.setText(
                "✓ No warnings. All parameters within acceptable ranges."
            )

        # Update chart
        self._update_chart()

    def _update_chart(self) -> None:
        """Update the pressure profile chart."""
        if self.results is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#313244")

        inlet_p = self.pressure_spin.value() * 1e5  # Convert bar to Pa
        outlet_p = self.results.get("outlet_pressure", inlet_p)
        length = self.pipe_length_spin.value()

        x = [0, length]
        y = [inlet_p / 1e5, outlet_p / 1e5]

        ax.plot(x, y, color="#89b4fa", linewidth=2, marker="o", markersize=8)
        ax.fill_between(x, y, alpha=0.3, color="#89b4fa")

        ax.set_xlabel("Distance (m)", color="#cdd6f4")
        ax.set_ylabel("Pressure (bar)", color="#cdd6f4")
        ax.set_title("Pressure Profile Along Pipe", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.grid(True, alpha=0.3, color="#585b70")

        for spine in ax.spines.values():
            spine.set_color("#585b70")

        self.figure.tight_layout()
        self.canvas.draw()

    def _apply_styling(self) -> None:
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #313244;
            }
            QGroupBox::title {
                color: #cba6f7;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #45475a;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cdd6f4;
            }
            QTableWidget {
                background-color: #313244;
                border: 1px solid #45475a;
                gridline-color: #45475a;
            }
            QHeaderView::section {
                background-color: #45475a;
                color: #cdd6f4;
                padding: 4px;
                border: none;
            }
            QTextEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #313244;
            }
            QTabBar::tab {
                background-color: #45475a;
                color: #cdd6f4;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
        """)
