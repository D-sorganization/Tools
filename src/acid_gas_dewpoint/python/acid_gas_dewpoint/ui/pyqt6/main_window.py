"""Acid Gas Dewpoint Calculator PyQt6 Main Window.

A comprehensive GUI for calculating acid gas dewpoints (HF, HCl, H2S)
using the AcidGasDewpointCalculator from the shared process calculators.
"""

from __future__ import annotations

import math
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


class AcidGasDewpointCalculatorWidget(QWidget):
    """Main widget for the Acid Gas Dewpoint Calculator application."""

    PRESET_COMPOSITIONS = {
        "Typical Syngas": {"H2O": 15.0, "HF": 0.01, "HCl": 0.02, "H2S": 0.1},
        "High Acid Content": {"H2O": 20.0, "HF": 0.1, "HCl": 0.2, "H2S": 0.5},
        "Coal Gasification": {"H2O": 12.0, "HF": 0.05, "HCl": 0.1, "H2S": 0.3},
        "Biomass Gasification": {"H2O": 18.0, "HF": 0.02, "HCl": 0.05, "H2S": 0.2},
        "Custom": {"H2O": 15.0, "HF": 0.01, "HCl": 0.02, "H2S": 0.1},
    }

    METHODS = ["antoine", "extended_antoine"]

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the Acid Gas Dewpoint Calculator widget."""
        super().__init__(parent)
        self.results: dict[str, Any] | None = None
        self._setup_ui()
        self._connect_signals()
        self._apply_styling()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Inputs
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)

        # Operating Conditions
        conditions_group = QGroupBox("Operating Conditions")
        conditions_layout = QFormLayout(conditions_group)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(-100, 400)
        self.temp_spin.setValue(150)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.setDecimals(1)
        conditions_layout.addRow("Temperature:", self.temp_spin)

        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.1, 300)
        self.pressure_spin.setValue(30)
        self.pressure_spin.setSuffix(" bar")
        self.pressure_spin.setDecimals(2)
        conditions_layout.addRow("Pressure:", self.pressure_spin)

        self.method_combo = QComboBox()
        self.method_combo.addItems(self.METHODS)
        conditions_layout.addRow("Calculation Method:", self.method_combo)

        input_layout.addWidget(conditions_group)

        # Preset Selection
        preset_group = QGroupBox("Composition Preset")
        preset_layout = QFormLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESET_COMPOSITIONS.keys()))
        preset_layout.addRow("Preset:", self.preset_combo)

        input_layout.addWidget(preset_group)

        # Gas Composition
        composition_group = QGroupBox("Gas Composition (mol%)")
        composition_layout = QGridLayout(composition_group)

        self.composition_spins: dict[str, QDoubleSpinBox] = {}
        components = [
            ("H2O", "Water Vapor", 0, 100),
            ("HF", "Hydrogen Fluoride", 0, 10),
            ("HCl", "Hydrogen Chloride", 0, 10),
            ("H2S", "Hydrogen Sulfide", 0, 10),
        ]

        for i, (abbr, name, min_val, max_val) in enumerate(components):
            label = QLabel(f"{name} ({abbr}):")
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setDecimals(4)
            spin.setSuffix(" %")
            self.composition_spins[abbr] = spin
            composition_layout.addWidget(label, i, 0)
            composition_layout.addWidget(spin, i, 1)

        # Set initial values
        self._load_preset("Typical Syngas")

        input_layout.addWidget(composition_group)

        # Calculate button
        self.calculate_btn = QPushButton("Calculate Dewpoints")
        self.calculate_btn.setMinimumHeight(40)
        input_layout.addWidget(self.calculate_btn)

        input_layout.addStretch()
        splitter.addWidget(input_widget)

        # Right panel - Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Dewpoint Results tab
        dewpoint_tab = QWidget()
        dewpoint_tab_layout = QVBoxLayout(dewpoint_tab)

        self.dewpoint_table = QTableWidget()
        self.dewpoint_table.setColumnCount(3)
        self.dewpoint_table.setHorizontalHeaderLabels(
            ["Component", "Dewpoint (°C)", "Partial Pressure (Pa)"]
        )
        dewpoint_header = self.dewpoint_table.horizontalHeader()
        if dewpoint_header is not None:
            dewpoint_header.setStretchLastSection(True)
        dewpoint_tab_layout.addWidget(self.dewpoint_table)

        self.tabs.addTab(dewpoint_tab, "Dewpoint Results")

        # Safety Analysis tab
        safety_tab = QWidget()
        safety_tab_layout = QVBoxLayout(safety_tab)

        self.safety_table = QTableWidget()
        self.safety_table.setColumnCount(2)
        self.safety_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        safety_header = self.safety_table.horizontalHeader()
        if safety_header is not None:
            safety_header.setStretchLastSection(True)
        safety_tab_layout.addWidget(self.safety_table)

        self.tabs.addTab(safety_tab, "Safety Analysis")

        # Warnings tab
        warnings_tab = QWidget()
        warnings_tab_layout = QVBoxLayout(warnings_tab)

        self.warnings_text = QTextEdit()
        self.warnings_text.setReadOnly(True)
        warnings_tab_layout.addWidget(self.warnings_text)

        self.tabs.addTab(warnings_tab, "Warnings & Sources")

        # Chart tab
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)

        self.figure = Figure(figsize=(8, 5), facecolor="#1e1e2e")
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)

        self.tabs.addTab(chart_tab, "Dewpoint Chart")

        results_layout.addWidget(self.tabs)
        splitter.addWidget(results_widget)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.calculate_btn.clicked.connect(self._calculate)
        self.preset_combo.currentTextChanged.connect(self._load_preset)

    def _load_preset(self, preset_name: str) -> None:
        """Load composition values from preset."""
        if preset_name in self.PRESET_COMPOSITIONS:
            preset = self.PRESET_COMPOSITIONS[preset_name]
            for comp, value in preset.items():
                if comp in self.composition_spins:
                    self.composition_spins[comp].setValue(value)

    def _calculate(self) -> None:
        """Perform the dewpoint calculation."""
        try:
            from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
                AcidGasComposition,
                AcidGasDewpointCalculator,
            )

            # Get composition values (convert from % to fraction)
            h2o = self.composition_spins["H2O"].value() / 100
            hf = self.composition_spins["HF"].value() / 100
            hcl = self.composition_spins["HCl"].value() / 100
            h2s = self.composition_spins["H2S"].value() / 100

            # Calculate other component fraction
            acid_gas_total = h2o + hf + hcl + h2s
            other = max(0, 1.0 - acid_gas_total)

            composition = AcidGasComposition(
                h2o=h2o, hf=hf, hcl=hcl, h2s=h2s, other=other
            )

            calculator = AcidGasDewpointCalculator()
            result = calculator.calculate_dewpoint_mixture(
                temperature_c=self.temp_spin.value(),
                pressure_bar=self.pressure_spin.value(),
                composition=composition,
                method=self.method_combo.currentText(),
            )

            # Store results as dict
            results_dict = result.to_dict()
            results_dict["result_obj"] = result
            self.results = results_dict
            self._update_display()

        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not import required modules:\n{e}\n\n"
                "Please ensure the shared library is available.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Calculation Error",
                f"An error occurred during calculation:\n{e}",
            )

    def _update_display(self) -> None:
        """Update the display with calculation results."""
        if self.results is None:
            return

        result_obj = self.results.get("result_obj")
        if result_obj is None:
            return

        # Update dewpoint table
        dewpoints = self.results.get("dewpoints", {})
        components = ["H2O", "HF", "HCl", "H2S"]

        self.dewpoint_table.setRowCount(len(components) + 1)

        for i, comp in enumerate(components):
            dp = dewpoints.get(comp, float("nan"))
            pp = getattr(result_obj, f"{comp.lower()}_partial_pressure_pa", 0)
            self.dewpoint_table.setItem(i, 0, QTableWidgetItem(comp))
            self.dewpoint_table.setItem(
                i, 1, QTableWidgetItem(f"{dp:.2f}" if dp == dp else "N/A")
            )
            self.dewpoint_table.setItem(i, 2, QTableWidgetItem(f"{pp:.2f}"))

        # Overall dewpoint row
        overall = dewpoints.get("overall", float("nan"))
        limiting = dewpoints.get("limiting_component", "Unknown")
        self.dewpoint_table.setItem(len(components), 0, QTableWidgetItem("OVERALL"))
        self.dewpoint_table.setItem(
            len(components),
            1,
            QTableWidgetItem(f"{overall:.2f} ({limiting})"),
        )
        self.dewpoint_table.setItem(len(components), 2, QTableWidgetItem("-"))

        # Update safety table
        safety = self.results.get("safety", {})
        safety_data = [
            ("Operating Temperature", f"{result_obj.temperature_c:.1f} °C"),
            ("Overall Dewpoint", f"{overall:.2f} °C"),
            ("Dewpoint Margin", f"{safety.get('dewpoint_margin_c', 0):.1f} °C"),
            ("Condensation Risk", safety.get("condensation_risk", "Unknown")),
            ("Limiting Component", limiting),
            ("Calculation Method", self.results.get("method", "antoine")),
        ]

        self.safety_table.setRowCount(len(safety_data))
        for i, (param, value) in enumerate(safety_data):
            self.safety_table.setItem(i, 0, QTableWidgetItem(param))
            self.safety_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Update warnings
        warnings = self.results.get("warnings", [])
        sources = self.results.get("sources", [])

        text_parts = []
        if warnings:
            text_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
        else:
            text_parts.append("No warnings - all parameters within acceptable ranges.")

        if sources:
            text_parts.append(
                "\n\nLiterature Sources:\n" + "\n".join(f"  - {s}" for s in sources)
            )

        self.warnings_text.setText("\n".join(text_parts))

        # Update chart
        self._update_chart()

    def _update_chart(self) -> None:
        """Update the dewpoint comparison chart."""
        if self.results is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#313244")

        dewpoints = self.results.get("dewpoints", {})
        components = ["H2O", "HF", "HCl", "H2S"]
        values = [dewpoints.get(c, float("nan")) for c in components]

        # Filter valid values (exclude NaN)
        valid_data = [
            (c, v) for c, v in zip(components, values, strict=True) if not math.isnan(v)
        ]

        if valid_data:
            comps, vals = zip(*valid_data, strict=True)
            colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8"]
            bars = ax.bar(comps, vals, color=colors[: len(comps)])

            # Add value labels on bars
            for bar, val in zip(bars, vals, strict=True):
                height = bar.get_height()
                ax.annotate(
                    f"{val:.1f}°C",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="#cdd6f4",
                    fontsize=9,
                )

            # Add operating temperature line
            temp = self.temp_spin.value()
            ax.axhline(y=temp, color="#f9e2af", linestyle="--", linewidth=2)
            ax.text(
                len(comps) - 0.5,
                temp + 2,
                f"Operating T: {temp}°C",
                color="#f9e2af",
                fontsize=9,
            )

        ax.set_xlabel("Component", color="#cdd6f4")
        ax.set_ylabel("Dewpoint Temperature (°C)", color="#cdd6f4")
        ax.set_title("Acid Gas Dewpoint Comparison", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.grid(True, alpha=0.3, color="#585b70", axis="y")

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
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QDoubleSpinBox, QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #89b4fa;
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
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QTableWidget {
                background-color: #313244;
                border: 1px solid #45475a;
                gridline-color: #45475a;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #45475a;
                padding: 5px;
                border: none;
            }
            QTextEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #313244;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
        """)
