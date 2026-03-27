#!/usr/bin/env python3
"""Inertia Calculator PyQt6 Main Window.

A PyQt6 GUI for calculating and validating inertia tensors.
"""

from __future__ import annotations

import sys

import numpy as np
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
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
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

QScrollArea {{
    border: none;
    background-color: {CATPPUCCIN_MOCHA["base"]};
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

QDoubleSpinBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QDoubleSpinBox:focus {{
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

QTextEdit {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
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
"""


class InertiaCalculatorWindow(BaseCalculatorWindow):
    """Main window for Inertia Calculator application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__(
            calculator_name="InertiaCalculator",
            window_title="Inertia Calculator",
            min_size=(650, 700),
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setStyleSheet(STYLESHEET)

        # Scroll area wrapping the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        self.main_layout.addWidget(scroll_area)

        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("Inertia Calculator")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Tab widget for different input modes
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_primitive_tab(), "Primitive Shapes")
        self.tab_widget.addTab(self._create_manual_tab(), "Manual Input")

        # Results
        main_layout.addWidget(self._create_results_group())

        main_layout.addStretch()

    def _create_primitive_tab(self) -> QWidget:
        """Create the primitive shapes calculation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Shape selection
        shape_group = QGroupBox("Shape Parameters")
        shape_layout = QGridLayout(shape_group)
        shape_layout.setSpacing(10)

        shape_layout.addWidget(QLabel("Shape:"), 0, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(
            ["Solid Box", "Solid Cylinder", "Solid Sphere", "Hollow Cylinder"]
        )
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        shape_layout.addWidget(self.shape_combo, 0, 1)

        # Mass
        shape_layout.addWidget(QLabel("Mass (kg):"), 1, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(0.001, 10000)
        self.mass_input.setDecimals(4)
        self.mass_input.setValue(1.0)
        self.mass_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        shape_layout.addWidget(self.mass_input, 1, 1)

        # Dimension inputs (labels change based on shape)
        self.dim1_label = QLabel("Length X (m):")
        shape_layout.addWidget(self.dim1_label, 2, 0)
        self.dim1_input = QDoubleSpinBox()
        self.dim1_input.setRange(0.001, 100)
        self.dim1_input.setDecimals(4)
        self.dim1_input.setValue(0.1)
        shape_layout.addWidget(self.dim1_input, 2, 1)

        self.dim2_label = QLabel("Length Y (m):")
        shape_layout.addWidget(self.dim2_label, 3, 0)
        self.dim2_input = QDoubleSpinBox()
        self.dim2_input.setRange(0.001, 100)
        self.dim2_input.setDecimals(4)
        self.dim2_input.setValue(0.1)
        shape_layout.addWidget(self.dim2_input, 3, 1)

        self.dim3_label = QLabel("Length Z (m):")
        shape_layout.addWidget(self.dim3_label, 4, 0)
        self.dim3_input = QDoubleSpinBox()
        self.dim3_input.setRange(0.001, 100)
        self.dim3_input.setDecimals(4)
        self.dim3_input.setValue(0.1)
        shape_layout.addWidget(self.dim3_input, 4, 1)

        layout.addWidget(shape_group)

        # Calculate button
        calc_btn = QPushButton("Calculate Inertia")
        calc_btn.clicked.connect(self._calculate_primitive)
        layout.addWidget(calc_btn)

        layout.addStretch()
        return tab

    def _create_manual_tab(self) -> QWidget:
        """Create the manual inertia input tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Inertia input
        inertia_group = QGroupBox("Inertia Tensor (kg*m²)")
        inertia_layout = QGridLayout(inertia_group)
        inertia_layout.setSpacing(10)

        # Principal moments
        inertia_layout.addWidget(QLabel("Ixx:"), 0, 0)
        self.ixx_input = QDoubleSpinBox()
        self.ixx_input.setRange(0.0, 10000)
        self.ixx_input.setDecimals(6)
        self.ixx_input.setValue(0.1)
        inertia_layout.addWidget(self.ixx_input, 0, 1)

        inertia_layout.addWidget(QLabel("Iyy:"), 1, 0)
        self.iyy_input = QDoubleSpinBox()
        self.iyy_input.setRange(0.0, 10000)
        self.iyy_input.setDecimals(6)
        self.iyy_input.setValue(0.1)
        inertia_layout.addWidget(self.iyy_input, 1, 1)

        inertia_layout.addWidget(QLabel("Izz:"), 2, 0)
        self.izz_input = QDoubleSpinBox()
        self.izz_input.setRange(0.0, 10000)
        self.izz_input.setDecimals(6)
        self.izz_input.setValue(0.1)
        inertia_layout.addWidget(self.izz_input, 2, 1)

        # Products of inertia
        inertia_layout.addWidget(QLabel("Ixy:"), 0, 2)
        self.ixy_input = QDoubleSpinBox()
        self.ixy_input.setRange(-10000, 10000)
        self.ixy_input.setDecimals(6)
        self.ixy_input.setValue(0.0)
        inertia_layout.addWidget(self.ixy_input, 0, 3)

        inertia_layout.addWidget(QLabel("Ixz:"), 1, 2)
        self.ixz_input = QDoubleSpinBox()
        self.ixz_input.setRange(-10000, 10000)
        self.ixz_input.setDecimals(6)
        self.ixz_input.setValue(0.0)
        inertia_layout.addWidget(self.ixz_input, 1, 3)

        inertia_layout.addWidget(QLabel("Iyz:"), 2, 2)
        self.iyz_input = QDoubleSpinBox()
        self.iyz_input.setRange(-10000, 10000)
        self.iyz_input.setDecimals(6)
        self.iyz_input.setValue(0.0)
        inertia_layout.addWidget(self.iyz_input, 2, 3)

        layout.addWidget(inertia_group)

        # Mass input
        mass_group = QGroupBox("Mass")
        mass_layout = QGridLayout(mass_group)
        mass_layout.addWidget(QLabel("Mass (kg):"), 0, 0)
        self.manual_mass_input = QDoubleSpinBox()
        self.manual_mass_input.setRange(0.001, 10000)
        self.manual_mass_input.setDecimals(4)
        self.manual_mass_input.setValue(1.0)
        mass_layout.addWidget(self.manual_mass_input, 0, 1)
        layout.addWidget(mass_group)

        # Validate button
        validate_btn = QPushButton("Validate Inertia")
        validate_btn.clicked.connect(self._validate_manual)
        layout.addWidget(validate_btn)

        layout.addStretch()
        return tab

    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Results will appear here...")
        layout.addWidget(self.results_text)

        return group

    def _on_shape_changed(self, shape: str) -> None:
        """Update dimension labels based on selected shape."""
        if shape == "Solid Box":
            self.dim1_label.setText("Length X (m):")
            self.dim2_label.setText("Length Y (m):")
            self.dim3_label.setText("Length Z (m):")
            self.dim1_input.setEnabled(True)
            self.dim2_input.setEnabled(True)
            self.dim3_input.setEnabled(True)
        elif shape == "Solid Cylinder":
            self.dim1_label.setText("Radius (m):")
            self.dim2_label.setText("Height (m):")
            self.dim3_label.setText("(unused)")
            self.dim1_input.setEnabled(True)
            self.dim2_input.setEnabled(True)
            self.dim3_input.setEnabled(False)
        elif shape == "Solid Sphere":
            self.dim1_label.setText("Radius (m):")
            self.dim2_label.setText("(unused)")
            self.dim3_label.setText("(unused)")
            self.dim1_input.setEnabled(True)
            self.dim2_input.setEnabled(False)
            self.dim3_input.setEnabled(False)
        elif shape == "Hollow Cylinder":
            self.dim1_label.setText("Outer Radius (m):")
            self.dim2_label.setText("Inner Radius (m):")
            self.dim3_label.setText("Height (m):")
            self.dim1_input.setEnabled(True)
            self.dim2_input.setEnabled(True)
            self.dim3_input.setEnabled(True)

    def _calculate_primitive(self) -> None:
        """Calculate inertia for primitive shape."""
        shape = self.shape_combo.currentText()
        mass = self.mass_input.value()

        try:
            if shape == "Solid Box":
                lx = self.dim1_input.value()
                ly = self.dim2_input.value()
                lz = self.dim3_input.value()
                ixx = (1 / 12) * mass * (ly**2 + lz**2)
                iyy = (1 / 12) * mass * (lx**2 + lz**2)
                izz = (1 / 12) * mass * (lx**2 + ly**2)
                desc = f"Solid Box: {lx}m x {ly}m x {lz}m"

            elif shape == "Solid Cylinder":
                r = self.dim1_input.value()
                h = self.dim2_input.value()
                ixx = (1 / 12) * mass * (3 * r**2 + h**2)
                iyy = ixx
                izz = (1 / 2) * mass * r**2
                desc = f"Solid Cylinder: r={r}m, h={h}m (axis along Z)"

            elif shape == "Solid Sphere":
                r = self.dim1_input.value()
                ixx = (2 / 5) * mass * r**2
                iyy = ixx
                izz = ixx
                desc = f"Solid Sphere: r={r}m"

            elif shape == "Hollow Cylinder":
                r_out = self.dim1_input.value()
                r_in = self.dim2_input.value()
                h = self.dim3_input.value()
                if r_in >= r_out:
                    raise ValueError("Inner radius must be less than outer radius")
                ixx = (1 / 12) * mass * (3 * (r_out**2 + r_in**2) + h**2)
                iyy = ixx
                izz = (1 / 2) * mass * (r_out**2 + r_in**2)
                desc = f"Hollow Cylinder: r_out={r_out}m, r_in={r_in}m, h={h}m"
            else:
                raise ValueError(f"Unknown shape: {shape}")

            self._display_results(ixx, iyy, izz, 0, 0, 0, mass, desc)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            self.results_text.setPlainText(f"Error: {e}")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _validate_manual(self) -> None:
        """Validate manually entered inertia values."""
        ixx = self.ixx_input.value()
        iyy = self.iyy_input.value()
        izz = self.izz_input.value()
        ixy = self.ixy_input.value()
        ixz = self.ixz_input.value()
        iyz = self.iyz_input.value()
        mass = self.manual_mass_input.value()

        self._display_results(ixx, iyy, izz, ixy, ixz, iyz, mass, "Manual Input")

    def _display_results(
        self,
        ixx: float,
        iyy: float,
        izz: float,
        ixy: float,
        ixz: float,
        iyz: float,
        mass: float,
        description: str,
    ) -> None:
        """Display calculation results."""
        if not (ixx is not None):
            raise ValueError("ixx must be provided")
        results = []
        results.append("Inertia Calculation Results")
        results.append("=" * 50)
        results.append(f"\nSource: {description}")
        results.append(f"Mass: {mass:.4f} kg")

        results.append("\nPrincipal Moments of Inertia (kg*m²):")
        results.append(f"  Ixx = {ixx:.6f}")
        results.append(f"  Iyy = {iyy:.6f}")
        results.append(f"  Izz = {izz:.6f}")

        results.append("\nProducts of Inertia (kg*m²):")
        results.append(f"  Ixy = {ixy:.6f}")
        results.append(f"  Ixz = {ixz:.6f}")
        results.append(f"  Iyz = {iyz:.6f}")

        # Matrix form
        results.append("\nInertia Tensor Matrix:")
        results.append(f"  [{ixx:12.6f}  {ixy:12.6f}  {ixz:12.6f}]")
        results.append(f"  [{ixy:12.6f}  {iyy:12.6f}  {iyz:12.6f}]")
        results.append(f"  [{ixz:12.6f}  {iyz:12.6f}  {izz:12.6f}]")

        # URDF format
        results.append("\nURDF Format:")
        results.append(
            f'  <inertia ixx="{ixx:.6f}" ixy="{ixy:.6f}" '
            f'ixz="{ixz:.6f}" iyy="{iyy:.6f}" '
            f'iyz="{iyz:.6f}" izz="{izz:.6f}"/>'
        )

        # Validation
        results.append("\nValidation:")
        errors = self._validate_inertia(ixx, iyy, izz, ixy, ixz, iyz)
        if errors:
            results.append("  INVALID - Issues found:")
            results.extend([f"    - {error}" for error in errors])
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['yellow']};")
        else:
            results.append("  VALID - All checks passed")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")

        self.results_text.setPlainText("\n".join(results))

    def _validate_inertia(
        self,
        ixx: float,
        iyy: float,
        izz: float,
        ixy: float,
        ixz: float,
        iyz: float,
    ) -> list[str]:
        """Validate inertia tensor."""
        if not (ixx is not None):
            raise ValueError("ixx must be provided")
        errors = []

        # Check positive diagonal
        if ixx <= 0:
            errors.append("Ixx must be positive")
        if iyy <= 0:
            errors.append("Iyy must be positive")
        if izz <= 0:
            errors.append("Izz must be positive")

        if errors:
            return errors

        # Check triangle inequality
        if not (abs(ixx - iyy) <= izz <= ixx + iyy):
            errors.append("Triangle inequality violated for Izz")
        if not (abs(iyy - izz) <= ixx <= iyy + izz):
            errors.append("Triangle inequality violated for Ixx")
        if not (abs(ixx - izz) <= iyy <= ixx + izz):
            errors.append("Triangle inequality violated for Iyy")

        # Check positive definite
        tensor = np.array(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ]
        )
        try:
            np.linalg.cholesky(tensor)
        except np.linalg.LinAlgError:
            errors.append("Inertia tensor is not positive definite")

        return errors


def main() -> int:
    """Run the Inertia Calculator application."""
    app = QApplication(sys.argv)
    window = InertiaCalculatorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
