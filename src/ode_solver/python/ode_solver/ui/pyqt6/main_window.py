#!/usr/bin/env python3
"""ODE Solver PyQt6 Main Window.

A PyQt6 GUI for solving systems of ordinary differential equations.
"""

from __future__ import annotations

import sys
from typing import Any

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
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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

QPushButton:disabled {{
    background-color: {CATPPUCCIN_MOCHA["surface2"]};
    color: {CATPPUCCIN_MOCHA["overlay0"]};
}}
"""

# Preset ODE examples
ODE_PRESETS: dict[str, dict[str, Any]] = {
    "Exponential Decay": {
        "derivatives": {"y": "-k*y"},
        "parameters": {"k": "0.1"},
        "initial": {"y": "100"},
        "description": "dy/dt = -k*y (exponential decay)",
    },
    "Heating/Cooling": {
        "derivatives": {"T": "k*(T_env - T)"},
        "parameters": {"k": "0.3", "T_env": "350"},
        "initial": {"T": "300"},
        "description": "dT/dt = k*(T_env - T) (Newton's law of cooling)",
    },
    "Harmonic Oscillator": {
        "derivatives": {"x": "v", "v": "-omega**2*x"},
        "parameters": {"omega": "1.0"},
        "initial": {"x": "1", "v": "0"},
        "description": "dx/dt=v, dv/dt=-omega^2*x (simple harmonic motion)",
    },
    "Damped Oscillator": {
        "derivatives": {"x": "v", "v": "-2*zeta*omega*v - omega**2*x"},
        "parameters": {"omega": "1.0", "zeta": "0.1"},
        "initial": {"x": "1", "v": "0"},
        "description": "Damped harmonic oscillator with damping ratio zeta",
    },
    "Lotka-Volterra": {
        "derivatives": {"x": "a*x - b*x*y", "y": "-c*y + d*x*y"},
        "parameters": {"a": "1.0", "b": "0.1", "c": "1.5", "d": "0.075"},
        "initial": {"x": "10", "y": "5"},
        "description": "Predator-prey model (x=prey, y=predators)",
    },
}


class ODESolverWindow(QMainWindow):
    """Main window for ODE Solver application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("ODE Solver")
        self.setMinimumSize(700, 800)
        self.setStyleSheet(STYLESHEET)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("ODE Solver")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Preset selector
        main_layout.addWidget(self._create_preset_group())

        # ODE definition
        main_layout.addWidget(self._create_ode_group())

        # Time parameters
        main_layout.addWidget(self._create_time_group())

        # Solve button
        self.solve_btn = QPushButton("Solve ODE System")
        self.solve_btn.clicked.connect(self._solve)
        main_layout.addWidget(self.solve_btn)

        # Results
        main_layout.addWidget(self._create_results_group())

        main_layout.addStretch()

    def _create_preset_group(self) -> QGroupBox:
        """Create the preset selector group."""
        group = QGroupBox("Preset Examples")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Select Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + list(ODE_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo, 0, 1)

        self.preset_description = QLabel("")
        self.preset_description.setWordWrap(True)
        self.preset_description.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['subtext0']}; font-style: italic;"
        )
        layout.addWidget(self.preset_description, 1, 0, 1, 2)

        return group

    def _create_ode_group(self) -> QGroupBox:
        """Create the ODE definition group."""
        group = QGroupBox("ODE System Definition")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Derivatives
        layout.addWidget(QLabel("Derivatives (var: expression):"), 0, 0, 1, 2)
        self.derivatives_edit = QTextEdit()
        self.derivatives_edit.setPlaceholderText(
            "Enter derivatives, one per line:\ny: -k*y\nExample: variable_name: expression"
        )
        self.derivatives_edit.setMaximumHeight(80)
        layout.addWidget(self.derivatives_edit, 1, 0, 1, 2)

        # Parameters
        layout.addWidget(QLabel("Parameters (name: value):"), 2, 0, 1, 2)
        self.parameters_edit = QTextEdit()
        self.parameters_edit.setPlaceholderText(
            "Enter parameters, one per line:\nk: 0.1\nExample: param_name: numerical_value"
        )
        self.parameters_edit.setMaximumHeight(80)
        layout.addWidget(self.parameters_edit, 3, 0, 1, 2)

        # Initial conditions
        layout.addWidget(QLabel("Initial Conditions (var: value):"), 4, 0, 1, 2)
        self.initial_edit = QTextEdit()
        self.initial_edit.setPlaceholderText(
            "Enter initial values, one per line:\ny: 100\nMust match derivative variables"
        )
        self.initial_edit.setMaximumHeight(80)
        layout.addWidget(self.initial_edit, 5, 0, 1, 2)

        return group

    def _create_time_group(self) -> QGroupBox:
        """Create the time parameters group."""
        group = QGroupBox("Time Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Start time
        layout.addWidget(QLabel("Start Time:"), 0, 0)
        self.t_start_input = QDoubleSpinBox()
        self.t_start_input.setRange(0, 1e6)
        self.t_start_input.setDecimals(2)
        self.t_start_input.setValue(0)
        layout.addWidget(self.t_start_input, 0, 1)

        # End time
        layout.addWidget(QLabel("End Time:"), 1, 0)
        self.t_end_input = QDoubleSpinBox()
        self.t_end_input.setRange(0.1, 1e6)
        self.t_end_input.setDecimals(2)
        self.t_end_input.setValue(20)
        layout.addWidget(self.t_end_input, 1, 1)

        # Number of points
        layout.addWidget(QLabel("Output Points:"), 2, 0)
        self.num_points_input = QSpinBox()
        self.num_points_input.setRange(10, 10000)
        self.num_points_input.setValue(100)
        layout.addWidget(self.num_points_input, 2, 1)

        return group

    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Solution Results")
        layout = QVBoxLayout(group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(250)
        self.results_text.setPlaceholderText(
            "Click 'Solve ODE System' to see results..."
        )
        layout.addWidget(self.results_text)

        return group

    def _on_preset_changed(self, preset_name: str) -> None:
        """Handle preset selection change."""
        if preset_name == "Custom":
            self.preset_description.setText("")
            return

        preset = ODE_PRESETS.get(preset_name)
        if not preset:
            return

        self.preset_description.setText(preset["description"])

        # Fill in derivatives
        deriv_lines = [f"{var}: {expr}" for var, expr in preset["derivatives"].items()]
        self.derivatives_edit.setPlainText("\n".join(deriv_lines))

        # Fill in parameters
        param_lines = [
            f"{name}: {value}" for name, value in preset["parameters"].items()
        ]
        self.parameters_edit.setPlainText("\n".join(param_lines))

        # Fill in initial conditions
        init_lines = [f"{var}: {value}" for var, value in preset["initial"].items()]
        self.initial_edit.setPlainText("\n".join(init_lines))

    def _parse_dict_input(self, text: str) -> dict[str, str]:
        """Parse colon-separated key-value pairs from text."""
        result = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    def _solve(self) -> None:
        """Solve the ODE system."""
        try:
            from upstream_drift_tools.process_calculators.ode_solver import (
                ODESolver,
            )

            # Parse inputs
            derivatives = self._parse_dict_input(self.derivatives_edit.toPlainText())
            parameters_str = self._parse_dict_input(self.parameters_edit.toPlainText())
            initial_str = self._parse_dict_input(self.initial_edit.toPlainText())

            if not derivatives:
                raise ValueError("No derivatives defined")

            # Convert parameters to floats
            parameters = {k: float(v) for k, v in parameters_str.items()}

            # Get initial conditions in order
            y0 = []
            for var in derivatives:
                if var not in initial_str:
                    raise ValueError(f"Missing initial condition for '{var}'")
                y0.append(float(initial_str[var]))

            # Time span
            t_start = self.t_start_input.value()
            t_end = self.t_end_input.value()
            num_points = self.num_points_input.value()
            t_eval = np.linspace(t_start, t_end, num_points)

            # Solve
            solver = ODESolver(derivatives, parameters)
            solution = solver.solve((t_start, t_end), y0, t_eval=t_eval)

            # Format results
            results = []
            results.append("ODE Solution Results")
            results.append("=" * 50)

            results.append("\nSystem Definition:")
            for var, expr in derivatives.items():
                results.append(f"  d{var}/dt = {expr}")

            results.append("\nParameters:")
            for name, value in parameters.items():
                results.append(f"  {name} = {value}")

            results.append("\nInitial Conditions:")
            for var, val in zip(derivatives.keys(), y0, strict=True):
                results.append(f"  {var}(0) = {val}")

            results.append(f"\nTime Range: [{t_start}, {t_end}]")
            results.append(f"Solution Points: {num_points}")

            results.append("\nFinal Values:")
            for idx, var in enumerate(derivatives.keys()):
                results.append(f"  {var}({t_end}) = {solution.y[idx][-1]:.6f}")

            results.append("\nSolution Summary:")
            for idx, var in enumerate(derivatives.keys()):
                results.append(
                    f"  {var}: min={np.min(solution.y[idx]):.4f}, "
                    f"max={np.max(solution.y[idx]):.4f}"
                )

            results.append("\nSample Data Points:")
            results.append("-" * 50)

            # Header
            header = "    t    |" + "|".join(
                f"  {var:^10}  " for var in derivatives.keys()
            )
            results.append(header)
            results.append("-" * len(header))

            # Show every 10th point
            step = max(1, len(solution.t) // 10)
            for i in range(0, len(solution.t), step):
                row = f"{solution.t[i]:8.3f} |"
                row += "|".join(
                    f"  {solution.y[idx][i]:12.6f}  " for idx in range(len(derivatives))
                )
                results.append(row)

            self.results_text.setPlainText("\n".join(results))
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")

        except ImportError as e:
            self.results_text.setPlainText(f"Error: {e}")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")


def main() -> int:
    """Run the ODE Solver application."""
    app = QApplication(sys.argv)
    window = ODESolverWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
