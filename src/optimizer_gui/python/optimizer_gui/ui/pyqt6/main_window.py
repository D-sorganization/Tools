from numba import jit

#!/usr/bin/env python3
"""Adam Optimizer PyQt6 Main Window.

A PyQt6 GUI for configuring and running Adam-based optimization.
"""

from __future__ import annotations  # noqa: E402, F404

import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QFont  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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

QDoubleSpinBox, QSpinBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 120px;
}}

QComboBox:hover {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    selection-background-color: {CATPPUCCIN_MOCHA["surface2"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
}}

QCheckBox {{
    color: {CATPPUCCIN_MOCHA["text"]};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
}}

QCheckBox::indicator:checked {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    border-color: {CATPPUCCIN_MOCHA["blue"]};
}}

QTableWidget {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    gridline-color: {CATPPUCCIN_MOCHA["surface1"]};
}}

QHeaderView::section {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    padding: 6px;
    border: none;
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

QPushButton#runBtn {{
    background-color: {CATPPUCCIN_MOCHA["green"]};
}}

QPushButton#runBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["teal"]};
}}
"""


@dataclass
class ParameterConfig:
    """Configuration for an optimization parameter."""

    name: str
    initial: float
    min_val: float
    max_val: float


class OptimizerWindow(QMainWindow):
    """Main window for Adam Optimizer application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._history: list[dict] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Adam Optimizer")
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
        title_label = QLabel("Adam Optimizer")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_parameters_tab(), "Parameters")
        self.tab_widget.addTab(self._create_adam_settings_tab(), "Adam Settings")
        self.tab_widget.addTab(self._create_results_tab(), "Results")

        # Run button
        run_btn = QPushButton("Run Optimization")
        run_btn.setObjectName("runBtn")
        run_btn.clicked.connect(self._run_optimization)
        main_layout.addWidget(run_btn)

        main_layout.addStretch()

    def _create_parameters_tab(self) -> QWidget:
        """Create the parameters configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Optimization goal
        goal_group = QGroupBox("Optimization Goal")
        goal_layout = QGridLayout(goal_group)
        goal_layout.setSpacing(10)

        goal_layout.addWidget(QLabel("Objective:"), 0, 0)
        self.maximize_checkbox = QCheckBox("Maximize (uncheck for minimize)")
        self.maximize_checkbox.setChecked(True)
        goal_layout.addWidget(self.maximize_checkbox, 0, 1)

        layout.addWidget(goal_group)

        # Parameter table
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QVBoxLayout(params_group)

        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(["Name", "Initial", "Min", "Max"])
        header = self.params_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        self.params_table.setRowCount(4)

        # Default parameters
        default_params = [
            ("Temperature", 800, 600, 1200),
            ("O2/Feed Ratio", 0.3, 0.1, 0.5),
            ("Steam/Feed Ratio", 0.5, 0.0, 1.0),
            ("Pressure", 1.0, 0.5, 2.0),
        ]

        for row, (name, initial, min_val, max_val) in enumerate(default_params):
            self.params_table.setItem(row, 0, QTableWidgetItem(name))
            self.params_table.setItem(row, 1, QTableWidgetItem(str(initial)))
            self.params_table.setItem(row, 2, QTableWidgetItem(str(min_val)))
            self.params_table.setItem(row, 3, QTableWidgetItem(str(max_val)))

        params_layout.addWidget(self.params_table)

        # Add/Remove buttons
        btn_layout = QGridLayout()
        add_btn = QPushButton("Add Parameter")
        add_btn.clicked.connect(self._add_parameter)
        btn_layout.addWidget(add_btn, 0, 0)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_parameter)
        btn_layout.addWidget(remove_btn, 0, 1)

        params_layout.addLayout(btn_layout)

        layout.addWidget(params_group)

        layout.addStretch()
        return tab

    def _create_adam_settings_tab(self) -> QWidget:
        """Create the Adam hyperparameters settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Adam hyperparameters
        adam_group = QGroupBox("Adam Hyperparameters")
        adam_layout = QGridLayout(adam_group)
        adam_layout.setSpacing(10)

        # Learning rate
        adam_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setDecimals(4)
        self.learning_rate_input.setValue(0.01)
        self.learning_rate_input.setSingleStep(0.001)
        self.learning_rate_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        adam_layout.addWidget(self.learning_rate_input, 0, 1)

        # Beta1
        adam_layout.addWidget(QLabel("Beta1 (momentum):"), 1, 0)
        self.beta1_input = QDoubleSpinBox()
        self.beta1_input.setRange(0.0, 0.999)
        self.beta1_input.setDecimals(3)
        self.beta1_input.setValue(0.9)
        adam_layout.addWidget(self.beta1_input, 1, 1)

        # Beta2
        adam_layout.addWidget(QLabel("Beta2 (RMSprop):"), 2, 0)
        self.beta2_input = QDoubleSpinBox()
        self.beta2_input.setRange(0.0, 0.9999)
        self.beta2_input.setDecimals(4)
        self.beta2_input.setValue(0.999)
        adam_layout.addWidget(self.beta2_input, 2, 1)

        # Epsilon
        adam_layout.addWidget(QLabel("Epsilon:"), 3, 0)
        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(1e-10, 1e-4)
        self.epsilon_input.setDecimals(10)
        self.epsilon_input.setValue(1e-8)
        adam_layout.addWidget(self.epsilon_input, 3, 1)

        layout.addWidget(adam_group)

        # Convergence settings
        conv_group = QGroupBox("Convergence Settings")
        conv_layout = QGridLayout(conv_group)
        conv_layout.setSpacing(10)

        # Max iterations
        conv_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 10000)
        self.max_iter_input.setValue(100)
        conv_layout.addWidget(self.max_iter_input, 0, 1)

        # Tolerance
        conv_layout.addWidget(QLabel("Tolerance:"), 1, 0)
        self.tolerance_input = QDoubleSpinBox()
        self.tolerance_input.setRange(1e-10, 1.0)
        self.tolerance_input.setDecimals(8)
        self.tolerance_input.setValue(1e-6)
        conv_layout.addWidget(self.tolerance_input, 1, 1)

        # Gradient step
        conv_layout.addWidget(QLabel("Gradient Step:"), 2, 0)
        self.grad_step_input = QDoubleSpinBox()
        self.grad_step_input.setRange(1e-8, 1.0)
        self.grad_step_input.setDecimals(6)
        self.grad_step_input.setValue(0.001)
        conv_layout.addWidget(self.grad_step_input, 2, 1)

        layout.addWidget(conv_group)

        # Alternative methods
        method_group = QGroupBox("Alternative Methods")
        method_layout = QGridLayout(method_group)

        method_layout.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "Adam",
                "Grid Search",
                "L-BFGS-B",
                "Differential Evolution",
            ]
        )
        method_layout.addWidget(self.method_combo, 0, 1)

        layout.addWidget(method_group)

        layout.addStretch()
        return tab

    def _create_results_tab(self) -> QWidget:
        """Create the results display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Best result
        best_group = QGroupBox("Best Result")
        best_layout = QGridLayout(best_group)
        best_layout.setSpacing(8)

        best_layout.addWidget(QLabel("Best Objective:"), 0, 0)
        self.best_objective_label = QLabel("-")
        self.best_objective_label.setStyleSheet(
            f"color: {CATPPUCCIN_MOCHA['green']}; font-weight: bold;"
        )
        best_layout.addWidget(self.best_objective_label, 0, 1)

        best_layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.iterations_label = QLabel("-")
        best_layout.addWidget(self.iterations_label, 1, 1)

        best_layout.addWidget(QLabel("Converged:"), 2, 0)
        self.converged_label = QLabel("-")
        best_layout.addWidget(self.converged_label, 2, 1)

        layout.addWidget(best_group)

        # Best parameters
        params_group = QGroupBox("Best Parameters")
        params_layout = QVBoxLayout(params_group)
        self.best_params_text = QTextEdit()
        self.best_params_text.setReadOnly(True)
        self.best_params_text.setMaximumHeight(120)
        self.best_params_text.setPlaceholderText("Best parameters will appear here...")
        params_layout.addWidget(self.best_params_text)
        layout.addWidget(params_group)

        # History
        history_group = QGroupBox("Optimization History")
        history_layout = QVBoxLayout(history_group)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText(
            "Optimization history will appear here...\n\n" "Click 'Run Optimization' to start."
        )
        history_layout.addWidget(self.history_text)
        layout.addWidget(history_group)

        return tab

    def _add_parameter(self) -> None:
        """Add a new parameter row."""
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)
        self.params_table.setItem(row, 0, QTableWidgetItem(f"Param{row + 1}"))
        self.params_table.setItem(row, 1, QTableWidgetItem("0.5"))
        self.params_table.setItem(row, 2, QTableWidgetItem("0.0"))
        self.params_table.setItem(row, 3, QTableWidgetItem("1.0"))

    def _remove_parameter(self) -> None:
        """Remove selected parameter rows."""
        selected_rows = set()
        for item in self.params_table.selectedItems():
            selected_rows.add(item.row())

        for row in sorted(selected_rows, reverse=True):
            self.params_table.removeRow(row)

    def _get_parameters(self) -> list[ParameterConfig]:
        """Get parameter configurations from the table."""
        params = []
        for row in range(self.params_table.rowCount()):
            name_item = self.params_table.item(row, 0)
            initial_item = self.params_table.item(row, 1)
            min_item = self.params_table.item(row, 2)
            max_item = self.params_table.item(row, 3)

            if (
                name_item is not None
                and initial_item is not None
                and min_item is not None
                and max_item is not None
            ):
                try:
                    params.append(
                        ParameterConfig(
                            name=name_item.text(),
                            initial=float(initial_item.text()),
                            min_val=float(min_item.text()),
                            max_val=float(max_item.text()),
                        )
                    )
                except ValueError:
                    continue

        return params

    def _run_optimization(self) -> None:
        """Run the optimization."""
        params = self._get_parameters()
        if not params:
            self.history_text.setPlainText("Error: No valid parameters defined.")
            self.history_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")
            return

        method = self.method_combo.currentText()

        if method == "Adam":
            self._run_adam_demo(params)
        else:
            self._run_surface_demo(params, method)

    @jit(nopython=True, fastmath=True)
    def _run_adam_demo(self, params: list[ParameterConfig]) -> None:
        """Run Adam optimization demo."""
        if not (params is not None):
            raise ValueError("params must be provided")
        maximize = self.maximize_checkbox.isChecked()
        learning_rate = self.learning_rate_input.value()
        beta1 = self.beta1_input.value()
        beta2 = self.beta2_input.value()
        epsilon = self.epsilon_input.value()
        max_iterations = self.max_iter_input.value()
        tolerance = self.tolerance_input.value()

        values, lower, upper = self._init_adam_arrays(params)
        m = np.zeros_like(values)
        v = np.zeros_like(values)

        self._history = []
        best_obj = np.inf if not maximize else -np.inf
        best_params: dict[str, float] = {}

        for iteration in range(1, max_iterations + 1):
            obj = self._eval_rosenbrock(values, maximize)

            if (maximize and obj > best_obj) or (not maximize and obj < best_obj):
                best_obj = obj
                best_params = {
                    p.name: float(values[i]) for i, p in enumerate(params[: len(values)])
                }

            self._history.append(
                {
                    "iteration": iteration,
                    "objective": obj,
                    "parameters": {
                        p.name: float(values[i]) for i, p in enumerate(params[: len(values)])
                    },
                }
            )

            gradient = self._compute_numerical_gradient(values, lower, upper, maximize)

            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)

            m_hat = m / (1 - beta1**iteration)
            v_hat = v / (1 - beta2**iteration)

            direction = 1.0 if maximize else -1.0
            update = direction * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            prev_values = values.copy()
            values = np.clip(values + update, lower, upper)

            if np.linalg.norm(values - prev_values) < tolerance:
                break

        self._display_adam_results(best_obj, best_params, max_iterations)

    @staticmethod
    def _init_adam_arrays(
        params: list[ParameterConfig],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize value, lower-bound, and upper-bound arrays from parameters."""
        subset = params[:2] if len(params) >= 2 else params
        values = np.array([p.initial for p in subset])
        lower = np.array([p.min_val for p in subset])
        upper = np.array([p.max_val for p in subset])
        return values, lower, upper

    @staticmethod
    def _eval_rosenbrock(values: np.ndarray, maximize: bool) -> float:
        """Evaluate the Rosenbrock demo objective function."""
        if not (values is not None):
            raise ValueError("values must be provided")
        if len(values) >= 2:
            x, y = values[0], values[1]
            obj = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        else:
            obj = (values[0] - 1) ** 2
        return -obj if maximize else obj

    @jit(nopython=True, fastmath=True)
    def _compute_numerical_gradient(
        self,
        values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        maximize: bool,
    ) -> np.ndarray:
        """Compute numerical gradient via central differences."""
        if not (values is not None):
            raise ValueError("values must be provided")
        gradient = np.zeros_like(values)
        step = self.grad_step_input.value()
        for i in range(len(values)):
            plus = values.copy()
            minus = values.copy()
            plus[i] = np.clip(values[i] + step, lower[i], upper[i])
            minus[i] = np.clip(values[i] - step, lower[i], upper[i])

            obj_plus = self._eval_rosenbrock(plus, maximize)
            obj_minus = self._eval_rosenbrock(minus, maximize)

            if plus[i] != minus[i]:
                gradient[i] = (obj_plus - obj_minus) / (plus[i] - minus[i])
        return gradient

    def _display_adam_results(
        self,
        best_obj: float,
        best_params: dict[str, float],
        max_iterations: int,
    ) -> None:
        """Update UI with Adam optimization results."""
        if not (best_obj is not None):
            raise ValueError("best_obj must be provided")
        self.best_objective_label.setText(f"{best_obj:.6f}")
        self.iterations_label.setText(str(len(self._history)))
        self.converged_label.setText(
            "Yes" if len(self._history) < max_iterations else "No (max iterations)"
        )

        params_text = "\n".join([f"{k}: {v:.6f}" for k, v in best_params.items()])
        self.best_params_text.setPlainText(params_text)

        history_lines = ["Iteration | Objective | Parameters"]
        history_lines.append("-" * 60)
        for entry in self._history[-20:]:
            param_str = ", ".join([f"{k}={v:.4f}" for k, v in entry["parameters"].items()])
            history_lines.append(
                f"{entry['iteration']:4d}      | {entry['objective']:10.6f} | {param_str}"
            )

        if len(self._history) > 20:
            history_lines.insert(2, f"... ({len(self._history) - 20} earlier entries)")

        self.history_text.setPlainText("\n".join(history_lines))
        self.history_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['text']};")
        self.tab_widget.setCurrentIndex(2)

    def _run_surface_demo(self, params: list[ParameterConfig], method: str) -> None:
        """Run surface optimization demo."""
        if not (params is not None):
            raise ValueError("params must be provided")
        if len(params) < 2:
            self.history_text.setPlainText(
                "Error: Surface optimization requires at least 2 parameters."
            )
            return

        # Create demo surface
        p1, p2 = params[0], params[1]
        x = np.linspace(p1.min_val, p1.max_val, 20)
        y = np.linspace(p2.min_val, p2.max_val, 20)
        X, Y = np.meshgrid(x, y)

        # Demo: negative Rosenbrock (so we maximize to find minimum)
        Z = -((1 - X) ** 2 + 100 * (Y - X**2) ** 2)

        # Find optimal
        maximize = self.maximize_checkbox.isChecked()
        if maximize:
            idx = np.unravel_index(np.argmax(Z), Z.shape)
        else:
            idx = np.unravel_index(np.argmin(Z), Z.shape)

        opt_x = X[idx]
        opt_y = Y[idx]
        opt_z = Z[idx]

        self.best_objective_label.setText(f"{opt_z:.6f}")
        self.iterations_label.setText("Grid: 400 evaluations")
        self.converged_label.setText("N/A (grid search)")

        self.best_params_text.setPlainText(f"{p1.name}: {opt_x:.6f}\n{p2.name}: {opt_y:.6f}")

        self.history_text.setPlainText(
            f"Surface Optimization Results ({method})\n"
            f"{'=' * 50}\n\n"
            f"Method: {method}\n"
            f"Grid Size: 20 x 20 = 400 points\n"
            f"Optimal {p1.name}: {opt_x:.6f}\n"
            f"Optimal {p2.name}: {opt_y:.6f}\n"
            f"Optimal Objective: {opt_z:.6f}\n\n"
            f"Note: This is a demo using the Rosenbrock function.\n"
            f"Real optimization requires an evaluation engine."
        )
        self.tab_widget.setCurrentIndex(2)


def main() -> int:
    """Run the Adam Optimizer application."""
    app = QApplication(sys.argv)
    window = OptimizerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
