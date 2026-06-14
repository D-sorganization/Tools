# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

#!/usr/bin/env python3
"""ODE Solver PyQt6 Main Window.

A PyQt6 GUI for solving systems of ordinary differential equations.
"""

from __future__ import annotations

import logging
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

from contracts import require
from ode_solver.timeout import SolverTimeoutError, with_timeout
from shared.python.theme.integration import ThemedWindowMixin

_log = logging.getLogger(__name__)

# Qt enum aliases — break LoD chains (Qt.X.Y is a 3-level access)
_SCROLL_BAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter


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


class ODESolverWindow(ThemedWindowMixin, QMainWindow):
    """Main window for ODE Solver application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._results_status = "default"
        self.setup_theme_support()
        self._notes_dock: Any | None = None
        self._setup_ui()

    # -- Notes integration (shared workspace) --
    def _toggle_notes(self) -> None:
        """Show/hide the shared notes dock widget."""
        try:
            from pathlib import Path

            from notes.integration import attach_notes_dock
        except ImportError:
            return
        if self._notes_dock is None:
            project_dir = Path(__file__).resolve().parents[4]
            self._notes_dock = attach_notes_dock(self, project_dir=project_dir)
        self._notes_dock.setVisible(not self._notes_dock.isVisible())

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("ODE Solver")
        self.setMinimumSize(700, 800)

        # Menu bar with Notes toggle
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("&View") if menu_bar is not None else None
        notes_action = (
            view_menu.addAction("Toggle &Notes") if view_menu is not None else None
        )
        if notes_action is not None:
            notes_action.triggered.connect(self._toggle_notes)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(_SCROLL_BAR_OFF)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self.title_label = QLabel("ODE Solver")
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(_ALIGN_CENTER)
        main_layout.addWidget(self.title_label)

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
        self._apply_theme_styles()

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
        layout.addWidget(self.preset_description, 1, 0, 1, 2)

        return group

    def _theme_color(self, key: str, fallback: str) -> str:
        manager = self.get_theme_manager()
        return str(manager.get_current_colors().get(key, fallback))

    def _apply_theme_styles(self) -> None:
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(
                f"color: {self._theme_color('accent', 'blue')};"
            )
        if hasattr(self, "preset_description"):
            self.preset_description.setStyleSheet(
                f"color: {self._theme_color('text_secondary', 'gray')}; "
                "font-style: italic;"
            )
        if hasattr(self, "results_text"):
            self._apply_results_style()

    def _on_theme_changed(self, theme_name: str) -> None:
        """Re-apply widget-local styles after the theme manager updates the window."""
        self._apply_theme_styles()

    def _apply_results_style(self) -> None:
        color_key_by_status = {
            "default": "text",
            "success": "success",
            "warning": "warning",
            "error": "error",
        }
        fallback_by_status = {
            "default": "black",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }
        color_key = color_key_by_status.get(self._results_status, "text")
        fallback = fallback_by_status.get(self._results_status, "black")
        self.results_text.setStyleSheet(
            f"color: {self._theme_color(color_key, fallback)};"
        )

    def _set_results_status(self, status: str) -> None:
        self._results_status = status
        self._apply_results_style()

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
        """Handle preset selection change.

        Preconditions:
            preset_name must be a non-empty str.

        Raises:
            TypeError: if preset_name is not a str.
            ValueError: if preset_name is an empty string.
        """
        if not isinstance(preset_name, str):
            raise TypeError(
                f"preset_name must be str, got {type(preset_name).__name__}"
            )
        if not preset_name:
            raise ValueError("preset_name must not be empty")
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
        """Parse colon-separated key-value pairs from text.

        Preconditions:
            text must be a str.

        Raises:
            TypeError: if text is not a str.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        result = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    # Default solver timeout in seconds (30 s is generous for typical ODEs)
    _SOLVER_TIMEOUT_S: float = 30.0

    def _solve(self) -> None:
        """Solve the ODE system with a timeout guard.

        Wraps the scipy integration call with ``with_timeout`` so that
        pathological or stiff systems cannot hang the GUI indefinitely.
        Raises ``SolverTimeoutError`` if the computation exceeds
        ``_SOLVER_TIMEOUT_S`` seconds.

        Invalid user input (a bad parameter expression, a missing initial
        condition, a non-monotonic time span, …) is reported in the results
        pane instead of propagating out of this slot. An unhandled exception in
        a Qt slot aborts the whole application under PyQt6, so a typo must never
        crash the app (issue #3321). While the (synchronous) solve runs, the
        Solve button is disabled and a wait cursor is shown so the user has
        feedback during the up-to-``_SOLVER_TIMEOUT_S`` blocking call.
        """
        self.solve_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._run_solve()
        finally:
            QApplication.restoreOverrideCursor()
            self.solve_btn.setEnabled(True)

    def _run_solve(self) -> None:
        """Execute the parse/solve/render pipeline (see :meth:`_solve`)."""
        try:
            from sidekick.process_calculators.ode_solver import (
                ODESolver,
            )

            # Parse inputs
            derivatives = self._parse_dict_input(self.derivatives_edit.toPlainText())
            parameters_str = self._parse_dict_input(self.parameters_edit.toPlainText())
            initial_str = self._parse_dict_input(self.initial_edit.toPlainText())

            require(bool(derivatives), "No derivatives defined")

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
            require(t_end > t_start, "t_end must be greater than t_start", t_end)
            require(num_points >= 2, "Need at least 2 points", num_points)
            t_eval = np.linspace(t_start, t_end, num_points)

            # Solve — guarded by timeout to prevent unbounded hangs
            solver = ODESolver(derivatives, parameters)
            solution = with_timeout(
                self._SOLVER_TIMEOUT_S,
                solver.solve,
                (t_start, t_end),
                y0,
                t_eval=t_eval,
            )

            # Format results
            results = []
            results.append("ODE Solution Results")
            results.append("=" * 50)

            results.append("\nSystem Definition:")
            results.extend(
                [f"  d{var}/dt = {expr}" for (var, expr) in derivatives.items()]
            )

            results.append("\nParameters:")
            results.extend(
                [f"  {name} = {value}" for (name, value) in parameters.items()]
            )

            results.append("\nInitial Conditions:")
            results.extend(
                [
                    f"  {var}(0) = {val}"
                    for (var, val) in zip(derivatives.keys(), y0, strict=True)
                ]
            )

            results.append(f"\nTime Range: [{t_start}, {t_end}]")
            results.append(f"Solution Points: {num_points}")

            results.append("\nFinal Values:")
            results.extend(
                [
                    f"  {var}({t_end}) = {solution.y[idx][-1]:.6f}"
                    for (idx, var) in enumerate(derivatives.keys())
                ]
            )

            results.append("\nSolution Summary:")
            results.extend(
                [
                    f"  {var}: min={np.min(solution.y[idx]):.4f}, max={np.max(solution.y[idx]):.4f}"
                    for (idx, var) in enumerate(derivatives.keys())
                ]
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
            self._set_results_status("success")

        except SolverTimeoutError as e:
            _log.warning("Solver timed out: %s", e)
            self.results_text.setPlainText(
                f"Timeout: {e}\n\nTry reducing the time span or simplifying the ODE system."
            )
            self._set_results_status("warning")
        except (ValueError, TypeError) as e:
            # Input-validation failures: bad parameter/initial-condition text,
            # a missing initial condition, or a contract precondition
            # (ContractViolationError subclasses ValueError). Report rather than
            # crash the application (issue #3321).
            _log.info("Invalid ODE input: %s", e)
            self.results_text.setPlainText(
                f"Invalid input: {e}\n\n"
                "Check that every parameter and initial condition is a number "
                "and that an initial condition is provided for each variable."
            )
            self._set_results_status("error")
        except ImportError as e:
            self.results_text.setPlainText(f"Error: {e}")
            self._set_results_status("error")


def main() -> int:
    """Run the ODE Solver application."""
    app = QApplication(sys.argv)
    window = ODESolverWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
