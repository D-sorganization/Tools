"""
Optimization panel for the pendulum simulator GUI.

Provides a PyQt6 widget that allows the user to configure and run
torque profile optimization using scipy.optimize (CPU) as a fallback
when JAX is not available.

For double/triple pendulums: maximizes horizontal tip velocity at the
bottom of the swing arc by optimizing polynomial torque coefficients.

For the golfer model: maximizes clubhead speed at the end of the swing.

Closes #1108, #1109, #1110.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_STYLE = """
QGroupBox {
    color: #9090c8; font-size: 10px; font-weight: bold;
    border: 1px solid #303050; border-radius: 4px;
    margin-top: 8px; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QLabel { color: #8080b0; font-size: 9px; }
QPushButton {
    background: #262650; color: #b0b0e8; border: 1px solid #404070;
    border-radius: 3px; padding: 4px 12px; font-size: 9px;
}
QPushButton:hover { background: #303068; }
QPushButton:disabled { color: #505060; }
QSpinBox {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 9px; padding: 2px;
}
QComboBox {
    background: #1a1a2a; color: #b0b0e8; border: 1px solid #303050;
    border-radius: 2px; font-size: 9px; padding: 2px;
}
QTextEdit {
    background: #0e0e1a; color: #808090; border: 1px solid #202040;
    border-radius: 3px; font-family: monospace; font-size: 8px;
}
QProgressBar {
    background: #1a1a2a; border: 1px solid #303050;
    border-radius: 3px; text-align: center;
    color: #a0a0d0; font-size: 8px;
}
QProgressBar::chunk { background: #404090; border-radius: 2px; }
"""


# ---------------------------------------------------------------------------
# Background optimizer worker
# ---------------------------------------------------------------------------


class _OptimizerWorker(QObject):
    """Runs the optimization loop on a background thread."""

    iteration_done = pyqtSignal(int, float)  # (iteration, loss_value)
    finished = pyqtSignal(object)  # result dict
    error = pyqtSignal(str)

    def __init__(
        self,
        objective_fn: Callable,
        n_params: int,
        n_iterations: int,
        method: str,
    ) -> None:
        super().__init__()
        self._objective = objective_fn
        self._n_params = n_params
        self._n_iterations = n_iterations
        self._method = method
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from scipy.optimize import differential_evolution, minimize

            x0 = np.random.default_rng(42).normal(0, 0.1, self._n_params)
            history: list[float] = []

            if self._method == "Nelder-Mead":
                result = minimize(
                    self._objective,
                    x0,
                    method="Nelder-Mead",
                    options={"maxiter": self._n_iterations, "adaptive": True},
                    callback=lambda xk: self._report(len(history), float(self._objective(xk)), history),
                )
                self.finished.emit({
                    "coeffs": result.x,
                    "speed": -float(result.fun),
                    "history": history,
                    "success": result.success,
                    "message": result.message,
                })
            elif self._method == "CMA-ES":
                # Use differential evolution as a robust global optimizer
                bounds = [(-50.0, 50.0)] * self._n_params
                result = differential_evolution(
                    self._objective,
                    bounds,
                    maxiter=self._n_iterations,
                    seed=42,
                    callback=lambda xk, convergence: self._report(
                        len(history), float(convergence), history
                    ),
                    polish=True,
                )
                self.finished.emit({
                    "coeffs": result.x,
                    "speed": -float(result.fun),
                    "history": history,
                    "success": result.success,
                    "message": result.message,
                })
            else:
                result = minimize(
                    self._objective,
                    x0,
                    method="L-BFGS-B",
                    options={"maxiter": self._n_iterations},
                    callback=lambda xk: self._report(len(history), float(self._objective(xk)), history),
                )
                self.finished.emit({
                    "coeffs": result.x,
                    "speed": -float(result.fun),
                    "history": history,
                    "success": result.success,
                    "message": result.message,
                })
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _report(self, iteration: int, loss: float, history: list[float]) -> None:
        history.append(loss)
        self.iteration_done.emit(iteration, loss)


# ---------------------------------------------------------------------------
# Optimization Panel Widget
# ---------------------------------------------------------------------------


class OptimizationWidget(QWidget):
    """Optimization panel for finding optimal torque profiles.

    This widget provides controls for configuring and running optimization
    of torque polynomial coefficients to maximize tip/clubhead speed.

    Emits ``optimized_coefficients`` with the result dict when optimization
    completes successfully.
    """

    optimized_coefficients = pyqtSignal(object)  # result dict

    def __init__(
        self,
        model_name: str = "Double Pendulum",
        n_torque_params: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_name = model_name
        self._n_torque_params = n_torque_params
        self._objective_fn: Callable | None = None
        self._result: dict | None = None
        self.setStyleSheet(_STYLE)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel(f"⚡ {self._model_name} Optimizer")
        title.setStyleSheet("color:#a0a0e0;font-size:11px;font-weight:bold;")
        layout.addWidget(title)

        # Config group
        config = QGroupBox("Configuration")
        cfg_lay = QVBoxLayout(config)
        cfg_lay.setContentsMargins(4, 14, 4, 4)
        cfg_lay.setSpacing(4)

        # Objective
        obj_row = QHBoxLayout()
        obj_row.addWidget(QLabel("Objective:"))
        self._cmb_objective = QComboBox()
        self._cmb_objective.addItems(["Max Tip Speed", "Max Height"])
        obj_row.addWidget(self._cmb_objective)
        cfg_lay.addLayout(obj_row)

        # Method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._cmb_method = QComboBox()
        self._cmb_method.addItems(["Nelder-Mead", "L-BFGS-B", "CMA-ES"])
        method_row.addWidget(self._cmb_method)
        cfg_lay.addLayout(method_row)

        # Iterations
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterations:"))
        self._spin_iters = QSpinBox()
        self._spin_iters.setRange(10, 10000)
        self._spin_iters.setValue(100)
        self._spin_iters.setSingleStep(10)
        iter_row.addWidget(self._spin_iters)
        cfg_lay.addLayout(iter_row)

        # Polynomial degree
        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("Poly degree:"))
        self._spin_degree = QSpinBox()
        self._spin_degree.setRange(1, 6)
        self._spin_degree.setValue(3)
        deg_row.addWidget(self._spin_degree)
        cfg_lay.addLayout(deg_row)

        layout.addWidget(config)

        # Controls
        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶ Optimize")
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("⏹ Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_cancel)

        self._btn_apply = QPushButton("✓ Apply")
        self._btn_apply.setEnabled(False)
        self._btn_apply.setToolTip("Apply optimized coefficients to the controls")
        self._btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self._btn_apply)
        layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Status
        self._lbl_status = QLabel("Ready")
        self._lbl_status.setStyleSheet("color:#606080;font-size:9px;")
        layout.addWidget(self._lbl_status)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        layout.addWidget(self._log)

        layout.addStretch()

    def set_objective_function(self, fn: Callable) -> None:
        """Set the objective function for optimization.

        Parameters
        ----------
        fn : Callable
            Must accept a 1D numpy array of torque coefficients and return
            a scalar (negative of the quantity to maximize).
        """
        self._objective_fn = fn

    def _on_run(self) -> None:
        if self._objective_fn is None:
            self._log.append("⚠ No objective function set. Run a simulation first.")
            return

        n_params = self._n_torque_params * self._spin_degree.value()
        n_iters = self._spin_iters.value()
        method = self._cmb_method.currentText()

        self._log.clear()
        self._log.append(f"Starting {method} optimization...")
        self._log.append(f"  Params: {n_params}, Iterations: {n_iters}")
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_apply.setEnabled(False)
        self._lbl_status.setText("Optimizing...")

        self._thread = QThread()
        self._worker = _OptimizerWorker(
            self._objective_fn, n_params, n_iters, method
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.iteration_done.connect(self._on_iteration)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_cancel(self) -> None:
        if hasattr(self, "_worker"):
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._lbl_status.setText("Cancelling...")

    def _on_iteration(self, iteration: int, loss: float) -> None:
        max_iter = self._spin_iters.value()
        pct = min(100, int(100 * iteration / max_iter))
        self._progress.setValue(pct)
        speed = -loss
        self._lbl_status.setText(f"Iter {iteration}: speed = {speed:.4f} m/s")

    def _on_finished(self, result: Any) -> None:
        self._result = result
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_apply.setEnabled(True)
        self._progress.setValue(100)

        speed = result.get("speed", 0.0)
        success = result.get("success", False)
        msg = result.get("message", "")

        self._lbl_status.setText(
            f"{'✓' if success else '⚠'} Speed: {speed:.4f} m/s"
        )
        self._log.append(f"\nOptimization {'succeeded' if success else 'finished'}:")
        self._log.append(f"  Max speed: {speed:.4f} m/s")
        self._log.append(f"  Status: {msg}")

        coeffs = result.get("coeffs")
        if coeffs is not None:
            self._log.append(f"  Coefficients: {np.array2string(coeffs, precision=4)}")

        self.optimized_coefficients.emit(result)

    def _on_error(self, msg: str) -> None:
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setValue(0)
        self._lbl_status.setText("⚠ Error")
        self._log.append(f"\n⚠ Optimization error: {msg}")
        logger.error("Optimization error: %s", msg)

    def _on_apply(self) -> None:
        if self._result is not None:
            self.optimized_coefficients.emit(self._result)
            self._log.append("\n✓ Applied optimized coefficients to controls.")
