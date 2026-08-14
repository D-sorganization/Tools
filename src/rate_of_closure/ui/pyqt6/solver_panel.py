"""The Solver panel: goal-driven optimization UI (epic #4103, #4109/#4110).

Lets the user pick launch-monitor goal targets (any subset, weighted),
partition the delivery / swing variables into optimizer-controlled
(with bounds) vs fixed, and run the ``swing_sim.solver`` multi-start
driver on a worker thread (:class:`SolverWorker`) with live progress,
cooperative cancellation, and a results view: achieved-vs-goal table
with per-goal errors, residual norm, convergence flag, expandable
per-start diagnostics, and an Apply button that hands the solved
variables back to the Simulation tab so the optimized swing/impact
shows up in the 3D scene immediately.

Validation errors from the solver's DbC layer (unknown names, bad
bounds, empty goal/free set) surface as friendly messages in the status
line — never tracebacks.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.solver_rows import GoalRow, VariableRow
from rate_of_closure.ui.pyqt6.solver_specs import GOAL_SPECS, VARIABLE_SPECS
from rate_of_closure.ui.pyqt6.solver_worker import SolverWorker
from rate_of_closure.ui.pyqt6.target_panel import TargetPanel
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver.goals import (
    SWING_DERIVED_VARIABLES,
    ImpactGoal,
    VariablePartition,
)
from shared.python.swing_sim.solver.solve import ProgressReport, SolverResult
from shared.python.swing_sim.solver.tuning import DEFAULT_N_STARTS

logger = logging.getLogger(__name__)

__all__ = ["SolverPanel"]

_SWING_MODE_GUIDANCE = (
    "Suggested range: off to optimize the delivery numbers directly; on "
    "to optimize double-pendulum swing variables (plane tilts, impact "
    "timing, joint damping) with clubhead speed/path/attack angle "
    "derived from the swing. Source: shared swing_sim solver "
    "documentation (movement_optimizer scaffolding)."
)
_STARTS_GUIDANCE = (
    "Suggested range: 1-12 multi-start seeds (6 default); start 0 is "
    "the bounds midpoint, the rest a Latin-hypercube sample. More "
    "starts are more robust to local minima but slower. Source: shared "
    "swing_sim solver tuning documentation."
)


class SolverPanel(QWidget):
    """Goal editor + variable partition + run/cancel + results + apply."""

    #: Emitted with (SolverResult, use_swing_source) when Apply is clicked.
    applyRequested = pyqtSignal(object, bool)  # noqa: N815 — Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: SolverWorker | None = None
        self._result: SolverResult | None = None
        self._goal_rows: dict[str, GoalRow] = {}
        self._var_rows: dict[str, VariableRow] = {}

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_goal_box())
        # Target region (#4125 H7b): region goal + 'Optimize to Target'.
        self._target_panel = TargetPanel()
        self._target_panel.optimizeRequested.connect(self._on_run_target)
        left_layout.addWidget(self._target_panel)
        left_layout.addWidget(self._build_variable_box())
        left_layout.addWidget(self._build_run_box())
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(380)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_results_column())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Sensible demo defaults: target ball speed, free clubhead speed.
        self._goal_rows["ball_speed_mph"].enabled.setChecked(True)
        self._var_rows["clubhead_speed_mps"].optimize.setChecked(True)
        self._sync_mode_rows()

    # ── construction ────────────────────────────────────────────────
    def _build_goal_box(self) -> QGroupBox:
        box = QGroupBox("Goals (Check the Targets to Hit)")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)
        for spec in GOAL_SPECS:
            row = GoalRow(spec)
            self._goal_rows[spec.name] = row
            layout.addWidget(row)
        return box

    def _build_variable_box(self) -> QGroupBox:
        box = QGroupBox("Variables (Optimize Within Bounds, or Fix)")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self._swing_check = QCheckBox("Optimize Swing Variables (Double Pendulum)")
        self._swing_check.setToolTip(_SWING_MODE_GUIDANCE)
        self._swing_check.toggled.connect(self._sync_mode_rows)
        layout.addWidget(self._swing_check)
        for spec in VARIABLE_SPECS:
            row = VariableRow(spec)
            self._var_rows[spec.name] = row
            layout.addWidget(row)
        return box

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Starts"))
        self._starts_spin = QSpinBox()
        self._starts_spin.setRange(1, 64)
        self._starts_spin.setValue(DEFAULT_N_STARTS)
        self._starts_spin.setToolTip(_STARTS_GUIDANCE)
        self._starts_spin.setMinimumWidth(64)  # readable at small windows
        row.addWidget(self._starts_spin)
        self._run_button = QPushButton("Run Solver")
        self._run_button.setToolTip(
            "Solve for the free-variable values that best achieve the "
            "checked goals (multi-start least squares on a worker thread)."
        )
        self._run_button.clicked.connect(self._on_run)
        row.addWidget(self._run_button, stretch=1)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(
            "Cooperatively cancel the running solve; in-flight starts "
            "stop at their next residual evaluation."
        )
        self._cancel_button.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_button)
        layout.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        return box

    def _build_results_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)

        results_box = QGroupBox("Achieved vs Goal")
        results_layout = QVBoxLayout(results_box)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Quantity", "Target", "Achieved", "Error"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self._table)
        self._summary = QLabel("No solution yet.")
        self._summary.setWordWrap(True)
        results_layout.addWidget(self._summary)
        layout.addWidget(results_box, stretch=2)

        starts_box = QGroupBox("Per-Start Diagnostics")
        starts_layout = QVBoxLayout(starts_box)
        self._starts_tree = QTreeWidget()
        self._starts_tree.setHeaderLabels(["Start", "Cost", "Evals", "Status"])
        starts_layout.addWidget(self._starts_tree)
        layout.addWidget(starts_box, stretch=1)

        self._apply_button = QPushButton("Apply to Simulation")
        self._apply_button.setEnabled(False)
        self._apply_button.setToolTip(
            "Load the solved variables into the Simulation tab (scenario "
            "speed and impact offsets; swing-plane tilts and impact "
            "timing in swing mode) and rerun so the optimized swing and "
            "impact appear in the 3D scene."
        )
        self._apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_button)
        return column

    # ── editor -> solver inputs ─────────────────────────────────────
    def use_swing_source(self) -> bool:
        """Whether swing-source mode is selected."""
        return self._swing_check.isChecked()

    def target_panel(self) -> TargetPanel:
        """The target-region editor (#4125 H7b) — wiring/test seam."""
        return self._target_panel

    def _sync_mode_rows(self, *_args: object) -> None:
        """Show only the variables valid for the selected mode."""
        swing = self.use_swing_source()
        for name, row in self._var_rows.items():
            if row.spec.swing_only:
                row.setVisible(swing)
            elif name in SWING_DERIVED_VARIABLES:
                row.setVisible(not swing)

    def build_goal(self, include_target: bool = False) -> ImpactGoal:
        """The ImpactGoal described by the checked rows (DbC-validated).

        With ``include_target`` the target panel's region joins the goal
        additively (#4125 H7b) — any checked quantity goals still apply.
        """
        # ``Any`` values: ImpactGoal.of mixes named parameters with
        # ``**targets``, and a ``**`` unpack is checked against every
        # parameter it could fill — each value is a (target, weight) pair.
        targets: dict[str, Any] = {
            name: (row.target.value(), row.weight.value())
            for name, row in self._goal_rows.items()
            if row.enabled.isChecked()
        }
        if include_target:
            return ImpactGoal.of(
                target_region=self._target_panel.region(),
                target_region_weight=self._target_panel.weight(),
                **targets,
            )
        return ImpactGoal.of(**targets)

    def build_partition(self) -> VariablePartition:
        """The VariablePartition described by the variable rows."""
        free: dict[str, tuple[float, float]] = {}
        fixed: dict[str, float] = {}
        for name, row in self._var_rows.items():
            if row.isHidden():
                continue
            if row.optimize.isChecked():
                free[name] = (row.low.value(), row.high.value())
            else:
                fixed[name] = row.fixed_value.value()
        return VariablePartition(
            free=free, fixed=fixed, use_swing_source=self.use_swing_source()
        )

    # ── run / cancel ────────────────────────────────────────────────
    def _on_run(self) -> None:
        self._run(include_target=False)

    def _on_run_target(self) -> None:
        """'Optimize to Target' (#4125 H7b): region goal + checked goals."""
        self._run(include_target=True)

    def _run(self, include_target: bool) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            goal = self.build_goal(include_target)
            partition = self.build_partition()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot solve: {exc}")
            return
        self._result = None
        self._apply_button.setEnabled(False)
        self._run_button.setEnabled(False)
        self._target_panel.set_running(True)
        self._cancel_button.setEnabled(True)
        worker = SolverWorker(goal, partition, self._starts_spin.value())
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(self._on_succeeded)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._progress.setRange(0, worker.max_evaluations)
        self._progress.setValue(0)
        self._status.setText("Solving…")
        worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    def stop(self) -> None:
        """Cancel and join any running worker (window close and tests)."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(10_000)

    # ── worker callbacks (GUI thread) ───────────────────────────────
    def _on_progress(self, report: ProgressReport) -> None:
        self._progress.setValue(min(report.iteration, self._progress.maximum()))
        stalled = f" — {report.stall_reason}" if report.is_stalled else ""
        self._status.setText(
            f"Evaluations {report.iteration}, best cost "
            f"{report.best_cost:.3g}, {report.elapsed_s:.1f} s{stalled}"
        )

    def _on_succeeded(self, result: SolverResult) -> None:
        self._result = result
        self._populate_results(result)
        self._apply_button.setEnabled(True)
        flag = "converged" if result.converged else "did NOT converge"
        self._status.setText(
            f"Done: best start {flag}, residual norm "
            f"{result.residual_norm:.3g}, {result.n_evals} evaluations in "
            f"{result.elapsed_s:.1f} s."
        )

    def _on_cancelled(self) -> None:
        self._status.setText("Cancelled before any start completed.")

    def _on_failed(self, message: str) -> None:
        self._status.setText(f"Solver failed: {message}")

    def _on_finished(self) -> None:
        self._run_button.setEnabled(True)
        self._target_panel.set_running(False)
        self._cancel_button.setEnabled(False)
        self._progress.setValue(self._progress.maximum())

    # ── results view ────────────────────────────────────────────────
    def _populate_results(self, result: SolverResult) -> None:
        goals = list(result.per_goal_errors)
        labels = {spec.name: (spec.label, spec.unit) for spec in GOAL_SPECS}
        # Region "error" row (#4125 H7b): signed distance, target <= 0.
        labels["target_region_m"] = ("Target Region (signed dist)", " m")
        rows = self._goal_rows
        self._table.setRowCount(len(goals))
        for i, name in enumerate(goals):
            label, unit = labels[name]
            target = 0.0 if name == "target_region_m" else rows[name].target.value()
            achieved = result.achieved[
                "target_distance_m" if name == "target_region_m" else name
            ]
            error = result.per_goal_errors[name]
            for col, text in enumerate(
                (
                    label,
                    f"{target:+.2f}{unit}",
                    f"{achieved:+.2f}{unit}",
                    f"{error:+.3f}{unit}",
                )
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)
        var_text = ", ".join(
            f"{name} = {result.variables[name]:.3f}" for name in result.free_names
        )
        self._summary.setText(
            f"Solved variables: {var_text or '(none)'}. Residual norm "
            f"{result.residual_norm:.3g}; "
            f"{'converged' if result.converged else 'not converged'}."
        )
        self._starts_tree.clear()
        for start in result.starts:
            status = (
                "cancelled"
                if start.cancelled
                else ("converged" if start.converged else "stopped")
            )
            node = QTreeWidgetItem(
                [
                    str(start.seed),
                    f"{start.cost:.4g}",
                    str(start.n_evals),
                    status,
                ]
            )
            node.addChild(QTreeWidgetItem([f"message: {start.message}", "", "", ""]))
            if start.x is not None:
                solution = ", ".join(f"{v:.4g}" for v in start.x.tolist())
                node.addChild(QTreeWidgetItem([f"x: [{solution}]", "", "", ""]))
            self._starts_tree.addTopLevelItem(node)

    # ── apply ───────────────────────────────────────────────────────
    def result(self) -> SolverResult | None:
        """The most recent successful SolverResult, if any."""
        return self._result

    def _on_apply(self) -> None:
        if self._result is not None:
            self.applyRequested.emit(self._result, self.use_swing_source())
