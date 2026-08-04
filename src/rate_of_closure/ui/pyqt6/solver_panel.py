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

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
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

from rate_of_closure.ui.pyqt6.solver_specs import (
    GOAL_SPECS,
    VARIABLE_SPECS,
    GoalSpec,
    VariableSpec,
)
from rate_of_closure.ui.pyqt6.solver_worker import SolverWorker
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


def _spin(
    lo: float, hi: float, value: float, decimals: int, suffix: str
) -> QDoubleSpinBox:
    """A no-arrow, typed QDoubleSpinBox in the app's input style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setSuffix(suffix)
    spin.setValue(value)
    spin.setMinimumWidth(84)  # readable at small windows (#4120)
    return spin


class _GoalRow(QWidget):
    """One goal quantity: enable checkbox + target + weight entries."""

    def __init__(self, spec: GoalSpec) -> None:
        super().__init__()
        self.spec = spec
        self.enabled = QCheckBox(spec.label)
        self.enabled.setToolTip(spec.guidance)
        self.target = _spin(
            spec.spin_range[0], spec.spin_range[1], spec.default_target, 1, spec.unit
        )
        self.target.setToolTip(spec.guidance)
        self.weight = _spin(0.01, 100.0, 1.0, 2, "")
        self.weight.setToolTip(
            "Suggested range: 0.1-10 relative weight (1 default); larger "
            "weights make the optimizer trade other goals away to hit "
            "this one. Source: shared swing_sim solver tuning "
            "documentation (launch-monitor-resolution residual scales)."
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enabled, stretch=1)
        layout.addWidget(self.target)
        layout.addWidget(QLabel("w"))
        layout.addWidget(self.weight)
        for widget in (self.target, self.weight):
            widget.setEnabled(False)
        self.enabled.toggled.connect(self.target.setEnabled)
        self.enabled.toggled.connect(self.weight.setEnabled)


class _VariableRow(QWidget):
    """One variable: radio Optimize (min/max bounds) | Fix (value)."""

    def __init__(self, spec: VariableSpec) -> None:
        super().__init__()
        self.spec = spec
        lo, hi = spec.spin_range
        self.optimize = QRadioButton("Optimize")
        self.fix = QRadioButton("Fix")
        self._group = QButtonGroup(self)
        self._group.addButton(self.optimize)
        self._group.addButton(self.fix)
        # NOTE: not named "lower"/"raise_" — those are QWidget methods.
        self.low = _spin(lo, hi, spec.default_bounds[0], spec.decimals, spec.unit)
        self.high = _spin(lo, hi, spec.default_bounds[1], spec.decimals, spec.unit)
        self.fixed_value = _spin(lo, hi, spec.default_value, spec.decimals, spec.unit)
        label = QLabel(spec.label)
        for widget in (label, self.optimize, self.fix, self.low, self.high):
            widget.setToolTip(spec.guidance)
        self.fixed_value.setToolTip(spec.guidance)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label, 0, 0, 1, 4)
        layout.addWidget(self.optimize, 1, 0)
        layout.addWidget(self.low, 1, 1)
        layout.addWidget(self.high, 1, 2)
        layout.addWidget(self.fix, 1, 3)
        layout.addWidget(self.fixed_value, 1, 4)
        self.optimize.toggled.connect(self._sync_enabled)
        self.fix.setChecked(True)
        self._sync_enabled()

    def _sync_enabled(self, *_args: object) -> None:
        free = self.optimize.isChecked()
        self.low.setEnabled(free)
        self.high.setEnabled(free)
        self.fixed_value.setEnabled(not free)


class SolverPanel(QWidget):
    """Goal editor + variable partition + run/cancel + results + apply."""

    #: Emitted with (SolverResult, use_swing_source) when Apply is clicked.
    applyRequested = pyqtSignal(object, bool)  # noqa: N815 — Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: SolverWorker | None = None
        self._result: SolverResult | None = None
        self._goal_rows: dict[str, _GoalRow] = {}
        self._var_rows: dict[str, _VariableRow] = {}

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_goal_box())
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
            row = _GoalRow(spec)
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
            row = _VariableRow(spec)
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

    def _sync_mode_rows(self, *_args: object) -> None:
        """Show only the variables valid for the selected mode."""
        swing = self.use_swing_source()
        for name, row in self._var_rows.items():
            if row.spec.swing_only:
                row.setVisible(swing)
            elif name in SWING_DERIVED_VARIABLES:
                row.setVisible(not swing)

    def build_goal(self) -> ImpactGoal:
        """The ImpactGoal described by the checked rows (DbC-validated)."""
        targets = {
            name: (row.target.value(), row.weight.value())
            for name, row in self._goal_rows.items()
            if row.enabled.isChecked()
        }
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
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            goal = self.build_goal()
            partition = self.build_partition()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot solve: {exc}")
            return
        self._result = None
        self._apply_button.setEnabled(False)
        self._run_button.setEnabled(False)
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
        self._cancel_button.setEnabled(False)
        self._progress.setValue(self._progress.maximum())

    # ── results view ────────────────────────────────────────────────
    def _populate_results(self, result: SolverResult) -> None:
        goals = list(result.per_goal_errors)
        labels = {spec.name: (spec.label, spec.unit) for spec in GOAL_SPECS}
        rows = self._goal_rows
        self._table.setRowCount(len(goals))
        for i, name in enumerate(goals):
            label, unit = labels[name]
            target = rows[name].target.value()
            achieved = result.achieved[name]
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
