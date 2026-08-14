"""Decision-ready PyQt presentation for conditional chip forgiveness studies."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.variation_distribution_matrix import (
    DistributionMatrixView,
)
from rate_of_closure.ui.pyqt6.variation_visualizations import DatasetScatterView
from rate_of_closure.variation import (
    ChipForgivenessStudy,
    ChipStudySummary,
    ChipTrialCohort,
    forgiveness_variation_dataset,
)
from rate_of_closure.variation.plot_labels import OUTPUT_LABELS, OUTPUT_UNITS

_COHORT_LABELS = {
    ChipTrialCohort.BALL_FIRST: "Ball First",
    ChipTrialCohort.BALL_ONLY: "Ball Only",
    ChipTrialCohort.GROUND_FIRST: "Ground First",
    ChipTrialCohort.SIMULTANEOUS: "Simultaneous / Grazing",
    ChipTrialCohort.GROUND_ONLY_MISS: "Ground Only — Ball Missed",
    ChipTrialCohort.NO_CONTACT_MISS: "No Contact — Ball Missed",
    ChipTrialCohort.NUMERICAL_FAILURE: "Numerical Failure",
}


def _item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class ChipForgivenessView(QWidget):
    """Show all-trial probability, risk, convergence, and metric evidence."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary: ChipStudySummary | None = None
        layout = QVBoxLayout(self)
        title = QLabel("Conditional Chip-Shot Forgiveness")
        title.setStyleSheet("font-weight: 700; font-size: 15px;")
        layout.addWidget(title)
        self._scope = QLabel(
            "Run a swing variation study with a selected wedge to compare the "
            "declared 30-yard objective."
        )
        self._scope.setWordWrap(True)
        self._scope.setToolTip(
            "Rankings are conditional on the plan, seed, turf calibration, "
            "objective, and solver. Misses and failures remain in all denominators."
        )
        layout.addWidget(self._scope)
        layout.addWidget(self._build_decision_box())
        self._cohorts = QTableWidget(0, 4)
        self._cohorts.setHorizontalHeaderLabels(
            ["Cohort", "Count", "Probability", "95% Wilson CI"]
        )
        self._cohorts.setToolTip(
            "Mutually exclusive contact outcomes using all configured trials as "
            "the denominator, including misses and numerical failures."
        )
        layout.addWidget(self._cohorts)
        self._metrics = QTableWidget(0, 6)
        self._metrics.setHorizontalHeaderLabels(
            ["Metric", "Unit", "P5", "Median", "P95", "Support / Unavailable"]
        )
        self._metrics.setToolTip(
            "Quantiles use available physical values only; support and unavailable "
            "counts disclose censoring rather than filling missing values with zero."
        )
        layout.addWidget(self._metrics)
        self._convergence = QTableWidget(0, 3)
        self._convergence.setHorizontalHeaderLabels(
            ["Trials", "Running Mean Loss", "Standard Error"]
        )
        self._convergence.setToolTip(
            "Prefix checkpoints disclose whether the all-trial expected-loss "
            "estimate is stabilizing at the declared sample count."
        )
        layout.addWidget(self._convergence)
        self._scatter = DatasetScatterView()
        self._matrix = DistributionMatrixView()
        self._scatter.selectionChanged.connect(self._matrix.set_selected_trial)
        self._matrix.selectionChanged.connect(self._scatter.set_selected_trial)
        layout.addWidget(QLabel("Forgiveness Metric Scatter"))
        layout.addWidget(self._scatter)
        layout.addWidget(QLabel("Forgiveness Scatter Matrix / Marginals"))
        layout.addWidget(self._matrix)

    def _build_decision_box(self) -> QGroupBox:
        box = QGroupBox("Declared Decision Metrics")
        grid = QGridLayout(box)
        self._expected_loss = QLabel("—")
        self._cvar = QLabel("—")
        self._clean = QLabel("—")
        self._violations = QLabel("—")
        entries = (
            ("Expected Loss (95% bootstrap CI)", self._expected_loss),
            ("Worst-Tail CVaR", self._cvar),
            ("Clean-Contact Probability", self._clean),
            ("Constraint-Violation Rate", self._violations),
        )
        for index, (name, value) in enumerate(entries):
            grid.addWidget(QLabel(name), index, 0)
            value.setStyleSheet("font-family: monospace; font-weight: 600;")
            grid.addWidget(value, index, 1)
        return box

    def summary(self) -> ChipStudySummary | None:
        """Return the currently displayed immutable summary."""
        return self._summary

    def scope_text(self) -> str:
        """Return the visible ranking qualification for tests and accessibility."""
        return self._scope.text()

    def set_summary(self, summary: ChipStudySummary) -> None:
        """Populate all decision and advanced-metric surfaces."""
        if not isinstance(summary, ChipStudySummary):
            raise TypeError("summary must be ChipStudySummary")
        self._summary = summary
        self._scope.setText(
            f"{summary.ranking_scope} {summary.metadata.limitations} "
            f"Inference: {summary.metadata.sampling_design}; "
            f"{summary.metadata.inference_method_id}."
        )
        self._expected_loss.setText(
            f"{summary.expected_loss:.3f} "
            f"[{summary.expected_loss_ci_low:.3f}, "
            f"{summary.expected_loss_ci_high:.3f}]"
        )
        tail_percent = 100.0 * summary.cvar_tail_fraction
        self._cvar.setText(f"{summary.cvar_loss:.3f} (worst {tail_percent:.0f}%)")
        self._clean.setText(f"{100.0 * summary.clean_contact_probability:.1f}%")
        self._violations.setText(f"{100.0 * summary.constraint_violation_rate:.1f}%")
        self._populate_cohorts(summary)
        self._populate_metrics(summary)
        self._populate_convergence(summary)

    def set_study(self, study: ChipForgivenessStudy) -> None:
        """Populate summary and shared linked scatter/marginal visualizations."""
        if not isinstance(study, ChipForgivenessStudy):
            raise TypeError("study must be ChipForgivenessStudy")
        self.set_summary(study.summary)
        dataset = forgiveness_variation_dataset(study)
        self._scatter.set_variation_dataset(dataset)
        self._matrix.set_variation_dataset(dataset)

    def _populate_cohorts(self, summary: ChipStudySummary) -> None:
        self._cohorts.setRowCount(len(ChipTrialCohort))
        for row, cohort in enumerate(ChipTrialCohort):
            estimate = summary.cohorts[cohort]
            values = (
                _COHORT_LABELS[cohort],
                str(estimate.count),
                f"{100.0 * estimate.probability:.2f}%",
                f"{100.0 * estimate.ci_low:.2f}% — {100.0 * estimate.ci_high:.2f}%",
            )
            for column, value in enumerate(values):
                self._cohorts.setItem(row, column, _item(value))
        self._cohorts.resizeColumnsToContents()

    def _populate_metrics(self, summary: ChipStudySummary) -> None:
        distributions = summary.metric_distributions
        self._metrics.setRowCount(len(distributions))
        for row, distribution in enumerate(distributions):
            values = (
                OUTPUT_LABELS.get(distribution.name, distribution.name),
                OUTPUT_UNITS.get(distribution.name, "—"),
                self._quantile(distribution.p05),
                self._quantile(distribution.p50),
                self._quantile(distribution.p95),
                f"{distribution.support_count} / {distribution.unavailable_count}",
            )
            for column, value in enumerate(values):
                self._metrics.setItem(row, column, _item(value))
        self._metrics.resizeColumnsToContents()

    def _populate_convergence(self, summary: ChipStudySummary) -> None:
        self._convergence.setRowCount(len(summary.convergence))
        for row, point in enumerate(summary.convergence):
            values = (
                str(point.sample_count),
                f"{point.mean_loss:.5g}",
                (
                    "Unavailable"
                    if point.standard_error is None
                    else f"{point.standard_error:.5g}"
                ),
            )
            for column, value in enumerate(values):
                self._convergence.setItem(row, column, _item(value))
        self._convergence.resizeColumnsToContents()

    @staticmethod
    def _quantile(value: float | None) -> str:
        """Format an available quantile without inventing a zero."""
        return "Unavailable" if value is None else f"{value:.5g}"


__all__ = ["ChipForgivenessView"]
