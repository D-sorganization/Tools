"""Ranked alternatives and observation diagnostics for capability runs."""

from __future__ import annotations

from functools import partial

from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
from rate_of_closure.variation.scalar_ensemble_contract import (
    ScalarEnsembleDataset,
    ScalarScatterData,
)
from rate_of_closure.variation.scalar_ensemble_io import non_complete_reason_summary
from shared.python.swing_sim.flight.capability_result import (
    OptimizationAlternative,
    OptimizationResult,
)

_PAGE_SIZE = 25
_COLORS = {"complete": "#2f8bd6", "no_impact": "#eb9f3c", "failed": "#d35f5f"}


def _parameter_units(dataset: ScalarEnsembleDataset) -> dict[str, str]:
    return {
        variable.key.removeprefix("nominal."): variable.unit
        for variable in dataset.variables
        if variable.key.startswith("nominal.")
    }


def _alternative_values(
    item: OptimizationAlternative, units: dict[str, str]
) -> tuple[str, ...]:
    recommendation = " · ".join(
        f"{key}={value:.5g} {units[key]}" for key, value in item.parameters
    )
    outcomes = (
        f"{item.successful_count}/{item.sample_count} complete · "
        f"{item.no_impact_count} no impact · {item.failed_count} failed"
    )
    evidence = (
        f"{100 * item.confidence:.1f}% · "
        f"{'extrapolated' if item.extrapolated else 'within envelope'} · "
        f"{', '.join(item.limiting_constraints) or 'no limits'}"
    )
    return (
        f"{item.rank}. {item.club_id}",
        recommendation,
        f"{item.score:.5g}",
        f"{item.mean_carry_m:.2f} m",
        f"{item.expected_miss_m:.2f} m",
        f"{item.dispersion_rms_m:.2f} m",
        f"{100 * item.target_hold_probability:.1f}%",
        f"{item.cvar_miss_m:.2f} m",
        f"{item.downside_carry_m:.2f} m",
        outcomes,
        f"{100 * item.failure_fraction:.1f}%",
        evidence,
        "efficient" if item.pareto_efficient else "dominated",
    )


def _draw_scatter(figure: Figure, scatter: ScalarScatterData) -> None:
    figure.clear()
    axes = figure.add_subplot(111)
    for cohort, color in _COLORS.items():
        points = tuple(point for point in scatter.points if point.cohort == cohort)
        if points:
            axes.scatter(
                [point.x for point in points],
                [point.y for point in points],
                s=18,
                alpha=0.7,
                label=cohort.replace("_", " ").title(),
                color=color,
            )
    axes.set_xlabel(f"{scatter.x_variable.label} [{scatter.x_variable.unit}]")
    axes.set_ylabel(f"{scatter.y_variable.label} [{scatter.y_variable.unit}]")
    axes.set_title("Capability Observation Scatter")
    axes.grid(alpha=0.25)
    if axes.collections:
        axes.legend(loc="best", fontsize=8)


class CapabilityResults(QWidget):
    """Bounded presentation of ranked alternatives and every raw observation."""

    def __init__(self) -> None:
        super().__init__()
        self._dataset: ScalarEnsembleDataset | None = None
        self._page = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.summary = QLabel("No current optimization result.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.alternatives = QTableWidget(0, 13)
        self.alternatives.setHorizontalHeaderLabels(
            [
                "Rank / Club",
                "Recommendation",
                "Score",
                "Carry",
                "Mean miss",
                "Dispersion",
                "Target hold",
                "Miss CVaR",
                "Downside carry",
                "Outcomes",
                "Failure rate",
                "Evidence",
                "Pareto",
            ]
        )
        layout.addWidget(self.alternatives)
        layout.addLayout(self._build_axis_controls())
        self.availability = QLabel("No current ensemble result.")
        layout.addWidget(self.availability)
        self.plot = PlotCanvasPane("Capability Observation Scatter")
        layout.addWidget(self.plot, stretch=1)
        self.raw_rows = QTableWidget(0, 4)
        self.raw_rows.setHorizontalHeaderLabels(
            ["Row", "Cohort", "Series", "Available scalars"]
        )
        layout.addWidget(self.raw_rows)
        layout.addLayout(self._build_paging_controls())
        self.x_axis.currentIndexChanged.connect(self._redraw)
        self.y_axis.currentIndexChanged.connect(self._redraw)

    def _build_axis_controls(self) -> QHBoxLayout:
        axes = QHBoxLayout()
        self.x_axis = QComboBox()
        self.y_axis = QComboBox()
        self.x_axis.setAccessibleName("Capability scatter horizontal axis")
        self.y_axis.setAccessibleName("Capability scatter vertical axis")
        self.x_axis.setToolTip(
            "Choose any retained scalar for the scatter plot horizontal axis."
        )
        self.y_axis.setToolTip(
            "Choose any retained scalar for the scatter plot vertical axis."
        )
        axes.addWidget(QLabel("Horizontal axis"))
        axes.addWidget(self.x_axis)
        axes.addWidget(QLabel("Vertical axis"))
        axes.addWidget(self.y_axis)
        return axes

    def _build_paging_controls(self) -> QHBoxLayout:
        paging = QHBoxLayout()
        self.previous = QPushButton("Previous rows")
        self.next = QPushButton("Next rows")
        self.previous.setToolTip("Show the previous 25 retained observation rows.")
        self.next.setToolTip("Show the next 25 retained observation rows.")
        self.page_label = QLabel("Page 0 of 0")
        paging.addWidget(self.previous)
        paging.addWidget(self.page_label)
        paging.addWidget(self.next)
        self.previous.clicked.connect(lambda: self._change_page(-1))
        self.next.clicked.connect(lambda: self._change_page(1))
        return paging

    def set_output(
        self, result: OptimizationResult, dataset: ScalarEnsembleDataset
    ) -> None:
        """Replace the complete immutable result and reset derived views."""
        self._dataset = dataset
        self._page = 0
        self.summary.setText(
            f"Attempted {result.evaluations_attempted}; complete "
            f"{result.evaluations_completed}; failed {result.failed_count}; "
            f"no impact {result.no_impact_count}. Status: {result.status}."
            f"{non_complete_reason_summary(dataset)}"
        )
        self._populate_alternatives(result, dataset)
        self._populate_axes(dataset)
        self._populate_page()

    def _populate_alternatives(
        self, result: OptimizationResult, dataset: ScalarEnsembleDataset
    ) -> None:
        self.alternatives.setRowCount(len(result.alternatives))
        units = _parameter_units(dataset)
        for row, item in enumerate(result.alternatives):
            for column, value in enumerate(_alternative_values(item, units)):
                self.alternatives.setItem(row, column, QTableWidgetItem(value))
        self.alternatives.resizeColumnsToContents()

    def _populate_axes(self, dataset: ScalarEnsembleDataset) -> None:
        duplicates = {
            (variable.label, variable.unit)
            for variable in dataset.variables
            if sum(
                item.label == variable.label and item.unit == variable.unit
                for item in dataset.variables
            )
            > 1
        }
        for combo in (self.x_axis, self.y_axis):
            combo.blockSignals(True)
            combo.clear()
            for variable in dataset.variables:
                prefix = (
                    f"{variable.stage_key.title()} · "
                    if (variable.label, variable.unit) in duplicates
                    else ""
                )
                combo.addItem(
                    f"{prefix}{variable.label} [{variable.unit}]", variable.key
                )
            combo.blockSignals(False)
        self.x_axis.setCurrentIndex(
            max(0, self.x_axis.findData("perturbed.ball_speed"))
        )
        self.y_axis.setCurrentIndex(
            max(0, self.y_axis.findData("metric.carry_distance"))
        )
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        if self._dataset is None or self.x_axis.currentIndex() < 0:
            return
        scatter = self._dataset.scatter(
            str(self.x_axis.currentData()), str(self.y_axis.currentData())
        )
        self.plot.render_custom(partial(_draw_scatter, scatter=scatter))
        count = scatter.availability.overall
        self.availability.setText(
            f"Paired finite {count.paired_finite}/{count.total_rows}; "
            f"unavailable {count.unavailable}."
        )

    def _change_page(self, offset: int) -> None:
        if self._dataset is None:
            return
        pages = max(1, (len(self._dataset.rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._page = max(0, min(pages - 1, self._page + offset))
        self._populate_page()

    def _populate_page(self) -> None:
        if self._dataset is None:
            return
        rows = self._dataset.rows[
            self._page * _PAGE_SIZE : (self._page + 1) * _PAGE_SIZE
        ]
        pages = max(1, (len(self._dataset.rows) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self.raw_rows.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (
                row.row_id,
                row.cohort,
                row.series_id or "",
                str(sum(value is not None for value in row.values.values())),
            )
            for column, value in enumerate(values):
                self.raw_rows.setItem(index, column, QTableWidgetItem(value))
        self.page_label.setText(f"Page {self._page + 1} of {pages}")
        self.previous.setEnabled(self._page > 0)
        self.next.setEnabled(self._page + 1 < pages)


__all__ = ["CapabilityResults"]
