"""Results widgets for the Variation tab (#4120 V3).

Three read-only views over a
:class:`~shared.python.swing_sim.variation.engine.VariationDataset`:

- :class:`SummaryTable` — per-output mean/std/percentiles;
- :class:`SensitivityTable` — heat-shaded matrix (inputs x outputs) for
  the one-at-a-time result or the Spearman rank-correlation check;
- :class:`LandingCanvas` — its own small themed matplotlib scatter of
  the landing points with the 2-sigma dispersion ellipse (allowed here:
  this is the Variation tab's dedicated canvas, not the plotting suite).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas
from rate_of_closure.units import DISTANCE_UNITS, display_distance_unit
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import (
    DispersionEllipse,
    OutputStats,
    TruncationShiftNote,
    VariationDataset,
    variable_registry,
)

__all__ = ["LandingCanvas", "SensitivityTable", "SummaryTable", "short_label"]

_HEAT_LOW = (37, 66, 96)  # muted blue
_HEAT_HIGH = (235, 106, 60)  # hot orange


def short_label(registry_key: str) -> str:
    """``category.name`` -> ``"Label (last category segment)"`` for headers."""
    definition = variable_registry().get(registry_key)
    if definition is None:
        return registry_key
    segment = definition.category.rsplit(".", 1)[1]
    return f"{definition.label} ({segment})"


def _read_only(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class SummaryTable(QTableWidget):
    """Per-output dispersion statistics (successful runs only)."""

    _COLUMNS = ("Output", "Mean", "Std", "P5", "Median", "P95", "N")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self._COLUMNS), parent)
        self.setHorizontalHeaderLabels(list(self._COLUMNS))
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setToolTip(
            "Dispersion of every pipeline output over the successful runs: "
            "mean, sample standard deviation, and the 5th / 50th / 95th "
            "percentiles."
        )

    def set_stats(
        self,
        stats: tuple[OutputStats, ...],
        truncation_notes: tuple[TruncationShiftNote, ...] | None = None,
    ) -> None:
        """Populate from :func:`swing_sim.variation.summary_stats` output.

        Landing distances (carry/lateral) follow the session's distance
        display unit (#4125 H6 — yards default; the row name gains the
        unit so the numbers are unambiguous). Apex stays in metres.
        """
        self.setRowCount(len(stats))
        for i, s in enumerate(stats):
            distance = s.name in ("carry_m", "lateral_m")
            unit = display_distance_unit() if distance else ""
            factor = DISTANCE_UNITS[unit] if distance else 1.0
            name = f"{s.name} [{unit}]" if distance else s.name
            cells = (
                name,
                f"{s.mean / factor:+.2f}",
                f"{s.std / factor:.3f}",
                f"{s.p5 / factor:+.2f}",
                f"{s.p50 / factor:+.2f}",
                f"{s.p95 / factor:+.2f}",
                str(s.n),
            )
            for col, text in enumerate(cells):
                self.setItem(i, col, _read_only(text))
        if truncation_notes:
            notes_str = "\n".join(f"• {n.note}" for n in truncation_notes)
            self.setToolTip(
                "Dispersion of every pipeline output over successful runs.\n"
                f"Truncation mean-shifts detected:\n{notes_str}"
            )
        self.resizeColumnsToContents()


class SensitivityTable(QTableWidget):
    """Heat-shaded inputs-x-outputs matrix (normalized to column max)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 0, parent)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    @staticmethod
    def _heat(fraction: float) -> QColor:
        f = min(max(fraction, 0.0), 1.0)
        return QColor(
            *(
                round(lo + f * (hi - lo))
                for lo, hi in zip(_HEAT_LOW, _HEAT_HIGH, strict=True)
            )
        )

    def set_matrix(
        self,
        input_keys: tuple[str, ...],
        output_names: tuple[str, ...],
        values: np.ndarray,
        normalized: np.ndarray,
        value_format: str = "{:.3g}",
        significant: np.ndarray | None = None,
        p_values: np.ndarray | None = None,
    ) -> None:
        """Show ``values`` shaded by ``normalized`` with significance check."""
        self.setRowCount(len(input_keys))
        self.setColumnCount(len(output_names))
        self.setVerticalHeaderLabels([short_label(key) for key in input_keys])
        self.setHorizontalHeaderLabels(list(output_names))
        for i in range(len(input_keys)):
            for j in range(len(output_names)):
                value = float(values[i, j])
                frac = float(normalized[i, j])
                text = "—" if np.isnan(value) else value_format.format(value)
                item = _read_only(text)
                is_sig = True if significant is None else bool(significant[i, j])
                pval = (
                    float(p_values[i, j])
                    if p_values is not None and not np.isnan(p_values[i, j])
                    else None
                )
                if not np.isnan(frac):
                    if is_sig:
                        item.setBackground(self._heat(frac))
                        item.setForeground(QColor(255, 255, 255))
                    else:
                        # Insignificant cell: grey/suppress heat shading
                        item.setBackground(QColor(48, 54, 66))
                        item.setForeground(QColor(140, 150, 165))
                sig_note = ""
                if pval is not None:
                    status = "significant" if is_sig else "insignificant"
                    sig_note = f"; p={pval:.3g} ({status})"
                norm_str = f"{0.0 if np.isnan(frac) else frac:.2f}"
                item.setToolTip(
                    f"{short_label(input_keys[i])} → {output_names[j]}: "
                    f"{text} (column-normalized {norm_str}{sig_note})"
                )
                self.setItem(i, j, item)
        self.resizeColumnsToContents()


class LandingCanvas(LifecycleSafeFigureCanvas):
    """Top-down landing scatter (lateral vs carry) with the 2σ ellipse."""

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.0, 4.0), layout="constrained")
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)
        self._axes = figure.add_subplot(111)
        self._outcome_counts: Counter[TrialEvaluationStatus] | None = None
        self.setToolTip(
            "Landing positions of every successful run, viewed from above "
            "(x = lateral, + right of target; y = carry). The dashed "
            "ellipse is the 2-sigma dispersion fit."
        )
        self._apply_theme()
        self.clear_view()

    def set_outcomes(self, outcomes: tuple[TrialEvaluationStatus, ...]) -> None:
        """Retain typed trial counts for the next landing render."""
        self._outcome_counts = Counter(outcomes)

    def _apply_theme(self) -> None:
        """Follow the widget palette so light/dark themes both read well."""
        palette = self.palette()
        window = palette.window().color()
        text = palette.text().color()
        self.figure.set_facecolor(window.name())
        self._axes.set_facecolor(window.lighter(105).name())
        for spine in self._axes.spines.values():
            spine.set_color(text.name())
        self._axes.tick_params(colors=text.name(), labelsize=8)
        self._axes.xaxis.label.set_color(text.name())
        self._axes.yaxis.label.set_color(text.name())
        self._axes.title.set_color(text.name())  # type: ignore[attr-defined]

    def clear_view(self) -> None:
        """Empty state before the first run."""
        self._axes.clear()
        self._apply_theme()
        self._axes.set_title("Run a variation study to see landing dispersion")
        self._axes.set_xlabel("Lateral [m] (+ right)")
        self._axes.set_ylabel("Carry [m]")
        self.draw_idle()

    def set_dataset(
        self, dataset: VariationDataset, ellipse: DispersionEllipse | None
    ) -> None:
        """Scatter the successful landings and overlay the fit ellipse / convex hull."""
        self._axes.clear()
        self._apply_theme()
        landing_points = dataset.finite_output_rows("carry_m", "lateral_m")
        carry = landing_points[:, 0]
        lateral = landing_points[:, 1]
        self._axes.scatter(
            lateral, carry, s=14, alpha=0.65, color="#2f8bd6", edgecolors="none"
        )
        if ellipse is not None:
            # Check bivariate normality and render convex hull fallback if non-normal
            if (
                ellipse.diagnostic is not None
                and not ellipse.diagnostic.is_normal
                and ellipse.convex_hull is not None
                and len(ellipse.convex_hull) >= 3
            ):
                hull = ellipse.convex_hull
                closed_hull = np.vstack([hull, hull[0]])
                self._axes.plot(
                    closed_hull[:, 0],
                    closed_hull[:, 1],
                    color="#f59e0b",
                    linestyle="-",
                    linewidth=1.8,
                    label="Convex Hull",
                )
                patch = Ellipse(
                    (ellipse.center_lateral_m, ellipse.center_carry_m),
                    width=2.0 * ellipse.semi_major_m,
                    height=2.0 * ellipse.semi_minor_m,
                    angle=90.0 - ellipse.angle_deg,
                    fill=False,
                    linestyle=":",
                    linewidth=1.2,
                    edgecolor="#eb6a3c",
                    alpha=0.45,
                )
                self._axes.add_patch(patch)
            else:
                patch = Ellipse(
                    (ellipse.center_lateral_m, ellipse.center_carry_m),
                    width=2.0 * ellipse.semi_major_m,
                    height=2.0 * ellipse.semi_minor_m,
                    angle=90.0 - ellipse.angle_deg,
                    fill=False,
                    linestyle="--",
                    linewidth=1.6,
                    edgecolor="#eb6a3c",
                )
                self._axes.add_patch(patch)
            self._axes.plot(
                [ellipse.center_lateral_m],
                [ellipse.center_carry_m],
                marker="+",
                markersize=10,
                color="#eb6a3c",
            )
        if self._outcome_counts is None:
            summary = f"{carry.size} landings / {dataset.plan.n_runs} trials"
        else:
            summary = (
                f"Hits {self._outcome_counts[TrialEvaluationStatus.EVALUATED_HIT]} · "
                "No Impact "
                f"{self._outcome_counts[TrialEvaluationStatus.EVALUATED_NO_IMPACT]} · "
                "Failures "
                f"{self._outcome_counts[TrialEvaluationStatus.NUMERICAL_FAILURE]} · "
                f"Landings {carry.size}"
            )
        if (
            ellipse is not None
            and ellipse.diagnostic is not None
            and not ellipse.diagnostic.is_normal
        ):
            norm_tag = (
                f"Non-normal (Mardia p={ellipse.diagnostic.p_value:.2g}): Convex Hull"
            )
        else:
            norm_tag = "2σ ellipse"
        self._axes.set_title(f"Landing dispersion — {summary} ({norm_tag})")
        self._axes.set_xlabel("Lateral [m] (+ right)")
        self._axes.set_ylabel("Carry [m]")
        self._axes.set_aspect("equal", adjustable="datalim")
        self.draw_idle()
