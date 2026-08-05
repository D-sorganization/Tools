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

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas
from rate_of_closure.units import DISTANCE_UNITS, display_distance_unit
from shared.python.swing_sim.variation import (
    DispersionEllipse,
    OutputStats,
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

    def set_stats(self, stats: tuple[OutputStats, ...]) -> None:
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
    ) -> None:
        """Show ``values`` shaded by ``normalized`` (both inputs x outputs)."""
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
                if not np.isnan(frac):
                    item.setBackground(self._heat(frac))
                    item.setForeground(QColor(255, 255, 255))
                item.setToolTip(
                    f"{short_label(input_keys[i])} → {output_names[j]}: "
                    f"{text} (column-normalized {0.0 if np.isnan(frac) else frac:.2f}; "
                    "1.00 marks the input that dominates this output)"
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
        self.setToolTip(
            "Landing positions of every successful run, viewed from above "
            "(x = lateral, + right of target; y = carry). The dashed "
            "ellipse is the 2-sigma dispersion fit."
        )
        self._apply_theme()
        self.clear_view()

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
        self._axes.title.set_color(text.name())

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
        """Scatter the successful landings and overlay the fit ellipse."""
        self._axes.clear()
        self._apply_theme()
        carry = dataset.output_column("carry_m")
        lateral = dataset.output_column("lateral_m")
        self._axes.scatter(
            lateral, carry, s=14, alpha=0.65, color="#2f8bd6", edgecolors="none"
        )
        if ellipse is not None:
            # Engine angle is CCW from the carry axis toward +lateral; in
            # plot coordinates (x = lateral, y = carry) that is 90° - angle.
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
        self._axes.set_title(
            f"Landing dispersion — {dataset.n_success}/{dataset.plan.n_runs} "
            "runs (2σ ellipse)"
        )
        self._axes.set_xlabel("Lateral [m] (+ right)")
        self._axes.set_ylabel("Carry [m]")
        self._axes.set_aspect("equal", adjustable="datalim")
        self.draw_idle()
