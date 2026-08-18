"""Kinetics sub-tab of the Simulation tab (#4125 H2).

Shows the three built-in kinetics plots for the current run (Joint
Torques, Joint Power, Reaction Forces) plus a peak-value table (peak
torque / power / force per joint with timing as % of the downswing —
start of the swing to the impact instant) and click-through
explanations wired to the glossary.

Plot styling mirrors the movement optimizer's time-series conventions
(``src/movement_optimizer/gui/plot_renderer.py``): "Time (s)" x-axis,
parenthesised units with the middle dot ("Torque (N·m)"), a faint zero
line on signed quantities, per-joint colors from the shared chart
cycle, and a dashed total overlay on the power plot.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.derivation import KINETICS_EXPLANATIONS
from rate_of_closure.simulation import (
    KineticsSeries,
    SimulationRun,
)
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.ui.pyqt6.presentation_kinetics import kinetics_for_presentation
from rate_of_closure.ui.pyqt6.result_row import explanation_html

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


logger = logging.getLogger(__name__)

__all__ = ["PEAK_TABLE_COLUMNS", "KineticsPanel"]

#: Peak-table column headers, in display order.
PEAK_TABLE_COLUMNS: tuple[str, ...] = (
    "Joint",
    "Peak |Torque| (N·m)",
    "Torque Timing (% downswing)",
    "Peak |Power| (W)",
    "Power Timing (% downswing)",
    "Peak |Force| (N)",
    "Force Timing (% downswing)",
)

_UNAVAILABLE_TEXT = (
    "Kinetics need the pendulum joint states — select the Double "
    "Pendulum swing source and run the simulation to populate this "
    "panel (manual and triple-pendulum sources are not supported)."
)


def _downswing_peak(
    t: np.ndarray, values: np.ndarray, tau: float
) -> tuple[float, float]:
    """(peak |value|, timing as % of the downswing [0, tau])."""
    mask = t <= max(tau, float(t[1]))
    window = np.abs(values[mask])
    index = int(np.argmax(window))
    timing = 100.0 * float(t[mask][index]) / tau if tau > 0.0 else 0.0
    return float(window[index]), timing


class KineticsPanel(QWidget):
    """Kinetics plots + peak table for the current simulation run."""

    #: Emitted with a glossary term key when an explanation link is used.
    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(6.0, 6.5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)

        self._status = QLabel(_UNAVAILABLE_TEXT)
        self._status.setWordWrap(True)

        self._table = QTableWidget(0, len(PEAK_TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(PEAK_TABLE_COLUMNS))
        header = self._table.verticalHeader()
        if header is not None:
            header.setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(140)
        self._table.setToolTip(
            "Peak torque, power, and reaction-force magnitudes per joint "
            "with their timing as a percentage of the downswing (start of "
            "the swing to the impact instant)."
        )

        self._explanation = QTextBrowser()
        self._explanation.setOpenExternalLinks(False)
        self._explanation.setOpenLinks(False)
        self._explanation.setMaximumHeight(170)
        self._explanation.setToolTip(
            "What the kinetics plots show; the Glossary links jump to the "
            "matching terms."
        )
        self._explanation.anchorClicked.connect(self._on_explanation_link)
        self._explanation.setHtml(
            "".join(
                explanation_html(title, KINETICS_EXPLANATIONS[field], field)
                for field, title in (
                    ("joint_torques", "Joint Torques"),
                    ("joint_power", "Joint Power"),
                    ("reaction_forces", "Reaction Forces"),
                    (
                        "zero_torque_counterfactual",
                        "Zero-Torque Counterfactual (ZTCF)",
                    ),
                )
            )
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._status)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._table)
        bottom = QHBoxLayout()
        bottom.addWidget(self._explanation)
        layout.addLayout(bottom)

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a run (or clear with ``None``) and redraw."""
        series = kinetics_for_presentation(run) if run is not None else None
        if series is None:
            self._status.setText(_UNAVAILABLE_TEXT)
            self._status.setVisible(True)
            self._figure.clear()
            self._canvas.draw_idle()
            self._table.setRowCount(0)
            return
        is_miss = run is not None and not run.impact_outcome.is_hit
        if is_miss:
            self._status.setText(
                "No impact occurred. Complete-swing kinetics remain available; "
                "the dashed closest approach marker is a timing reference, not "
                "an impact."
            )
            self._status.setVisible(True)
        else:
            self._status.setVisible(False)
        self._draw(series, "Closest Approach" if is_miss else "Impact")
        self._fill_table(series)

    def table(self) -> QTableWidget:
        """The peak-values table (tests)."""
        return self._table

    # ── internals ──────────────────────────────────────────────────
    def _styled_axis(self, ax, ylabel: str, title: str) -> None:  # type: ignore[no-untyped-def]
        """Movement-optimizer axis conventions (plot_renderer.py)."""
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.12)
        ax.legend(fontsize=7, loc="best")

    def _draw(self, series: KineticsSeries, reference_label: str) -> None:
        self._figure.clear()
        axes = self._figure.subplots(3, 1, sharex=True)
        t, tau = series.t, series.impact_time_s
        names = series.joint_names

        for j, name in enumerate(names):
            axes[0].plot(
                t,
                series.torque_inertial_nm[:, j],
                color=get_chart_color(j),
                lw=1.8,
                label=name,
            )
            axes[0].plot(
                t,
                series.torque_gravity_nm[:, j],
                color=get_chart_color(j),
                lw=1.0,
                ls=":",
                alpha=0.7,
                label=f"{name} gravity",
            )
            axes[0].plot(
                t,
                series.ztcf_inertial_torque_nm[:, j],
                color=get_chart_color(j),
                lw=1.2,
                ls="--",
                alpha=0.9,
                label=f"{name} ZTCF",
            )
        axes[0].axhline(0, lw=0.5, alpha=0.3)
        axes[0].axvline(tau, ls="--", lw=1.5, alpha=0.8, label=reference_label)
        self._styled_axis(
            axes[0],
            "Torque (N·m)",
            "Joint Torques and State-Matched ZTCF",
        )

        for j, name in enumerate(names):
            axes[1].plot(
                t,
                series.power_w[:, j],
                color=get_chart_color(j),
                lw=1.8,
                label=name,
            )
        # Dashed total overlay: the movement-optimizer power convention.
        axes[1].plot(
            t,
            np.sum(series.power_w, axis=1),
            "--",
            lw=2.0,
            alpha=0.7,
            label="Total",
        )
        axes[1].axhline(0, lw=0.5, alpha=0.3)
        axes[1].axvline(tau, ls="--", lw=1.5, alpha=0.8)
        self._styled_axis(axes[1], "Power (W)", "Joint Power")

        for j, which in enumerate(("shoulder", "wrist", "clubhead")):
            axes[2].plot(
                t,
                series.force_magnitude_n(which),
                color=get_chart_color(j),
                lw=1.8,
                label=which,
            )
            axes[2].plot(
                t,
                series.ztcf_force_magnitude_n(which),
                color=get_chart_color(j),
                lw=1.1,
                ls="--",
                alpha=0.85,
                label=f"{which} ZTCF",
            )
        axes[2].axvline(tau, ls="--", lw=1.5, alpha=0.8)
        self._styled_axis(
            axes[2],
            "Force (N)",
            "Reaction Forces and State-Matched ZTCF",
        )
        self._canvas.draw_idle()

    def _fill_table(self, series: KineticsSeries) -> None:
        t, tau = series.t, series.impact_time_s
        rows: list[tuple[str, ...]] = []
        for j, name in enumerate(series.joint_names):
            torque, torque_at = _downswing_peak(t, series.torque_inertial_nm[:, j], tau)
            power, power_at = _downswing_peak(t, series.power_w[:, j], tau)
            force, force_at = _downswing_peak(t, series.force_magnitude_n(name), tau)
            rows.append(
                (
                    name,
                    f"{torque:.1f}",
                    f"{torque_at:.0f}%",
                    f"{power:.0f}",
                    f"{power_at:.0f}%",
                    f"{force:.0f}",
                    f"{force_at:.0f}%",
                )
            )
        head, head_at = _downswing_peak(t, series.force_magnitude_n("clubhead"), tau)
        rows.append(("clubhead", "—", "—", "—", "—", f"{head:.0f}", f"{head_at:.0f}%"))

        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()

    def _on_explanation_link(self, url) -> None:  # type: ignore[no-untyped-def]
        """Forward ``glossary:<term>`` links to the main window."""
        text = url.toString()
        if text.startswith("glossary:"):
            self.glossaryRequested.emit(text.partition(":")[2])
