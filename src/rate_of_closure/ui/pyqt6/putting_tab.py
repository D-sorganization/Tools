"""Putting tab — putter, stroke, green, roll-out (#4125 H3).

Left: putter picker (H1 club-library putters via
:func:`rate_of_closure.putting.putter_specs`), stroke pace (clubhead
speed directly, or a backstroke length through the pendulum proxy),
green conditions (stimp, grade, downhill aspect), hole distance, and
clickable result rows with explanations. Right: a top-down green view
(path colour-coded by skid/pure-roll phase, hole, downhill arrow) over
a speed-vs-distance plot with the capture-speed bound marked.

The physics lives in ``shared.python.swing_sim.putting``; this widget
is presentation only. Distances are SI (metres) end to end through the
single ``_format_m`` chokepoint, ready for the units-quantity pass
(H6) to route through the shared conversion table.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.putting import PUTT_EXPLANATIONS, putter_specs
from rate_of_closure.putting_result_contract import (
    AcceptedPuttingContext,
    validate_putting_result_summary,
)
from rate_of_closure.putting_sample_inspector import (
    PuttingSamplePlan,
    PuttingSampleSeries,
    plan_putting_samples,
)
from rate_of_closure.ui.pyqt6.putting_result_presentation import putting_result_values
from rate_of_closure.ui.pyqt6.putting_visuals import PuttingPlotView
from rate_of_closure.ui.pyqt6.result_row import ResultRow, explanation_html
from rate_of_closure.units import format_distance_m
from shared.python.swing_sim.putting import (
    GreenConditions,
    PuttResult,
    clubhead_speed_from_backstroke,
    simulate_putt,
    strike,
)

logger = logging.getLogger(__name__)

__all__ = ["PuttingTab"]

#: (result field, Title Case label) in display order; every field has
#: a PUTT_EXPLANATIONS entry (contract-tested).
_ROWS: tuple[tuple[str, str], ...] = (
    ("putt_rollout_m", "Roll-Out Distance"),
    ("putt_skid_m", "Skid Distance"),
    ("putt_skid_pct", "Skid Share of Putt"),
    ("putt_time_s", "Time To Rest"),
    ("putt_break_m", "Break"),
    ("putt_speed_at_hole_mps", "Speed At The Hole"),
    ("putt_margin", "Holed / Miss Margin"),
)


class PuttingTab(QWidget):
    """Interactive putting laboratory on a uniform sloped green."""

    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal style

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._putters = putter_specs()
        self._rows: dict[str, ResultRow] = {}
        self._result: PuttResult | None = None
        self._accepted_context: AcceptedPuttingContext | None = None
        self._accepted_plan: PuttingSamplePlan | None = None
        self._accepted_generation: object | None = None

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_controls_box())
        left_layout.addWidget(self._build_rows_box())
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(300)

        self._plot_view = PuttingPlotView()
        self._canvas = self._plot_view.canvas()

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self._plot_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._recompute()
        self._show_explanation(_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
    def _spin(
        self,
        low: float,
        high: float,
        value: float,
        step: float,
        suffix: str,
        tooltip: str,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setToolTip(tooltip)
        box.valueChanged.connect(self._recompute)
        return box

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("Putt Setup")
        form = QFormLayout(box)

        self._putter_combo = QComboBox()
        self._putter_combo.addItems(list(self._putters))
        self._putter_combo.setToolTip(
            "Putter head used for the impact model. Library putters "
            "(H1 club library) when available; head mass and loft "
            "drive the ball-speed transfer and launch spin."
        )
        self._putter_combo.currentTextChanged.connect(self._recompute)
        form.addRow("Putter", self._putter_combo)

        self._pace_mode = QComboBox()
        self._pace_mode.addItems(["Clubhead speed", "Backstroke length"])
        self._pace_mode.setToolTip(
            "How the stroke pace is set: the head speed at impact "
            "directly, or a pendulum backstroke length through "
            "v = A·sqrt(g/L) (simple-pendulum proxy)."
        )
        form.addRow("Pace input", self._pace_mode)

        self._speed_spin = self._spin(
            0.2,
            6.0,
            1.8,
            0.05,
            " m/s",
            "Clubhead speed at impact. Suggested range: 0.5-3 m/s for "
            "putts inside 15 m. Source: pendulum-stroke kinematics "
            "(swing_sim.putting.impact).",
        )
        self._backstroke_spin = self._spin(
            5.0,
            100.0,
            30.0,
            1.0,
            " cm",
            "Backstroke arc length; converted to head speed with the "
            "simple-pendulum proxy v = A·sqrt(g/L). Suggested range: "
            "10-60 cm. Source: swing_sim.putting.impact derivation.",
            decimals=0,
        )
        self._pace_stack = QStackedWidget()
        self._pace_stack.setToolTip(
            "Stroke pace entry — switches with the pace-input mode."
        )
        for widget in (self._speed_spin, self._backstroke_spin):
            holder = QWidget()
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.addWidget(widget)
            self._pace_stack.addWidget(holder)
        self._pace_stack.setCurrentIndex(0)
        self._pace_mode.currentIndexChanged.connect(self._pace_stack.setCurrentIndex)
        self._pace_mode.currentIndexChanged.connect(self._recompute)
        form.addRow("Stroke pace", self._pace_stack)

        self._stimp_spin = self._spin(
            3.0,
            16.0,
            10.0,
            0.5,
            " ft",
            "Green speed as a stimpmeter reading. Suggested range: "
            "7 (slow) - 13 (tournament fast). Source: USGA stimpmeter "
            "geometry (swing_sim.putting.roll derivation).",
            decimals=1,
        )
        form.addRow("Green speed (stimp)", self._stimp_spin)

        self._grade_spin = self._spin(
            0.0,
            10.0,
            0.0,
            0.25,
            " %",
            "Uniform slope grade of the green. Suggested range: 0-4 % "
            "(greens rarely exceed ~5 %). Source: course-architecture "
            "norms (swing_sim.putting.green).",
        )
        form.addRow("Slope grade", self._grade_spin)

        self._aspect_spin = self._spin(
            -360.0,
            360.0,
            90.0,
            5.0,
            "°",
            "Downhill direction relative to the putt line: 0° = "
            "downhill straight ahead, +90° = low side on your left, "
            "180° = uphill putt. Source: swing_sim.putting.green frame.",
            decimals=0,
        )
        form.addRow("Downhill direction", self._aspect_spin)

        self._distance_spin = self._spin(
            0.1,
            40.0,
            3.0,
            0.1,
            " m",
            "Distance from the ball to the hole centre along the "
            "starting line. Suggested range: 1-15 m. Source: "
            "swing_sim.putting.green.",
            decimals=1,
        )
        form.addRow("Distance to hole", self._distance_spin)
        return box

    def _build_rows_box(self) -> QGroupBox:
        box = QGroupBox("Putt Results")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for field, label in _ROWS:
            row = ResultRow(field, label)
            row.setToolTip(
                "Click for a plain-language explanation of this number "
                "(with a glossary link)."
            )
            row.clicked.connect(self._show_explanation)
            self._rows[field] = row
            layout.addWidget(row)
        return box

    def _build_explanation_box(self) -> QGroupBox:
        box = QGroupBox("What This Number Means")
        layout = QVBoxLayout(box)
        self._explanation = QTextBrowser()
        self._explanation.setOpenExternalLinks(False)
        self._explanation.setOpenLinks(False)
        self._explanation.setToolTip(
            "Explanation of the selected result row; the Glossary link "
            "jumps to the matching term."
        )
        self._explanation.anchorClicked.connect(self._on_explanation_link)
        self._explanation.setMinimumHeight(110)
        self._explanation.setMaximumHeight(170)
        layout.addWidget(self._explanation)
        return box

    # ── behaviour ───────────────────────────────────────────────────
    def _show_explanation(self, field: str) -> None:
        labels = dict(_ROWS)
        for row_field, row in self._rows.items():
            row.set_selected(row_field == field)
        self._explanation.setHtml(
            explanation_html(labels[field], PUTT_EXPLANATIONS[field], field)
        )

    def _on_explanation_link(self, url) -> None:  # type: ignore[no-untyped-def]
        text = url.toString()
        if text.startswith("glossary:"):
            self.glossaryRequested.emit(text.partition(":")[2])

    def _clubhead_speed(self) -> float:
        if self._pace_mode.currentIndex() == 1:
            putter_length_m = 0.889  # standard 35 in putter
            return float(
                clubhead_speed_from_backstroke(
                    self._backstroke_spin.value() / 100.0, putter_length_m
                )
            )
        return float(self._speed_spin.value())

    @staticmethod
    def _format_m(value: float) -> str:
        """Single distance-format chokepoint — follows the session's
        distance display unit (#4125 H6: yards default, metres option)."""
        return str(format_distance_m(value, decimals=2))

    def result(self) -> PuttResult | None:
        """The last computed putt (LoD seam for tests)."""
        return self._result

    def refresh_units(self) -> None:
        """Re-render rows and axes in the distance display unit (H6)."""
        if self._result is not None:
            self._update_rows(self._result)
            if (
                self._accepted_context is None
                or self._accepted_plan is None
                or self._accepted_generation is None
            ):
                raise RuntimeError("accepted putting display bundle is unavailable")
            context = self._accepted_context
            self._plot_view.set_result(
                self._result,
                self._accepted_plan,
                generation=self._accepted_generation,
                hole_x=context.hole_m,
                grade=context.grade_percent,
                aspect=context.aspect_deg,
                context_text=context.label(),
            )

    def _recompute(self) -> None:
        putter = self._putters[self._putter_combo.currentText()]
        speed = self._clubhead_speed()
        context = AcceptedPuttingContext(
            putter.name,
            putter.head_mass_kg,
            putter.loft_deg,
            putter.cor,
            speed,
            self._stimp_spin.value(),
            self._grade_spin.value(),
            self._aspect_spin.value(),
            self._distance_spin.value(),
        )
        try:
            launch = strike(putter, speed)
            green = GreenConditions(
                stimp_ft=self._stimp_spin.value(),
                grade_percent=self._grade_spin.value(),
                aspect_deg=self._aspect_spin.value(),
            )
            result = simulate_putt(launch, green, self._distance_spin.value())
            plan = plan_putting_samples(PuttingSampleSeries.from_result(result))
            validate_putting_result_summary(result, plan)
            row_values = putting_result_values(result, self._format_m)
            generation = object()
            self._plot_view.set_result(
                result,
                plan,
                generation=generation,
                hole_x=context.hole_m,
                grade=context.grade_percent,
                aspect=context.aspect_deg,
                context_text=context.label(),
            )
        except Exception as error:
            logger.exception("putt inputs rejected")
            self._plot_view.set_error(
                f"Attempted configuration rejected ({context.label()}): {error}"
            )
            return
        self._result = result
        self._accepted_plan = plan
        self._accepted_context = context
        self._accepted_generation = generation
        self._publish_rows(row_values)

    def _publish_rows(self, values: dict[str, str]) -> None:
        for field, text in values.items():
            self._rows[field].value_label.setText(text)

    def _update_rows(self, result: PuttResult) -> None:
        self._publish_rows(putting_result_values(result, self._format_m))
