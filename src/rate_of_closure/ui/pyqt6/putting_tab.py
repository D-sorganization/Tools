"""Putting tab — stroke, green, results, and 3-D playback (#4800 P6).

Left column: the delivered stroke
(:mod:`~rate_of_closure.ui.pyqt6.putting_stroke_controls` — putter head,
pace, aim/face/path/attack, strike location), the green
(:mod:`~rate_of_closure.ui.pyqt6.putting_green_controls` — stimp, planar
grade and aspect or an imported heightfield, hole distance, capture
model), then clickable result rows with explanations. Right column: the
top-down green with the break trajectory and the hole-capture geometry
over a speed-vs-distance plot, and the orbitable 3-D playback of the
same recorded samples.

Every number on screen comes from one solve per recompute:
``strike_with_head`` (P1 impact through P3's head document) →
``simulate_putt_on_surface`` (P2 surface integration and capture) →
``putting_result_document`` (the ``swing_sim.putting_result/2`` record,
P5). The wire record is the presentation authority for the 2-D fields;
the tab never recomputes a summary the record already carries, and the
playback view replays the retained samples rather than re-integrating.

The physics lives in ``shared.python.swing_sim.putting`` and
``shared.python.golf_club.putter_head``; this widget is presentation
only. Distances are SI (metres) end to end through the single
``_format_m`` chokepoint, which follows the session's display unit.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.putting import PUTT_EXPLANATIONS
from rate_of_closure.putting_result_contract import (
    AcceptedPuttingContext,
    validate_putting_result_summary,
)
from rate_of_closure.putting_sample_inspector import (
    PuttingSamplePlan,
    PuttingSampleSeries,
    plan_putting_samples,
)
from rate_of_closure.ui.pyqt6.putting_green_controls import PuttingGreenControls
from rate_of_closure.ui.pyqt6.putting_playback import PuttPlaybackView
from rate_of_closure.ui.pyqt6.putting_result_presentation import (
    putting_document_values,
    putting_result_values,
)
from rate_of_closure.ui.pyqt6.putting_stroke_controls import PuttingStrokeControls
from rate_of_closure.ui.pyqt6.putting_visuals import PuttingPlotView
from rate_of_closure.ui.pyqt6.result_row import ResultRow, explanation_html
from rate_of_closure.units import format_distance_m
from shared.python.swing_sim.putting import (
    PuttingResultDocument,
    PuttingResultProvenance,
    PuttResult,
    putting_result_document,
    simulate_putt_on_surface,
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
    ("putt_start_azimuth_deg", "Start Line"),
    ("putt_break_m", "Break At Rest"),
    ("putt_apex_break_m", "Apex Break"),
    ("putt_entry_azimuth_deg", "Entry Direction"),
    ("putt_speed_at_hole_mps", "Speed At The Hole"),
    ("putt_capture_margin_m", "Capture Margin"),
    ("putt_face_twist_deg", "Face Twist At Strike"),
    ("putt_margin", "Holed / Miss Margin"),
)

#: ``putter_head/1`` provenance kind -> ``putting_result/2`` putter
#: source. The two wires name the same origins with their own
#: vocabularies; mapping them here keeps the record honest about which
#: kind of head actually solved the impact.
_PUTTER_SOURCES = {"mesh": "mesh", "library": "library"}


class PuttingTab(QWidget):
    """Interactive putting laboratory on a planar or imported green."""

    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal style

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, ResultRow] = {}
        self._result: PuttResult | None = None
        self._accepted_context: AcceptedPuttingContext | None = None
        self._accepted_plan: PuttingSamplePlan | None = None
        self._accepted_generation: object | None = None
        self._accepted_document: PuttingResultDocument | None = None

        self._stroke_controls = PuttingStrokeControls()
        self._green_controls = PuttingGreenControls()
        for controls in (self._stroke_controls, self._green_controls):
            controls.changed.connect(self._recompute)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._stroke_controls)
        left_layout.addWidget(self._green_controls)
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
        self._playback_view = PuttPlaybackView()
        visuals = QSplitter()
        visuals.setOrientation(Qt.Orientation.Vertical)
        visuals.addWidget(self._plot_view)
        visuals.addWidget(self._playback_view)
        visuals.setStretchFactor(0, 3)
        visuals.setStretchFactor(1, 2)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(visuals)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._recompute()
        self._show_explanation(_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
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

    @staticmethod
    def _format_m(value: float) -> str:
        """Single distance-format chokepoint — follows the session's
        distance display unit (#4125 H6: yards default, metres option)."""
        return str(format_distance_m(value, decimals=2))

    def result(self) -> PuttResult | None:
        """The last computed putt (LoD seam for tests)."""
        return self._result

    def document(self) -> PuttingResultDocument | None:
        """The last ``putting_result/2`` record (LoD seam for tests)."""
        return self._accepted_document

    def playback_view(self) -> PuttPlaybackView:
        """The 3-D playback surface (transport and probe seam)."""
        return self._playback_view

    def stroke_controls(self) -> PuttingStrokeControls:
        """The delivered-stroke control group (LoD seam for tests)."""
        return self._stroke_controls

    def green_controls(self) -> PuttingGreenControls:
        """The green control group (LoD seam for tests)."""
        return self._green_controls

    def refresh_units(self) -> None:
        """Re-render rows and axes in the distance display unit (H6)."""
        if self._result is None:
            return
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
        from shared.python.golf_club.putter_head import strike_with_head

        head = self._stroke_controls.head_document()
        stroke = self._stroke_controls.stroke()
        green = self._green_controls.green()
        context = AcceptedPuttingContext(
            head.name,
            head.provenance.source_kind,
            head.head_mass_kg,
            head.loft_deg,
            head.cor,
            stroke.label(),
            green.label(),
            green.grade_percent,
            green.aspect_deg,
            green.hole_distance_m,
        )
        try:
            solved = strike_with_head(
                head,
                stroke.clubhead_speed_mps,
                stroke.shaft_lean_deg,
                aim_deg=stroke.aim_deg,
                face_angle_deg=stroke.face_angle_deg,
                path_angle_deg=stroke.path_angle_deg,
                attack_angle_deg=stroke.attack_angle_deg,
                strike_offset_toe_mm=stroke.strike_offset_toe_mm,
                strike_offset_high_mm=stroke.strike_offset_high_mm,
            )
            result = simulate_putt_on_surface(
                solved.launch,
                green.surface,
                stimp_ft=green.stimp_ft,
                hole_distance_m=green.hole_distance_m,
                capture_model=green.capture_model,
            )
            document = putting_result_document(
                solved.launch,
                result,
                PuttingResultProvenance(
                    putter_source=_PUTTER_SOURCES[head.provenance.source_kind],
                    putter_name=head.name,
                    stroke_source="declared",
                    capture_model=green.capture_model,
                    putter_mesh_sha256=head.provenance.mesh_sha256,
                    putter_library_name=head.provenance.library_name,
                ),
                hole_distance_m=green.hole_distance_m,
            )
            plan = plan_putting_samples(PuttingSampleSeries.from_result(result))
            validate_putting_result_summary(result, plan)
            row_values = putting_result_values(result, self._format_m)
            row_values.update(
                putting_document_values(document, solved.twist, self._format_m)
            )
            generation = object()
            self._playback_view.set_putt(
                result, green.surface, hole_distance_m=green.hole_distance_m
            )
            self._plot_view.set_result(
                result,
                plan,
                generation=generation,
                hole_x=green.hole_distance_m,
                grade=green.grade_percent,
                aspect=green.aspect_deg,
                context_text=context.label(),
                document=document,
            )
        except Exception as error:
            logger.exception("putt inputs rejected")
            # The 2-D view is the retained-evidence authority and
            # restores itself; the derived playback is dropped so it
            # can never show a putt the tab has not accepted.
            self._playback_view.clear()
            self._plot_view.set_error(
                f"Attempted configuration rejected ({context.label()}): {error}"
            )
            return
        self._result = result
        self._accepted_plan = plan
        self._accepted_context = context
        self._accepted_generation = generation
        self._accepted_document = document
        self._publish_rows(row_values)

    def _publish_rows(self, values: dict[str, str]) -> None:
        for field, text in values.items():
            self._rows[field].value_label.setText(text)

    def _update_rows(self, result: PuttResult) -> None:
        self._publish_rows(putting_result_values(result, self._format_m))
