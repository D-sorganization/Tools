"""Calculation Description tab — every model's derivation, step by step.

Renders :func:`rate_of_closure.derivation_models.derivation_sections`
as a scrollable sequence of titled sections (closure chain, impact
model, ball flight, and — when a pendulum source is selected — the
swing model): section heading, intro, then each step's Title Case
heading, plain-language narrative, and the symbolic formula plus live
numeric substitution typeset with matplotlib mathtext (the house LaTeX
renderer — no external TeX dependency). The steps re-substitute
whenever the scenario changes, and sections toggle with the live
configuration (:meth:`DerivationView.set_config`).
"""

from __future__ import annotations

import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.derivation_models import (
    DerivationConfig,
    derivation_sections,
)
from rate_of_closure.model import ImpactScenario

logger = logging.getLogger(__name__)

__all__ = ["DerivationView"]

_FORMULA_HEIGHT_PX = 96


class _FormulaCanvas(FigureCanvas):
    """A fixed-height mathtext rendering of formula + numeric lines."""

    def __init__(self, latex: str, values: str) -> None:
        figure = Figure(figsize=(6.4, 1.0))
        figure.patch.set_alpha(0.0)
        super().__init__(figure)
        self.setStyleSheet("background: transparent;")
        self.setFixedHeight(_FORMULA_HEIGHT_PX)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("Symbolic formula with the live numeric substitution")
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        try:
            axes.text(0.02, 0.68, latex, fontsize=12, va="center")
            axes.text(0.02, 0.22, values, fontsize=11, va="center", alpha=0.85)
        except ValueError:  # malformed mathtext must never break the tab
            logger.exception("mathtext rendering failed")
            axes.text(0.02, 0.5, latex.replace("$", ""), fontsize=10)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Pass wheel events through so the surrounding tab scrolls.

        Matplotlib's Qt canvas normally accepts wheel events for its own
        zoom/scroll machinery, which silently ate scrolling on this tab.
        """
        event.ignore()


class DerivationView(QWidget):
    """Scrollable, live-substituted derivation of every model in use."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)
        self._scenario: ImpactScenario | None = None
        self._config = DerivationConfig()

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Rebuild every step with the new scenario's numbers."""
        if scenario == self._scenario:
            return
        self._scenario = scenario
        self._rebuild()

    def set_config(self, config: DerivationConfig) -> None:
        """Adopt the live configuration; sections toggle to match."""
        if config == self._config:
            return
        self._config = config
        self._rebuild()

    def config(self) -> DerivationConfig:
        """The configuration currently rendered (used by tests)."""
        return self._config

    def section_keys(self) -> tuple[str, ...]:
        """Keys of the sections rendered for the current state."""
        if self._scenario is None:
            return ()
        return tuple(
            section.key for section in derivation_sections(self._scenario, self._config)
        )

    # ── internals ──────────────────────────────────────────────────
    def _rebuild(self) -> None:
        if self._scenario is None:
            return
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Every number in the app traces to one of the sections "
            "below — the closure chain, the impact model, the active "
            "ball-flight model, and (when a pendulum source is "
            "selected) the swing model. The numeric line under each "
            "formula substitutes the current scenario and "
            "configuration. Sources: the AffineDrift Launch Monitor "
            "Technology Review, the Cheetham 2014 closure-rate "
            "dossier, and the swing_sim impact/flight/reference "
            "derivations."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for section in derivation_sections(self._scenario, self._config):
            heading = QLabel(section.title)
            font = heading.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 3)
            heading.setFont(font)
            layout.addWidget(heading)

            section_intro = QLabel(section.intro)
            section_intro.setWordWrap(True)
            layout.addWidget(section_intro)

            for index, step in enumerate(section.steps, start=1):
                step_heading = QLabel(f"Step {index} — {step.title}")
                font = step_heading.font()
                font.setBold(True)
                font.setPointSize(font.pointSize() + 1)
                step_heading.setFont(font)
                layout.addWidget(step_heading)

                narrative = QLabel(step.narrative)
                narrative.setWordWrap(True)
                narrative.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                layout.addWidget(narrative)

                layout.addWidget(_FormulaCanvas(step.latex, step.values))

                rule = QFrame()
                rule.setFrameShape(QFrame.Shape.HLine)
                rule.setFrameShadow(QFrame.Shadow.Sunken)
                layout.addWidget(rule)

        layout.addStretch(1)
        self._scroll.setWidget(content)
