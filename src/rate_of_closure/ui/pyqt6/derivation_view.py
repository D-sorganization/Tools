"""Derivation & Traceability tab — the calculation, step by step.

Renders :func:`rate_of_closure.derivation.derivation_steps` as a
scrollable sequence: Title Case heading, plain-language narrative, and
the symbolic formula plus live numeric substitution typeset with
matplotlib mathtext (the house LaTeX renderer — no external TeX
dependency). Every number in the results panel can be traced to its
step here, and the steps re-substitute whenever the scenario changes.
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

from rate_of_closure.derivation import derivation_steps
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
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        try:
            axes.text(0.02, 0.68, latex, fontsize=12, va="center")
            axes.text(0.02, 0.22, values, fontsize=11, va="center", alpha=0.85)
        except ValueError:  # malformed mathtext must never break the tab
            logger.exception("mathtext rendering failed")
            axes.text(0.02, 0.5, latex.replace("$", ""), fontsize=10)


class DerivationView(QWidget):
    """Scrollable, live-substituted derivation of the whole calculation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)
        self._scenario: ImpactScenario | None = None

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Rebuild every step with the new scenario's numbers."""
        if scenario == self._scenario:
            return
        self._scenario = scenario

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        intro = QLabel(
            "Every result in the left panel traces to one of the steps "
            "below; the numeric line under each formula substitutes the "
            "current scenario. Sources: AffineDrift Launch Monitor "
            "Technology Review (frame and sign conventions), the "
            "closure-rate derivation (d / R_ISA, deg/ft), and the "
            "Cheetham 2014 closure-rate dossier."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for index, step in enumerate(derivation_steps(scenario), start=1):
            heading = QLabel(f"Step {index} — {step.title}")
            font = heading.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1)
            heading.setFont(font)
            layout.addWidget(heading)

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
