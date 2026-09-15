"""Tracker phase and the last putt, as labels.

:class:`PuttReadout` is a passive display: :meth:`show_event` copies the
phase and, when the event carries a putt, its speed, HLA, points, r² and
the sink's verdict. It never computes anything about the putt.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from putting_launch_monitor.monitor import FrameEvent
from putting_launch_monitor.track import Phase, Putt
from shared.python.theme import Sizes, Weights, get_display_font, get_mono_font

from .painting import TOKENS

PHASE_TEXT: dict[Phase, str] = {
    Phase.WAITING: "waiting for a resting ball",
    Phase.ARMED: "armed",
    Phase.ROLLING: "rolling",
}
EMPTY = "—"


class PuttReadout(QGroupBox):
    """Phase and last-putt figures."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Putt", parent)
        form = QFormLayout(self)
        form.setSpacing(TOKENS.spacing_px)
        pad = TOKENS.padding_px
        form.setContentsMargins(pad, pad, pad, pad)
        self.phase = self._label("Tracker state: waiting, armed or rolling.")
        self.phase.setFont(get_display_font(Sizes.MD, Weights.SEMIBOLD))
        self.speed = self._label("Launch speed of the last putt, in mph.")
        self.speed.setFont(get_display_font(Sizes.XXL, Weights.BOLD))
        self.hla = self._label(
            "Horizontal launch angle of the last putt: positive to the right."
        )
        self.hla.setFont(get_display_font(Sizes.XL, Weights.SEMIBOLD))
        self.points = self._label("Frames used in the launch fit.")
        self.r2 = self._label("Coefficient of determination of the launch fit.")
        self.outcome = self._label("What happened to the putt: logged, sent or held.")
        self.outcome.setWordWrap(True)
        self.frames = self._label("Sequence number of the last frame processed.")
        for widget in (self.points, self.r2, self.frames):
            widget.setFont(get_mono_font(Sizes.BASE))
        form.addRow("Phase", self.phase)
        form.addRow("Speed (mph)", self.speed)
        form.addRow("HLA (deg)", self.hla)
        form.addRow("Points", self.points)
        form.addRow("r²", self.r2)
        form.addRow("Outcome", self.outcome)
        form.addRow("Frame", self.frames)
        self.clear()

    @staticmethod
    def _label(tooltip: str) -> QLabel:
        label = QLabel(EMPTY)
        label.setToolTip(tooltip)
        return label

    def clear(self) -> None:
        self.phase.setText(PHASE_TEXT[Phase.WAITING])
        for widget in (self.speed, self.hla, self.points, self.r2, self.outcome):
            widget.setText(EMPTY)
        self.frames.setText("0")

    def show_event(self, event: FrameEvent) -> None:
        self.phase.setText(PHASE_TEXT[event.phase])
        self.frames.setText(str(event.sequence))
        if event.putt is not None:
            self.show_putt(event.putt, event.outcome)

    def show_putt(self, putt: Putt, outcome: str) -> None:
        self.speed.setText(f"{putt.speed_mph:.2f}")
        self.hla.setText(f"{putt.hla_deg:+.1f}")
        self.points.setText(str(putt.launch.points))
        self.r2.setText(f"{putt.launch.r2:.3f}")
        self.outcome.setText(outcome or putt.reason or "accepted")
