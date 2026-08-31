"""Putting binding of the shared transport controls (#4800 P8).

The putting half of Amendment 1's one playback architecture. Every
transport behaviour — the single owned timer, scrub quantization, the
canonical speed set, and the wall-clock advance — already lives in
:class:`~rate_of_closure.ui.pyqt6.playback_transport_controls.PlaybackTransportControls`
and the runtime-neutral
:mod:`rate_of_closure.simulation.playback_transport` model it delegates
to, whose TypeScript twin drives the React surfaces. Nothing here
re-implements any of it: this module binds putting vocabulary and
connects the transport's ``timeChanged`` to P6's
:class:`~rate_of_closure.ui.pyqt6.putting_playback.PuttPlaybackView`,
which takes a physical time and owns no transport of its own.

Event wording: the second jump is the last *recorded* sample, which is
capture for a holed putt and rest for a missed one, so it is named
"Finish" exactly as P6's :meth:`PuttPlaybackView.event_times_s`
documents it. The view's status line names the outcome.

Camera seam (#4571): the orbit is the 3-D axes' own; no camera state is
owned here or in the view.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from rate_of_closure.ui.pyqt6.playback_transport_controls import (
    PlaybackTransportControls,
)
from rate_of_closure.ui.pyqt6.putting_playback import PuttPlaybackView
from shared.python.swing_sim.putting import GreenSurface, PuttResult

_EVENT_STRIKE = 0
_EVENT_FINISH = 1


class PuttPlaybackControls(PlaybackTransportControls):
    """Play, pause, scrub, restart, and jump through one putt timeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            subject_label="Putt",
            subject_phrase="putt",
            event_labels=("Strike", "Finish"),
            scrub_tooltip=(
                "Scrub physical putt time [s] from strike to rest or capture. "
                "Source: the retained integrator sample times; positions are "
                "green-frame metres (x target line, y left, z elevation)."
            ),
            help_text=(
                "Drag the 3D green to rotate; use the wheel to zoom. Axes: x "
                "along the target line, y to the left, z elevation; distances "
                "are physical metres."
            ),
            help_tooltip=(
                "Frames are the recorded integrator samples read off the same "
                "green surface the putt was solved on; nothing is "
                "re-simulated during playback."
            ),
            parent=parent,
        )

    @property
    def strike_button(self) -> QPushButton:
        """The jump-to-strike transport button (test seam)."""
        button: QPushButton = self.event_buttons[_EVENT_STRIKE]
        return button

    @property
    def finish_button(self) -> QPushButton:
        """The jump-to-finish transport button (test seam)."""
        button: QPushButton = self.event_buttons[_EVENT_FINISH]
        return button

    def jump_to_strike(self) -> None:
        """Pause at the launch sample."""
        self.jump_to_event(_EVENT_STRIKE)

    def jump_to_finish(self) -> None:
        """Pause at the terminal capture-or-rest sample."""
        self.jump_to_event(_EVENT_FINISH)


class PuttPlaybackPanel(QWidget):
    """Compose P6's 3-D green view with the shared transport controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = PuttPlaybackView()
        self.controls = PuttPlaybackControls()
        self.controls.timeChanged.connect(self.view.set_time)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.controls)

    def set_putt(
        self,
        result: PuttResult,
        surface: GreenSurface,
        *,
        hole_distance_m: float,
    ) -> None:
        """Adopt one integrated putt and its recorded transport timeline."""
        self.view.set_putt(result, surface, hole_distance_m=hole_distance_m)
        self.controls.set_transport_timeline(
            self.view.duration_s(), self.view.event_times_s()
        )

    def clear(self) -> None:
        """Drop the scene and collapse the transport to an empty timeline."""
        self.view.clear()
        self.controls.set_transport_timeline(0.0, (0.0, 0.0))


__all__ = ["PuttPlaybackControls", "PuttPlaybackPanel"]
