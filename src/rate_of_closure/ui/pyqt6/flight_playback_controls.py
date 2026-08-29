"""Ball-flight binding of the shared transport controls (#4800 P8).

All transport behavior — the single owned timer, scrub quantization,
speed multipliers, and wall-clock advance — lives in
:class:`~rate_of_closure.ui.pyqt6.playback_transport_controls.PlaybackTransportControls`
and the runtime-neutral :mod:`rate_of_closure.simulation.playback_transport`
model it delegates to. This module only binds the flight vocabulary:
Launch/Apex/Landing events and the ball-flight wording the accessibility
gates pin.
"""

from __future__ import annotations

import math

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.playback_transport_controls import (
    PlaybackTransportControls,
)

_EVENT_LAUNCH = 0
_EVENT_APEX = 1
_EVENT_LANDING = 2


class FlightPlaybackControls(PlaybackTransportControls):
    """Play, pause, scrub, restart, and jump through one flight timeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            subject_label="Ball Flight",
            subject_phrase="ball-flight",
            event_labels=("Launch", "Apex", "Landing"),
            scrub_tooltip=(
                "Scrub physical trajectory time [s] from launch to landing. "
                "Source: solver trajectory timestamps; positions use the app "
                "frame (x target, y up, z right) in metres."
            ),
            help_text=(
                "Drag the 3D plot to rotate; use the wheel to zoom. Axes: x "
                "target, y up, z right; distances are physical metres."
            ),
            help_tooltip=(
                "The 3D axes retain one physical scale per metre; display-unit "
                "labels do not distort the trajectory geometry."
            ),
            parent=parent,
        )

    @property
    def launch_button(self) -> QPushButton:
        """The jump-to-launch transport button (test seam)."""
        button: QPushButton = self.event_buttons[_EVENT_LAUNCH]
        return button

    @property
    def apex_button(self) -> QPushButton:
        """The jump-to-apex transport button (test seam)."""
        button: QPushButton = self.event_buttons[_EVENT_APEX]
        return button

    @property
    def landing_button(self) -> QPushButton:
        """The jump-to-landing transport button (test seam)."""
        button: QPushButton = self.event_buttons[_EVENT_LANDING]
        return button

    def set_timeline(self, duration_s: float, apex_time_s: float) -> None:
        """Adopt finite landing/apex event times and reset to launch."""
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and >= 0")
        if not math.isfinite(apex_time_s) or not 0.0 <= apex_time_s <= duration_s:
            raise ValueError("apex_time_s must be finite and within the timeline")
        self.set_transport_timeline(duration_s, (0.0, apex_time_s, duration_s))

    def jump_to_launch(self) -> None:
        """Pause at the launch sample."""
        self.jump_to_event(_EVENT_LAUNCH)

    def jump_to_apex(self) -> None:
        """Pause at the first maximum-height sample."""
        self.jump_to_event(_EVENT_APEX)

    def jump_to_landing(self) -> None:
        """Pause at the terminal ground-contact/landing sample."""
        self.jump_to_event(_EVENT_LANDING)


class FlightPlaybackPanel(QWidget):
    """Compose an existing flight view with its reusable transport controls."""

    def __init__(self, flight_view: FlightView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controls = FlightPlaybackControls()
        flight_view.timelineChanged.connect(self.controls.set_timeline)
        self.controls.timeChanged.connect(flight_view.set_playback_time)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(flight_view, stretch=1)
        layout.addWidget(self.controls)


__all__ = ["FlightPlaybackControls", "FlightPlaybackPanel"]
