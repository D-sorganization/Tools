"""Subject-neutral, lifecycle-safe transport controls for 3D shot playback.

The single Qt implementation of the #4800 P8 playback architecture:
play/pause, restart, event jumps, scrub, and speed for one recorded
timeline. Every semantic decision — scrub quantization, speed set, and
wall-clock advance — delegates to
:mod:`rate_of_closure.simulation.playback_transport`, whose TypeScript
twin drives the React surfaces, so both runtimes share one timeline
model.

The putting seam (#4800 P6/P7): this widget carries no flight
vocabulary. The flight binding
(:class:`~rate_of_closure.ui.pyqt6.flight_playback_controls.FlightPlaybackControls`)
instantiates it with "Ball Flight" wording and Launch/Apex/Landing
events; the putting vertical passes its own wording and events (for
example Strike/Holed) and consumes this class unchanged.

Camera seam (#4571): camera state stays with
``rate_of_closure.application.camera_commands`` and the viewport mixins;
this widget owns only the timeline and its sole animation timer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from PyQt6.QtCore import QElapsedTimer, QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation.playback_transport import (
    DEFAULT_SPEED,
    PLAYBACK_SPEEDS,
    SCRUB_STEPS,
    advance_playback,
    scrub_value,
    time_at_scrub,
)

_FRAME_INTERVAL_MS = 33


class PlaybackTransportControls(QWidget):
    """Play, pause, scrub, restart, and jump through one recorded timeline."""

    timeChanged = pyqtSignal(float)  # noqa: N815 - Qt signal convention

    def __init__(
        self,
        *,
        subject_label: str,
        subject_phrase: str,
        event_labels: Sequence[str],
        scrub_tooltip: str,
        help_text: str,
        help_tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not subject_label or not subject_phrase:
            raise ValueError("subject_label and subject_phrase must be non-empty")
        if not event_labels:
            raise ValueError("event_labels must name at least one jump event")
        self._subject_label = subject_label
        self._subject_phrase = subject_phrase
        self._event_labels = tuple(event_labels)
        self._duration_s = 0.0
        self._event_times_s: tuple[float, ...] = tuple(0.0 for _ in self._event_labels)
        self._current_time_s = 0.0
        self._speed = DEFAULT_SPEED
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance)
        self._build_ui(scrub_tooltip, help_text, help_tooltip)

    def _build_ui(self, scrub_tooltip: str, help_text: str, help_tooltip: str) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(4, 3, 4, 0)
        first_event = self._event_button(0)
        self.play_button = self._button("Play", self._toggle)
        self.play_button.setAccessibleName(f"Play or Pause {self._subject_label}")
        self.restart_button = self._button("Restart", self.restart)
        later_events = tuple(
            self._event_button(index) for index in range(1, len(self._event_labels))
        )
        self.event_buttons: tuple[QPushButton, ...] = (first_event, *later_events)
        for button in (
            first_event,
            self.play_button,
            self.restart_button,
            *later_events,
        ):
            row.addWidget(button)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, SCRUB_STEPS)
        self.scrubber.setAccessibleName(f"{self._subject_label} Time")
        self.scrubber.setToolTip(scrub_tooltip)
        self.scrubber.valueChanged.connect(self._scrub)
        row.addWidget(self.scrubber, stretch=1)
        self.speed_combo = QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(PLAYBACK_SPEEDS.index(DEFAULT_SPEED))
        self.speed_combo.setAccessibleName("Playback Speed")
        self.speed_combo.setToolTip(
            "Playback rate multiplier; physical timestamps are unchanged."
        )
        self.speed_combo.currentIndexChanged.connect(self._change_speed)
        row.addWidget(self.speed_combo)
        self.time_label = QLabel("0.00 / 0.00 s")
        self.time_label.setMinimumWidth(94)
        self.time_label.setAccessibleName(f"{self._subject_label} Playback Time")
        row.addWidget(self.time_label)
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setToolTip(help_tooltip)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(help_label)
        self.set_transport_timeline(0.0, self._event_times_s)

    def _button(self, label: str, callback: Callable[[], None]) -> QPushButton:
        """Build one sourced transport button."""
        button = QPushButton(label)
        button.clicked.connect(callback)
        button.setToolTip(
            f"{label} {self._subject_phrase} playback using physical "
            "trajectory time [s]."
        )
        return button

    def _event_button(self, index: int) -> QPushButton:
        return self._button(
            self._event_labels[index], lambda: self.jump_to_event(index)
        )

    def set_transport_timeline(
        self, duration_s: float, event_times_s: Sequence[float]
    ) -> None:
        """Adopt finite event times on ``[0, duration]`` and reset to the start."""
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and >= 0")
        events = tuple(float(time_s) for time_s in event_times_s)
        if len(events) != len(self._event_labels):
            raise ValueError("event_times_s must provide one time per event label")
        for time_s in events:
            if not math.isfinite(time_s) or not 0.0 <= time_s <= duration_s:
                raise ValueError("event times must be finite and within the timeline")
        self.pause()
        self._duration_s = float(duration_s)
        self._event_times_s = events
        self._set_time(0.0)
        enabled = duration_s > 0.0
        for control in (self.play_button, self.restart_button, *self.event_buttons):
            control.setEnabled(enabled)

    def play(self) -> None:
        """Start the single owned timer; repeated calls are idempotent."""
        if self._duration_s <= 0.0 or self._timer.isActive():
            return
        if self._current_time_s >= self._duration_s:
            self._set_time(0.0)
        self._elapsed.start()
        self._timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        """Stop playback without changing the displayed physical time."""
        self._timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")

    def restart(self) -> None:
        """Jump to the timeline start and play from the beginning."""
        self.pause()
        self._set_time(0.0)
        self.play()

    def jump_to_event(self, index: int) -> None:
        """Pause at one recorded event timestamp by event position."""
        if not 0 <= index < len(self._event_times_s):
            raise ValueError("event index out of range")
        self.pause()
        self._set_time(self._event_times_s[index])

    def jump_to_time(self, time_s: float) -> None:
        """Pause at one exact accepted solver timestamp."""
        if not math.isfinite(time_s):
            raise ValueError("playback time must be finite")
        self.pause()
        self._set_time(time_s)

    def current_time_s(self) -> float:
        """Current physical playback time [s]."""
        return self._current_time_s

    def duration_s(self) -> float:
        """Adopted timeline duration [s]."""
        return self._duration_s

    def timer(self) -> QTimer:
        """The sole animation timer (test and lifecycle inspection seam)."""
        return self._timer

    def _toggle(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    def _advance(self) -> None:
        elapsed_s = self._elapsed.restart() / 1000.0
        step = advance_playback(
            self._current_time_s, elapsed_s, self._speed, self._duration_s
        )
        self._set_time(step.time_s)
        if step.finished:
            self.pause()

    def _scrub(self, value: int) -> None:
        self.pause()
        self._set_time(time_at_scrub(value, self._duration_s), update_slider=False)

    def _change_speed(self) -> None:
        self._speed = float(self.speed_combo.currentData())
        if self._timer.isActive():
            self._elapsed.restart()

    def _set_time(self, time_s: float, *, update_slider: bool = True) -> None:
        self._current_time_s = min(max(time_s, 0.0), self._duration_s)
        if update_slider:
            with QSignalBlocker(self.scrubber):
                self.scrubber.setValue(
                    scrub_value(self._current_time_s, self._duration_s)
                )
        self.time_label.setText(
            f"{self._current_time_s:.2f} / {self._duration_s:.2f} s"
        )
        self.timeChanged.emit(self._current_time_s)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Stop the owned timer before Qt destroys the widget."""
        self.pause()
        super().closeEvent(event)


__all__ = ["PlaybackTransportControls"]
