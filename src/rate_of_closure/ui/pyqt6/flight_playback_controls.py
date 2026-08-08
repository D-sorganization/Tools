"""Accessible, lifecycle-safe transport controls for ball-flight playback."""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QElapsedTimer, QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.flight_view import FlightView

_SLIDER_STEPS = 10_000
_FRAME_INTERVAL_MS = 33
_SPEEDS: tuple[tuple[str, float], ...] = (
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
)


class FlightPlaybackControls(QWidget):
    """Play, pause, scrub, restart, and jump through one flight timeline."""

    timeChanged = pyqtSignal(float)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration_s = 0.0
        self._apex_time_s = 0.0
        self._current_time_s = 0.0
        self._speed = 1.0
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance)
        self._build_ui()

    def _build_ui(self) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(4, 3, 4, 0)
        self.launch_button = self._button("Launch", self.jump_to_launch)
        self.play_button = self._button("Play", self._toggle)
        self.restart_button = self._button("Restart", self.restart)
        self.apex_button = self._button("Apex", self.jump_to_apex)
        self.landing_button = self._button("Landing", self.jump_to_landing)
        for button in (
            self.launch_button,
            self.play_button,
            self.restart_button,
            self.apex_button,
            self.landing_button,
        ):
            row.addWidget(button)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, _SLIDER_STEPS)
        self.scrubber.setAccessibleName("Ball Flight Time")
        self.scrubber.setToolTip(
            "Scrub physical trajectory time [s] from launch to landing. "
            "Source: solver trajectory timestamps; positions use the app frame "
            "(x target, y up, z right) in metres."
        )
        self.scrubber.valueChanged.connect(self._scrub)
        row.addWidget(self.scrubber, stretch=1)
        self.speed_combo = QComboBox()
        for label, speed in _SPEEDS:
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.setAccessibleName("Playback Speed")
        self.speed_combo.setToolTip(
            "Playback rate multiplier; physical timestamps are unchanged."
        )
        self.speed_combo.currentIndexChanged.connect(self._change_speed)
        row.addWidget(self.speed_combo)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setAccessibleName("Loop Ball Flight Playback")
        self.loop_check.setToolTip("Restart at launch when the flight reaches landing.")
        row.addWidget(self.loop_check)
        self.time_label = QLabel("0.00 / 0.00 s")
        self.time_label.setMinimumWidth(94)
        self.time_label.setAccessibleName("Ball Flight Playback Time")
        row.addWidget(self.time_label)
        help_label = QLabel(
            "Drag the 3D plot to rotate; use the wheel to zoom. Axes: x target, "
            "y up, z right; distances are physical metres."
        )
        help_label.setWordWrap(True)
        help_label.setToolTip(
            "The 3D axes retain one physical scale per metre; display-unit labels "
            "do not distort the trajectory geometry."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(help_label)

    @staticmethod
    def _button(label: str, callback: Callable[[], None]) -> QPushButton:
        """Build one sourced transport button."""
        button = QPushButton(label)
        button.clicked.connect(callback)
        button.setToolTip(
            f"{label} ball-flight playback using physical trajectory time [s]."
        )
        return button

    def set_timeline(self, duration_s: float, apex_time_s: float) -> None:
        """Adopt finite landing/apex event times and reset to launch."""
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and >= 0")
        if not math.isfinite(apex_time_s) or not 0.0 <= apex_time_s <= duration_s:
            raise ValueError("apex_time_s must be finite and within the timeline")
        self.pause()
        self._duration_s = float(duration_s)
        self._apex_time_s = float(apex_time_s)
        self._set_time(0.0)
        enabled = duration_s > 0.0
        for control in (
            self.play_button,
            self.restart_button,
            self.apex_button,
            self.landing_button,
        ):
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
        self.play_button.setAccessibleName("Play or Pause Ball Flight")

    def pause(self) -> None:
        """Stop playback without changing the displayed physical time."""
        self._timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")
            self.play_button.setAccessibleName("Play or Pause Ball Flight")

    def restart(self) -> None:
        """Jump to launch and play from the beginning."""
        self.pause()
        self._set_time(0.0)
        self.play()

    def jump_to_launch(self) -> None:
        """Pause at the launch sample."""
        self.pause()
        self._set_time(0.0)

    def jump_to_apex(self) -> None:
        """Pause at the first maximum-height sample."""
        self.pause()
        self._set_time(self._apex_time_s)

    def jump_to_landing(self) -> None:
        """Pause at the terminal ground-contact/landing sample."""
        self.pause()
        self._set_time(self._duration_s)

    def current_time_s(self) -> float:
        """Current physical playback time [s]."""
        return self._current_time_s

    def timer(self) -> QTimer:
        """The sole animation timer (test and lifecycle inspection seam)."""
        return self._timer

    def set_looping(self, looping: bool) -> None:
        """Set whether playback wraps from landing back to launch."""
        self.loop_check.setChecked(looping)

    def _toggle(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    def _advance(self) -> None:
        next_time_s = self._current_time_s + self._elapsed_seconds() * self._speed
        if next_time_s < self._duration_s:
            self._set_time(next_time_s)
            return
        if self.loop_check.isChecked() and self._duration_s > 0.0:
            self._set_time(next_time_s % self._duration_s)
            return
        self._set_time(self._duration_s)
        self.pause()

    def _elapsed_seconds(self) -> float:
        """Return elapsed wall time through a deterministic test seam."""
        return self._elapsed.restart() / 1000.0

    def _scrub(self, value: int) -> None:
        self.pause()
        fraction = value / _SLIDER_STEPS
        self._set_time(fraction * self._duration_s, update_slider=False)

    def _change_speed(self) -> None:
        self._speed = float(self.speed_combo.currentData())
        if self._timer.isActive():
            self._elapsed.restart()

    def _set_time(self, time_s: float, *, update_slider: bool = True) -> None:
        self._current_time_s = min(max(time_s, 0.0), self._duration_s)
        if update_slider:
            value = (
                round(_SLIDER_STEPS * self._current_time_s / self._duration_s)
                if self._duration_s > 0.0
                else 0
            )
            with QSignalBlocker(self.scrubber):
                self.scrubber.setValue(value)
        self.time_label.setText(
            f"{self._current_time_s:.2f} / {self._duration_s:.2f} s"
        )
        self.timeChanged.emit(self._current_time_s)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Stop the owned timer before Qt destroys the widget."""
        self.pause()
        super().closeEvent(event)


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
