"""Focused playback-control construction for the PyQt ground viewer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
)

PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
SLIDER_STEPS = 10_000


@dataclass(frozen=True)
class GroundPlaybackActions:
    """Callbacks owned by the playback tab and invoked by this view."""

    restart: Callable[[], None]
    previous: Callable[[], None]
    toggle: Callable[[], None]
    next: Callable[[], None]
    end: Callable[[], None]
    jump_phase: Callable[[str], None]
    change_speed: Callable[[int], None]
    change_loop: Callable[[bool], None]
    reset_view: Callable[[], None]


class GroundPlaybackControls(QGroupBox):
    """Accessible controls for deterministic ground-result playback."""

    def __init__(self, actions: GroundPlaybackActions) -> None:
        super().__init__("Playback")
        layout = QGridLayout(self)
        self.first_button = QPushButton("First Contact")
        self.previous_button = QPushButton("−1 frame")
        self.play_button = QPushButton("Play")
        self.play_button.setAccessibleName("Play ground result")
        self.next_button = QPushButton("+1 frame")
        self.end_button = QPushButton("End")
        tooltips = (
            (self.first_button, "Pause and seek to the exact first-contact sample."),
            (
                self.previous_button,
                "Pause and seek to the previous exact trajectory sample.",
            ),
            (self.play_button, "Play or pause the imported ground-result timeline."),
            (self.next_button, "Pause and seek to the next exact trajectory sample."),
            (self.end_button, "Pause and seek to the final observed sample."),
        )
        for button, tooltip in tooltips:
            button.setToolTip(tooltip)
        layout.addWidget(self.first_button, 0, 0)
        layout.addWidget(self.play_button, 0, 1)
        layout.addWidget(self.end_button, 0, 2)
        layout.addWidget(self.previous_button, 1, 0)
        layout.addWidget(self.next_button, 1, 1, 1, 2)
        self.first_button.clicked.connect(actions.restart)
        self.previous_button.clicked.connect(actions.previous)
        self.play_button.clicked.connect(actions.toggle)
        self.next_button.clicked.connect(actions.next)
        self.end_button.clicked.connect(actions.end)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, SLIDER_STEPS)
        self.scrubber.setAccessibleName("Ground playback timeline")
        self.scrubber.setToolTip(
            "Seek using absolute solver time from first contact to observed end."
        )
        layout.addWidget(self.scrubber, 2, 0, 1, 3)
        self.phase_label = QLabel("No phase")
        self.phase_label.setAccessibleName("Current ground phase")
        self.time_label = QLabel("Absolute — · elapsed —")
        layout.addWidget(self.phase_label, 3, 0)
        layout.addWidget(self.time_label, 3, 1, 1, 2)
        phase_row = QHBoxLayout()
        self.phase_buttons: dict[str, QPushButton] = {}
        for phase in ("bounce", "skid", "roll"):
            button = QPushButton(phase.title())
            button.setToolTip(
                f"Pause and seek to the first exact {phase} sample, when available."
            )
            button.clicked.connect(
                lambda _checked=False, name=phase: actions.jump_phase(name)
            )
            self.phase_buttons[phase] = button
            phase_row.addWidget(button)
        layout.addLayout(phase_row, 4, 0, 1, 3)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems([f"{speed:g}×" for speed in PLAYBACK_SPEEDS])
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.setAccessibleName("Ground playback speed")
        self.speed_combo.setToolTip(
            "Set playback speed from quarter-speed to four-times real time."
        )
        self.speed_combo.currentIndexChanged.connect(actions.change_speed)
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setToolTip(
            "Restart at first contact after the observed end."
        )
        self.loop_checkbox.toggled.connect(actions.change_loop)
        reset_view = QPushButton("Reset 3D view")
        reset_view.setToolTip(
            "Restore the documented camera and auto-fit locked physical axes."
        )
        reset_view.clicked.connect(actions.reset_view)
        layout.addWidget(self.speed_combo, 5, 0)
        layout.addWidget(self.loop_checkbox, 5, 1)
        layout.addWidget(reset_view, 5, 2)

    def set_playback_enabled(self, enabled: bool) -> None:
        """Enable or disable every control that requires a loaded timeline."""
        controls = (
            self.first_button,
            self.previous_button,
            self.play_button,
            self.next_button,
            self.end_button,
            self.scrubber,
            self.speed_combo,
            self.loop_checkbox,
            *self.phase_buttons.values(),
        )
        for control in controls:
            control.setEnabled(enabled)


__all__ = [
    "GroundPlaybackActions",
    "GroundPlaybackControls",
    "PLAYBACK_SPEEDS",
    "SLIDER_STEPS",
]
