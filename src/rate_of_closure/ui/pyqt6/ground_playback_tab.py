"""Standalone PyQt ground-result import and playback workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation.ground_playback import (
    DEFAULT_IMPORT_MAX_BYTES,
    GroundPlaybackTimeline,
    load_ground_result_json,
)
from rate_of_closure.ui.pyqt6.ground_playback_tables import (
    EVENT_HEADERS,
    TRAJECTORY_HEADERS,
    create_ground_table,
    populate_ground_tables,
)
from rate_of_closure.ui.pyqt6.ground_playback_view import GroundPlayback3DView

_SLIDER_STEPS = 10_000
_TIMER_INTERVAL_MS = 16
_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


class GroundPlaybackTab(QWidget):
    """Inspect one strict Python-generated ground result without simulating it."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._timeline: GroundPlaybackTimeline | None = None
        self.current_time_s = 0.0
        self._playback_anchor_wall_s: float | None = None
        self._playback_anchor_time_s = 0.0
        self._playback_speed = 1.0
        self._playback_loop = False
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(_TIMER_INTERVAL_MS)
        self.playback_timer.timeout.connect(self._advance)
        self._build_ui()
        self._set_controls_enabled(False)

    @property
    def timeline(self) -> GroundPlaybackTimeline:
        """Return the last successfully imported playback timeline."""
        if self._timeline is None:
            raise RuntimeError("no ground result has been imported")
        return self._timeline

    def import_json_text(self, text: str, *, source_name: str = "result JSON") -> None:
        """Atomically import strict text while retaining the last good result."""
        try:
            candidate = GroundPlaybackTimeline(load_ground_result_json(text))
        except (TypeError, ValueError) as exc:
            self.status_label.setText(f"Could not import {source_name}: {exc}")
            self.status_label.setProperty("state", "error")
            raise
        self.pause()
        self._timeline = candidate
        self.current_time_s = candidate.start_time_s
        self.view.set_timeline(candidate)
        populate_ground_tables(
            candidate,
            summary_table=self.summary_table,
            trajectory_table=self.trajectory_table,
            events_table=self.events_table,
            warnings_table=self.warnings_table,
        )
        for phase, button in self.phase_buttons.items():
            button.setVisible(candidate.phase_time(phase) is not None)
        self.end_button.setText(candidate.end_label)
        self._set_controls_enabled(True)
        self.set_time(candidate.start_time_s)
        self.status_label.setText(
            f"Loaded {source_name} — {candidate.result.status.value}; "
            f"{len(candidate.result.trajectory)} samples."
        )
        self.status_label.setProperty("state", "ready")

    def set_time(self, time_s: float) -> None:
        """Seek to a clamped absolute ground-result time."""
        if self._timeline is None:
            return
        frame = self._timeline.frame_at(time_s)
        self.current_time_s = frame.time_s
        duration = self._timeline.duration_s
        ratio = 0.0 if duration == 0.0 else frame.elapsed_s / duration
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(round(ratio * _SLIDER_STEPS))
        self.scrubber.blockSignals(False)
        self.phase_label.setText(frame.phase.title())
        self.time_label.setText(
            f"Absolute {frame.time_s:.4f} s · elapsed {frame.elapsed_s:.4f} s"
        )
        self.view.set_position(frame.position_m)

    def play(self) -> None:
        """Start or restart playback using exactly one owned timer."""
        if self._timeline is None:
            return
        if self.current_time_s >= self._timeline.end_time_s:
            self.set_time(self._timeline.start_time_s)
        self._playback_anchor_wall_s = self._clock()
        self._playback_anchor_time_s = self.current_time_s
        self._playback_speed = _SPEEDS[self.speed_combo.currentIndex()]
        self._playback_loop = self.loop_checkbox.isChecked()
        self.playback_timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        """Pause playback without changing the current frame."""
        if self.playback_timer.isActive():
            self._advance()
        self._stop_playback()

    def _stop_playback(self) -> None:
        """Stop the owned timer without performing a final time sample."""
        self.playback_timer.stop()
        self._playback_anchor_wall_s = None
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")

    def restart(self) -> None:
        """Return to exact first contact and pause."""
        self.pause()
        if self._timeline is not None:
            self.set_time(self._timeline.start_time_s)

    def previous_frame(self) -> None:
        """Seek to the previous exact trajectory sample."""
        if self._timeline is not None:
            self.pause()
            self.set_time(self._timeline.step_time(self.current_time_s, -1))

    def next_frame(self) -> None:
        """Seek to the next exact trajectory sample."""
        if self._timeline is not None:
            self.pause()
            self.set_time(self._timeline.step_time(self.current_time_s, 1))

    def jump_to_phase(self, phase: str) -> None:
        """Seek to the first exact sample for an available phase."""
        if self._timeline is None:
            return
        time_s = self._timeline.phase_time(phase)
        if time_s is not None:
            self.pause()
            self.set_time(time_s)

    def stop(self) -> None:
        """Stop timers and queued redraws during application teardown."""
        self.pause()
        self.view.stop()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.disclosure_label = QLabel(
            "Import a strict flight-to-ground-result/v1 JSON generated by the Python "
            "reference executor. This viewer does not execute ground physics."
        )
        self.disclosure_label.setWordWrap(True)
        self.disclosure_label.setObjectName("groundPlaybackDisclosure")
        root.addWidget(self.disclosure_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        self.view = GroundPlayback3DView()
        details = QSplitter(Qt.Orientation.Vertical)
        details.addWidget(self.view)
        details.addWidget(self._build_detail_tabs())
        details.setStretchFactor(0, 3)
        details.setStretchFactor(1, 2)
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 760])
        root.addWidget(splitter, 1)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        self.import_button = QPushButton("Import Ground Result JSON…")
        self.import_button.setAccessibleName("Import strict ground result JSON")
        self.import_button.setToolTip(
            "Open one exact flight-to-ground-result/v1 JSON record (maximum 5 MiB)."
        )
        self.import_button.clicked.connect(self._choose_result_file)
        layout.addWidget(self.import_button)
        self.status_label = QLabel("No result loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Ground result import status")
        layout.addWidget(self.status_label)
        layout.addWidget(self._build_playback_group())
        self.summary_table = create_ground_table(
            ("Metric", "Value"), "Ground result summary"
        )
        self.summary_table.setMaximumHeight(220)
        layout.addWidget(self.summary_table)
        self.geometry_note = QLabel(
            "Result v1 does not embed surface geometry. The view uses neutral axes "
            "and does not claim an exact terrain plane."
        )
        self.geometry_note.setWordWrap(True)
        layout.addWidget(self.geometry_note)
        layout.addStretch(1)
        return panel

    def _build_playback_group(self) -> QGroupBox:
        group = QGroupBox("Playback")
        layout = QGridLayout(group)
        self.first_button = QPushButton("First Contact")
        self.previous_button = QPushButton("−1 frame")
        self.play_button = QPushButton("Play")
        self.play_button.setAccessibleName("Play ground result")
        self.next_button = QPushButton("+1 frame")
        self.end_button = QPushButton("End")
        self.first_button.setToolTip(
            "Pause and seek to the exact first-contact sample."
        )
        self.previous_button.setToolTip(
            "Pause and seek to the previous exact trajectory sample."
        )
        self.play_button.setToolTip(
            "Play or pause the imported ground-result timeline."
        )
        self.next_button.setToolTip(
            "Pause and seek to the next exact trajectory sample."
        )
        self.end_button.setToolTip("Pause and seek to the final observed sample.")
        layout.addWidget(self.first_button, 0, 0)
        layout.addWidget(self.play_button, 0, 1)
        layout.addWidget(self.end_button, 0, 2)
        layout.addWidget(self.previous_button, 1, 0)
        layout.addWidget(self.next_button, 1, 1, 1, 2)
        self.first_button.clicked.connect(self.restart)
        self.previous_button.clicked.connect(self.previous_frame)
        self.play_button.clicked.connect(self._toggle_playback)
        self.next_button.clicked.connect(self.next_frame)
        self.end_button.clicked.connect(self._jump_to_end)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, _SLIDER_STEPS)
        self.scrubber.setAccessibleName("Ground playback timeline")
        self.scrubber.setToolTip(
            "Seek using absolute solver time from first contact to observed end."
        )
        self.scrubber.valueChanged.connect(self._seek_from_slider)
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
                lambda _checked=False, name=phase: self.jump_to_phase(name)
            )
            self.phase_buttons[phase] = button
            phase_row.addWidget(button)
        layout.addLayout(phase_row, 4, 0, 1, 3)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems([f"{speed:g}×" for speed in _SPEEDS])
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.setAccessibleName("Ground playback speed")
        self.speed_combo.setToolTip(
            "Set playback speed from quarter-speed to four-times real time."
        )
        self.speed_combo.currentIndexChanged.connect(self._change_speed)
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setToolTip(
            "Restart at first contact after the observed end."
        )
        self.loop_checkbox.toggled.connect(self._change_loop)
        reset_view = QPushButton("Reset 3D view")
        reset_view.setToolTip(
            "Restore the documented camera and auto-fit locked physical axes."
        )
        reset_view.clicked.connect(lambda: self.view.reset_view())
        layout.addWidget(self.speed_combo, 5, 0)
        layout.addWidget(self.loop_checkbox, 5, 1)
        layout.addWidget(reset_view, 5, 2)
        return group

    def _build_detail_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.trajectory_table = create_ground_table(
            TRAJECTORY_HEADERS,
            "Ground trajectory samples",
        )
        self.events_table = create_ground_table(
            EVENT_HEADERS,
            "Ground events",
        )
        self.warnings_table = create_ground_table(
            ("Severity", "Code", "Message"), "Warnings and provenance"
        )
        tabs.addTab(self.trajectory_table, "Trajectory")
        tabs.addTab(self.events_table, "Events")
        tabs.addTab(self.warnings_table, "Warnings / provenance")
        return tabs

    def _set_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.first_button,
            self.previous_button,
            self.play_button,
            self.next_button,
            self.end_button,
            self.scrubber,
            self.speed_combo,
            self.loop_checkbox,
        ):
            control.setEnabled(enabled)
        for button in self.phase_buttons.values():
            button.setEnabled(enabled)

    def _choose_result_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import Ground Result", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            file_path = Path(path)
            if file_path.stat().st_size > DEFAULT_IMPORT_MAX_BYTES:
                raise ValueError("ground result JSON exceeds the import size limit")
            self.import_json_text(
                file_path.read_text(encoding="utf-8"), source_name=file_path.name
            )
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            self.status_label.setText(f"Could not import {Path(path).name}: {exc}")
            self.status_label.setProperty("state", "error")

    def _toggle_playback(self) -> None:
        self.pause() if self.playback_timer.isActive() else self.play()

    def _seek_from_slider(self, value: int) -> None:
        if self._timeline is None:
            return
        self.pause()
        self.set_time(
            self._timeline.start_time_s
            + self._timeline.duration_s * value / _SLIDER_STEPS
        )

    def _jump_to_end(self) -> None:
        if self._timeline is not None:
            self.pause()
            self.set_time(self._timeline.end_time_s)

    def _advance(self) -> None:
        timeline = self._timeline
        if timeline is None:
            self._stop_playback()
            return
        if not self.playback_timer.isActive():
            return
        if self._playback_anchor_wall_s is None:
            self._playback_anchor_wall_s = self._clock()
            self._playback_anchor_time_s = self.current_time_s
            return
        wall_elapsed_s = max(0.0, self._clock() - self._playback_anchor_wall_s)
        candidate = self._playback_anchor_time_s + wall_elapsed_s * self._playback_speed
        if self._playback_loop and timeline.duration_s > 0.0:
            offset_s = (candidate - timeline.start_time_s) % timeline.duration_s
            self.set_time(timeline.start_time_s + offset_s)
        elif candidate < timeline.end_time_s:
            self.set_time(candidate)
        else:
            self.set_time(timeline.end_time_s)
            self._stop_playback()

    def _change_speed(self, index: int) -> None:
        """Re-anchor active playback when its wall-clock multiplier changes."""
        new_speed = _SPEEDS[index]
        if self.playback_timer.isActive():
            self._advance()
            if self.playback_timer.isActive():
                self._playback_anchor_wall_s = self._clock()
                self._playback_anchor_time_s = self.current_time_s
        self._playback_speed = new_speed

    def _change_loop(self, enabled: bool) -> None:
        """Re-anchor active playback before changing end-of-run semantics."""
        if self.playback_timer.isActive():
            self._advance()
            if self.playback_timer.isActive():
                self._playback_anchor_wall_s = self._clock()
                self._playback_anchor_time_s = self.current_time_s
        self._playback_loop = enabled


__all__ = ["GroundPlaybackTab"]
