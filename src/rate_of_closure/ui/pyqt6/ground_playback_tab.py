"""Standalone PyQt ground-result import and playback workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation.ground_playback import (
    DEFAULT_IMPORT_MAX_BYTES,
    GroundPlaybackTimeline,
    load_ground_result_json,
    timeline_from_regional_execution,
)
from rate_of_closure.ui.pyqt6.ground_playback_controls import (
    PLAYBACK_SPEEDS,
    SLIDER_STEPS,
    GroundPlaybackActions,
    GroundPlaybackControls,
)
from rate_of_closure.ui.pyqt6.ground_playback_tables import (
    EVENT_HEADERS,
    TRAJECTORY_HEADERS,
    create_ground_table,
    populate_ground_tables,
)
from rate_of_closure.ui.pyqt6.ground_playback_view import GroundPlayback3DView
from shared.python.swing_sim.ground import (
    MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES,
    regional_ground_execution_result_from_json,
)

_TIMER_INTERVAL_MS = 16


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
        self._playback_controls.set_playback_enabled(False)

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
        self._accept_timeline(candidate, source_name)

    def import_regional_json_text(
        self, text: str, *, source_name: str = "regional execution JSON"
    ) -> None:
        """Import a strict regional envelope and reuse its nested ground result."""
        try:
            execution = regional_ground_execution_result_from_json(text)
            candidate = timeline_from_regional_execution(execution)
        except (TypeError, ValueError) as exc:
            self.status_label.setText(f"Could not import {source_name}: {exc}")
            self.status_label.setProperty("state", "error")
            raise
        self._accept_timeline(candidate, source_name)

    def _accept_timeline(
        self, candidate: GroundPlaybackTimeline, source_name: str
    ) -> None:
        """Atomically install one validated timeline in every playback surface."""
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
        self._playback_controls.set_playback_enabled(True)
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
        self.scrubber.setValue(round(ratio * SLIDER_STEPS))
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
        self._playback_speed = PLAYBACK_SPEEDS[self.speed_combo.currentIndex()]
        self._playback_loop = self.loop_checkbox.isChecked()
        self.playback_timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        """Pause playback without changing the current frame."""
        if self.playback_timer.isActive():
            self._advance()
        self._stop_playback()

    def _stop_playback(self) -> None:
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
            "Import a strict flight-to-ground-result/v1 JSON or validated "
            "ground-regional-execution-result/v1 JSON. This viewer reuses existing "
            "evidence and does not execute ground physics."
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
        self.regional_import_button = QPushButton("Import Regional Execution JSON…")
        self.regional_import_button.setAccessibleName(
            "Import strict regional ground execution JSON"
        )
        self.regional_import_button.setToolTip(
            "Open one exact ground-regional-execution-result/v1 envelope."
        )
        self.regional_import_button.clicked.connect(self._choose_regional_file)
        layout.addWidget(self.regional_import_button)
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

    def _build_playback_group(self) -> GroundPlaybackControls:
        actions = GroundPlaybackActions(
            restart=self.restart,
            previous=self.previous_frame,
            toggle=self._toggle_playback,
            next=self.next_frame,
            end=self._jump_to_end,
            jump_phase=self.jump_to_phase,
            change_speed=self._change_speed,
            change_loop=self._change_loop,
            reset_view=lambda: self.view.reset_view(),
        )
        controls = GroundPlaybackControls(actions)
        self._playback_controls = controls
        self.first_button = controls.first_button
        self.previous_button = controls.previous_button
        self.play_button = controls.play_button
        self.next_button = controls.next_button
        self.end_button = controls.end_button
        self.scrubber = controls.scrubber
        self.phase_label = controls.phase_label
        self.time_label = controls.time_label
        self.phase_buttons = controls.phase_buttons
        self.speed_combo = controls.speed_combo
        self.loop_checkbox = controls.loop_checkbox
        self.scrubber.valueChanged.connect(self._seek_from_slider)
        return controls

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

    def _choose_result_file(self) -> None:
        self._choose_file(regional=False)

    def _choose_regional_file(self) -> None:
        self._choose_file(regional=True)

    def _choose_file(self, *, regional: bool) -> None:
        title = (
            "Import Regional Ground Execution" if regional else "Import Ground Result"
        )
        path, _filter = QFileDialog.getOpenFileName(
            self, title, "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            file_path = Path(path)
            limit = (
                MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES
                if regional
                else DEFAULT_IMPORT_MAX_BYTES
            )
            if file_path.stat().st_size > limit:
                raise ValueError("selected JSON exceeds the import size limit")
            text = file_path.read_text(encoding="utf-8")
            importer = (
                self.import_regional_json_text if regional else self.import_json_text
            )
            importer(text, source_name=file_path.name)
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
            + self._timeline.duration_s * value / SLIDER_STEPS
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
        new_speed = PLAYBACK_SPEEDS[index]
        if self.playback_timer.isActive():
            self._advance()
            if self.playback_timer.isActive():
                self._playback_anchor_wall_s = self._clock()
                self._playback_anchor_time_s = self.current_time_s
        self._playback_speed = new_speed

    def _change_loop(self, enabled: bool) -> None:
        if self.playback_timer.isActive():
            self._advance()
            if self.playback_timer.isActive():
                self._playback_anchor_wall_s = self._clock()
                self._playback_anchor_time_s = self.current_time_s
        self._playback_loop = enabled


__all__ = ["GroundPlaybackTab"]
