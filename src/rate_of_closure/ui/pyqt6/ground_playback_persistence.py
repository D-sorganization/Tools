"""Persistence behavior shared by the PyQt ground playback tab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PyQt6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from rate_of_closure.simulation.ground_playback import (
    DEFAULT_IMPORT_MAX_BYTES,
    GroundPlaybackTimeline,
    load_ground_result_json,
)
from rate_of_closure.simulation.ground_playback_workspace import (
    GroundPlaybackState,
    GroundPlaybackViewState,
    GroundPlaybackWorkspace,
    ground_event_csv,
    ground_result_json,
    ground_trajectory_csv,
    ground_workspace_from_json,
    ground_workspace_to_json,
)
from rate_of_closure.ui.pyqt6.ground_playback_persistence_controls import (
    GroundPlaybackPersistenceControls,
)
from rate_of_closure.ui.pyqt6.ground_playback_tables import populate_ground_tables

_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


class GroundPlaybackPersistenceMixin:
    """Atomic imports and deterministic exports for a playback tab host."""

    _timeline: GroundPlaybackTimeline | None
    current_time_s: float
    status_label: Any
    view: Any
    summary_table: Any
    trajectory_table: Any
    events_table: Any
    warnings_table: Any
    phase_buttons: dict[str, Any]
    end_button: Any
    speed_combo: Any
    loop_checkbox: Any
    persistence_controls: GroundPlaybackPersistenceControls
    pause: Callable[[], None]
    set_time: Callable[[float], None]
    _set_controls_enabled: Callable[[bool], None]

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
            self._report_file_error(f"Could not import {source_name}: {exc}")
            raise
        self._apply_timeline(candidate, source_name)

    def import_workspace_json_text(
        self, text: str, *, source_name: str = "workspace JSON"
    ) -> None:
        """Atomically restore one strict workspace while retaining last-good state."""
        try:
            workspace = ground_workspace_from_json(text)
            candidate = GroundPlaybackTimeline(workspace.result)
        except (TypeError, ValueError) as exc:
            self._report_file_error(f"Could not import {source_name}: {exc}")
            raise
        self._apply_timeline(candidate, f"workspace {source_name}")
        self.speed_combo.setCurrentText(f"{workspace.playback.speed:g}×")
        self.loop_checkbox.setChecked(workspace.playback.loop)
        self.set_time(workspace.playback.time_s)
        self.view.apply_workspace_view(
            yaw_deg=workspace.view.yaw_deg,
            pitch_deg=workspace.view.pitch_deg,
            zoom=workspace.view.zoom,
        )

    def _apply_timeline(
        self, candidate: GroundPlaybackTimeline, source_name: str
    ) -> None:
        """Commit one fully validated candidate to every dependent view."""
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

    def workspace_json(self) -> str:
        """Pause and serialize the exact result plus portable UI state."""
        self.pause()
        view = self.view.workspace_view()
        workspace = GroundPlaybackWorkspace(
            result=self.timeline.result,
            playback=GroundPlaybackState(
                self.current_time_s,
                _SPEEDS[self.speed_combo.currentIndex()],
                self.loop_checkbox.isChecked(),
            ),
            view=GroundPlaybackViewState(view.yaw_deg, view.pitch_deg, view.zoom),
        )
        document: str = ground_workspace_to_json(workspace)
        return document

    def result_json(self) -> str:
        """Return the canonical loaded strict result JSON."""
        document: str = ground_result_json(self.timeline.result)
        return document

    def trajectory_csv(self) -> str:
        """Return every loaded trajectory field as deterministic CSV."""
        document: str = ground_trajectory_csv(self.timeline.result)
        return document

    def event_csv(self) -> str:
        """Return every loaded event field as deterministic CSV."""
        document: str = ground_event_csv(self.timeline.result)
        return document

    def attach_persistence_controls(self, panel: QWidget, layout: QVBoxLayout) -> None:
        """Attach accessible workspace and evidence file controls."""
        self.persistence_controls = GroundPlaybackPersistenceControls(
            panel,
            import_workspace=lambda text, name: self.import_workspace_json_text(
                text, source_name=name
            ),
            exports={
                "workspace": ("ground-playback-workspace.json", self.workspace_json),
                "result": ("ground-result.json", self.result_json),
                "trajectory": ("ground-trajectory.csv", self.trajectory_csv),
                "events": ("ground-events.csv", self.event_csv),
            },
            report_error=self._report_file_error,
        )
        self.import_workspace_button = self.persistence_controls.import_workspace_button
        buttons = self.persistence_controls.export_buttons
        self.save_workspace_button = buttons["workspace"]
        self.export_result_button = buttons["result"]
        self.export_trajectory_button = buttons["trajectory"]
        self.export_events_button = buttons["events"]
        layout.addWidget(self.persistence_controls)

    def _report_file_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("state", "error")

    def _choose_result_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            cast(QWidget, self), "Import Ground Result", "", "JSON files (*.json)"
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
            self._report_file_error(f"Could not import {Path(path).name}: {exc}")


__all__ = ["GroundPlaybackPersistenceMixin"]
