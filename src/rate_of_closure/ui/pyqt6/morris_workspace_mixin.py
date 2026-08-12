"""Lossless workspace persistence actions for the PyQt Morris tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QWidget

from rate_of_closure.application.morris._response_types import MorrisResponseJob
from rate_of_closure.application.morris.request_document import (
    CANONICAL_MORRIS_FACTOR_KEYS,
    base_document,
)
from rate_of_closure.application.morris.workspace import (
    MORRIS_WORKSPACE_EXPORT_SCOPE,
    MORRIS_WORKSPACE_SCHEMA_ID,
    MorrisCompletedEvidence,
    MorrisWorkspace,
    MorrisWorkspaceFactorDraft,
    MorrisWorkspaceSetup,
    loads_morris_workspace,
    morris_workspace_dict,
    parse_morris_workspace,
    write_morris_csv,
    write_morris_workspace,
)
from rate_of_closure.application.morris.workspace_validation import request_from_setup
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.morris_factor_row import MorrisFactorEditor


class MorrisWorkspaceMixin:
    """Separate file I/O and atomic restoration from run orchestration."""

    _config: SimulationConfig
    _factor_rows: list[MorrisFactorEditor]
    _last_job: MorrisResponseJob | None
    _status: QLabel
    _workspace_detached_drafts: dict[str, MorrisWorkspaceFactorDraft]

    def _invalidate_active_run(self) -> None:
        raise NotImplementedError

    def _populate_targets(self, job: MorrisResponseJob) -> None:
        raise NotImplementedError

    def _set_running(self, running: bool) -> None:
        raise NotImplementedError

    def _run_workers_running(self) -> bool:
        raise NotImplementedError

    def _workspace_setup(self) -> MorrisWorkspaceSetup:
        visible = {row.variable_key: row.workspace_draft() for row in self._factor_rows}
        detached = getattr(self, "_workspace_detached_drafts", {})
        tee_key = CANONICAL_MORRIS_FACTOR_KEYS[-1]
        fallback = MorrisWorkspaceFactorDraft(
            tee_key,
            False,
            "0.0",
            "0.006",
            None,
        )
        drafts = tuple(
            visible.get(key) or detached.get(key) or fallback
            for key in CANONICAL_MORRIS_FACTOR_KEYS
        )
        return MorrisWorkspaceSetup(
            MORRIS_WORKSPACE_EXPORT_SCOPE,
            base_document(self._config),
            drafts,
            self._trajectories.value(),  # type: ignore[attr-defined]
            self._levels.value(),  # type: ignore[attr-defined]
            self._seed.value(),  # type: ignore[attr-defined]
            self._minimum_effects.value(),  # type: ignore[attr-defined]
            self._workers.value(),  # type: ignore[attr-defined]
        )

    def workspace_document(self) -> MorrisWorkspace:
        """Capture controls and completed evidence without runtime credentials."""
        setup = self._workspace_setup()
        evidence = None
        if self._last_job is not None and self._last_job.report is not None:
            request = request_from_setup(setup, self._last_job.request_id)
            if request.total_samples == self._last_job.total_samples:
                evidence = MorrisCompletedEvidence(request, self._last_job)
        document = MorrisWorkspace(MORRIS_WORKSPACE_SCHEMA_ID, 1, setup, evidence)
        return parse_morris_workspace(morris_workspace_dict(document))

    def _preflight_workspace_ui(self, workspace: MorrisWorkspace) -> None:
        """Reject values that cannot be restored exactly before cancellation."""
        setup = workspace.setup
        controls = (
            (self._trajectories, setup.trajectories),  # type: ignore[attr-defined]
            (self._levels, setup.levels),  # type: ignore[attr-defined]
            (self._seed, setup.seed),  # type: ignore[attr-defined]
            (self._minimum_effects, setup.minimum_effects),  # type: ignore[attr-defined]
            (self._workers, setup.worker_count),  # type: ignore[attr-defined]
        )
        if any(
            not editor.minimum() <= value <= editor.maximum()
            for editor, value in controls
        ):
            raise ValueError("Morris workspace control is not representable by this UI")
        visible = {row.variable_key: row for row in self._factor_rows}
        for draft in setup.factor_drafts:
            row = visible.get(draft.variable_key)
            if row is None or draft.validation_error is not None:
                continue
            lower, upper = float(draft.lower), float(draft.upper)
            if not (
                row.lower_editor.minimum() <= lower <= row.lower_editor.maximum()
                and row.upper_editor.minimum() <= upper <= row.upper_editor.maximum()
            ):
                raise ValueError("Morris factor bound is not representable by this UI")

    def load_workspace_text(self, text: str) -> None:
        """Parse and validate fully, then atomically replace editor state."""
        workspace = loads_morris_workspace(text)
        if workspace.base_config() != self._config:
            raise ValueError("Morris workspace host base mismatch")
        self._preflight_workspace_ui(workspace)
        self._invalidate_active_run()
        setup = workspace.setup
        self._minimum_effects.setMaximum(setup.trajectories)  # type: ignore[attr-defined]
        controls = (
            (self._trajectories, setup.trajectories),  # type: ignore[attr-defined]
            (self._levels, setup.levels),  # type: ignore[attr-defined]
            (self._seed, setup.seed),  # type: ignore[attr-defined]
            (self._minimum_effects, setup.minimum_effects),  # type: ignore[attr-defined]
            (self._workers, setup.worker_count),  # type: ignore[attr-defined]
        )
        for editor, value in controls:
            editor.blockSignals(True)
            editor.setValue(value)
            editor.blockSignals(False)
        by_key = {draft.variable_key: draft for draft in setup.factor_drafts}
        visible_keys = {row.variable_key for row in self._factor_rows}
        self._workspace_detached_drafts = {
            key: draft for key, draft in by_key.items() if key not in visible_keys
        }
        for row in self._factor_rows:
            draft = by_key[row.variable_key]
            row.blockSignals(True)
            row.load_workspace_draft(draft)
            row.blockSignals(False)
        self._last_job = (
            None
            if workspace.completed_evidence is None
            else workspace.completed_evidence.job
        )
        if self._last_job is not None:
            self._populate_targets(self._last_job)
            self._status.setText(
                "Archived completed evidence loaded (unverified live; identifiers "
                "are inert)."
            )
        else:
            self._status.setText(
                "Morris setup loaded; no completed evidence was stored."
            )
        self._set_running(self._run_workers_running())

    def _build_workspace_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        for label, callback, tip in (
            (
                "Save Workspace…",
                self._on_save_workspace,
                "Save the exact Morris setup and optional archived aggregate evidence.",
            ),
            (
                "Load Workspace…",
                self._on_load_workspace,
                "Load only after strict full-document validation.",
            ),
            (
                "Export Aggregate CSV…",
                self._on_export_workspace_csv,
                "Export aggregate effects and provenance; raw samples are not "
                "retained.",
            ),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            layout.addWidget(button)
        return layout

    def _on_save_workspace(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Save Morris Workspace",
            "morris_workspace.json",
            "JSON (*.json)",
        )
        if selected:
            try:
                write_morris_workspace(self.workspace_document(), selected)
            except (OSError, TypeError, ValueError) as exc:
                self._status.setText(f"Cannot save Morris workspace: {exc}")
            else:
                self._status.setText(f"Morris workspace saved to {selected}.")

    def _on_load_workspace(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            cast(QWidget, self), "Load Morris Workspace", "", "JSON (*.json)"
        )
        if not selected:
            return
        try:
            self.load_workspace_text(Path(selected).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._status.setText(f"Cannot load Morris workspace: {exc}")

    def _on_export_workspace_csv(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Morris Aggregate CSV",
            "morris_aggregate_effects.csv",
            "CSV (*.csv)",
        )
        if not selected:
            return
        try:
            write_morris_csv(self.workspace_document(), selected)
        except ValueError as exc:
            self._status.setText(f"Cannot export Morris CSV: {exc}")
            return
        self._status.setText(f"Archived aggregate Morris CSV written to {selected}.")


__all__ = ["MorrisWorkspaceMixin"]
