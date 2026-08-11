"""Native File-command adapter for the Rate of Closure workspace."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from rate_of_closure.application.workspace_files import (
    read_workspace,
    write_text_atomic,
    write_workspace_atomic,
)
from rate_of_closure.application.workspace_session import (
    ExplorerWorkspaceState,
    WorkspaceSessionMetadata,
    document_from_state,
    state_from_document,
)
from rate_of_closure.view_workspace import (
    workspace_from_document,
    workspace_to_document,
)
from shared.python.compatibility import UTC

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.app_toolstrip import ApplicationToolstrip
    from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
    from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
    from rate_of_closure.ui.pyqt6.variation_tab import VariationTab

_APP_VERSION = "1.14.34"
_WORKSPACE_FILTER = "Rate Workspace (*.roc-workspace.json);;JSON files (*.json)"
_VIEW_FILTER = "Rate View Layout (*.roc-view.json);;JSON files (*.json)"
_RECENT_PATHS_KEY = "workspace/recent_paths_v1"
_RECENT_PATH_LIMIT = 8
_PYQT_TO_CANONICAL = {
    "clubhead": "explorer",
    "plots": "plots",
    "calculation_description": "calculation",
    "simulation": "simulation",
    "flight_explorer": "flight",
    "launch_monitor_analytics": "launch-monitor-analytics",
    "capability_optimization": "capability-optimization",
    "variation": "variation",
    "putting": "putting",
    "glossary": "glossary",
}
_CANONICAL_TO_PYQT = {value: key for key, value in _PYQT_TO_CANONICAL.items()}


class MainWindowFileCommandsMixin:
    """Implement supported whole-session and compositor File commands."""

    _app_toolstrip: ApplicationToolstrip
    _controls: ControlsPanel
    _simulation_tab: SimulationTab
    _variation_tab: VariationTab
    _workspace_path: Path | None
    _workspace_metadata: WorkspaceSessionMetadata
    _workspace_baseline: str
    _default_workspace_state: ExplorerWorkspaceState
    _recent_workspace_paths: list[Path]
    _navigation_settings: QSettings

    if TYPE_CHECKING:

        def primary_tab_ids(self) -> list[str]: ...
        def visible_primary_tab_ids(self) -> list[str]: ...
        def current_primary_module_id(self) -> str: ...
        def apply_primary_navigation(
            self, order: tuple[str, ...], visible: tuple[str, ...], active: str
        ) -> None: ...
        def statusBar(self): ...  # type: ignore[no-untyped-def]

    def initialize_workspace_files(self) -> None:
        """Capture first-run defaults and initialize an untitled session."""
        self._workspace_path = None
        self._workspace_metadata = self._new_metadata("Untitled Workspace")
        self._default_workspace_state = self._capture_workspace_state()
        self._workspace_baseline = self._fingerprint(self._default_workspace_state)
        self._recent_workspace_paths = self._load_recent_paths()
        self._refresh_recent_action()

    def new_workspace(self) -> None:
        """Reset to first-run state after resolving unsaved changes."""
        if not self._confirm_destructive_action("create a new workspace"):
            return
        self._apply_workspace_state(self._default_workspace_state)
        self._workspace_path = None
        self._workspace_metadata = self._new_metadata("Untitled Workspace")
        self._mark_saved("New workspace created")

    def open_workspace(self) -> None:
        """Choose, validate, and atomically apply a whole workspace file."""
        selected, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(), "Open Workspace", "", _WORKSPACE_FILTER
        )
        if selected:
            self._open_workspace_path(Path(selected))

    def open_recent_workspace(self) -> None:
        """Open the newest locally persisted native workspace path."""
        if not self._recent_workspace_paths:
            self._show_error("Open Recent", "No recent workspace is available yet.")
            return
        path = self._recent_workspace_paths[0]
        if not path.is_file():
            self._recent_workspace_paths.pop(0)
            self._persist_recent_paths()
            self._refresh_recent_action()
            self._show_error("Open Recent", f"Workspace file no longer exists: {path}")
            return
        self._open_workspace_path(path)

    def save_workspace(self) -> bool:
        """Atomically save to the current path, choosing one when necessary."""
        if self._workspace_path is None:
            return self.save_workspace_as()
        return self._save_to_path(self._workspace_path)

    def save_workspace_as(self) -> bool:
        """Choose a new destination and atomically save the whole workspace."""
        selected, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Save Workspace As",
            "workspace.roc-workspace.json",
            _WORKSPACE_FILTER,
        )
        return bool(selected) and self._save_to_path(Path(selected))

    def import_workspace(self) -> None:
        """Import a strict compositor document without partial mutation."""
        selected, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(), "Import View Layout", "", _VIEW_FILTER
        )
        if not selected:
            return
        try:
            raw = json.loads(Path(selected).read_text(encoding="utf-8"))
            imported = workspace_from_document(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._show_error("Import Failed", str(exc))
            return
        self._simulation_tab.compositor().import_workspace_document(
            workspace_to_document(imported)
        )
        self._show_status("View layout imported")

    def export_workspace(self) -> None:
        """Export the strict cross-client compositor document atomically."""
        selected, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Export View Layout",
            "layout.roc-view.json",
            _VIEW_FILTER,
        )
        if not selected:
            return
        try:
            document = self._simulation_tab.compositor().export_workspace_document()
            write_text_atomic(
                json.dumps(document, indent=2, sort_keys=True) + "\n", selected
            )
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("Export Failed", str(exc))
            return
        self._show_status("View layout exported")

    def close_workspace(self) -> None:
        """Close the named session and leave a clean untitled workspace."""
        if not self._confirm_destructive_action("close this workspace"):
            return
        self._apply_workspace_state(self._default_workspace_state)
        self._workspace_path = None
        self._workspace_metadata = self._new_metadata("Untitled Workspace")
        self._mark_saved("Workspace closed; clean defaults loaded")

    def workspace_is_dirty(self) -> bool:
        """Return whether supported live state differs from the saved baseline."""
        return (
            self._fingerprint(self._capture_workspace_state())
            != self._workspace_baseline
        )

    def _open_workspace_path(self, path: Path) -> None:
        try:
            document = read_workspace(path)
            session_version = document.model_session.schema_version
            legacy_simulation = session_version == 1
            legacy_torque = session_version < 3
            legacy_variation = session_version < 4
            current = (
                self._capture_workspace_state()
                if legacy_torque or legacy_variation
                else None
            )
            state = state_from_document(
                document,
                legacy_simulation_fallback=(
                    current.simulation if legacy_simulation and current else None
                ),
                legacy_torque_fallback=(
                    current.torque if legacy_torque and current else None
                ),
                legacy_variation_fallback=(
                    current.variation if legacy_variation and current else None
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("Open Failed", str(exc))
            return
        if not self._confirm_destructive_action("open another workspace"):
            return
        try:
            self._apply_workspace_state(state)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._show_error("Open Failed", str(exc))
            return
        metadata = document.metadata
        self._workspace_path = path
        self._workspace_metadata = WorkspaceSessionMetadata(
            metadata.document_id,
            metadata.title,
            metadata.created_at_utc,
            metadata.modified_at_utc,
            metadata.app_version,
        )
        self._remember_workspace(path)
        preserved: list[str] = []
        if legacy_simulation:
            preserved.append("ball setup and spatial target")
        if legacy_torque:
            preserved.append("torque-profile library and selection")
        if legacy_variation:
            preserved.append("variation plan and analysis selection")
        suffix = (
            "; legacy session preserved " + " plus ".join(preserved)
            if preserved
            else ""
        )
        self._mark_saved(f"Opened {path.name}{suffix}")

    def _save_to_path(self, path: Path) -> bool:
        now = self._utc_now()
        metadata = WorkspaceSessionMetadata(
            self._workspace_metadata.document_id,
            path.stem.removesuffix(".roc-workspace"),
            self._workspace_metadata.created_at_utc,
            now,
            _APP_VERSION,
        )
        try:
            document = document_from_state(self._capture_workspace_state(), metadata)
            write_workspace_atomic(document, path)
        except (OSError, TypeError, ValueError) as exc:
            self._show_error("Save Failed", str(exc))
            return False
        self._workspace_path = path
        self._workspace_metadata = metadata
        self._remember_workspace(path)
        self._mark_saved(f"Saved {path.name}")
        return True

    def _capture_workspace_state(self) -> ExplorerWorkspaceState:
        module_order = tuple(
            _PYQT_TO_CANONICAL[item] for item in self.primary_tab_ids()
        )
        visible = tuple(
            _PYQT_TO_CANONICAL[item] for item in self.visible_primary_tab_ids()
        )
        return ExplorerWorkspaceState(
            scenario=self._controls.scenario(),
            club=self._controls.club_spec(),
            units=self._controls.unit_selections(),
            simulation=self._simulation_tab.simulation_workspace_state(),
            torque=self._simulation_tab.torque_workspace_state(),
            variation=self._variation_tab.variation_workspace_state(),
            module_order=module_order,
            visible_module_ids=visible,
            active_module_id=_PYQT_TO_CANONICAL[self.current_primary_module_id()],
            view_workspace=self._simulation_tab.compositor().workspace(),
        )

    def _apply_workspace_state(self, state: ExplorerWorkspaceState) -> None:
        prior = self._capture_workspace_state()
        try:
            self._apply_workspace_state_unchecked(state)
        except Exception:
            self._apply_workspace_state_unchecked(prior)
            raise

    def _apply_workspace_state_unchecked(self, state: ExplorerWorkspaceState) -> None:
        order = tuple(_CANONICAL_TO_PYQT[item] for item in state.module_order)
        visible = tuple(_CANONICAL_TO_PYQT[item] for item in state.visible_module_ids)
        active = _CANONICAL_TO_PYQT[state.active_module_id]
        self._controls.apply_workspace_state(
            state.scenario, state.club, dict(state.units)
        )
        self.apply_primary_navigation(order, visible, active)
        self._simulation_tab.apply_simulation_workspace_state(state.simulation)
        self._simulation_tab.apply_torque_workspace_state(state.torque)
        self._variation_tab.apply_variation_workspace_state(state.variation)
        self._simulation_tab.compositor().import_workspace_document(
            workspace_to_document(state.view_workspace)
        )

    def _confirm_destructive_action(self, action: str) -> bool:
        if not self.workspace_is_dirty():
            return True
        choice = QMessageBox.warning(
            self._dialog_parent(),
            "Unsaved Workspace",
            f"Save changes before you {action}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        return choice == QMessageBox.StandardButton.Discard or self.save_workspace()

    def _mark_saved(self, message: str) -> None:
        self._workspace_baseline = self._fingerprint(self._capture_workspace_state())
        self._show_status(message)

    def _show_status(self, message: str) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, 5000)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self._dialog_parent(), title, message)

    def _dialog_parent(self) -> QWidget:
        """Return the concrete window type required by Qt dialog overloads."""
        return cast(QWidget, self)

    def _load_recent_paths(self) -> list[Path]:
        raw = self._navigation_settings.value(_RECENT_PATHS_KEY)
        if not isinstance(raw, str):
            return []
        try:
            values = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            return []
        return [Path(value) for value in values[:_RECENT_PATH_LIMIT]]

    def _remember_workspace(self, path: Path) -> None:
        resolved = path.resolve()
        self._recent_workspace_paths = [
            resolved,
            *(item for item in self._recent_workspace_paths if item != resolved),
        ][:_RECENT_PATH_LIMIT]
        self._persist_recent_paths()
        self._refresh_recent_action()

    def _persist_recent_paths(self) -> None:
        self._navigation_settings.setValue(
            _RECENT_PATHS_KEY,
            json.dumps([str(path) for path in self._recent_workspace_paths]),
        )

    def _refresh_recent_action(self) -> None:
        path = (
            str(self._recent_workspace_paths[0]) if self._recent_workspace_paths else ""
        )
        self._app_toolstrip.set_open_recent_available(bool(path), path)

    def _new_metadata(self, title: str) -> WorkspaceSessionMetadata:
        now = self._utc_now()
        return WorkspaceSessionMetadata(
            f"workspace.{uuid4()}", title, now, now, _APP_VERSION
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _fingerprint(self, state: ExplorerWorkspaceState) -> str:
        document = document_from_state(state, self._workspace_metadata).to_json_dict()
        document.pop("metadata")
        return json.dumps(document, sort_keys=True, separators=(",", ":"))


__all__ = ["MainWindowFileCommandsMixin"]
