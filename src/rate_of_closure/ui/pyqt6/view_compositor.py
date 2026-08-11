"""Responsive PyQt host for synchronized Impact, Swing, and Flight views."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

from PyQt6.QtCore import QSettings, QSignalBlocker, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.camera_commands import CameraState
from rate_of_closure.application.camera_preferences import (
    CameraPreferences,
    preference_from_camera_state,
)
from rate_of_closure.ui.pyqt6.camera_controls import CameraViewportMixin
from rate_of_closure.view_workspace import (
    PlaybackState,
    ViewKind,
    ViewLayout,
    ViewSlot,
    ViewWorkspace,
    workspace_from_document,
    workspace_to_document,
)
from rate_of_closure.view_workspace_recovery import (
    SUPPORTED_VIEW_KINDS,
    normalized_workspace_layout,
    recover_workspace_document,
)

_SETTINGS_KEY = "view_compositor/layout_v2"
_LEGACY_SETTINGS_KEY = "view_compositor/layout_v1"
_PLAYBACK_PERSIST_DEBOUNCE_MS = 200
_LABELS = {
    ViewKind.IMPACT: "Impact",
    ViewKind.SWING: "Swing",
    ViewKind.FLIGHT: "Flight",
}


class ViewCompositor(QWidget):
    """Arrange persistent real view instances without recreating their state."""

    def __init__(
        self,
        views: Mapping[ViewKind, QWidget],
        settings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if set(views) != set(SUPPORTED_VIEW_KINDS):
            raise ValueError("compositor requires Impact, Swing, and Flight views")
        self._views: dict[ViewKind, QWidget] = dict(views)
        self._settings = settings
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(_PLAYBACK_PERSIST_DEBOUNCE_MS)
        self._persist_timer.timeout.connect(lambda: self._persist())
        self._hosts = {kind: self._host(kind, view) for kind, view in views.items()}
        self._layout_combo = QComboBox(self)
        self._checks: dict[ViewKind, QCheckBox] = {}
        self._grid = QGridLayout()
        self._workspace = self._load_workspace()
        self._build()
        self._apply_workspace(self._workspace, persist=False)
        self._bind_camera_preference_listeners()

    def workspace(self) -> ViewWorkspace:
        """Return the current immutable workspace description."""
        return self._workspace

    def visible_view_ids(self) -> tuple[str, ...]:
        """Return visible stable view identities in focus order."""
        return tuple(slot.id for slot in self._workspace.slots)

    def view(self, kind: ViewKind) -> QWidget:
        """Return the persistent view instance registered for ``kind``."""
        return self._views[kind]

    def export_workspace_document(self) -> dict[str, object]:
        """Return a detached, strict version-1 compositor document."""
        return workspace_to_document(self._workspace)

    def import_workspace_document(self, document: Mapping[str, object]) -> None:
        """Atomically apply one strict version-1 compositor document."""
        if not isinstance(document, Mapping):
            raise TypeError("workspace document must be a mapping")
        workspace = workspace_from_document(document)
        if any(slot.kind not in SUPPORTED_VIEW_KINDS for slot in workspace.slots):
            raise ValueError("workspace document contains an unsupported view kind")
        self._apply_workspace(workspace)

    def show_single_view(self, kind: ViewKind) -> None:
        """Select one real host while preserving all other host-owned state."""
        if kind not in SUPPORTED_VIEW_KINDS:
            raise ValueError(f"unsupported compositor view: {kind!r}")
        self._apply_workspace(
            replace(
                self._workspace,
                layout=ViewLayout.SINGLE,
                slots=(self._slot_for_kind(kind),),
                active_slot_id=kind.value,
            )
        )

    def update_playback(self, playback: PlaybackState) -> None:
        """Update the owned transport snapshot and debounce durable writes."""
        playback.validate()
        if playback == self._workspace.playback:
            return
        self._workspace = replace(self._workspace, playback=playback)
        if self._settings is not None:
            self._persist_timer.start()

    def _build(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Viewport Layout"))
        self._layout_combo.setAccessibleName("Viewport Layout")
        self._layout_combo.setObjectName("viewportLayoutCombo")
        self._layout_combo.setToolTip(
            "Arrange the selected synchronized views as a single panel, split, or grid."
        )
        for layout in ViewLayout:
            self._layout_combo.addItem(layout.value.replace("_", " ").title(), layout)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        controls.addWidget(self._layout_combo)
        for kind in SUPPORTED_VIEW_KINDS:
            check = QCheckBox(_LABELS[kind])
            check.setObjectName(f"{kind.value}ViewportToggle")
            check.setAccessibleName(f"Show {_LABELS[kind]} viewport")
            check.setToolTip(
                f"Show or hide the synchronized {_LABELS[kind].lower()} viewport."
            )
            check.toggled.connect(
                lambda checked, item=kind: self._toggle(item, checked)
            )
            self._checks[kind] = check
            controls.addWidget(check)
        controls.addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(controls)
        focus_order = [self._layout_combo, *self._checks.values()]
        for current, following in zip(focus_order, focus_order[1:], strict=False):
            self.setTabOrder(current, following)
        viewport_surface = QWidget(self)
        viewport_surface.setLayout(self._grid)
        viewport = QScrollArea(self)
        viewport.setObjectName("viewCompositorScrollArea")
        viewport.setAccessibleName("Synchronized viewport workspace")
        viewport.setWidgetResizable(True)
        viewport.setWidget(viewport_surface)
        root.addWidget(viewport, stretch=1)

    def _host(self, kind: ViewKind, view: QWidget) -> QGroupBox:
        host = QGroupBox(f"{_LABELS[kind]} View", self)
        host.setObjectName(f"{kind.value}ViewportHost")
        host.setAccessibleName(f"{_LABELS[kind]} synchronized viewport")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(view)
        return host

    def _apply_workspace(
        self, workspace: ViewWorkspace, *, persist: bool = True
    ) -> None:
        workspace.validate()
        self._restore_camera_preferences(workspace)
        self._workspace = workspace
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.hide()
        visible = {slot.kind for slot in workspace.slots}
        for kind, check in self._checks.items():
            with QSignalBlocker(check):
                check.setChecked(kind in visible)
        with QSignalBlocker(self._layout_combo):
            self._layout_combo.setCurrentIndex(
                self._layout_combo.findData(workspace.layout)
            )
        for index, slot in enumerate(workspace.slots):
            row, column = self._position(index, workspace.layout)
            host = self._hosts[slot.kind]
            self._grid.addWidget(host, row, column)
            host.show()
        if persist:
            self._persist_timer.stop()
            self._persist()

    def _restore_camera_preferences(self, workspace: ViewWorkspace) -> None:
        """Apply validated preferences to native 3D adapters only."""
        for kind, view in self._views.items():
            if isinstance(view, CameraViewportMixin):
                view.restore_camera_preference(
                    workspace.camera_preferences.viewports[kind.value]
                )

    def _bind_camera_preference_listeners(self) -> None:
        for kind, view in self._views.items():
            if isinstance(view, CameraViewportMixin):
                view.set_camera_preference_listener(
                    partial(self._camera_preference_changed, kind)
                )

    def _camera_preference_changed(self, kind: ViewKind, state: CameraState) -> None:
        current = self._workspace.camera_preferences
        preference = preference_from_camera_state(state, current.viewports[kind.value])
        if preference == current.viewports[kind.value]:
            return
        viewports = dict(current.viewports)
        viewports[kind.value] = preference
        self._workspace = replace(
            self._workspace,
            camera_preferences=CameraPreferences(viewports),
        )
        if self._settings is not None:
            self._persist_timer.start()

    @staticmethod
    def _position(index: int, layout: ViewLayout) -> tuple[int, int]:
        if layout is ViewLayout.SPLIT_HORIZONTAL:
            return 0, index
        if layout is ViewLayout.GRID:
            return divmod(index, 2)
        return index, 0

    def _on_layout_changed(self) -> None:
        layout = self._layout_combo.currentData()
        if not isinstance(layout, ViewLayout):
            return
        if layout is ViewLayout.SINGLE:
            self.show_single_view(ViewKind(self._workspace.active_slot_id))
            return
        count = 3 if layout is ViewLayout.GRID else 2
        current = [slot.kind for slot in self._workspace.slots]
        kinds = (
            current + [kind for kind in SUPPORTED_VIEW_KINDS if kind not in current]
        )[:count]
        self._set_kinds(layout, kinds)

    def _toggle(self, kind: ViewKind, checked: bool) -> None:
        current = [slot.kind for slot in self._workspace.slots]
        kinds = (
            current + [kind]
            if checked and kind not in current
            else [item for item in current if item is not kind]
        )
        if not kinds:
            with QSignalBlocker(self._checks[kind]):
                self._checks[kind].setChecked(True)
            return
        layout = normalized_workspace_layout(self._workspace.layout, len(kinds))
        self._set_kinds(layout, kinds)

    def _set_kinds(self, layout: ViewLayout, kinds: list[ViewKind]) -> None:
        active = self._workspace.active_slot_id
        identifiers = [kind.value for kind in kinds]
        self._apply_workspace(
            replace(
                self._workspace,
                layout=normalized_workspace_layout(layout, len(kinds)),
                slots=tuple(self._slot_for_kind(kind) for kind in kinds),
                active_slot_id=active if active in identifiers else identifiers[0],
            )
        )

    def _slot_for_kind(self, kind: ViewKind) -> ViewSlot:
        return next(
            (slot for slot in self._workspace.slots if slot.kind is kind),
            ViewSlot(id=kind.value, kind=kind),
        )

    def _load_workspace(self) -> ViewWorkspace:
        if self._settings is None:
            return ViewWorkspace.default()
        raw = self._settings.value(_SETTINGS_KEY)
        if not isinstance(raw, str):
            raw = self._settings.value(_LEGACY_SETTINGS_KEY)
        if not isinstance(raw, str):
            return ViewWorkspace.default()
        try:
            return recover_workspace_document(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ViewWorkspace.default()

    def _persist(self) -> None:
        if self._settings is not None:
            self._settings.setValue(
                _SETTINGS_KEY,
                json.dumps(workspace_to_document(self._workspace), sort_keys=True),
            )


__all__ = ["ViewCompositor"]
