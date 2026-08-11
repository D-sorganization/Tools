"""Responsive PyQt host for synchronized Impact, Swing, and Flight views."""

from __future__ import annotations

import json
from collections.abc import Mapping

from PyQt6.QtCore import QSettings, QSignalBlocker
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.view_workspace import (
    ViewKind,
    ViewLayout,
    ViewSlot,
    ViewWorkspace,
    workspace_to_document,
)
from rate_of_closure.view_workspace_recovery import (
    SUPPORTED_VIEW_KINDS,
    recover_workspace_document,
)

_SETTINGS_KEY = "view_compositor/layout_v1"
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
        self._hosts = {kind: self._host(kind, view) for kind, view in views.items()}
        self._layout_combo = QComboBox(self)
        self._checks: dict[ViewKind, QCheckBox] = {}
        self._grid = QGridLayout()
        self._workspace = self._load_workspace()
        self._build()
        self._apply_workspace(self._workspace, persist=False)

    def workspace(self) -> ViewWorkspace:
        """Return the current immutable workspace description."""
        return self._workspace

    def visible_view_ids(self) -> tuple[str, ...]:
        """Return visible stable view identities in focus order."""
        return tuple(slot.id for slot in self._workspace.slots)

    def view(self, kind: ViewKind) -> QWidget:
        """Return the persistent view instance registered for ``kind``."""
        return self._views[kind]

    def show_single_view(self, kind: ViewKind) -> None:
        """Select one real host while preserving all other host-owned state."""
        if kind not in SUPPORTED_VIEW_KINDS:
            raise ValueError(f"unsupported compositor view: {kind!r}")
        self._apply_workspace(
            ViewWorkspace(
                layout=ViewLayout.SINGLE,
                slots=(ViewSlot(id=kind.value, kind=kind),),
                active_slot_id=kind.value,
                playback=self._workspace.playback,
            )
        )

    def _build(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Viewport Layout"))
        self._layout_combo.setAccessibleName("Viewport Layout")
        for layout in ViewLayout:
            self._layout_combo.addItem(layout.value.replace("_", " ").title(), layout)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        controls.addWidget(self._layout_combo)
        for kind in SUPPORTED_VIEW_KINDS:
            check = QCheckBox(_LABELS[kind])
            check.setAccessibleName(f"Show {_LABELS[kind]} viewport")
            check.toggled.connect(
                lambda checked, item=kind: self._toggle(item, checked)
            )
            self._checks[kind] = check
            controls.addWidget(check)
        controls.addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(controls)
        root.addLayout(self._grid, stretch=1)

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
            self._persist()

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
        layout = self._workspace.layout
        if len(kinds) == 1:
            layout = ViewLayout.SINGLE
        elif layout is ViewLayout.SINGLE:
            layout = ViewLayout.SPLIT_HORIZONTAL
        self._set_kinds(layout, kinds)

    def _set_kinds(self, layout: ViewLayout, kinds: list[ViewKind]) -> None:
        active = self._workspace.active_slot_id
        identifiers = [kind.value for kind in kinds]
        self._apply_workspace(
            ViewWorkspace(
                layout=layout,
                slots=tuple(ViewSlot(id=kind.value, kind=kind) for kind in kinds),
                active_slot_id=active if active in identifiers else identifiers[0],
                playback=self._workspace.playback,
            )
        )

    def _load_workspace(self) -> ViewWorkspace:
        if self._settings is None:
            return ViewWorkspace.default()
        raw = self._settings.value(_SETTINGS_KEY)
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
