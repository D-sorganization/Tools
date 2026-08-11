"""Contextual application-toolstrip commands for the combined request."""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QWidget

from rate_of_closure.application.commands import AppCommandId

_MODULES = frozenset(("regional_surfaces", "variation"))
_DISABLED_REASON = "Available in the Ground Surfaces and Variation modules."


class RegionalGroundFileHost(Protocol):
    """Two host callbacks bound to contextual file actions."""

    def open_regional_ground_variation_request(self) -> None: ...

    def save_regional_ground_variation_request_as(self) -> None: ...


class RegionalGroundFileCommandGroup:
    """Own contextual actions while sharing the toolstrip registry."""

    def __init__(
        self,
        host: RegionalGroundFileHost,
        parent: QWidget,
        registry: dict[AppCommandId, QAction],
    ) -> None:
        self._host = host
        self._parent = parent
        self._registry = registry
        self._actions: tuple[QAction, ...] = ()

    def add_to(self, menu: QMenu) -> None:
        """Create the two stable commands and add them to ``menu``."""
        definitions = (
            (
                AppCommandId.FILE_OPEN_REGIONAL_GROUND_VARIATION_REQUEST,
                "Open Regional-Ground Variation Request…",
                self._host.open_regional_ground_variation_request,
            ),
            (
                AppCommandId.FILE_SAVE_REGIONAL_GROUND_VARIATION_REQUEST_AS,
                "Save Regional-Ground Variation Request As…",
                self._host.save_regional_ground_variation_request_as,
            ),
        )
        actions = []
        for command_id, label, callback in definitions:
            action = QAction(label, self._parent)
            action.setObjectName(command_id.value)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self._registry[command_id] = action
            actions.append(action)
            menu.addAction(action)
        self._actions = tuple(actions)
        self.set_active_module("")

    def set_active_module(self, module_id: str) -> None:
        """Apply contextual availability and accessible explanations."""
        enabled = module_id in _MODULES
        reason = "" if enabled else _DISABLED_REASON
        for action in self._actions:
            action.setEnabled(enabled)
            action.setToolTip(reason)
            action.setStatusTip(reason)


__all__ = ["RegionalGroundFileCommandGroup"]
