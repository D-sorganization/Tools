"""Persistent primary-module navigation for the standalone PyQt shell."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from PyQt6.QtWidgets import QTabBar, QTabWidget

logger = logging.getLogger(__name__)

_DEFAULT_TAB_IDS: tuple[str, ...] = (
    "clubhead",
    "plots",
    "calculation_description",
    "simulation",
    "flight_explorer",
    "ground_playback",
    "launch_monitor_analytics",
    "capability_optimization",
    "variation",
    "putting",
    "glossary",
)
_REQUIRED_TAB_IDS: tuple[str, ...] = ("clubhead",)
_NAVIGATION_SETTINGS_ORG = "D-sorganization"
_NAVIGATION_SETTINGS_APP = "RateOfClosureImpactExplorer"
_NAVIGATION_STATE_KEY = "ui/primary-tabs/v1"
_NAVIGATION_STATE_VERSION = 1


class NavigationSettings(Protocol):
    """Minimal settings boundary used by primary-module persistence."""

    def value(self, key: str, default_value: object = None) -> object:
        """Return a persisted value."""

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        """Persist a value."""


@dataclass(frozen=True)
class PrimaryModuleEntry:
    """Presentation metadata for one registered workspace module."""

    module_id: str
    label: str
    visible: bool
    required: bool


class WorkspaceNavigationMixin:
    """Manage tab order, visibility, active fallback, and persistence."""

    _navigation_settings: NavigationSettings
    _tabs: QTabWidget

    def primary_tab_ids(self) -> list[str]:
        """Return stable primary-module IDs in current visual order."""
        bar = self._primary_tab_bar()
        return [str(bar.tabData(index)) for index in range(self._tabs.count())]

    def visible_primary_tab_ids(self) -> list[str]:
        """Return visible module IDs in current visual order."""
        module_ids = self.primary_tab_ids()
        return [
            module_id
            for index, module_id in enumerate(module_ids)
            if self._tabs.isTabVisible(index)
        ]

    def primary_module_entries(self) -> tuple[PrimaryModuleEntry, ...]:
        """Return module metadata for the workspace-management UI."""
        return tuple(
            PrimaryModuleEntry(
                module_id=module_id,
                label=self._tabs.tabText(index),
                visible=self._tabs.isTabVisible(index),
                required=module_id in _REQUIRED_TAB_IDS,
            )
            for index, module_id in enumerate(self.primary_tab_ids())
        )

    def current_primary_module_id(self) -> str:
        """Return the selected module's stable identifier."""
        return self._current_primary_tab_id()

    def set_primary_module_active(self, module_id: str) -> None:
        """Select a visible registered module.

        Raises:
            ValueError: If the ID is unknown or its module is hidden.
        """
        index = self._module_index(module_id)
        if not self._tabs.isTabVisible(index):
            raise ValueError(f"primary module is hidden: {module_id}")
        self._tabs.setCurrentIndex(index)

    def show_primary_module(self, module_id: str) -> None:
        """Make a registered module visible and select it."""
        self.set_primary_module_visible(module_id, True)
        self.set_primary_module_active(module_id)

    def set_primary_module_visible(self, module_id: str, visible: bool) -> bool:
        """Apply module visibility, rejecting attempts to hide required modules."""
        index = self._module_index(module_id)
        if not visible and module_id in _REQUIRED_TAB_IDS:
            return False
        was_active = index == self._tabs.currentIndex()
        self._tabs.setTabVisible(index, visible)
        if was_active and not visible:
            fallback = self.visible_primary_tab_ids()[0]
            self._tabs.setCurrentIndex(self._module_index(fallback))
        self._persist_primary_navigation()
        return True

    def move_primary_module(self, module_id: str, offset: int) -> bool:
        """Move a module by one or more slots inside the registered order."""
        if offset == 0:
            return False
        source = self._module_index(module_id)
        destination = max(0, min(self._tabs.count() - 1, source + offset))
        if source == destination:
            return False
        self._primary_tab_bar().moveTab(source, destination)
        self._persist_primary_navigation()
        return True

    def restore_default_workspace(self) -> None:
        """Restore declared order, visibility, and the default active module."""
        bar = self._primary_tab_bar()
        for destination, module_id in enumerate(_DEFAULT_TAB_IDS):
            bar.moveTab(self._module_index(module_id), destination)
        for module_id in _DEFAULT_TAB_IDS:
            self._tabs.setTabVisible(self._module_index(module_id), True)
        self._tabs.setCurrentIndex(self._module_index(_DEFAULT_TAB_IDS[0]))
        self._persist_primary_navigation()

    def _current_primary_tab_id(self) -> str:
        """Return the selected stable ID, with a deterministic safe fallback."""
        index = self._tabs.currentIndex()
        if index >= 0:
            module_id = str(self._primary_tab_bar().tabData(index))
            if module_id in _DEFAULT_TAB_IDS and self._tabs.isTabVisible(index):
                return module_id
        return self.visible_primary_tab_ids()[0]

    def _primary_tab_bar(self) -> QTabBar:
        """Return the main tab bar, enforcing the QTabWidget invariant."""
        bar = self._tabs.tabBar()
        if bar is None:  # pragma: no cover - Qt always creates its tab bar
            raise RuntimeError("Primary tab bar is unavailable")
        return bar

    def _module_index(self, module_id: str) -> int:
        """Resolve one registered module ID to its current tab index."""
        if module_id not in _DEFAULT_TAB_IDS:
            raise ValueError(f"unknown primary module: {module_id}")
        return self.primary_tab_ids().index(module_id)

    def _restore_primary_navigation(self) -> None:
        """Restore valid order, visibility, and selection from persisted state."""
        state = self._read_navigation_state()
        if state is None:
            return
        supplied_order = state.get("order")
        order = self._sanitized_order(supplied_order)
        bar = self._primary_tab_bar()
        for destination, module_id in enumerate(order):
            bar.moveTab(self._module_index(module_id), destination)
        visible = self._sanitized_visibility(state.get("visible"), supplied_order)
        for module_id in _DEFAULT_TAB_IDS:
            self._tabs.setTabVisible(
                self._module_index(module_id), module_id in visible
            )
        active = state.get("active")
        selected = active if active in visible else self.visible_primary_tab_ids()[0]
        self._tabs.setCurrentIndex(self._module_index(str(selected)))

    def _read_navigation_state(self) -> dict[str, object] | None:
        """Return a supported persisted state or ``None`` on invalid input."""
        raw = self._navigation_settings.value(_NAVIGATION_STATE_KEY)
        if not isinstance(raw, str):
            return None
        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ignoring corrupt primary-module navigation state")
            return None
        if (
            not isinstance(state, dict)
            or state.get("version") != _NAVIGATION_STATE_VERSION
        ):
            return None
        return state

    @staticmethod
    def _sanitized_order(supplied: object) -> list[str]:
        """Ignore unknown/duplicate IDs and append newly registered modules."""
        raw_order = supplied if isinstance(supplied, list) else []
        order = list(
            dict.fromkeys(item for item in raw_order if item in _DEFAULT_TAB_IDS)
        )
        order.extend(item for item in _DEFAULT_TAB_IDS if item not in order)
        return order

    @staticmethod
    def _sanitized_visibility(supplied: object, supplied_order: object) -> set[str]:
        """Migrate legacy state, revealing modules absent from the saved order."""
        if not isinstance(supplied, list):
            return set(_DEFAULT_TAB_IDS)
        visible = {item for item in supplied if item in _DEFAULT_TAB_IDS}
        known = (
            {item for item in supplied_order if item in _DEFAULT_TAB_IDS}
            if isinstance(supplied_order, list)
            else set()
        )
        visible.update(item for item in _DEFAULT_TAB_IDS if item not in known)
        visible.update(_REQUIRED_TAB_IDS)
        return visible

    def _persist_primary_navigation(self, _index: int = -1) -> None:
        """Persist known stable IDs; unknown future IDs are intentionally ignored."""
        state = {
            "version": _NAVIGATION_STATE_VERSION,
            "order": self.primary_tab_ids(),
            "visible": self.visible_primary_tab_ids(),
            "active": self._current_primary_tab_id(),
        }
        self._navigation_settings.setValue(_NAVIGATION_STATE_KEY, json.dumps(state))
