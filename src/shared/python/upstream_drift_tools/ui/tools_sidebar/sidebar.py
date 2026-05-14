"""Unified dockable tools sidebar for Qt host applications."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtWidgets, Signal, all_sidebar_dock_features, dock_area
from .registry import WorkspaceRegistry
from .state import SidebarState

DEFAULT_TABS = (
    "files",
    "workspace",
    "chat",
    "terminal",
    "calculator",
    "units",
    "notes",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolsSidebarInstallResult:
    """Result returned by the shared host install helper."""

    installed: bool
    reason: str
    sidebar: UnifiedToolsSidebar | None = None
    dock_widget: QtWidgets.QDockWidget | None = None


class UnifiedToolsSidebar(QtWidgets.QWidget):
    """Tabbed sidebar that can be installed as a tear-off dock widget."""

    file_open_requested = Signal(str)
    context_updated = Signal(dict)

    def __init__(
        self,
        project_root: str | Path | None = None,
        registry: WorkspaceRegistry | None = None,
        state: SidebarState | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry or WorkspaceRegistry()
        self._state = state or SidebarState()
        self._dock_widget: QtWidgets.QDockWidget | None = None
        self._tab_ids: list[str] = []

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.currentChanged.connect(self._emit_context)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self.file_explorer = ProjectFileExplorer(project_root or Path.cwd(), self)
        self.file_explorer.file_open_requested.connect(self.file_open_requested.emit)
        self.add_tab("files", "Files", self.file_explorer)
        self.add_tab("workspace", "Workspace", self._build_workspace_tab())
        self.add_tab("chat", "Chat", self._placeholder("Chat panel"))
        self.add_tab("terminal", "Terminal", self._placeholder("Terminal panel"))
        self.add_tab("calculator", "Calculator", self._placeholder("Calculator panel"))
        self.add_tab("units", "Units", self._build_unit_converter_tab())
        self.add_tab("notes", "Notes", self._placeholder("Notepad"))

        self.apply_state(self._state)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget | None:
        """Return the installed dock widget, if any."""
        return self._dock_widget

    def add_tab(self, tab_id: str, title: str, widget: QtWidgets.QWidget) -> None:
        """Add a tab with a stable persistence id."""
        if tab_id in self._tab_ids:
            raise ValueError(f"Duplicate sidebar tab id: {tab_id}")
        self._tab_ids.append(tab_id)
        self.tabs.addTab(widget, title)

    def install_as_dock(
        self,
        main_window: QtWidgets.QMainWindow,
        *,
        area: str | None = None,
        title: str = "Tools",
        state_path: str | Path | None = None,
    ) -> QtWidgets.QDockWidget:
        """Install this sidebar into ``main_window`` as a QDockWidget."""
        if state_path is not None:
            self.apply_state(SidebarState.load_json(state_path))

        dock = QtWidgets.QDockWidget(title, main_window)
        dock.setObjectName("UnifiedToolsSidebarDock")
        dock.setFeatures(all_sidebar_dock_features())
        dock.setWidget(self)
        dock.setFloating(self._state.floating)
        dock.resize(self._state.width, self._state.height)
        main_window.addDockWidget(dock_area(area or self._state.dock_area), dock)
        self._dock_widget = dock
        self._emit_context()
        return dock

    def snapshot_state(self) -> SidebarState:
        """Return current dock, size, and active-tab state."""
        dock_area_name = self._state.dock_area
        floating = self._state.floating
        width = self.width()
        height = self.height()

        if self._dock_widget is not None:
            floating = self._dock_widget.isFloating()
            width = self._dock_widget.width()
            height = self._dock_widget.height()
            host = self._dock_widget.parent()
            if hasattr(host, "dockWidgetArea"):
                area = host.dockWidgetArea(self._dock_widget)
                if area == dock_area("left"):
                    dock_area_name = "left"
                elif area == dock_area("right"):
                    dock_area_name = "right"

        return SidebarState(
            dock_area=dock_area_name,
            floating=floating,
            width=width,
            height=height,
            active_tab=self.active_tab_id(),
        )

    def save_state(self, path: str | Path) -> SidebarState:
        """Persist current sidebar state and return it."""
        state = self.snapshot_state()
        state.save_json(path)
        self._state = state
        return state

    def apply_state(self, state: SidebarState) -> None:
        """Apply active-tab and size state."""
        self._state = state
        self.resize(state.width, state.height)
        self.set_active_tab(state.active_tab)
        if self._dock_widget is not None:
            self._dock_widget.setFloating(state.floating)
            self._dock_widget.resize(state.width, state.height)

    def active_tab_id(self) -> str:
        """Return the stable id for the active tab."""
        index = int(self.tabs.currentIndex())
        if 0 <= index < len(self._tab_ids):
            return self._tab_ids[index]
        return self._tab_ids[0]

    def set_active_tab(self, tab_id: str) -> bool:
        """Activate a tab by stable id. Returns whether it was found."""
        if tab_id not in self._tab_ids:
            return False
        self.tabs.setCurrentIndex(self._tab_ids.index(tab_id))
        return True

    def set_context_variable(self, name: str, value: Any) -> None:
        """Update the workspace registry and notify host applications."""
        self.registry.set(name, value)
        self._refresh_workspace_list()
        self._emit_context()

    def set_project_root(self, project_root: str | Path) -> None:
        """Update the project explorer root."""
        self.file_explorer.set_project_root(project_root)
        self._emit_context()

    def _build_workspace_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(widget)
        self._workspace_list = QtWidgets.QListWidget(widget)
        layout.addWidget(self._workspace_list)
        self._refresh_workspace_list()
        return widget

    def _refresh_workspace_list(self) -> None:
        if not hasattr(self, "_workspace_list"):
            return
        self._workspace_list.clear()
        for variable in self.registry.variables():
            label = f"{variable.name}: {variable.type_name} ({variable.summary})"
            self._workspace_list.addItem(label)

    def _build_unit_converter_tab(self) -> QtWidgets.QWidget:
        if QT_API != "PyQt6":
            return self._placeholder("Unit converter")
        try:
            from upstream_drift_tools.ui.widgets.unit_converter_widget import (
                UnitConverterWidget,
            )

            return UnitConverterWidget(self)
        except Exception:
            return self._placeholder("Unit converter")

    def _placeholder(self, title: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(widget)
        label = QtWidgets.QLabel(title, widget)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _emit_context(self) -> None:
        self.context_updated.emit(
            {
                "active_tab": self.active_tab_id(),
                "project_root": str(self.file_explorer.project_root),
                "workspace_variables": [
                    variable.to_metadata() for variable in self.registry.variables()
                ],
            }
        )


def create_tools_sidebar(
    project_root: str | Path | None = None,
    registry: WorkspaceRegistry | None = None,
    state: SidebarState | None = None,
    parent: QtWidgets.QWidget | None = None,
    context_provider: Callable[[], Any] | None = None,
    **_: Any,
) -> UnifiedToolsSidebar:
    """Create a sidebar widget using the stable shared factory contract."""
    sidebar = UnifiedToolsSidebar(
        project_root=project_root,
        registry=registry,
        state=state,
        parent=parent,
    )
    if context_provider is not None:
        try:
            sidebar.set_context_variable("host_context", context_provider())
        except Exception as exc:  # noqa: BLE001 - host context is optional
            logger.debug("Tools sidebar context provider failed: %s", exc)
    return sidebar


def install_tools_sidebar(
    main_window: QtWidgets.QMainWindow,
    *,
    project_root: str | Path | None = None,
    registry: WorkspaceRegistry | None = None,
    state: SidebarState | None = None,
    context_provider: Callable[[], Any] | None = None,
    area: str | None = None,
    title: str = "Tools",
    state_path: str | Path | None = None,
    **_: Any,
) -> ToolsSidebarInstallResult:
    """Install the shared sidebar as a dock widget in a Qt main window."""
    if main_window is None or not hasattr(main_window, "addDockWidget"):
        return ToolsSidebarInstallResult(False, "main_window does not support docks")

    sidebar = create_tools_sidebar(
        project_root=project_root,
        registry=registry,
        state=state,
        parent=main_window,
        context_provider=context_provider,
    )
    dock_widget = sidebar.install_as_dock(
        main_window,
        area=area,
        title=title,
        state_path=state_path,
    )
    return ToolsSidebarInstallResult(
        True,
        "installed",
        sidebar=sidebar,
        dock_widget=dock_widget,
    )
