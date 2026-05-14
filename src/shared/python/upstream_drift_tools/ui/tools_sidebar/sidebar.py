"""Unified dockable tools sidebar for Qt host applications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtWidgets, Signal, all_sidebar_dock_features, dock_area
from .registry import WorkspaceRegistry
from .runtime_tabs import (
    build_calculator_tab,
    build_chat_tab,
    build_notes_tab,
    build_terminal_tab,
)
from .state import SidebarState


@dataclass(frozen=True)
class SidebarTabDefinition:
    """Configurable Sidekick tab contract."""

    tab_id: str
    title: str
    factory: Callable[[UnifiedToolsSidebar], QtWidgets.QWidget]
    visible: bool = True
    popout_enabled: bool = True
    duplicate_enabled: bool = False


class UnifiedToolsSidebar(QtWidgets.QWidget):
    """Tabbed sidebar that can be installed as a tear-off dock widget."""

    file_open_requested = Signal(str)
    context_updated = Signal(dict)

    def __init__(
        self,
        project_root: str | Path | None = None,
        registry: WorkspaceRegistry | None = None,
        state: SidebarState | None = None,
        tab_definitions: list[SidebarTabDefinition] | None = None,
        design_tokens: theme.SidekickDesignTokens | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(theme.SIDEKICK_SIDEBAR_OBJECT_NAME)
        self.registry = registry or WorkspaceRegistry()
        self._state = state or SidebarState()
        self._design_tokens = design_tokens or theme.SIDEKICK_DESIGN_TOKENS
        self._dock_widget: QtWidgets.QDockWidget | None = None
        self._expanded_width = self._state.width
        self._tab_ids: list[str] = []
        self._tab_definitions: dict[str, SidebarTabDefinition] = {}
        self._tab_widgets: dict[str, QtWidgets.QWidget] = {}
        self._popout_windows: dict[str, QtWidgets.QMainWindow] = {}
        self._duplicate_counts: dict[str, int] = {}
        self._project_root = Path(project_root or Path.cwd()).expanduser().resolve()

        self.toolbar = self._build_toolbar()
        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setObjectName(theme.SIDEKICK_TABS_OBJECT_NAME)
        self.tabs.tabBar().setObjectName(theme.SIDEKICK_TAB_BAR_OBJECT_NAME)
        self.tabs.setMovable(True)
        self.tabs.currentChanged.connect(self._emit_context)
        self.tabs.tabBar().tabMoved.connect(self._sync_tab_order_from_widget)
        self.setStyleSheet(theme.sidekick_qss(self._design_tokens))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.tabs)

        self.configure_tabs(tab_definitions or self._default_tab_definitions())

        self.apply_state(self._state)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget | None:
        """Return the installed dock widget, if any."""
        return self._dock_widget

    @property
    def project_root(self) -> Path:
        """Return the current project root used by runtime tabs."""
        return self._project_root

    def add_tab(self, tab_id: str, title: str, widget: QtWidgets.QWidget) -> None:
        """Add a tab with a stable persistence id."""
        if tab_id in self._tab_ids:
            raise ValueError(f"Duplicate sidebar tab id: {tab_id}")
        self._tab_ids.append(tab_id)
        self._tab_widgets[tab_id] = widget
        self.tabs.addTab(widget, title)

    def configure_tabs(self, definitions: list[SidebarTabDefinition]) -> None:
        """Configure the available tab set for this Sidekick instance."""
        self.tabs.clear()
        self._tab_ids.clear()
        self._tab_widgets.clear()
        self._tab_definitions = {
            definition.tab_id: definition for definition in definitions
        }
        for definition in definitions:
            if definition.visible:
                self._add_defined_tab(definition)

    def visible_tab_ids(self) -> list[str]:
        """Return tab ids currently docked in the sidebar."""
        return list(self._tab_ids)

    def hidden_tab_ids(self) -> list[str]:
        """Return configured tabs that are currently hidden."""
        return [
            tab_id
            for tab_id in self._tab_definitions
            if tab_id not in self._tab_ids and tab_id not in self._popout_windows
        ]

    def move_tab(self, tab_id: str, index: int) -> bool:
        """Move a visible tab to ``index``."""
        if tab_id not in self._tab_ids:
            return False
        current = self._tab_ids.index(tab_id)
        target = max(0, min(index, len(self._tab_ids) - 1))
        if current == target:
            return True
        self.tabs.tabBar().moveTab(current, target)
        self._sync_tab_order_from_widget()
        return True

    def set_tab_visible(self, tab_id: str, visible: bool) -> bool:
        """Show or hide a configured tab."""
        if visible:
            if tab_id in self._tab_ids:
                return True
            definition = self._tab_definitions.get(tab_id)
            if definition is None:
                return False
            self._add_defined_tab(definition)
            self._sync_tab_order_from_widget()
            self._emit_context()
            return True

        if tab_id not in self._tab_ids:
            return tab_id in self._tab_definitions
        index = self._tab_ids.index(tab_id)
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self._tab_ids.pop(index)
        self._tab_widgets.pop(tab_id, None)
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        self._emit_context()
        return True

    def set_minimized(self, minimized: bool) -> None:
        """Collapse or expand the sidebar without destroying tab state."""
        self._state.minimized = minimized
        if minimized:
            self._expanded_width = max(self.width(), self._expanded_width)
            self.tabs.setVisible(False)
            self.setMaximumWidth(56)
        else:
            self.tabs.setVisible(True)
            self.setMaximumWidth(16777215)
            self.resize(max(self._expanded_width, 240), self.height())
        self._emit_context()

    def set_dock_area(self, area: str) -> bool:
        """Move the installed dock widget to the left or right side."""
        if area not in {"left", "right"}:
            return False
        self._state.dock_area = area
        if self._dock_widget is not None:
            host = self._dock_widget.parent()
            if hasattr(host, "addDockWidget"):
                host.addDockWidget(dock_area(area), self._dock_widget)
        self._emit_context()
        return True

    def pop_out_tab(self, tab_id: str) -> QtWidgets.QMainWindow | None:
        """Move one visible tab into a standalone utility window."""
        definition = self._tab_definitions.get(tab_id)
        if (
            definition is None
            or not definition.popout_enabled
            or tab_id not in self._tab_ids
        ):
            return None
        index = self._tab_ids.index(tab_id)
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self._tab_ids.pop(index)
        self._tab_widgets.pop(tab_id, None)

        window = QtWidgets.QMainWindow(self)
        window.setObjectName(f"SidekickPopout_{tab_id}")
        window.setWindowTitle(f"Sidekick - {definition.title}")
        window.setCentralWidget(widget)
        window.resize(max(self._state.width, 360), max(self._state.height, 360))
        window.closeEvent = self._redock_close_event(tab_id, window)
        self._popout_windows[tab_id] = window
        window.show()
        self._emit_context()
        return window

    def redock_tab(self, tab_id: str) -> bool:
        """Return a popped-out tab to the sidebar."""
        window = self._popout_windows.pop(tab_id, None)
        if window is None:
            return tab_id in self._tab_ids
        widget = window.centralWidget()
        window.setCentralWidget(None)
        window.hide()
        definition = self._tab_definitions[tab_id]
        self._tab_ids.append(tab_id)
        self._tab_widgets[tab_id] = widget
        self.tabs.addTab(widget, definition.title)
        self.set_active_tab(tab_id)
        self._sync_tab_order_from_widget()
        self._emit_context()
        return True

    def duplicate_tab(self, tab_id: str) -> str | None:
        """Create a second docked instance of a duplicable tab."""
        definition = self._tab_definitions.get(tab_id)
        if definition is None or not definition.duplicate_enabled:
            return None
        count = self._duplicate_counts.get(tab_id, 0) + 1
        self._duplicate_counts[tab_id] = count
        duplicate_id = f"{tab_id}#{count}"
        duplicate = replace(
            definition,
            tab_id=duplicate_id,
            title=f"{definition.title} {count + 1}",
        )
        self._tab_definitions[duplicate_id] = duplicate
        self._add_defined_tab(duplicate)
        self.set_active_tab(duplicate_id)
        self._sync_tab_order_from_widget()
        self._emit_context()
        return duplicate_id

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
        dock.setObjectName(theme.SIDEKICK_DOCK_OBJECT_NAME)
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
            minimized=self._state.minimized,
            tab_order=self.visible_tab_ids(),
            hidden_tabs=self.hidden_tab_ids(),
            popped_out_tabs=list(self._popout_windows),
        )

    def save_state(self, path: str | Path) -> SidebarState:
        state = self.snapshot_state()
        state.save_json(path)
        self._state = state
        return state

    def apply_state(self, state: SidebarState) -> None:
        self._state = state
        self.resize(state.width, state.height)
        self._apply_tab_state(state)
        self.set_minimized(state.minimized)
        self.set_active_tab(state.active_tab)
        if self._dock_widget is not None:
            self._dock_widget.setFloating(state.floating)
            self._dock_widget.resize(state.width, state.height)

    def active_tab_id(self) -> str:
        index = int(self.tabs.currentIndex())
        if 0 <= index < len(self._tab_ids):
            return self._tab_ids[index]
        return self._tab_ids[0]

    def set_active_tab(self, tab_id: str) -> bool:
        if tab_id not in self._tab_ids:
            return False
        self.tabs.setCurrentIndex(self._tab_ids.index(tab_id))
        return True

    def set_context_variable(self, name: str, value: Any) -> None:
        self.registry.set(name, value)
        self._refresh_workspace_list()
        self._emit_context()

    def set_design_tokens(self, design_tokens: theme.SidekickDesignTokens) -> None:
        """Apply a new Sidekick token set to this sidebar."""
        self._design_tokens = design_tokens
        self.setStyleSheet(theme.sidekick_qss(self._design_tokens))
        self._emit_context()

    def set_theme(self, theme_name: str) -> None:
        """Apply a shared fleet theme by name to this sidebar."""
        self.set_design_tokens(theme.SidekickDesignTokens.from_shared_theme(theme_name))

    def set_project_root(self, project_root: str | Path) -> None:
        self._project_root = Path(project_root).expanduser().resolve()
        file_explorer = self._tab_widgets.get("files")
        if isinstance(file_explorer, ProjectFileExplorer):
            file_explorer.set_project_root(self._project_root)
        self._emit_context()

    def _default_tab_definitions(self) -> list[SidebarTabDefinition]:
        return [
            SidebarTabDefinition(
                "files", "Files", self._build_file_explorer_tab, duplicate_enabled=True
            ),
            SidebarTabDefinition(
                "workspace",
                "Workspace",
                lambda _sidebar: self._build_workspace_tab(),
            ),
            SidebarTabDefinition(
                "chat",
                "Chat",
                build_chat_tab,
            ),
            SidebarTabDefinition(
                "terminal",
                "Terminal",
                build_terminal_tab,
                duplicate_enabled=True,
            ),
            SidebarTabDefinition(
                "calculator",
                "Calculator",
                build_calculator_tab,
                duplicate_enabled=True,
            ),
            SidebarTabDefinition(
                "units",
                "Units",
                lambda _sidebar: self._build_unit_converter_tab(),
                duplicate_enabled=True,
            ),
            SidebarTabDefinition(
                "notes",
                "Notes",
                build_notes_tab,
                duplicate_enabled=True,
            ),
        ]

    def _add_defined_tab(self, definition: SidebarTabDefinition) -> None:
        widget = definition.factory(self)
        self.add_tab(definition.tab_id, definition.title, widget)

    def _build_file_explorer_tab(
        self, _sidebar: UnifiedToolsSidebar
    ) -> QtWidgets.QWidget:
        explorer = ProjectFileExplorer(self._project_root, self)
        explorer.file_open_requested.connect(self.file_open_requested.emit)
        return explorer

    def _build_toolbar(self) -> QtWidgets.QToolBar:
        toolbar = QtWidgets.QToolBar("Sidekick", self)
        toolbar.setObjectName(theme.SIDEKICK_TOOLBAR_OBJECT_NAME)
        toolbar.setMovable(False)
        left_action = toolbar.addAction("Left")
        left_action.triggered.connect(lambda: self.set_dock_area("left"))
        right_action = toolbar.addAction("Right")
        right_action.triggered.connect(lambda: self.set_dock_area("right"))
        toolbar.addSeparator()
        popout_action = toolbar.addAction("Pop Out")
        popout_action.triggered.connect(lambda: self.pop_out_tab(self.active_tab_id()))
        duplicate_action = toolbar.addAction("Duplicate")
        duplicate_action.triggered.connect(
            lambda: self.duplicate_tab(self.active_tab_id())
        )
        toolbar.addSeparator()
        minimize_action = toolbar.addAction("Minimize")
        minimize_action.triggered.connect(
            lambda: self.set_minimized(not self._state.minimized)
        )
        return toolbar

    def _build_workspace_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        widget.setObjectName(theme.SIDEKICK_WORKSPACE_TAB_OBJECT_NAME)
        layout = QtWidgets.QVBoxLayout(widget)
        self._workspace_list = QtWidgets.QListWidget(widget)
        self._workspace_list.setObjectName(theme.SIDEKICK_WORKSPACE_LIST_OBJECT_NAME)
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
        widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
        layout = QtWidgets.QVBoxLayout(widget)
        label = QtWidgets.QLabel(title, widget)
        label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _emit_context(self) -> None:
        self.context_updated.emit(
            {
                "active_tab": self.active_tab_id(),
                "project_root": str(self._project_root),
                "dock_area": self._state.dock_area,
                "minimized": self._state.minimized,
                "visible_tabs": self.visible_tab_ids(),
                "hidden_tabs": self.hidden_tab_ids(),
                "popped_out_tabs": list(self._popout_windows),
                "workspace_variables": [
                    variable.to_metadata() for variable in self.registry.variables()
                ],
            }
        )

    def _sync_tab_order_from_widget(self, *_args: Any) -> None:
        ordered: list[str] = []
        for index in range(self.tabs.count()):
            widget = self.tabs.widget(index)
            for tab_id, tab_widget in self._tab_widgets.items():
                if tab_widget is widget:
                    ordered.append(tab_id)
                    break
        if len(ordered) == len(self._tab_ids):
            self._tab_ids = ordered
        self._emit_context()

    def _apply_tab_state(self, state: SidebarState) -> None:
        for tab_id in list(self._tab_ids):
            if tab_id in state.hidden_tabs:
                self.set_tab_visible(tab_id, False)
        for tab_id in state.tab_order:
            if tab_id in self._tab_definitions and tab_id not in self._tab_ids:
                self.set_tab_visible(tab_id, True)
        for index, tab_id in enumerate(state.tab_order):
            self.move_tab(tab_id, index)

    def _redock_close_event(
        self,
        tab_id: str,
        window: QtWidgets.QMainWindow,
    ) -> Callable[[Any], None]:
        def _handle_close(event: Any) -> None:
            self.redock_tab(tab_id)
            event.ignore()
            window.hide()

        return _handle_close


SidekickSidebar = UnifiedToolsSidebar
