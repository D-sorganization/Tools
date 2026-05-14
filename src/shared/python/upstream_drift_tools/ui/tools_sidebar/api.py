"""Public Sidekick sidebar factory and host-install helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qt_compat import QtWidgets
from .registry import WorkspaceRegistry
from .sidebar import SidebarTabDefinition, UnifiedToolsSidebar
from .state import SidebarState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolsSidebarInstallResult:
    """Result returned by the shared host install helper."""

    installed: bool
    reason: str
    sidebar: UnifiedToolsSidebar | None = None
    dock_widget: QtWidgets.QDockWidget | None = None


def create_tools_sidebar(
    project_root: str | Path | None = None,
    registry: WorkspaceRegistry | None = None,
    state: SidebarState | None = None,
    tab_definitions: list[SidebarTabDefinition] | None = None,
    parent: QtWidgets.QWidget | None = None,
    context_provider: Callable[[], Any] | None = None,
    **_: Any,
) -> UnifiedToolsSidebar:
    """Create a sidebar widget using the stable shared factory contract."""
    sidebar = UnifiedToolsSidebar(
        project_root=project_root,
        registry=registry,
        state=state,
        tab_definitions=tab_definitions,
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
    tab_definitions: list[SidebarTabDefinition] | None = None,
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
        tab_definitions=tab_definitions,
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
