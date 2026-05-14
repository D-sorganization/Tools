"""Unified tools sidebar public API.

The backend registry/state classes import without Qt. Widget classes are loaded
on demand so non-GUI hosts can still use persistence and workspace contracts.
"""

from __future__ import annotations

from .command_history import (
    DEFAULT_COMMAND_HISTORY_LIMIT,
    CommandHistoryController,
)
from .design_tokens import (
    SIDEKICK_DESIGN_TOKENS,
    SIDEKICK_DOCK_OBJECT_NAME,
    SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME,
    SIDEKICK_PLACEHOLDER_OBJECT_NAME,
    SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME,
    SIDEKICK_PROJECT_TREE_OBJECT_NAME,
    SIDEKICK_SIDEBAR_OBJECT_NAME,
    SIDEKICK_TAB_BAR_OBJECT_NAME,
    SIDEKICK_TABS_OBJECT_NAME,
    SIDEKICK_TOKEN_NAMES,
    SIDEKICK_TOOLBAR_OBJECT_NAME,
    SIDEKICK_WORKSPACE_LIST_OBJECT_NAME,
    SIDEKICK_WORKSPACE_TAB_OBJECT_NAME,
    SidekickDesignTokens,
    sidekick_qss,
)
from .registry import WorkspaceRegistry, WorkspaceVariable
from .state import SidebarState

__all__ = [
    "SIDEKICK_DESIGN_TOKENS",
    "DEFAULT_COMMAND_HISTORY_LIMIT",
    "CommandHistoryController",
    "ProjectFileExplorer",
    "SidebarState",
    "SidebarTabDefinition",
    "SIDEKICK_DOCK_OBJECT_NAME",
    "SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME",
    "SIDEKICK_PLACEHOLDER_OBJECT_NAME",
    "SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME",
    "SIDEKICK_PROJECT_TREE_OBJECT_NAME",
    "SIDEKICK_SIDEBAR_OBJECT_NAME",
    "SIDEKICK_TAB_BAR_OBJECT_NAME",
    "SIDEKICK_TABS_OBJECT_NAME",
    "SIDEKICK_TOKEN_NAMES",
    "SIDEKICK_TOOLBAR_OBJECT_NAME",
    "SIDEKICK_WORKSPACE_LIST_OBJECT_NAME",
    "SIDEKICK_WORKSPACE_TAB_OBJECT_NAME",
    "SidekickDesignTokens",
    "SidekickSidebar",
    "ToolsSidebarInstallResult",
    "UnifiedToolsSidebar",
    "WorkspaceRegistry",
    "WorkspaceVariable",
    "create_tools_sidebar",
    "install_tools_sidebar",
    "sidekick_qss",
]


def __getattr__(name: str) -> object:
    if name in {
        "UnifiedToolsSidebar",
        "SidekickSidebar",
        "SidebarTabDefinition",
    }:
        from . import sidebar

        return getattr(sidebar, name)
    if name in {
        "ToolsSidebarInstallResult",
        "create_tools_sidebar",
        "install_tools_sidebar",
    }:
        from . import api

        return getattr(api, name)
    if name == "ProjectFileExplorer":
        from .project_file_explorer import ProjectFileExplorer

        return ProjectFileExplorer
    raise AttributeError(name)
