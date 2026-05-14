"""Unified tools sidebar public API.

The backend registry/state classes import without Qt. Widget classes are loaded
on demand so non-GUI hosts can still use persistence and workspace contracts.
"""

from __future__ import annotations

from .registry import WorkspaceRegistry, WorkspaceVariable
from .state import SidebarState

__all__ = [
    "ProjectFileExplorer",
    "SidebarState",
    "ToolsSidebarInstallResult",
    "UnifiedToolsSidebar",
    "WorkspaceRegistry",
    "WorkspaceVariable",
    "create_tools_sidebar",
    "install_tools_sidebar",
]


def __getattr__(name: str) -> object:
    if name in {
        "UnifiedToolsSidebar",
        "ToolsSidebarInstallResult",
        "create_tools_sidebar",
        "install_tools_sidebar",
    }:
        from . import sidebar

        return getattr(sidebar, name)
    if name == "ProjectFileExplorer":
        from .project_file_explorer import ProjectFileExplorer

        return ProjectFileExplorer
    raise AttributeError(name)
