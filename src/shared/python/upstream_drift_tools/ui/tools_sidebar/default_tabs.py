"""Default Sidekick tab builders kept outside the sidebar controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtWidgets
from .runtime_tabs import (
    build_calculator_tab,
    build_chat_tab,
    build_notes_tab,
    build_terminal_tab,
)

logger = logging.getLogger(__name__)

TabDefinitionFactory = Callable[..., Any]


def build_default_tab_definitions(
    sidebar: Any,
    tab_definition: TabDefinitionFactory,
) -> list[Any]:
    """Return the standard Sidekick tabs for a host sidebar."""
    return [
        tab_definition(
            "files",
            "Files",
            build_file_explorer_tab,
            duplicate_enabled=True,
        ),
        tab_definition("workspace", "Workspace", build_workspace_tab),
        tab_definition("chat", "Chat", build_chat_tab),
        tab_definition(
            "terminal",
            "Terminal",
            build_terminal_tab,
            duplicate_enabled=True,
        ),
        tab_definition(
            "calculator",
            "Calculator",
            build_calculator_tab,
            duplicate_enabled=True,
        ),
        tab_definition(
            "units",
            "Units",
            build_unit_converter_tab,
            duplicate_enabled=True,
        ),
        tab_definition("notes", "Notes", build_notes_tab, duplicate_enabled=True),
    ]


def build_file_explorer_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the project file explorer tab and forward open-file signals."""
    explorer = ProjectFileExplorer(sidebar.project_root, sidebar)
    explorer.file_open_requested.connect(sidebar.file_open_requested.emit)
    return explorer


def set_project_explorer_root(
    widget: QtWidgets.QWidget | None,
    project_root: Path,
) -> None:
    """Update a file explorer widget when the host changes project roots."""
    if isinstance(widget, ProjectFileExplorer):
        widget.set_project_root(project_root)


def build_workspace_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the workspace variable list tab."""
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_WORKSPACE_TAB_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    workspace_list = QtWidgets.QListWidget(widget)
    workspace_list.setObjectName(theme.SIDEKICK_WORKSPACE_LIST_OBJECT_NAME)
    sidebar._workspace_list = workspace_list
    layout.addWidget(workspace_list)
    refresh_workspace_list(sidebar)
    return widget


def refresh_workspace_list(sidebar: Any) -> None:
    """Refresh the workspace list widget from the sidebar registry."""
    workspace_list = getattr(sidebar, "_workspace_list", None)
    if workspace_list is None:
        return
    workspace_list.clear()
    for variable in sidebar.registry.variables():
        label = f"{variable.name}: {variable.type_name} ({variable.summary})"
        workspace_list.addItem(label)


def build_unit_converter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the unit converter tab when the PyQt widget is available."""
    if QT_API != "PyQt6":
        return placeholder(sidebar, "Unit converter")
    try:
        from upstream_drift_tools.ui.widgets.unit_converter_widget import (
            UnitConverterWidget,
        )
    except Exception as exc:  # noqa: BLE001 - optional GUI widget
        logger.debug("Unit converter unavailable for Sidekick: %s", exc)
        return placeholder(sidebar, "Unit converter")
    return UnitConverterWidget(sidebar)


def placeholder(sidebar: Any, title: str) -> QtWidgets.QWidget:
    """Build a compact placeholder for optional tabs."""
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(title, widget)
    label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addStretch(1)
    return widget
