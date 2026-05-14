"""Default Sidekick tab builders kept outside the sidebar controller."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .calculator_assist import CALCULATOR_HELP
from .calculator_plotting import CALCULATOR_PLOT_TAB_ID
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
ROTATION_CONVERTER_TAB_ID = "rotation_converter"


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
            help_metadata=CALCULATOR_HELP.to_metadata(),
        ),
        tab_definition(
            CALCULATOR_PLOT_TAB_ID,
            "Calculator Plot",
            build_calculator_plot_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata={
                "title": "Calculator Plot",
                "summary": (
                    "Build validated plot requests from calculator expressions "
                    "and workspace variables using the shared PlotSpec contract."
                ),
                "source": "upstream_drift_tools.ui.tools_sidebar.calculator_plotting",
            },
        ),
        tab_definition(
            "units",
            "Units",
            build_unit_converter_tab,
            duplicate_enabled=True,
        ),
        tab_definition(
            ROTATION_CONVERTER_TAB_ID,
            "Rotation Converter",
            build_rotation_converter_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata={
                "title": "Rotation Converter",
                "summary": (
                    "Convert between rotation matrices, quaternions, Euler "
                    "angles, axis-angle, rigid transforms, twists, and frames."
                ),
                "source": "rotation_converter.gui_registration",
            },
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
        details = [variable.summary]
        if variable.dtype:
            details.append(variable.dtype)
        if variable.size is not None:
            details.append(f"size={variable.size}")
        label = (
            f"{variable.name}: {variable.type_name} ({', '.join(details)}) "
            f"{variable.preview}"
        )
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


def build_calculator_plot_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Calculator Plot tab with graceful optional dependency handling."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting requires the PyQt6 UI backend.",
        )
    try:
        from plot_engine.pyqt6_widget import PlotWidget
        from plot_engine.specs import PlotSpec
    except Exception as exc:  # noqa: BLE001 - optional plot UI dependencies
        logger.debug("Calculator plot tab unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting is unavailable because optional plot UI "
            "dependencies could not be loaded.",
        )

    widget = PlotWidget(parent=sidebar)
    widget.setObjectName("SidekickCalculatorPlotTab")
    widget.set_spec(
        PlotSpec(
            title="Calculator Plot",
            series=[],
        )
    )
    return widget


def build_rotation_converter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Rotation Converter tab when its PyQt6 surface is available."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Rotation Converter",
            "Rotation Converter requires the PyQt6 UI backend.",
        )
    try:
        module = importlib.import_module("rotation_converter.ui.pyqt6.main_window")
        window_type = module.RotationConverterMainWindow
        widget = window_type(sidebar)
    except Exception as exc:  # noqa: BLE001 - optional GUI surface
        logger.debug("Rotation converter unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Rotation Converter",
            "Rotation Converter is unavailable because optional UI dependencies "
            "could not be loaded.",
        )
    widget.setObjectName(theme.SIDEKICK_ROTATION_CONVERTER_OBJECT_NAME)
    return widget


def placeholder(
    sidebar: Any,
    title: str,
    message: str | None = None,
) -> QtWidgets.QWidget:
    """Build a compact placeholder for optional tabs."""
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(title, widget)
    label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
    label.setWordWrap(True)
    layout.addWidget(label)
    if message:
        detail = QtWidgets.QLabel(message, widget)
        detail.setWordWrap(True)
        layout.addWidget(detail)
    layout.addStretch(1)
    return widget
