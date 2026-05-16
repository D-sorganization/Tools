"""Default Sidekick tab builders kept outside the sidebar controller."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from . import design_tokens as theme
from .calculator_plotting import CALCULATOR_PLOT_TAB_ID
from .data_explorer_tab import (
    DATA_EXPLORER_TAB_ID,
    DATA_EXPLORER_TAB_SETTINGS,
    build_data_explorer_tab,
)
from .data_processor_tab import DATA_PROCESSOR_TAB_ID, build_data_processor_tab
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtWidgets
from .reporting_tab import build_reporting_tab
from .runtime_tabs import (
    build_calculator_tab,
    build_chat_tab,
    build_notes_tab,
    build_python_repl_tab,
    build_terminal_tab,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)

TabDefinitionFactory = Callable[..., T]
ROTATION_CONVERTER_TAB_ID = "rotation_converter"
FUNCTION_GENERATOR_TAB_ID = "function_generator"


def build_default_tab_definitions(
    sidebar: Any,
    tab_definition: TabDefinitionFactory,
) -> list[T]:
    """Return the standard Sidekick tabs for a host sidebar."""
    return [
        tab_definition(
            "files",
            "Files",
            build_file_explorer_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["files"]),
        ),
        tab_definition(
            "workspace",
            "Workspace",
            build_workspace_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["workspace"]),
        ),
        tab_definition(
            "chat",
            "Chat",
            build_chat_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["chat"]),
        ),
        tab_definition(
            "terminal",
            "Terminal",
            build_terminal_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["terminal"]),
        ),
        tab_definition(
            "python_repl",
            "Python REPL",
            build_python_repl_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["python_repl"]),
        ),
        tab_definition(
            "calculator",
            "Calculator",
            build_calculator_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["calculator"]),
        ),
        tab_definition(
            CALCULATOR_PLOT_TAB_ID,
            "Calculator Plot",
            build_calculator_plot_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["calculator_plot"]),
        ),
        tab_definition(
            DATA_EXPLORER_TAB_ID,
            "Data Explorer",
            build_data_explorer_tab,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]),
            settings=DATA_EXPLORER_TAB_SETTINGS,
        ),
        tab_definition(
            DATA_PROCESSOR_TAB_ID,
            "Data Processor",
            build_data_processor_tab,
            visible=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[DATA_PROCESSOR_TAB_ID]),
        ),
        tab_definition(
            "units",
            "Units",
            build_unit_converter_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["units"]),
        ),
        tab_definition(
            ROTATION_CONVERTER_TAB_ID,
            "Rotation Converter",
            build_rotation_converter_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["rotation_converter"]),
        ),
        tab_definition(
            FUNCTION_GENERATOR_TAB_ID,
            "Function Generator",
            build_function_generator_tab,
            visible=False,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[FUNCTION_GENERATOR_TAB_ID]),
        ),
        tab_definition(
            "notes",
            "Notes",
            build_notes_tab,
            duplicate_enabled=True,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["notes"]),
        ),
        tab_definition(
            "reporting",
            "Reporting",
            build_reporting_tab,
            duplicate_enabled=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP["reporting"]),
        ),
    ]


def build_file_explorer_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the project file explorer tab and forward open-file signals."""
    explorer = ProjectFileExplorer(sidebar.project_root, sidebar)
    explorer.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["files"]["summary"])
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
    workspace_list.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["workspace"]["summary"])
    sidebar.register_workspace_list_widget(workspace_list)
    layout.addWidget(workspace_list)
    refresh_workspace_list(sidebar)
    return widget


def refresh_workspace_list(sidebar: Any) -> None:
    """Refresh the workspace list widget from the sidebar registry."""
    workspace_list = sidebar.workspace_list_widget()
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
    widget = UnitConverterWidget(sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["units"]["summary"])
    return widget


def build_calculator_plot_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Calculator Plot tab with graceful optional dependency handling."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting requires the PyQt6 UI backend.",
        )
    try:
        plot_widget_module = importlib.import_module("plot_engine.pyqt6_widget")
        plot_specs_module = importlib.import_module("plot_engine.specs")
    except Exception as exc:  # noqa: BLE001 - optional plot UI dependencies
        logger.debug("Calculator plot tab unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Calculator Plot",
            "Calculator plotting is unavailable because optional plot UI "
            "dependencies could not be loaded.",
        )

    widget = plot_widget_module.PlotWidget(parent=sidebar)
    widget.setObjectName("SidekickCalculatorPlotTab")
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["calculator_plot"]["summary"])
    widget.set_spec(
        plot_specs_module.PlotSpec(
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
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["rotation_converter"]["summary"])
    return widget


def build_function_generator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Function Generator tab when its PyQt6 surface is available."""
    if QT_API != "PyQt6":
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator requires the PyQt6 UI backend.",
        )
    try:
        registration = importlib.import_module("function_generator.gui_registration")
        gui_info = registration.get_gui_info()
        pyqt_info = gui_info["pyqt6"]
        module = importlib.import_module(pyqt_info["module"])
        widget_type = getattr(module, pyqt_info["class"])
        widget = widget_type(sidebar, use_builtin_theme=False)
    except Exception as exc:  # noqa: BLE001 - optional GUI surface
        logger.debug("Function Generator unavailable for Sidekick: %s", exc)
        return placeholder(
            sidebar,
            "Function Generator",
            "Function Generator is unavailable because optional UI dependencies "
            "could not be loaded.",
        )
    widget.setObjectName(theme.SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[FUNCTION_GENERATOR_TAB_ID]["summary"])
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
