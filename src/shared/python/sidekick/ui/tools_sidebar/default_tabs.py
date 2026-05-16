"""Default Sidekick tab builders kept outside the sidebar controller."""

from __future__ import annotations

import contextlib
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
from .jupyter_tab import JUPYTER_TAB_ID
from .project_file_explorer import ProjectFileExplorer
from .qt_compat import QT_API, QtCore, QtGui, QtWidgets, Signal
from .registry import WorkspaceRegistry
from .reporting_tab import build_reporting_tab
from .runtime_tabs import (
    build_calculator_tab,
    build_chat_tab,
    build_notes_tab,
    build_python_repl_tab,
    build_terminal_tab,
)

QtGui_QStandardItemModel = QtGui.QStandardItemModel
QtGui_QStandardItem = QtGui.QStandardItem

WORKSPACE_TABLE_COLUMNS: tuple[str, ...] = ("Name", "Type", "Size", "Preview")

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
        tab_definition(
            JUPYTER_TAB_ID,
            "Jupyter",
            build_jupyter_tab,
            visible=False,
            help_metadata=dict(DEFAULT_SIDEBAR_TAB_HELP[JUPYTER_TAB_ID]),
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


class WorkspaceTableWidget(QtWidgets.QWidget):
    """MATLAB-style workspace table bound to a :class:`WorkspaceRegistry`.

    Columns: Name / Type / Size / Preview. The widget subscribes to the
    registry so changes are reflected without manual ``refresh()`` calls
    (LOD: receives only the registry reference; no reach into a
    sidebar/theme tree).

    Args:
        registry: Workspace registry providing variables. Required.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``registry`` is missing or not a :class:`WorkspaceRegistry`.
    """

    inspect_requested = Signal(str)
    _COLUMNS = WORKSPACE_TABLE_COLUMNS

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise TypeError("registry must be provided")
        if not isinstance(registry, WorkspaceRegistry):
            raise TypeError("registry must be a WorkspaceRegistry")
        super().__init__(parent)
        self.setObjectName(theme.SIDEKICK_WORKSPACE_TAB_OBJECT_NAME)
        self._registry = registry
        self._build_ui()
        self._subscription = registry.subscribe(self._on_registry_event)
        self.refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._table = QtWidgets.QTableView(self)
        self._table.setObjectName("SidekickWorkspaceTable")
        self._table.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["workspace"]["summary"])
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSortingEnabled(True)
        self._table.doubleClicked.connect(self._on_double_clicked)

        self._model = QtGui_QStandardItemModel(0, len(self._COLUMNS), self)
        self._model.setHorizontalHeaderLabels(list(self._COLUMNS))
        self._table.setModel(self._model)
        layout.addWidget(self._table, stretch=3)

        self._history_label = QtWidgets.QLabel("Command History", self)
        self._history_label.setObjectName("SidekickWorkspaceHistoryLabel")
        layout.addWidget(self._history_label)
        self._history = QtWidgets.QListWidget(self)
        self._history.setObjectName("SidekickWorkspaceHistory")
        self._history.setToolTip("Recent commands run from this sidebar's REPL.")
        layout.addWidget(self._history, stretch=1)

    def column_headers(self) -> tuple[str, ...]:
        """Return column header labels."""
        return tuple(self._COLUMNS)

    def row_data(self) -> list[tuple[str, str, str, str]]:
        """Return current rows as ``(name, type, size, preview)`` tuples."""
        rows: list[tuple[str, str, str, str]] = []
        for row in range(self._model.rowCount()):
            cells: list[str] = []
            for col in range(len(self._COLUMNS)):
                item = self._model.item(row, col)
                cells.append(item.text() if item else "")
            rows.append(tuple(cells))  # type: ignore[arg-type]
        return rows

    def sort_by_column(self, column: int, *, ascending: bool = True) -> None:
        """Sort the table by ``column`` (0=Name)."""
        if not 0 <= column < len(self._COLUMNS):
            raise ValueError("column out of range")
        order = (
            QtCore.Qt.SortOrder.AscendingOrder
            if ascending
            else QtCore.Qt.SortOrder.DescendingOrder
        )
        self._table.sortByColumn(column, order)

    def trigger_inspect(self, name: str) -> None:
        """Emit ``inspect_requested`` for ``name`` (test/programmatic hook)."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty str")
        self.inspect_requested.emit(name)

    def append_history(self, command: str) -> None:
        """Record a command line in the history pane."""
        if not isinstance(command, str) or not command.strip():
            return
        self._history.addItem(command.strip())

    def refresh(self) -> None:
        """Rebuild the table from the current registry contents."""
        self._model.removeRows(0, self._model.rowCount())
        for variable in self._registry.variables():
            self._append_row(variable)

    def _append_row(self, variable: Any) -> None:
        size_text = "" if variable.size is None else str(variable.size)
        items = [
            QtGui_QStandardItem(variable.name),
            QtGui_QStandardItem(variable.type_name),
            QtGui_QStandardItem(size_text),
            QtGui_QStandardItem(variable.preview or ""),
        ]
        for item in items:
            item.setEditable(False)
        self._model.appendRow(items)

    def _on_registry_event(self, event: str, name: str) -> None:
        # Re-render whole table: small N, simple, avoids row-index drift.
        del event, name
        self.refresh()

    def _on_double_clicked(self, index: Any) -> None:
        name_item = self._model.item(index.row(), 0)
        if name_item is not None:
            self.inspect_requested.emit(name_item.text())

    def deleteLater(self) -> None:  # type: ignore[override]
        with contextlib.suppress(Exception):
            self._subscription.unsubscribe()
        super().deleteLater()


def build_workspace_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the workspace variable inspector tab (MATLAB-style table)."""
    widget = WorkspaceTableWidget(registry=sidebar.registry, parent=sidebar)
    sidebar.register_workspace_table_widget(widget)
    return widget


def refresh_workspace_list(sidebar: Any) -> None:
    """Refresh the workspace table widget from the sidebar registry."""
    table_widget = getattr(sidebar, "workspace_table_widget", None)
    table = table_widget() if callable(table_widget) else None
    if table is not None:
        table.refresh()
        return
    # Legacy path: some tests register a plain QListWidget via the old API.
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


def build_jupyter_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Sidekick Jupyter notebook tab.

    The factory is unconditionally registered so the tab is always
    discoverable. When the optional ``nbformat`` dependency is missing,
    the factory returns :class:`JupyterUnavailableWidget` which shows
    an actionable install hint. When the dependency is present the
    tab opens an empty :class:`JupyterNotebookWidget`; loading a
    specific notebook into the tab is wired in Phase 3 (#2877).
    """
    from .jupyter_tab import (
        JupyterNotebookWidget,
        JupyterTabAvailability,
        JupyterUnavailableWidget,
        NotebookDocument,
    )

    available, install_hint = JupyterTabAvailability.check()
    if not available:
        widget = JupyterUnavailableWidget(install_hint=install_hint, parent=sidebar)
    else:
        widget = JupyterNotebookWidget(document=NotebookDocument(), parent=sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[JUPYTER_TAB_ID]["summary"])
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
