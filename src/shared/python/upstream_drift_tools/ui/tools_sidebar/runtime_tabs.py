"""Embedded Sidekick runtime widgets for shared utility tabs."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .calculator_assist import (
    calculator_predictive_text_enabled,
    calculator_startup_config,
    set_calculator_predictive_text_enabled,
)
from .calculator_runtime import (
    SidekickCalculatorWidget,
)
from .calculator_startup import (
    apply_calculator_startup_imports,
    default_calculator_startup_config,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QT_API, QtCore, QtWidgets
from .registry import WorkspaceRegistry

logger = logging.getLogger(__name__)

SIDEKICK_CHAT_RUNTIME_OBJECT_NAME = "SidekickChatRuntimeTab"
SIDEKICK_TERMINAL_OBJECT_NAME = "SidekickTerminalTab"
SIDEKICK_NOTES_OBJECT_NAME = "SidekickNotesTab"

_RESERVED_NAMESPACE_NAMES = {
    "__builtins__",
    "np",
    "numpy",
    "pd",
    "pandas",
    "scipy",
}

SetVariable = Callable[[str, Any], None]


def build_chat_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the embedded chat tab for a Sidekick sidebar."""
    if QT_API == "PyQt6":
        widget = _build_pyqt_chat_dock(sidebar)
        if widget is not None:
            widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
            return widget
    return _build_chat_status_tab(sidebar)


def build_terminal_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded Python terminal tab bound to the workspace registry."""
    widget = SidekickTerminalWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        terminal_theme=theme.SidekickTerminalTheme.inherited(
            getattr(sidebar, "_design_tokens", None),
        ),
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["terminal"]["summary"])
    return widget


def build_calculator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded symbolic calculator tab bound to workspace state."""
    widget = SidekickCalculatorWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        predictive_text_enabled=calculator_predictive_text_enabled(sidebar),
        startup_import_config=calculator_startup_config(sidebar),
        set_predictive_text_enabled=partial(
            set_calculator_predictive_text_enabled,
            sidebar,
        ),
        refresh_workspace=sidebar.refresh_workspace,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["calculator"]["summary"])
    return widget


def build_notes_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a project-persistent notes tab."""
    widget = SidekickNotesWidget(project_root=sidebar.project_root, parent=sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["notes"]["summary"])
    return widget


class SidekickTerminalWidget(QtWidgets.QWidget):
    """Small Python execution surface sharing values with Workspace."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        terminal_theme: theme.SidekickTerminalTheme | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if set_variable is None:
            raise ValueError("set_variable must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_TERMINAL_OBJECT_NAME)
        self._registry = registry
        self._set_variable = set_variable
        self._terminal_theme = terminal_theme or theme.SidekickTerminalTheme.inherited()
        self._namespace: dict[str, Any] = {}
        self._load_workspace_namespace()
        _preload_scientific_namespace(self._namespace)
        self._build_ui()
        self.apply_terminal_theme(self._terminal_theme)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._input = QtWidgets.QPlainTextEdit(self)
        self._input.setObjectName("SidekickTerminalInput")
        self._input.setPlaceholderText("result = np.array([1, 2, 3]).sum()")
        self._input.setToolTip(
            "Enter Python code that can read and write shared workspace variables."
        )
        layout.addWidget(self._input, stretch=2)

        self._run_button = QtWidgets.QPushButton("Run", self)
        self._run_button.setObjectName("SidekickTerminalRun")
        self._run_button.setToolTip(
            "Execute the current terminal script and export assigned variables."
        )
        self._run_button.clicked.connect(self.execute_script)
        layout.addWidget(self._run_button)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickTerminalOutput")
        self._output.setReadOnly(True)
        self._output.setToolTip("Shows terminal stdout, stderr, and execution errors.")
        layout.addWidget(self._output, stretch=3)

    def execute_script(self) -> None:
        """Execute the current script and export user variables."""
        script = self._input.toPlainText()
        if not script.strip():
            self._append_output("No code to run.")
            return
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            compiled = compile(script, "<sidekick-terminal>", "exec")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, self._namespace, self._namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - terminal reports user code errors
            logger.debug("Sidekick terminal execution failed: %s", exc)
            self._append_output(_format_terminal_output(stdout, stderr, exc))
            return

        self._sync_namespace_to_registry()
        self._append_output(_format_terminal_output(stdout, stderr, None))

    def _load_workspace_namespace(self) -> None:
        for name in self._registry.list_names():
            self._namespace[name] = self._registry.get(name)

    def _sync_namespace_to_registry(self) -> None:
        for name, value in _exportable_values(self._namespace).items():
            self._set_variable(name, value)

    def _append_output(self, text: str) -> None:
        existing = self._output.toPlainText().strip()
        combined = f"{existing}\n{text}" if existing else text
        self._output.setPlainText(combined.strip())

    def apply_terminal_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply terminal-scoped colors without changing global Sidekick QSS."""
        if terminal_theme is None:
            raise ValueError("terminal_theme must be provided")
        self._terminal_theme = terminal_theme
        self.setStyleSheet(terminal_theme.qss(SIDEKICK_TERMINAL_OBJECT_NAME))


class SidekickNotesWidget(QtWidgets.QWidget):
    """Project notes editor with explicit save and debounced persistence."""

    def __init__(
        self,
        *,
        project_root: Path,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if project_root is None:
            raise ValueError("project_root must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_NOTES_OBJECT_NAME)
        self._storage = _notes_storage(project_root)
        self._autosave = QtCore.QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(500)
        self._autosave.timeout.connect(self.save_notes)
        self._build_ui()
        self._editor.setPlainText(self._storage.load_text())

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status = QtWidgets.QLabel("Ready", self)
        self._status.setObjectName("SidekickNotesStatus")
        self._status.setToolTip("Reports the latest notes persistence status.")
        layout.addWidget(self._status)

        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setObjectName("SidekickNotesEditor")
        self._editor.setPlaceholderText("Project notes")
        self._editor.setToolTip(
            "Edit project-scoped notes that persist beside the workspace."
        )
        self._editor.textChanged.connect(self._schedule_autosave)
        layout.addWidget(self._editor, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self._save = QtWidgets.QPushButton("Save", self)
        self._save.setObjectName("SidekickNotesSave")
        self._save.setToolTip("Persist the current notes text immediately.")
        self._save.clicked.connect(self.save_notes)
        row.addWidget(self._save)

        clear = QtWidgets.QPushButton("Clear", self)
        clear.setObjectName("SidekickNotesClear")
        clear.setToolTip(
            "Clear the current note text while keeping the notes file available."
        )
        clear.clicked.connect(self.clear_notes)
        row.addWidget(clear)

        restore = QtWidgets.QPushButton("Restore", self)
        restore.setObjectName("SidekickNotesRestore")
        restore.setToolTip(
            "Restore the latest recycled notes snapshot when one exists."
        )
        restore.clicked.connect(self.restore_latest)
        row.addWidget(restore)
        layout.addLayout(row)

    def save_notes(self) -> None:
        """Persist the current notes text to the project notes file."""
        self._storage.save_text(self._editor.toPlainText())
        self._status.setText("Saved")

    def clear_notes(self) -> None:
        """Clear notes while preserving the storage file."""
        self._editor.setPlainText("")
        self._storage.clear()
        self._status.setText("Cleared")

    def restore_latest(self) -> None:
        """Restore the latest recycled note file when available."""
        item_id = self._storage.latest_recycled_id()
        if item_id is None or self._storage.restore(item_id) is None:
            self._status.setText("Nothing to restore")
            return
        self._editor.setPlainText(self._storage.load_text())
        self._status.setText("Restored")

    def _schedule_autosave(self) -> None:
        self._autosave.start()


def _build_pyqt_chat_dock(sidebar: Any) -> QtWidgets.QWidget | None:
    try:
        chat_module = importlib.import_module("chat.chat_dock_widget")
    except Exception as exc:  # noqa: BLE001 - optional chat dependency
        logger.debug("PyQt chat dock unavailable for Sidekick: %s", exc)
        return None

    dock = chat_module.ChatDockWidget(
        app_context="sidekick",
        app_name="sidekick",
        project_root=sidebar.project_root,
        parent=sidebar,
    )
    dock.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    dock.setTitleBarWidget(QtWidgets.QWidget(dock))
    _disable_dock_chrome(dock)
    return dock


def _build_chat_status_tab(sidebar: Any) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    label = QtWidgets.QLabel(
        "Shared chat is available when the PyQt chat dock is loaded.",
        widget,
    )
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addStretch(1)
    return widget


def _disable_dock_chrome(dock: Any) -> None:
    feature_type = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", None)
    if feature_type is not None:
        dock.setFeatures(feature_type.NoDockWidgetFeatures)
        return
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)


def _notes_storage(project_root: Path) -> Any:
    from notes.storage import NotesStorage

    return NotesStorage(project_dir=project_root)


def _preload_scientific_namespace(namespace: dict[str, Any]) -> None:
    apply_calculator_startup_imports(
        namespace,
        default_calculator_startup_config(),
    )


def _exportable_values(namespace: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in namespace.items()
        if _is_exportable_name(name) and _is_exportable_value(value)
    }


def _is_exportable_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("_")
        and name not in _RESERVED_NAMESPACE_NAMES
    )


def _is_exportable_value(value: Any) -> bool:
    return not isinstance(value, types.ModuleType) and not callable(value)


def _format_terminal_output(
    stdout: io.StringIO,
    stderr: io.StringIO,
    exc: Exception | None,
) -> str:
    parts = [text for text in (stdout.getvalue(), stderr.getvalue()) if text]
    if exc is not None:
        parts.append(f"{type(exc).__name__}: {exc}")
    if not parts:
        return "Executed."
    return "".join(parts).strip()
