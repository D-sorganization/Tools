"""Embedded Sidekick runtime widgets for shared utility tabs."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .qt_compat import QT_API, QtCore, QtWidgets
from .registry import WorkspaceRegistry

logger = logging.getLogger(__name__)

SIDEKICK_CHAT_RUNTIME_OBJECT_NAME = "SidekickChatRuntimeTab"
SIDEKICK_TERMINAL_OBJECT_NAME = "SidekickTerminalTab"
SIDEKICK_CALCULATOR_OBJECT_NAME = "SidekickCalculatorTab"
SIDEKICK_NOTES_OBJECT_NAME = "SidekickNotesTab"

_CALCULATOR_RESULT_NAME = "calculator_result"
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
            return widget
    return _build_chat_status_tab(sidebar)


def build_terminal_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded Python terminal tab bound to the workspace registry."""
    return SidekickTerminalWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        parent=sidebar,
    )


def build_calculator_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build an embedded symbolic calculator tab bound to workspace state."""
    return SidekickCalculatorWidget(
        set_variable=sidebar.set_context_variable,
        parent=sidebar,
    )


def build_notes_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a project-persistent notes tab."""
    return SidekickNotesWidget(project_root=sidebar.project_root, parent=sidebar)


class SidekickCalculatorWidget(QtWidgets.QWidget):
    """Compact symbolic calculator suitable for narrow sidebars."""

    def __init__(
        self,
        *,
        set_variable: SetVariable,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if set_variable is None:
            raise ValueError("set_variable must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_CALCULATOR_OBJECT_NAME)
        self._set_variable = set_variable
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._input = QtWidgets.QLineEdit(self)
        self._input.setObjectName("SidekickCalculatorInput")
        self._input.setPlaceholderText("2 + 2, sin(pi/2), diff(x**2, x)")
        self._input.returnPressed.connect(self.evaluate_expression)
        layout.addWidget(self._input)

        self._run_button = QtWidgets.QPushButton("Evaluate", self)
        self._run_button.setObjectName("SidekickCalculatorRun")
        self._run_button.clicked.connect(self.evaluate_expression)
        layout.addWidget(self._run_button)

        self._result = QtWidgets.QLabel("", self)
        self._result.setObjectName("SidekickCalculatorResult")
        self._result.setWordWrap(True)
        layout.addWidget(self._result)
        layout.addStretch(1)

    def evaluate_expression(self) -> None:
        """Evaluate the current expression and publish the result."""
        expression = self._input.text().strip()
        if not expression:
            self._result.setText("Enter an expression.")
            return
        try:
            calculator = _calculator()
            result = calculator.evaluate(expression)
        except Exception as exc:  # noqa: BLE001 - user-facing calculator errors
            logger.debug("Sidekick calculator evaluation failed: %s", exc)
            self._result.setText(f"Error: {exc}")
            return

        text = str(result.result)
        self._result.setText(text)
        self._set_variable(_CALCULATOR_RESULT_NAME, text)


class SidekickTerminalWidget(QtWidgets.QWidget):
    """Small Python execution surface sharing values with Workspace."""

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
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
        self._namespace: dict[str, Any] = {}
        self._load_workspace_namespace()
        _preload_scientific_namespace(self._namespace)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._input = QtWidgets.QPlainTextEdit(self)
        self._input.setObjectName("SidekickTerminalInput")
        self._input.setPlaceholderText("result = np.array([1, 2, 3]).sum()")
        layout.addWidget(self._input, stretch=2)

        self._run_button = QtWidgets.QPushButton("Run", self)
        self._run_button.setObjectName("SidekickTerminalRun")
        self._run_button.clicked.connect(self.execute_script)
        layout.addWidget(self._run_button)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickTerminalOutput")
        self._output.setReadOnly(True)
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
        layout.addWidget(self._status)

        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setObjectName("SidekickNotesEditor")
        self._editor.setPlaceholderText("Project notes")
        self._editor.textChanged.connect(self._schedule_autosave)
        layout.addWidget(self._editor, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self._save = QtWidgets.QPushButton("Save", self)
        self._save.setObjectName("SidekickNotesSave")
        self._save.clicked.connect(self.save_notes)
        row.addWidget(self._save)

        clear = QtWidgets.QPushButton("Clear", self)
        clear.setObjectName("SidekickNotesClear")
        clear.clicked.connect(self.clear_notes)
        row.addWidget(clear)

        restore = QtWidgets.QPushButton("Restore", self)
        restore.setObjectName("SidekickNotesRestore")
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
        from chat.chat_dock_widget import ChatDockWidget
    except Exception as exc:  # noqa: BLE001 - optional chat dependency
        logger.debug("PyQt chat dock unavailable for Sidekick: %s", exc)
        return None

    dock = ChatDockWidget(
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


def _calculator() -> Any:
    from web_applications.calculator.calculator import TI89Calculator

    return TI89Calculator()


def _notes_storage(project_root: Path) -> Any:
    from notes.storage import NotesStorage

    return NotesStorage(project_dir=project_root)


def _preload_scientific_namespace(namespace: dict[str, Any]) -> None:
    for module_name, alias in (
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scipy", "scipy"),
    ):
        with contextlib.suppress(ImportError):
            module = importlib.import_module(module_name)
            namespace[alias] = module
            namespace[module_name] = module


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
