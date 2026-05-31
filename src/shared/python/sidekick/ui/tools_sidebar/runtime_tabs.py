"""Embedded Sidekick runtime widgets for shared utility tabs."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import traceback
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

from . import design_tokens as theme
from .appearance import (
    DEFAULT_DARK_PANEL_APPEARANCE,
    PanelAppearance,
    coerce_appearance,
    panel_qss,
)
from .calculator_assist import (
    calculator_predictive_text_enabled,
    calculator_startup_config,
    set_calculator_predictive_text_enabled,
)
from .calculator_runtime import (
    SidekickCalculatorWidget,
)
from .calculator_startup import (
    CalculatorStartupConfig,
    CalculatorStartupResult,
    apply_calculator_startup_imports,
    default_calculator_startup_config,
    default_repl_startup_config,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QT_API, QtCore, QtWidgets
from .registry import WorkspaceRegistry

__all__ = [
    "PythonReplWidget",
    "SIDEKICK_CHAT_RUNTIME_OBJECT_NAME",
    "SIDEKICK_CHAT_STATUS_OBJECT_NAME",
    "SIDEKICK_NOTES_OBJECT_NAME",
    "SIDEKICK_PYTHON_REPL_OBJECT_NAME",
    "SIDEKICK_TERMINAL_OBJECT_NAME",
    "SetVariable",
    "SidekickNotesWidget",
    "SidekickPythonReplWidget",
    "build_calculator_tab",
    "build_chat_tab",
    "build_notes_tab",
    "build_python_repl_tab",
    "build_terminal_tab",
]

_logger = logging.getLogger(__name__)

SIDEKICK_CHAT_RUNTIME_OBJECT_NAME = "SidekickChatRuntimeTab"
SIDEKICK_CHAT_STATUS_OBJECT_NAME = "SidekickChatStatusTab"
# UpstreamDrift #5617: this object name is retained for backward compatibility
# with downstream tests and styling. The widget is now SidekickPythonReplWidget;
# the new OS terminal uses SIDEKICK_OS_TERMINAL_OBJECT_NAME (os_terminal module).
SIDEKICK_TERMINAL_OBJECT_NAME = "SidekickTerminalTab"
SIDEKICK_PYTHON_REPL_OBJECT_NAME = SIDEKICK_TERMINAL_OBJECT_NAME
SIDEKICK_NOTES_OBJECT_NAME = "SidekickNotesTab"

_DEFAULT_CHAT_ACCENT_COLOR = "#FF8800"
_CHAT_INSTALL_HINT = (
    "Install the chat extras to enable the embedded dock: "
    "pip install 'upstream-drift-tools[chat]' "
    "(or the minimum: pip install PyQt6)."
)


def _resolve_accent_color(theme_provider: Any) -> str:
    """Return the accent color for the chat dock from ``theme_provider``.

    Falls back to :data:`_DEFAULT_CHAT_ACCENT_COLOR` when ``theme_provider``
    is ``None`` or exposes none of the supported color APIs. Never raises;
    a misshaped provider must not crash chat-tab construction.
    """
    if theme_provider is None:
        return _DEFAULT_CHAT_ACCENT_COLOR

    # Preferred path: dict-style color map via get_current_colors() (the
    # ThemeProviderProtocol used by ChatDockWidget and theme.theme_manager).
    try:
        getter = getattr(theme_provider, "get_current_colors", None)
        if callable(getter):
            colors = getter()
            if isinstance(colors, dict):
                accent = colors.get("accent")
                if isinstance(accent, str) and accent:
                    return accent
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider.get_current_colors() failed: %s", exc)

    # Token-style providers occasionally expose tokens().accent or accent_color().
    try:
        tokens = getattr(theme_provider, "tokens", None)
        if callable(tokens):
            token_obj = tokens()
            accent = getattr(token_obj, "accent", None)
            if isinstance(accent, str) and accent:
                return accent
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider.tokens() failed: %s", exc)

    try:
        accent_attr = getattr(theme_provider, "accent_color", None)
        if callable(accent_attr):
            accent = accent_attr()
            if isinstance(accent, str) and accent:
                return accent
        elif isinstance(accent_attr, str) and accent_attr:
            return accent_attr
    except Exception as exc:  # noqa: BLE001 - optional theme surface
        _logger.debug("theme_provider.accent_color failed: %s", exc)

    return _DEFAULT_CHAT_ACCENT_COLOR


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
    """Build the OS-level terminal tab (UpstreamDrift #5617).

    The widget launches a real interactive shell (bash, zsh, pwsh,
    powershell, cmd, or a WSL distro) backed by a PTY when ``ptyprocess``
    or ``pywinpty`` is installed, and falls back to plain subprocess
    pipes otherwise.
    """
    # Local import keeps the heavy os_terminal module out of the import path
    # for headless hosts that don't reach this code path.
    from .os_terminal import SidekickOsTerminalWidget

    widget = SidekickOsTerminalWidget(
        project_root=sidebar.project_root,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["terminal"]["summary"])
    return widget


def build_python_repl_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Python REPL tab (UpstreamDrift #5617).

    This is the widget formerly named ``SidekickTerminalWidget``. It runs
    bounded Python snippets against the shared workspace registry; see
    :class:`SidekickPythonReplWidget`.
    """
    startup_config, appearance = _repl_settings_from_sidebar(sidebar)
    widget = SidekickPythonReplWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        terminal_theme=theme.SidekickTerminalTheme.inherited(
            getattr(sidebar, "_design_tokens", None),
        ),
        startup_config=startup_config,
        appearance=appearance,
        parent=sidebar,
    )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["python_repl"]["summary"])
    return widget


def _repl_settings_from_sidebar(
    sidebar: Any,
) -> tuple[CalculatorStartupConfig, PanelAppearance]:
    """Read persisted REPL startup imports + appearance from ``sidebar``.

    Defensive: if the settings store is not yet configured or the payload is
    malformed, fall back to the default scientific bundle and dark panel
    appearance so tab construction never fails.
    """
    values: dict[str, Any] = {}
    getter = getattr(sidebar, "tab_settings", None)
    if callable(getter):
        try:
            payload = getter("python_repl")
            if isinstance(payload, dict) and isinstance(payload.get("values"), dict):
                values = payload["values"]
        except Exception:  # noqa: BLE001 - degrade to defaults on store error
            _logger.debug("Reading python_repl tab settings failed", exc_info=True)
    raw_imports = values.get("startup_imports")
    if raw_imports in (None, ""):
        startup_config = default_repl_startup_config()
    else:
        try:
            startup_config = CalculatorStartupConfig.from_list(raw_imports)
        except (TypeError, ValueError):
            startup_config = default_repl_startup_config()
    return startup_config, coerce_appearance(values, DEFAULT_DARK_PANEL_APPEARANCE)


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


class _ReplWorker(QtCore.QThread):
    """Execute a Python script off the GUI thread (F6).

    Emits ``finished(output)`` on the GUI thread when the script
    completes so PythonReplWidget can safely update its output pane.
    The worker does NOT hold a reference to the widget; all UI updates
    go through the signal.
    """

    finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        script: str,
        namespace: dict[str, Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._script = script
        self._namespace = namespace
        # Captured formatted output, populated by run() before finished is
        # emitted.  Exposed so callers can fetch the result deterministically
        # after wait(), without relying on event-loop signal delivery
        # (UpstreamDrift #5616 / issue #3138 REPL output regression).
        self._output: str = ""

    def run(self) -> None:  # noqa: ANN201 - Qt override
        """Run the user script and emit ``finished`` with formatted output."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        exception: Exception | None = None
        last_result: Any = _SENTINEL
        compiled_exec = None
        compiled_eval = None
        try:
            with contextlib.suppress(SyntaxError):
                compiled_eval = compile(self._script, "<sidekick-repl>", "eval")
            if compiled_eval is None:
                compiled_exec = compile(self._script, "<sidekick-repl>", "exec")
        except Exception as exc:  # noqa: BLE001 - report compile errors
            exception = exc

        if exception is None:
            try:
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    if compiled_eval is not None:
                        last_result = eval(  # noqa: S307  # nosec B307
                            compiled_eval, self._namespace, self._namespace
                        )
                    else:
                        assert compiled_exec is not None
                        exec(  # noqa: S102  # nosec B102
                            compiled_exec, self._namespace, self._namespace
                        )
            except Exception as exc:  # noqa: BLE001 - REPL reports user errors
                _logger.debug("Sidekick REPL execution failed: %s", exc)
                exception = exc

        output = _format_terminal_output(stdout, stderr, exception)
        if (
            exception is None
            and last_result is not _SENTINEL
            and last_result is not None
        ):
            output = f"{repr(last_result)}\n{output}" if output else repr(last_result)
        self._output = output
        self.finished.emit(output)


class PythonReplWidget(QtWidgets.QWidget):
    """Reusable Python REPL bound to a :class:`WorkspaceRegistry`.

    DRY: this is the single Python execution surface — both the Terminal tab
    and the MATLAB-home command window embed an instance of this widget.

    Args:
        registry: Workspace registry holding shared variables. Required.
        set_variable: Callback ``(name, value)`` used to export user
            assignments back to the host registry. Required.
        object_name: Qt object name; defaults to a stable widget id.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``registry`` or ``set_variable`` is missing or wrong type.
    """

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        object_name: str = SIDEKICK_PYTHON_REPL_OBJECT_NAME,
        startup_config: CalculatorStartupConfig | None = None,
        appearance: PanelAppearance | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise TypeError("registry must be provided")
        if not isinstance(registry, WorkspaceRegistry):
            raise TypeError("registry must be a WorkspaceRegistry")
        if set_variable is None:
            raise TypeError("set_variable must be provided")
        if not callable(set_variable):
            raise TypeError("set_variable must be callable")
        super().__init__(parent)
        self.setObjectName(object_name)
        self._registry = registry
        self._set_variable = set_variable
        self._namespace: dict[str, Any] = {}
        self._history: list[str] = []
        self._startup_config = startup_config or default_repl_startup_config()
        self._appearance = appearance or DEFAULT_DARK_PANEL_APPEARANCE
        self._load_workspace_namespace()
        _preload_scientific_namespace(self._namespace, self._startup_config)
        self._build_ui()
        self.apply_appearance(self._appearance)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._input_label = QtWidgets.QLabel("Python input", self)
        self._input_label.setObjectName("SidekickPythonReplInputLabel")
        self._input_label.setToolTip(
            "Type Python here. Assigned names are exported to the Workspace tab."
        )
        layout.addWidget(self._input_label)

        self._input = QtWidgets.QPlainTextEdit(self)
        self._input.setObjectName("SidekickPythonReplInput")
        self._input.setPlaceholderText(
            "Type Python here, then press Run …  e.g.  result = np.array([1, 2]).sum()"
        )
        self._input.setToolTip(
            "Enter Python code that can read and write shared workspace variables."
        )
        layout.addWidget(self._input, stretch=2)

        # F6: run-row — Run + Cancel buttons side-by-side.
        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(6)

        self._run_button = QtWidgets.QPushButton("Run", self)
        self._run_button.setObjectName("SidekickPythonReplRun")
        self._run_button.setToolTip(
            "Execute the current script and export assigned variables."
        )
        self._run_button.clicked.connect(self._on_run_clicked)
        run_row.addWidget(self._run_button)

        self._cancel_button = QtWidgets.QPushButton("Cancel", self)
        self._cancel_button.setObjectName("SidekickPythonReplCancel")
        self._cancel_button.setToolTip(
            "Attempt to interrupt a long-running script (best-effort)."
        )
        self._cancel_button.setEnabled(False)
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self._cancel_button)

        layout.addLayout(run_row)

        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setObjectName("SidekickPythonReplStatus")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._output_label = QtWidgets.QLabel("Output", self)
        self._output_label.setObjectName("SidekickPythonReplOutputLabel")
        layout.addWidget(self._output_label)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickPythonReplOutput")
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("stdout, stderr, and results appear here.")
        self._output.setToolTip("Shows stdout, stderr, and execution errors.")
        layout.addWidget(self._output, stretch=3)

        # F6: worker thread reference (None when idle).
        self._worker: _ReplWorker | None = None

    def _on_run_clicked(self) -> None:
        self.execute(self._input.toPlainText())

    def execute(self, script: str) -> None:
        """Execute ``script`` against the shared namespace.

        Runs the script on a background QThread (F6) so the GUI stays
        responsive during long computations. The Run button is disabled and
        a 'Running…' label is shown while the worker is active; a Cancel
        button allows the user to terminate the worker thread (best-effort).

        Args:
            script: Python source to execute.  Must be a ``str``.

        Raises:
            TypeError: If ``script`` is not a ``str``.
        """
        if not isinstance(script, str):
            raise TypeError("script must be a str")
        if not script.strip():
            self._append_output("No code to run.")
            return

        # Prevent re-entrant execution.
        if self._worker is not None and self._worker.isRunning():
            _logger.debug(
                "REPL execution already in progress; ignoring re-entrant call"
            )
            return

        self._history.append(script.strip())
        self._set_running(True)

        # Snapshot the namespace into the worker so the GUI thread and
        # the worker thread never share a mutable reference at the same time.
        worker = _ReplWorker(script, self._namespace, parent=self)
        self._worker = worker
        worker.start()
        # Block until the worker finishes and deliver its output directly.
        # Relying on the queued ``finished`` signal alone left the output pane
        # empty whenever no Qt event loop was spinning (e.g. unit tests and
        # synchronous callers), regressing REPL/Run output (issue #3138).
        worker.wait()
        self._on_execution_finished(worker._output)  # noqa: SLF001

    def _set_running(self, running: bool) -> None:
        """Toggle Run/Cancel controls and the 'Running…' status label (F6)."""
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._cancel_button.setVisible(running)
        self._status_label.setText("Running\u2026" if running else "")
        self._status_label.setVisible(running)

    def _on_cancel_clicked(self) -> None:
        """Best-effort cancel: terminate the worker thread (F6)."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(500)
            self._append_output("[Cancelled]")
        self._set_running(False)

    def _on_execution_finished(self, output: str) -> None:
        """Slot called on the GUI thread when the worker emits ``finished`` (F6)."""
        # Sync namespace changes made by the worker back into self._namespace
        # so the workspace registry and subsequent executions share the updates.
        if self._worker is not None:
            self._namespace.update(self._worker._namespace)  # noqa: SLF001
        self._sync_namespace_to_registry()
        self._set_running(False)
        self._append_output(output)

    def output_text(self) -> str:
        """Return the current output pane text."""
        return str(self._output.toPlainText())

    def history(self) -> tuple[str, ...]:
        """Return submitted scripts in oldest-to-newest order."""
        return tuple(self._history)

    def apply_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply REPL-scoped colors. Single-token theme handoff (LOD)."""
        if terminal_theme is None:
            raise TypeError("terminal_theme must be provided")
        self.setStyleSheet(terminal_theme.qss(self.objectName()))

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the REPL surfaces.

        Single-value handoff (LOD): the panel passes a validated
        :class:`PanelAppearance`; the widget renders it via shared QSS.
        """
        if not isinstance(appearance, PanelAppearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._appearance = appearance
        self.setStyleSheet(panel_qss(self.objectName(), appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._appearance

    def startup_config(self) -> CalculatorStartupConfig:
        """Return the startup-import config currently backing the REPL."""
        return self._startup_config

    def apply_startup_config(
        self, config: CalculatorStartupConfig
    ) -> CalculatorStartupResult:
        """Re-import the configured packages into the live namespace.

        Lets the user change preloaded scientific packages without
        rebuilding the tab. Missing optional packages degrade to warnings.
        """
        if not isinstance(config, CalculatorStartupConfig):
            raise TypeError("config must be a CalculatorStartupConfig")
        self._startup_config = config
        result = apply_calculator_startup_imports(self._namespace, config)
        self._sync_namespace_to_registry()
        return result

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


_SENTINEL = object()


class SidekickPythonReplWidget(QtWidgets.QWidget):
    """Small Python execution surface sharing values with Workspace.

    UpstreamDrift #5617: renamed from ``SidekickTerminalWidget``. The name
    is more honest — this widget runs a bounded Python REPL, not an OS
    shell. The new ``SidekickOsTerminalWidget`` (in
    :mod:`upstream_drift_tools.ui.tools_sidebar.os_terminal`) provides the
    real PTY-backed shell. Object name, child widget object names, and
    tooltips are preserved so existing styling and tests continue to work.

    Kept as a thin shell so existing tests and Terminal-tab plumbing
    (theming, object-name lookup) continue to work. All REPL behaviour
    lives in :class:`PythonReplWidget` (DRY).
    """

    def __init__(
        self,
        *,
        registry: WorkspaceRegistry,
        set_variable: SetVariable,
        terminal_theme: theme.SidekickTerminalTheme | None = None,
        startup_config: CalculatorStartupConfig | None = None,
        appearance: PanelAppearance | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        if set_variable is None:
            raise ValueError("set_variable must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_TERMINAL_OBJECT_NAME)
        self._terminal_theme = terminal_theme or theme.SidekickTerminalTheme.inherited()
        self._repl = PythonReplWidget(
            registry=registry,
            set_variable=set_variable,
            startup_config=startup_config,
            appearance=appearance,
            parent=self,
        )
        # Preserve legacy attribute names that hosts/tests inspect.
        self._registry = registry
        self._set_variable = set_variable
        self._namespace = self._repl._namespace  # noqa: SLF001 - intentional alias
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._repl)
        # Expose the historic input/output/run object names by re-tagging the
        # nested widgets so existing findChild() queries keep working.
        self._repl._input.setObjectName("SidekickTerminalInput")  # noqa: SLF001
        self._repl._output.setObjectName("SidekickTerminalOutput")  # noqa: SLF001
        self._repl._run_button.setObjectName("SidekickTerminalRun")  # noqa: SLF001
        self.apply_terminal_theme(self._terminal_theme)
        # Appearance (visible borders + user colours) is applied last so it is
        # authoritative over the legacy inherited terminal theme.
        self.apply_appearance(self._repl.appearance())

    def execute_script(self) -> None:
        """Execute the current script and export user variables (legacy API)."""
        self._repl.execute(self._repl._input.toPlainText())  # noqa: SLF001

    def apply_terminal_theme(self, terminal_theme: theme.SidekickTerminalTheme) -> None:
        """Apply terminal-scoped colors without changing global Sidekick QSS."""
        if terminal_theme is None:
            raise ValueError("terminal_theme must be provided")
        self._terminal_theme = terminal_theme
        self.setStyleSheet(terminal_theme.qss(SIDEKICK_TERMINAL_OBJECT_NAME))

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the REPL (delegates inward)."""
        if not isinstance(appearance, PanelAppearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._repl.apply_appearance(appearance)
        self.setStyleSheet(panel_qss(SIDEKICK_TERMINAL_OBJECT_NAME, appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._repl.appearance()

    def startup_config(self) -> CalculatorStartupConfig:
        """Return the startup-import config backing the REPL."""
        return self._repl.startup_config()

    def apply_startup_config(
        self, config: CalculatorStartupConfig
    ) -> CalculatorStartupResult:
        """Re-import configured packages into the live namespace."""
        return self._repl.apply_startup_config(config)


class SidekickNotesWidget(QtWidgets.QWidget):
    """Project note-card editor with explicit save and debounced persistence."""

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
        self._store = _note_card_store(project_root)
        self._active_card_id: str | None = None
        self._autosave = QtCore.QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(500)
        self._autosave.timeout.connect(self.save_notes)
        self._build_ui()
        self._load_first_card()
        self._apply_board_style()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status = QtWidgets.QLabel("Ready", self)
        self._status.setObjectName("SidekickNotesStatus")
        self._status.setToolTip("Reports the latest notes persistence status.")
        layout.addWidget(self._status)

        self._card_frame = QtWidgets.QFrame(self)
        self._card_frame.setObjectName("SidekickNotesCard")
        card_layout = QtWidgets.QVBoxLayout(self._card_frame)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        self._color = QtWidgets.QLineEdit(self._card_frame)
        self._color.setObjectName("SidekickNotesCardColor")
        self._color.setPlaceholderText("#fff7cc")
        self._color.setToolTip("Sets the active note card color as a #RRGGBB value.")
        card_layout.addWidget(self._color)

        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setObjectName("SidekickNotesEditor")
        self._editor.setPlaceholderText("Project notes")
        self._editor.setToolTip("Edit the active project-scoped markdown note card.")
        self._editor.textChanged.connect(self._schedule_autosave)
        card_layout.addWidget(self._editor, stretch=1)
        layout.addWidget(self._card_frame, stretch=1)

        self._board_color = QtWidgets.QLineEdit(self)
        self._board_color.setObjectName("SidekickNotesBoardColor")
        self._board_color.setPlaceholderText("#f7f7f7")
        self._board_color.setToolTip("Sets the notes screen background color.")
        layout.addWidget(self._board_color)

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

        apply_colors = QtWidgets.QPushButton("Apply Colors", self)
        apply_colors.setObjectName("SidekickNotesApplyColors")
        apply_colors.setToolTip("Validate and persist note and screen colors.")
        apply_colors.clicked.connect(self.apply_colors)
        row.addWidget(apply_colors)
        layout.addLayout(row)

    def save_notes(self) -> None:
        """Persist the current notes text to the active markdown card."""
        color = self._color.text().strip() or "#fff7cc"
        if self._active_card_id is None:
            card = self._store.create_note(
                "Project Notes",
                self._editor.toPlainText(),
                color=color,
            )
            self._active_card_id = card.note_id
        else:
            self._store.update_note(
                self._active_card_id,
                title="Project Notes",
                markdown_body=self._editor.toPlainText(),
                color=color,
            )
        self.apply_colors(save_note=False)
        self._status.setText("Saved")

    def clear_notes(self) -> None:
        """Clear notes while preserving the active markdown card."""
        self._editor.setPlainText("")
        self.save_notes()
        self._status.setText("Cleared")

    def restore_latest(self) -> None:
        """Restore the latest recycled note file when available."""
        item_id = self._store.latest_recycled_id()
        restored = None if item_id is None else self._store.restore_note(item_id)
        if restored is None:
            self._status.setText("Nothing to restore")
            return
        self._active_card_id = restored.note_id
        self._editor.setPlainText(restored.markdown_body)
        self._color.setText(restored.color)
        self._apply_card_style(restored.color)
        self._status.setText("Restored")

    def apply_colors(self, *, save_note: bool = True) -> None:
        """Validate and persist note-card and board colors."""
        from notes.models import NotesBoardSettings, normalize_color

        note_color = normalize_color(self._color.text().strip() or "#fff7cc")
        board = NotesBoardSettings(
            background_color=self._board_color.text().strip() or "#f7f7f7"
        )
        self._color.setText(note_color)
        self._board_color.setText(board.background_color)
        self._store.save_settings(board)
        self._apply_card_style(note_color)
        self._apply_board_style()
        if save_note:
            self.save_notes()

    def _load_first_card(self) -> None:
        card = self._store.migrate_legacy_text_note()
        if card is None:
            notes = self._store.list_notes()
            card = notes[0] if notes else None
        if card is not None:
            self._active_card_id = card.note_id
            self._editor.setPlainText(card.markdown_body)
            self._color.setText(card.color)
            self._apply_card_style(card.color)
        else:
            self._color.setText("#fff7cc")
        self._board_color.setText(self._store.load_settings().background_color)

    def _apply_board_style(self) -> None:
        color = self._store.load_settings().background_color
        self.setStyleSheet(f"#{SIDEKICK_NOTES_OBJECT_NAME} {{ background: {color}; }}")

    def _apply_card_style(self, color: str) -> None:
        self._card_frame.setStyleSheet(
            "#SidekickNotesCard { "
            f"background: {color}; border: 1px solid #d0d0d0; border-radius: 6px;"
            " }"
        )

    def _schedule_autosave(self) -> None:
        self._autosave.start()


class _SidebarWorkspaceAdapter:
    """Adapt a sidebar :class:`WorkspaceRegistry` to the chat workspace bridge.

    Tools issue #2849. The chat module depends only on the
    ``WorkspaceContextProtocol`` Protocol; this adapter implements that
    contract on top of the existing sidebar registry without leaking the
    Sidekick package back into the chat module.
    """

    def __init__(self, registry: WorkspaceRegistry) -> None:
        if registry is None:
            raise ValueError("registry must be provided")
        self._registry = registry

    def describe(self) -> list[Any]:
        """Return :class:`WorkspaceVariableInfo` snapshots for all variables.

        The return type is annotated as ``list[Any]`` so this module does
        not need to import from the chat package at module-import time;
        the chat dock duck-types against the actual values returned.
        """
        _WorkspaceVariableInfo = importlib.import_module(
            "chat._workspace_protocol"
        ).WorkspaceVariableInfo

        items: list[Any] = []
        for variable in self._registry.variables():
            items.append(
                _WorkspaceVariableInfo(
                    name=variable.name,
                    dtype=variable.dtype or variable.type_name,
                    shape=tuple(variable.shape) if variable.shape else None,
                    preview=variable.preview or "",
                )
            )
        return items

    def read(self, name: str) -> Any:
        """Return the registry value for ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._registry.list_names():
            raise KeyError(name)
        return self._registry.get(name)

    def write(self, name: str, value: Any) -> None:
        """Write ``value`` into the registry under ``name``.

        Raises:
            TypeError: If ``name`` is not a ``str``.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self._registry.set(name, value)


def _build_sidebar_plot_request_sink(sidebar: Any) -> Callable[[Any], None] | None:
    """Return a sink that routes plot requests to the Calculator Plot tab.

    The sink accepts either a dict in the
    :class:`CalculatorPlotRequest`-shaped JSON form or an already-built
    request object, and submits the resulting :class:`PlotSpec` to the
    sidebar's Calculator Plot tab widget. Returns ``None`` when any
    required sidebar attribute is missing (host without calculator
    plotting); the caller logs at DEBUG and degrades silently.
    """
    try:
        from .calculator_plotting import (
            CALCULATOR_PLOT_TAB_ID,
            CalculatorPlotRequest,
            CalculatorPlotSource,
            CalculatorPlotTabConfig,
            build_calculator_plot_spec,
        )
    except Exception as exc:  # noqa: BLE001 - optional plot dependency
        _logger.debug("Calculator plot module unavailable for chat: %s", exc)
        return None

    registry = getattr(sidebar, "registry", None)
    if registry is None:
        _logger.debug("Sidebar has no registry; chat plot sink disabled.")
        return None

    set_tab_visible = getattr(sidebar, "set_tab_visible", None)
    tab_widgets = getattr(sidebar, "_tab_widgets", None)
    if not callable(set_tab_visible) or tab_widgets is None:
        _logger.debug("Sidebar lacks tab APIs; chat plot sink disabled.")
        return None

    def _coerce_request(spec: Any) -> Any:
        if isinstance(spec, CalculatorPlotRequest):
            return spec
        if not isinstance(spec, dict):
            raise TypeError("plot spec must be a dict or CalculatorPlotRequest")
        source_val = spec.get("source")
        source: CalculatorPlotSource
        if isinstance(source_val, CalculatorPlotSource):
            source = source_val
        elif isinstance(source_val, str):
            try:
                source = CalculatorPlotSource(source_val)
            except ValueError:
                source = cast(
                    CalculatorPlotSource,
                    CalculatorPlotSource.WORKSPACE_RESULT,
                )
        else:
            source = cast(
                CalculatorPlotSource,
                CalculatorPlotSource.WORKSPACE_RESULT,
            )
        config_data = spec.get("config")
        config = (
            CalculatorPlotTabConfig(**config_data)
            if isinstance(config_data, dict)
            else CalculatorPlotTabConfig()
        )
        return CalculatorPlotRequest(
            source=source,
            x_ref=spec.get("x_ref"),
            y_ref=spec.get("y_ref"),
            expression=spec.get("expression"),
            x_min=spec.get("x_min"),
            x_max=spec.get("x_max"),
            points=spec.get("points"),
            title=spec.get("title"),
            config=config,
        )

    def _sink(spec: Any) -> None:
        request = _coerce_request(spec)
        plot_spec = build_calculator_plot_spec(request, registry)
        # Tools issue #2849: prefer a hidden-tab activation over dropping
        # the request silently. ``set_tab_visible`` is the canonical
        # sidebar API for this.
        if CALCULATOR_PLOT_TAB_ID not in tab_widgets:
            set_tab_visible(CALCULATOR_PLOT_TAB_ID, True)
        widget = tab_widgets.get(CALCULATOR_PLOT_TAB_ID)
        if widget is None:
            _logger.warning("Calculator Plot tab not available; dropping plot request.")
            return
        set_spec = getattr(widget, "set_spec", None)
        if not callable(set_spec):
            _logger.warning(
                "Calculator Plot tab does not implement set_spec; "
                "dropping plot request."
            )
            return
        set_spec(plot_spec)

    return _sink


def _build_pyqt_chat_dock(sidebar: Any) -> QtWidgets.QWidget | None:
    try:
        chat_module = importlib.import_module("chat.chat_dock_widget")
    except Exception as exc:  # noqa: BLE001 - optional chat dependency
        _logger.debug("PyQt chat dock unavailable for Sidekick: %s", exc)
        # Tools issue #2851: stash the import error so the fallback tab can
        # render a useful diagnostic and retry the import on demand.
        with contextlib.suppress(Exception):
            sidebar._chat_dock_import_error = exc
        return None

    # Tools issue #2766: chat dock no longer hard-imports theme.theme_manager.
    # Inject the manager explicitly so existing visuals are preserved when
    # the theme package is available; otherwise the dock falls back to its
    # built-in dark theme.
    theme_provider: Any = None
    try:
        theme_module = importlib.import_module("theme.theme_manager")
        theme_provider = theme_module.get_theme_manager()
    except Exception as exc:  # noqa: BLE001 - theme is optional at this layer
        _logger.debug("Theme manager unavailable for chat dock: %s", exc)

    # Tools issue #2849: wire optional workspace + plot bridges. Any
    # failure here degrades gracefully — the chat dock continues to work
    # with workspace_provider=None / plot_request_sink=None.
    workspace_provider: Any = None
    plot_request_sink: Callable[[Any], None] | None = None
    try:
        registry = getattr(sidebar, "registry", None)
        if registry is not None:
            workspace_provider = _SidebarWorkspaceAdapter(registry)
        plot_request_sink = _build_sidebar_plot_request_sink(sidebar)
    except Exception as exc:  # noqa: BLE001 - bridge is best-effort
        _logger.debug("Sidekick workspace bridge unavailable for chat: %s", exc)
        workspace_provider = None
        plot_request_sink = None

    # Tools issue #2850: forward sidebar-level overrides for the chat dock's
    # constructor params. Each value has a safe default so a bare sidebar
    # still builds the dock identically to today.
    dock = chat_module.ChatDockWidget(
        app_context="sidekick",
        app_name="sidekick",
        session_id=getattr(sidebar, "chat_session_id", None),
        accent_color=_resolve_accent_color(theme_provider),
        auto_index_on_open=bool(getattr(sidebar, "auto_index_on_open", False)),
        project_root=sidebar.project_root,
        terminal_registry=getattr(sidebar, "terminal_registry", None),
        theme_provider=theme_provider,
        workspace_provider=workspace_provider,
        plot_request_sink=plot_request_sink,
        parent=sidebar,
    )
    dock.setObjectName(SIDEKICK_CHAT_RUNTIME_OBJECT_NAME)
    dock.setTitleBarWidget(QtWidgets.QWidget(dock))
    _disable_dock_chrome(dock)
    # Clear any previously stashed import error: a successful build means the
    # chat module is reachable again.
    with contextlib.suppress(Exception):
        if hasattr(sidebar, "_chat_dock_import_error"):
            sidebar._chat_dock_import_error = None
    return dock


def _format_chat_import_error(exc: BaseException | None) -> str:
    """Return a human-readable explanation for a chat-dock import failure."""
    if exc is None:
        return "Chat dock module could not be loaded. Reason unknown."
    summary = traceback.format_exception_only(type(exc), exc)
    text = "".join(summary).strip()
    return text or repr(exc)


def _build_chat_status_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build a diagnostic fallback widget for the chat tab.

    Replaces the legacy single-label placeholder with a heading, the captured
    import-error traceback, an install hint, and a Retry button that re-runs
    the chat-dock import and swaps this widget out on success.
    """
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(SIDEKICK_CHAT_STATUS_OBJECT_NAME)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    heading = QtWidgets.QLabel("Chat unavailable", widget)
    heading.setObjectName("SidekickChatStatusHeading")
    heading_font = heading.font()
    heading_font.setBold(True)
    point_size = heading_font.pointSize()
    if point_size > 0:
        heading_font.setPointSize(point_size + 2)
    heading.setFont(heading_font)
    heading.setToolTip(
        "The embedded chat dock could not be loaded into this Sidekick session."
    )
    layout.addWidget(heading)

    error_view = QtWidgets.QPlainTextEdit(widget)
    error_view.setObjectName("SidekickChatStatusError")
    error_view.setReadOnly(True)
    monospace = _monospace_font()
    if monospace is not None:
        error_view.setFont(monospace)
    error_view.setToolTip("Captured chat-dock import error.")
    error_view.setPlainText(
        _format_chat_import_error(getattr(sidebar, "_chat_dock_import_error", None))
    )
    layout.addWidget(error_view, stretch=1)

    install_hint = QtWidgets.QLabel(_CHAT_INSTALL_HINT, widget)
    install_hint.setObjectName("SidekickChatStatusInstallHint")
    install_hint.setWordWrap(True)
    install_hint.setToolTip(
        "Suggested install command to enable the embedded chat dock."
    )
    layout.addWidget(install_hint)

    retry = QtWidgets.QPushButton("Retry", widget)
    retry.setObjectName("SidekickChatStatusRetry")
    retry.setToolTip("Re-attempt loading the embedded chat dock.")
    retry.clicked.connect(partial(_retry_chat_dock, sidebar, widget, error_view))
    layout.addWidget(retry)

    return widget


def _monospace_font() -> Any | None:
    """Return a monospace ``QFont`` when QtGui is reachable; else ``None``."""
    try:
        from .qt_compat import QtGui

        font = QtGui.QFont()
        style_hint = getattr(QtGui.QFont, "StyleHint", None)
        if style_hint is not None and hasattr(style_hint, "Monospace"):
            font.setStyleHint(style_hint.Monospace)
        font.setFamily("monospace")
        return font
    except Exception as exc:  # noqa: BLE001 - font tweak is cosmetic only
        _logger.debug("Monospace font unavailable for chat status tab: %s", exc)
        return None


def _retry_chat_dock(
    sidebar: Any,
    fallback_widget: QtWidgets.QWidget,
    error_view: QtWidgets.QPlainTextEdit,
) -> None:
    """Retry the chat-dock import; swap in the real dock on success."""
    dock = _build_pyqt_chat_dock(sidebar)
    if dock is None:
        error_view.setPlainText(
            _format_chat_import_error(getattr(sidebar, "_chat_dock_import_error", None))
        )
        return

    dock.setToolTip(DEFAULT_SIDEBAR_TAB_HELP["chat"]["summary"])
    replaced = _replace_sidebar_tab_widget(sidebar, fallback_widget, dock)
    if not replaced:
        # If we cannot swap (e.g. sidebar lacks the helper), leave the
        # fallback in place but record success so users can re-open the tab.
        _logger.debug(
            "Chat dock rebuilt but sidebar tab swap failed; leaving fallback."
        )
        dock.deleteLater()


def _replace_sidebar_tab_widget(
    sidebar: Any,
    old_widget: QtWidgets.QWidget,
    new_widget: QtWidgets.QWidget,
) -> bool:
    """Swap ``old_widget`` for ``new_widget`` inside ``sidebar.tabs``."""
    replace = getattr(sidebar, "replace_tab_widget", None)
    if callable(replace):
        try:
            return bool(replace(old_widget, new_widget))
        except Exception as exc:  # noqa: BLE001 - sidebar-defined helper
            _logger.debug("sidebar.replace_tab_widget failed: %s", exc)
            return False

    tabs = getattr(sidebar, "tabs", None)
    if tabs is None:
        return False
    index = tabs.indexOf(old_widget)
    if index < 0:
        return False
    title = tabs.tabText(index)
    tooltip = tabs.tabToolTip(index)
    tabs.removeTab(index)
    tabs.insertTab(index, new_widget, title)
    if tooltip:
        tabs.setTabToolTip(index, tooltip)
    tabs.setCurrentIndex(index)
    # Keep the sidebar's stable-id -> widget map in sync when present so that
    # future remove/popout/duplicate operations target the new widget.
    tab_widgets = getattr(sidebar, "_tab_widgets", None)
    if isinstance(tab_widgets, dict):
        for tab_id, widget in list(tab_widgets.items()):
            if widget is old_widget:
                tab_widgets[tab_id] = new_widget
                break
    old_widget.setParent(None)
    old_widget.deleteLater()
    return True


def _disable_dock_chrome(dock: Any) -> None:
    feature_type = getattr(QtWidgets.QDockWidget, "DockWidgetFeature", None)
    if feature_type is not None:
        dock.setFeatures(feature_type.NoDockWidgetFeatures)
        return
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)


def _note_card_store(project_root: Path) -> Any:
    from notes.card_store import NoteCardStore

    return NoteCardStore(project_dir=project_root)


def _preload_scientific_namespace(
    namespace: dict[str, Any],
    config: CalculatorStartupConfig | None = None,
) -> None:
    apply_calculator_startup_imports(
        namespace,
        config or default_calculator_startup_config(),
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
