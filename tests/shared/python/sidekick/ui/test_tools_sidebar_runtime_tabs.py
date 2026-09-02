# ruff: noqa: E501
"""Runtime tab contract tests for the unified tools sidebar."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


def test_sidekick_calculator_terminal_and_notes_runtime_flow(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_active_tab("calculator") is True
    calculator = sidebar.tabs.currentWidget()
    expression = calculator.findChild(QtWidgets.QLineEdit, "SidekickCalculatorInput")
    predictive = calculator.findChild(
        QtWidgets.QCheckBox,
        "SidekickCalculatorPredictiveText",
    )
    evaluate = calculator.findChild(QtWidgets.QPushButton, "SidekickCalculatorRun")
    save_workspace = calculator.findChild(
        QtWidgets.QPushButton,
        "SidekickCalculatorSaveWorkspace",
    )
    load_workspace = calculator.findChild(
        QtWidgets.QPushButton,
        "SidekickCalculatorLoadWorkspace",
    )
    workspace_command = calculator.findChild(
        QtWidgets.QLineEdit,
        "SidekickWorkspaceCommandInput",
    )
    run_workspace_command = calculator.findChild(
        QtWidgets.QPushButton,
        "SidekickWorkspaceCommandRun",
    )
    result = calculator.findChild(QtWidgets.QLabel, "SidekickCalculatorResult")
    assert expression is not None
    assert predictive is not None
    assert evaluate is not None
    assert save_workspace is not None
    assert load_workspace is not None
    assert workspace_command is not None
    assert run_workspace_command is not None
    assert result is not None
    widgets = (
        expression,
        predictive,
        evaluate,
        save_workspace,
        load_workspace,
        workspace_command,
        run_workspace_command,
        result,
    )
    for widget in widgets:
        assert widget.toolTip()

    assert calculator.predictive_text_enabled is False
    assert calculator.suggestions_for("sol") == ()
    predictive.setChecked(True)
    assert sidebar.snapshot_state().calculator_predictive_text_enabled is True
    assert "solve(" in calculator.suggestions_for("sol")
    sidebar.set_context_variable("solution_rate", 3.14)
    assert "solution_rate" in calculator.suggestions_for("sol")
    predictive.setChecked(False)
    assert sidebar.snapshot_state().calculator_predictive_text_enabled is False
    assert calculator.suggestions_for("sol") == ()

    expression.setText("2 + 2")
    evaluate.click()

    assert result.text() == "4"
    assert sidebar.registry.get("calculator_result") is None
    assert calculator._registry.get("calculator_result") == "4"  # noqa: SLF001

    expression.setText("Matrix([[1, 2], [3, 4]])")
    evaluate.click()

    matrix_result = calculator._registry.describe("calculator_result")  # noqa: SLF001
    assert result.text() == "[[1, 2], [3, 4]]"
    assert sidebar.registry.get("calculator_result") is None
    assert calculator._registry.get("calculator_result") == [
        [1, 2],
        [3, 4],
    ]  # noqa: SLF001
    assert matrix_result.shape == (2, 2)
    assert matrix_result.size == 4

    workspace_action_calls: list[str] = []
    calculator._workspace_actions.save_workspace = (  # noqa: SLF001
        lambda: workspace_action_calls.append("save")
    )
    calculator._workspace_actions.load_workspace = (  # noqa: SLF001
        lambda: workspace_action_calls.append("load")
    )
    save_workspace.click()
    load_workspace.click()

    assert workspace_action_calls == ["save", "load"]

    workspace_command.setText("global answer = 42")
    run_workspace_command.click()
    assert sidebar.registry.get("answer") == 42
    assert "answer" in result.text()

    workspace_command.setText("local gain = [1, 2, 3]")
    run_workspace_command.click()
    assert calculator._registry.get("gain") == [1, 2, 3]  # noqa: SLF001
    assert calculator._workspace_command_history.commands == (  # noqa: SLF001
        "global answer = 42",
        "local gain = [1, 2, 3]",
    )

    # UpstreamDrift #5617: the Python REPL now lives on the ``python_repl`` tab
    # id; the ``terminal`` tab hosts the new OS-level shell.
    assert sidebar.set_active_tab("python_repl") is True
    terminal = sidebar.tabs.currentWidget()
    script = terminal.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalInput")
    run = terminal.findChild(QtWidgets.QPushButton, "SidekickTerminalRun")
    output = terminal.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalOutput")
    assert script is not None
    assert run is not None
    assert output is not None

    script.setPlainText("answer = 21 * 2\nprint(answer)")
    run.click()

    assert "42" in output.toPlainText()
    assert sidebar.registry.get("answer") == 42

    assert sidebar.set_active_tab("notes") is True
    notes = sidebar.tabs.currentWidget()
    editor = notes.findChild(QtWidgets.QPlainTextEdit, "SidekickNotesEditor")
    save = notes.findChild(QtWidgets.QPushButton, "SidekickNotesSave")
    card_color = notes.findChild(QtWidgets.QLineEdit, "SidekickNotesCardColor")
    board_color = notes.findChild(QtWidgets.QLineEdit, "SidekickNotesBoardColor")
    apply_colors = notes.findChild(QtWidgets.QPushButton, "SidekickNotesApplyColors")
    assert editor is not None
    assert save is not None
    assert card_color is not None
    assert board_color is not None
    assert apply_colors is not None

    editor.setPlainText("persistent note")
    card_color.setText("#C7F9CC")
    board_color.setText("#EEE4FF")
    apply_colors.click()
    save.click()

    reloaded = UnifiedToolsSidebar(project_root=tmp_path)
    assert reloaded.set_active_tab("notes") is True
    reloaded_editor = reloaded.tabs.currentWidget().findChild(
        QtWidgets.QPlainTextEdit,
        "SidekickNotesEditor",
    )
    assert reloaded_editor is not None
    assert reloaded_editor.toPlainText() == "persistent note"
    reloaded_notes = reloaded.tabs.currentWidget()
    reloaded_card_color = reloaded_notes.findChild(
        QtWidgets.QLineEdit,
        "SidekickNotesCardColor",
    )
    reloaded_board_color = reloaded_notes.findChild(
        QtWidgets.QLineEdit,
        "SidekickNotesBoardColor",
    )
    assert reloaded_card_color is not None
    assert reloaded_board_color is not None
    assert reloaded_card_color.text() == "#c7f9cc"
    assert reloaded_board_color.text() == "#eee4ff"


def test_sidekick_tab_context_menu_exposes_help_text(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar
    from upstream_drift_tools.ui.tools_sidebar.tab_context_menu import (
        build_tab_context_menu,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    menu = build_tab_context_menu(sidebar, "calculator")
    actions = {
        action.text().replace("&", ""): action
        for action in menu.actions()
        if not action.isSeparator()
    }

    assert "Help" in actions
    assert actions["Help"].toolTip()
    assert actions["Help"].statusTip()
    assert actions["Rename"].toolTip()
    assert actions["Close"].statusTip()


def test_sidekick_open_tab_supports_launcher_facing_ids(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.open_tab("os_terminal") is True
    assert sidebar.active_tab_id() == "terminal"
    assert sidebar.open_tab("workspace") is True
    assert sidebar.active_tab_id() == "workspace"
    assert sidebar.open_tab("jupyter") is True
    assert sidebar.active_tab_id() == "jupyter"
    assert "jupyter" in sidebar.visible_tab_ids()
    assert sidebar.open_tab("missing") is False


def test_sidekick_terminal_and_notes_controls_have_tooltips(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    # UpstreamDrift #5617: the Python REPL controls live on the
    # ``python_repl`` tab id after the OS-terminal split.
    assert sidebar.set_active_tab("python_repl") is True
    terminal = sidebar.tabs.currentWidget()
    terminal_widgets = (
        terminal.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalInput"),
        terminal.findChild(QtWidgets.QPushButton, "SidekickTerminalRun"),
        terminal.findChild(QtWidgets.QPlainTextEdit, "SidekickTerminalOutput"),
    )
    for widget in terminal_widgets:
        assert widget is not None
        assert widget.toolTip()

    assert sidebar.set_active_tab("notes") is True
    notes = sidebar.tabs.currentWidget()
    note_widgets = (
        notes.findChild(QtWidgets.QLabel, "SidekickNotesStatus"),
        notes.findChild(QtWidgets.QPlainTextEdit, "SidekickNotesEditor"),
        notes.findChild(QtWidgets.QLineEdit, "SidekickNotesCardColor"),
        notes.findChild(QtWidgets.QLineEdit, "SidekickNotesBoardColor"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesSave"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesClear"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesRestore"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesApplyColors"),
    )
    for widget in note_widgets:
        assert widget is not None
        assert widget.toolTip()


def test_sidekick_data_explorer_preview_export_and_handoff(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "temperature,status\n293.15,ok\n294.0,warn\n295.2,ok\n",
        encoding="utf-8",
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    launches: list[tuple[str, dict[str, object]]] = []
    sidebar.tool_launch_requested.connect(
        lambda tool_id, payload: launches.append((tool_id, payload))
    )

    assert sidebar.set_active_tab("data_explorer") is True
    explorer = sidebar.tabs.currentWidget()
    assert explorer is not None

    path_input = explorer.findChild(QtWidgets.QLineEdit, "SidekickDataExplorerPath")
    load_button = explorer.findChild(
        QtWidgets.QPushButton,
        "SidekickDataExplorerLoad",
    )
    column_input = explorer.findChild(
        QtWidgets.QLineEdit,
        "SidekickDataExplorerColumns",
    )
    variable_input = explorer.findChild(
        QtWidgets.QLineEdit,
        "SidekickDataExplorerVariable",
    )
    export_button = explorer.findChild(
        QtWidgets.QPushButton,
        "SidekickDataExplorerExport",
    )
    handoff_button = explorer.findChild(
        QtWidgets.QPushButton,
        "SidekickDataExplorerSendToDataProcessor",
    )
    preview_table = explorer.findChild(
        QtWidgets.QTableWidget,
        "SidekickDataExplorerPreviewTable",
    )

    assert path_input is not None
    assert load_button is not None
    assert column_input is not None
    assert variable_input is not None
    assert export_button is not None
    assert handoff_button is not None
    assert preview_table is not None

    path_input.setText(str(csv_path))
    load_button.click()

    assert preview_table.rowCount() == 3
    assert preview_table.columnCount() == 2

    column_input.setText("temperature")
    variable_input.setText("temperature_preview")
    export_button.click()

    assert sidebar.registry.get("temperature_preview") == [293.15, 294.0, 295.2]

    handoff_button.click()

    assert launches == [
        (
            "data_processor",
            {
                "tool_id": "data_processor",
                "source_path": str(csv_path),
                "source_format": "csv",
                "selected_columns": ["temperature"],
                "row_limit": 20,
            },
        )
    ]


class _ChatDockSpy:
    """Container for the latest ChatDockWidget construction kwargs.

    The actual widget class is built lazily inside ``_install_fake_chat_module``
    so it can extend the real ``QDockWidget`` available at test time.
    """

    last_kwargs: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls.last_kwargs = {}


class _FakeThemeProvider:
    """Theme provider returning a known accent color via the dict protocol."""

    def __init__(self, accent: str = "#123456") -> None:
        self._accent = accent

    def get_current_colors(self) -> dict[str, str]:
        return {"accent": self._accent, "bg": "#222"}


def _build_fake_sidebar(
    project_root: Path,
    *,
    terminal_registry: Any = None,
    auto_index_on_open: bool = False,
    chat_session_id: str | None = None,
) -> Any:
    """Return a minimal QWidget-based sidebar double for runtime-tab tests."""
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    sidebar = QtWidgets.QWidget()
    sidebar.project_root = project_root
    sidebar.terminal_registry = terminal_registry
    sidebar.auto_index_on_open = auto_index_on_open
    sidebar.chat_session_id = chat_session_id
    return sidebar


_QT_APP_REF: list[Any] = []


def _ensure_qt_widgets() -> Any:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Hold a process-wide reference so the QApplication is not GC'd after the
    # helper returns. Without this, subsequent QWidget creations crash because
    # PyQt6 reaps the application instance when its last reference dies.
    if not _QT_APP_REF:
        _QT_APP_REF.append(app)
    return QtWidgets


def _install_fake_chat_module(
    monkeypatch: pytest.MonkeyPatch,
) -> type[_ChatDockSpy]:
    """Install a synthetic ``chat.chat_dock_widget`` exposing a spy widget."""
    from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets

    _ChatDockSpy.reset()

    class _SpyChatDockWidget(QtWidgets.QDockWidget):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__("AI Chat", kwargs.get("parent"))
            _ChatDockSpy.last_kwargs = dict(kwargs)

    from chat._chat_dock_widget_qt import (
        ChatConnectionConfig,
        ChatIntegrationHooks,
        ChatPresentationConfig,
    )

    module = types.ModuleType("chat.chat_dock_widget")
    module.ChatDockWidget = _SpyChatDockWidget  # type: ignore[attr-defined]
    module.ChatConnectionConfig = ChatConnectionConfig  # type: ignore[attr-defined]
    module.ChatPresentationConfig = ChatPresentationConfig  # type: ignore[attr-defined]
    module.ChatIntegrationHooks = ChatIntegrationHooks  # type: ignore[attr-defined]

    chat_pkg = sys.modules.get("chat")
    if chat_pkg is None or not isinstance(chat_pkg, types.ModuleType):
        chat_pkg = types.ModuleType("chat")
        chat_pkg.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chat", chat_pkg)
    monkeypatch.setitem(sys.modules, "chat.chat_dock_widget", module)
    return _ChatDockSpy


def test_chat_dock_forwards_sidebar_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2850: forward four extra params from sidebar to ChatDockWidget."""
    _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sentinel_registry = object()
    sidebar = _build_fake_sidebar(
        project_root=tmp_path,
        terminal_registry=sentinel_registry,
        auto_index_on_open=True,
        chat_session_id="abc",
    )
    sidebar_theme = _FakeThemeProvider(accent="#ABCDEF")

    spy_cls = _install_fake_chat_module(monkeypatch)

    # Force the theme.theme_manager import path to return our fake provider.
    fake_theme_module = types.ModuleType("theme.theme_manager")
    fake_theme_module.get_theme_manager = lambda: sidebar_theme  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "theme.theme_manager", fake_theme_module)
    theme_pkg = sys.modules.get("theme")
    if theme_pkg is None:
        theme_pkg = types.ModuleType("theme")
        theme_pkg.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "theme", theme_pkg)

    dock = runtime_tabs._build_pyqt_chat_dock(sidebar)
    assert dock is not None
    kwargs = spy_cls.last_kwargs
    connection = kwargs["connection"]
    presentation = kwargs["presentation"]
    integrations = kwargs["integrations"]

    assert integrations.terminal_registry is sentinel_registry
    assert presentation.auto_index_on_open is True
    assert connection.session_id == "abc"
    assert presentation.accent_color == "#ABCDEF"
    # Existing forwarded params remain intact.
    assert connection.project_root == tmp_path
    assert connection.app_context == "sidekick"
    assert connection.app_name == "sidekick"


def test_chat_dock_accent_color_falls_back_when_theme_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Misshaped theme providers must yield the documented fallback color."""
    _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sidebar = _build_fake_sidebar(project_root=tmp_path)
    spy_cls = _install_fake_chat_module(monkeypatch)

    # Make the theme import fail entirely.
    def _raise_theme(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("theme unavailable")

    monkeypatch.setattr(
        importlib,
        "import_module",
        _wrap_import_module(_raise_theme, {"theme.theme_manager"}),
    )

    dock = runtime_tabs._build_pyqt_chat_dock(sidebar)
    assert dock is not None
    assert spy_cls.last_kwargs["presentation"].accent_color == "#FF8800"


def _wrap_import_module(
    raiser: Any,
    targets: set[str],
) -> Any:
    """Return an ``import_module`` shim that raises for ``targets`` only."""
    real = importlib.import_module

    def _shim(name: str, package: str | None = None) -> Any:
        if name in targets:
            return raiser(name)
        return real(name, package)

    return _shim


def test_chat_dock_stashes_import_error_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2851: a failed import must be stashed on the sidebar."""
    _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sidebar = _build_fake_sidebar(project_root=tmp_path)
    raised = ImportError("PyQt6 not installed")

    def _shim(name: str, package: str | None = None) -> Any:
        if name == "chat.chat_dock_widget":
            raise raised
        return importlib.import_module(name, package)

    monkeypatch.setattr(runtime_tabs.importlib, "import_module", _shim)

    result = runtime_tabs._build_pyqt_chat_dock(sidebar)
    assert result is None
    assert sidebar._chat_dock_import_error is raised


def test_chat_status_tab_shows_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2851: fallback tab must surface the captured error + install hint."""
    QtWidgets = _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sidebar = _build_fake_sidebar(project_root=tmp_path)
    sidebar._chat_dock_import_error = ImportError("PyQt6 not installed")

    widget = runtime_tabs._build_chat_status_tab(sidebar)
    try:
        assert widget.objectName() == runtime_tabs.SIDEKICK_CHAT_STATUS_OBJECT_NAME

        error_view = widget.findChild(
            QtWidgets.QPlainTextEdit, "SidekickChatStatusError"
        )
        assert error_view is not None
        assert "PyQt6 not installed" in error_view.toPlainText()

        install_hint = widget.findChild(
            QtWidgets.QLabel, "SidekickChatStatusInstallHint"
        )
        assert install_hint is not None
        assert "pip install" in install_hint.text()

        retry = widget.findChild(QtWidgets.QPushButton, "SidekickChatStatusRetry")
        assert retry is not None
        assert retry.text() == "Retry"
    finally:
        widget.deleteLater()


def test_chat_status_tab_default_message_when_no_error(
    tmp_path: Path,
) -> None:
    """Without a stashed error the fallback widget should still build cleanly."""
    QtWidgets = _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sidebar = _build_fake_sidebar(project_root=tmp_path)
    widget = runtime_tabs._build_chat_status_tab(sidebar)
    try:
        error_view = widget.findChild(
            QtWidgets.QPlainTextEdit, "SidekickChatStatusError"
        )
        assert error_view is not None
        assert "Reason unknown" in error_view.toPlainText()
    finally:
        widget.deleteLater()


def test_chat_status_tab_retry_button_retries_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2851: Retry triggers another chat-module import attempt."""
    QtWidgets = _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    sidebar = _build_fake_sidebar(project_root=tmp_path)

    raised = ImportError("PyQt6 not installed")
    call_log: list[str] = []

    def _failing_shim(name: str, package: str | None = None) -> Any:
        if name == "chat.chat_dock_widget":
            call_log.append(name)
            raise raised
        return importlib.import_module(name, package)

    monkeypatch.setattr(runtime_tabs.importlib, "import_module", _failing_shim)

    # Initial build fails and stashes the error.
    assert runtime_tabs._build_pyqt_chat_dock(sidebar) is None
    assert sidebar._chat_dock_import_error is raised

    widget = runtime_tabs._build_chat_status_tab(sidebar)
    try:
        retry = widget.findChild(QtWidgets.QPushButton, "SidekickChatStatusRetry")
        assert retry is not None

        # Retry while the import still fails -> another import attempt occurs
        # and the error text refreshes.
        retry.click()
        assert call_log.count("chat.chat_dock_widget") >= 2
        error_view = widget.findChild(
            QtWidgets.QPlainTextEdit, "SidekickChatStatusError"
        )
        assert error_view is not None
        assert "PyQt6 not installed" in error_view.toPlainText()
    finally:
        widget.deleteLater()


def test_chat_status_tab_retry_swaps_widget_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful retry swaps the fallback widget for the real dock."""
    QtWidgets = _ensure_qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar import runtime_tabs

    raised = ImportError("PyQt6 not installed")

    class _Sidebar(QtWidgets.QWidget):
        def __init__(self, project_root: Path) -> None:
            super().__init__()
            self.project_root = project_root
            self.tabs = QtWidgets.QTabWidget(self)
            self._tab_widgets: dict[str, QtWidgets.QWidget] = {}
            self._chat_dock_import_error = raised

    sidebar = _Sidebar(tmp_path)

    fallback = runtime_tabs._build_chat_status_tab(sidebar)
    sidebar.tabs.addTab(fallback, "Chat")
    sidebar._tab_widgets["chat"] = fallback

    # Install a spy chat module so the second import succeeds.
    _install_fake_chat_module(monkeypatch)

    retry = fallback.findChild(QtWidgets.QPushButton, "SidekickChatStatusRetry")
    assert retry is not None
    retry.click()

    # Index 0 should now hold something other than the original fallback widget.
    swapped = sidebar.tabs.widget(0)
    assert swapped is not fallback
    assert sidebar._tab_widgets["chat"] is swapped
    sidebar.deleteLater()
