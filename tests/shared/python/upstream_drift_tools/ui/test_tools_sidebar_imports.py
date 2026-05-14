"""Import and optional Qt contract tests for the unified tools sidebar."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

QT_BINDINGS = {"PyQt6", "PySide6", "PyQt5", "PySide2"}


def test_tools_sidebar_backend_imports_without_qt() -> None:
    qt_modules_before = {
        name for name in sys.modules if name.partition(".")[0] in QT_BINDINGS
    }

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_TOKEN_NAMES,
        CommandHistoryController,
        SidebarState,
        SidekickDesignTokens,
        SidekickTerminalTheme,
        WorkspaceRegistry,
    )

    assert CommandHistoryController().commands == ()
    assert SidebarState().active_tab == "files"
    assert WorkspaceRegistry().list() == []
    assert SidekickDesignTokens()["color.accent"] == "#2563eb"
    assert SidekickTerminalTheme.inherited()["background"].startswith("#")
    assert "color.background" in SIDEKICK_TOKEN_NAMES

    qt_modules_after = {
        name for name in sys.modules if name.partition(".")[0] in QT_BINDINGS
    }
    assert qt_modules_after == qt_modules_before


def test_tools_sidebar_backend_imports_without_qt_in_clean_python() -> None:
    env = os.environ.copy()
    pythonpath = [
        str(Path("src").resolve()),
        str(Path("src/shared/python").resolve()),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    script = """
import sys
from upstream_drift_tools.ui.tools_sidebar import (
    CommandHistoryController,
    SidebarState,
    SidekickDesignTokens,
    WorkspaceRegistry,
)

assert CommandHistoryController().commands == ()
assert SidebarState().active_tab == "files"
assert WorkspaceRegistry().list() == []
assert SidekickDesignTokens()["color.accent"] == "#2563eb"
loaded = [
    name for name in sys.modules
    if name.partition(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}
]
assert loaded == [], loaded
rotation_converter_ui = [
    name for name in sys.modules
    if name.startswith("rotation_converter.ui")
]
assert rotation_converter_ui == [], rotation_converter_ui
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_tools_sidebar_public_widget_api_is_lazy() -> None:
    import upstream_drift_tools.ui.tools_sidebar as tools_sidebar

    assert "create_tools_sidebar" in tools_sidebar.__all__
    assert "install_tools_sidebar" in tools_sidebar.__all__
    assert "SidekickSidebar" in tools_sidebar.__all__
    assert "SidebarTabDefinition" in tools_sidebar.__all__
    assert "SidekickDesignTokens" in tools_sidebar.__all__
    assert "CommandHistoryController" in tools_sidebar.__all__
    assert "SidekickTerminalTheme" in tools_sidebar.__all__
    assert "sidekick_qss" in tools_sidebar.__all__


def test_sidekick_token_contract_spans_pyqt_and_web_css() -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidekickDesignTokens

    tokens = SidekickDesignTokens.from_sidekick_tokens(
        {
            "sidekick.color.canvas": "#0f172a",
            "sidekick.color.surface.elevated": "#111827",
            "sidekick.radius.lg": "10px",
        }
    )

    assert tokens["color.background"] == "#0f172a"
    assert tokens["color.surface.raised"] == "#111827"
    assert tokens["radius.panel"] == "10px"
    assert tokens.css_variables()["--sidekick-color-background"] == "#0f172a"
    assert tokens.qss_variables()["sidekick-color-background"] == "#0f172a"

    css_path = Path("src/shared/typescript/theme/theme-variables.css")
    css = css_path.read_text(encoding="utf-8")
    expected_aliases = {
        "--sidekick-color-canvas: var(--theme-bg);",
        "--sidekick-color-surface: var(--theme-group-bg);",
        "--sidekick-color-border: var(--theme-border);",
        "--sidekick-color-text: var(--theme-text);",
        "--sidekick-color-accent: var(--theme-accent);",
        "--sidekick-color-focus: var(--theme-focus);",
        "--sidekick-control-height: 28px;",
    }

    for alias in expected_aliases:
        assert alias in css

    theme_definitions = Path(
        "src/shared/typescript/theme/themeDefinitions.ts"
    ).read_text(encoding="utf-8")
    expected_exports = {
        "export function generateSidekickCSSVariables(",
        "export function applySidekickThemeToElement(",
        "'--sidekick-color-canvas': theme.bg,",
        "'--sidekick-color-surface': theme.groupBg,",
        "'--sidekick-color-accent': theme.accent,",
        "'--sidekick-radius-control': '6px',",
    }
    for export in expected_exports:
        assert export in theme_definitions

    theme_index = Path("src/shared/typescript/theme/index.ts").read_text(
        encoding="utf-8"
    )
    assert "generateSidekickCSSVariables" in theme_index
    assert "applySidekickThemeToElement" in theme_index


def test_tools_sidebar_widget_contract_when_qt_available(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_DOCK_OBJECT_NAME,
        SIDEKICK_PROJECT_TREE_OBJECT_NAME,
        SIDEKICK_SIDEBAR_OBJECT_NAME,
        SIDEKICK_TAB_BAR_OBJECT_NAME,
        SIDEKICK_TABS_OBJECT_NAME,
        SidebarState,
        SidebarTabDefinition,
        SidekickDesignTokens,
        SidekickSidebar,
        UnifiedToolsSidebar,
        create_tools_sidebar,
        install_tools_sidebar,
        sidekick_qss,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = QtWidgets.QMainWindow()
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    dock = sidebar.install_as_dock(host, area="left")

    assert sidebar.objectName() == SIDEKICK_SIDEBAR_OBJECT_NAME
    assert sidebar.tabs.objectName() == SIDEKICK_TABS_OBJECT_NAME
    assert sidebar.tabs.tabBar().objectName() == SIDEKICK_TAB_BAR_OBJECT_NAME
    assert sidebar.styleSheet() == sidekick_qss()
    assert dock.objectName() == SIDEKICK_DOCK_OBJECT_NAME
    assert dock.widget() is sidebar
    assert sidebar.active_tab_id() == "files"
    assert "rotation_converter" not in sidebar.visible_tab_ids()
    assert "rotation_converter" in sidebar.hidden_tab_ids()
    calculator_definition = sidebar._tab_definitions["calculator"]
    assert calculator_definition.help_metadata["title"] == "Calculator"
    assert "solve(x**2 - 4, x)" in calculator_definition.help_metadata["examples"]
    assert "Workspace" in calculator_definition.help_metadata["tips"]
    plot_definition = sidebar._tab_definitions["calculator_plot"]
    assert plot_definition.title == "Calculator Plot"
    assert plot_definition.visible is False
    assert plot_definition.duplicate_enabled is True
    assert plot_definition.help_metadata["title"] == "Calculator Plot"
    assert "workspace variables" in plot_definition.help_metadata["summary"]
    rotation_definition = sidebar._tab_definitions["rotation_converter"]
    assert rotation_definition.visible is False
    assert rotation_definition.duplicate_enabled is True
    assert rotation_definition.help_metadata["title"] == "Rotation Converter"
    assert (
        sidebar.findChild(QtWidgets.QTreeView, SIDEKICK_PROJECT_TREE_OBJECT_NAME)
        is not None
    )
    assert sidebar.set_active_tab("terminal") is True
    assert sidebar.snapshot_state().active_tab == "terminal"
    assert sidebar.set_active_tab("missing") is False

    sidebar.set_context_variable("case", {"id": 1})
    assert sidebar.registry.get("case") == {"id": 1}

    created = create_tools_sidebar(project_root=tmp_path, parent=host)
    assert isinstance(created, UnifiedToolsSidebar)
    assert SidekickSidebar is UnifiedToolsSidebar

    result = install_tools_sidebar(host, project_root=tmp_path)
    assert result.installed is True
    assert result.sidebar is not None
    assert result.dock_widget is not None
    assert result.dock_widget.widget() is result.sidebar

    state = SidebarState(
        dock_area="left",
        active_tab="notes",
        tab_order=["notes", "files"],
        hidden_tabs=["chat"],
    )
    configured = UnifiedToolsSidebar(project_root=tmp_path, state=state)
    assert configured.visible_tab_ids()[0] == "notes"
    assert "chat" in configured.hidden_tab_ids()
    assert configured.set_tab_visible("rotation_converter", False) is True

    assert configured.move_tab("files", 0) is True
    assert configured.visible_tab_ids()[0] == "files"
    configured.set_minimized(True)
    assert configured.snapshot_state().minimized is True
    configured.set_minimized(False)
    assert configured.set_dock_area("right") is True
    assert configured.snapshot_state().dock_area == "right"

    popped = configured.pop_out_tab("notes")
    assert popped is not None
    assert "notes" in configured.snapshot_state().popped_out_tabs
    assert configured.redock_tab("notes") is True
    assert "notes" in configured.visible_tab_ids()

    duplicate_id = configured.duplicate_tab("calculator")
    assert duplicate_id is not None
    assert duplicate_id in configured.visible_tab_ids()

    configured.rename_tab("calculator", "  Steam calc  ")
    assert configured.tab_display_name("calculator") == "Steam calc"
    assert "calculator" in configured.visible_tab_ids()
    calculator_index = configured.visible_tab_ids().index("calculator")
    assert configured.tabs.tabText(calculator_index) == "Steam calc"
    assert configured.active_tab_id() == duplicate_id
    assert configured.snapshot_state().tab_display_names == {"calculator": "Steam calc"}

    with pytest.raises(ValueError):
        configured.rename_tab("calculator", "   ")
    with pytest.raises(KeyError):
        configured.rename_tab("missing", "Name")

    duplicated_from_custom = configured.duplicate_tab("calculator")
    assert duplicated_from_custom is not None
    duplicate_index = configured.visible_tab_ids().index(duplicated_from_custom)
    assert configured.tabs.tabText(duplicate_index) == "Calculator 3"

    configured.reset_tab_display_name("calculator")
    assert configured.tab_display_name("calculator") == "Calculator"
    assert configured.tabs.tabText(calculator_index) == "Calculator"
    assert configured.snapshot_state().tab_display_names == {}

    restored = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState(
            tab_display_names={
                "calculator": "Restored calc",
                "missing": "Stale custom name",
            }
        ),
    )
    restored_index = restored.visible_tab_ids().index("calculator")
    assert restored.tabs.tabText(restored_index) == "Restored calc"
    assert restored.snapshot_state().tab_display_names == {
        "calculator": "Restored calc"
    }

    custom = UnifiedToolsSidebar(
        project_root=tmp_path,
        design_tokens=SidekickDesignTokens({"color.background": "#ffffff"}),
        tab_definitions=[
            SidebarTabDefinition(
                "scratch",
                "Scratch",
                lambda sidebar: QtWidgets.QLabel("scratch", sidebar),
                duplicate_enabled=True,
            )
        ],
    )
    assert custom.visible_tab_ids() == ["scratch"]

    installed = install_tools_sidebar(
        host,
        project_root=tmp_path,
        sidekick_tokens={"sidekick.color.canvas": "#0f172a"},
    )
    assert installed.sidebar is not None
    assert "#0f172a" in installed.sidebar.styleSheet()

    themed = create_tools_sidebar(project_root=tmp_path, theme_name="dark")
    assert "#1a1d23" in themed.styleSheet()
    assert "#e1e4e8" in themed.styleSheet()

    installed_theme = install_tools_sidebar(
        host,
        project_root=tmp_path,
        theme_name="dark",
    )
    assert installed_theme.sidebar is not None
    assert "#1a1d23" in installed_theme.sidebar.styleSheet()

    themed.set_theme("light")
    assert "#ffffff" in themed.styleSheet()
    assert "#212529" in themed.styleSheet()
    themed.set_design_tokens(SidekickDesignTokens({"color.background": "#123456"}))
    assert "#123456" in themed.styleSheet()
    assert custom.duplicate_tab("scratch") == "scratch#1"


def test_sidekick_custom_tab_names_update_popout_titles(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    sidebar.rename_tab("notes", "Run notes")
    popped = sidebar.pop_out_tab("notes")

    assert popped is not None
    assert popped.windowTitle() == "Sidekick - Run notes"

    sidebar.rename_tab("notes", "Session notes")
    assert popped.windowTitle() == "Sidekick - Session notes"


def test_sidekick_rotation_converter_import_is_lazy(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    before = {name for name in sys.modules if name.startswith("rotation_converter.ui")}

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    after = {name for name in sys.modules if name.startswith("rotation_converter.ui")}
    assert after == before
    assert "rotation_converter" in sidebar.hidden_tab_ids()


def test_sidekick_rotation_converter_unavailable_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_PLACEHOLDER_OBJECT_NAME,
        UnifiedToolsSidebar,
        default_tabs,
    )

    def fail_rotation_converter_import(name: str) -> object:
        if name == "rotation_converter.ui.pyqt6.main_window":
            raise ImportError("missing optional rotation converter UI")
        return original_import_module(name)

    original_import_module = default_tabs.importlib.import_module
    monkeypatch.setattr(
        default_tabs.importlib,
        "import_module",
        fail_rotation_converter_import,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("rotation_converter", True) is True
    assert sidebar.set_active_tab("rotation_converter") is True
    tab = sidebar.tabs.currentWidget()
    assert tab is not None
    assert tab.objectName() == SIDEKICK_PLACEHOLDER_OBJECT_NAME
    assert "rotation_converter" in sidebar.visible_tab_ids()


def test_sidekick_default_runtime_tabs_are_real_widgets(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_PLACEHOLDER_OBJECT_NAME,
        UnifiedToolsSidebar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    for tab_id in ("chat", "terminal", "calculator", "notes"):
        assert sidebar.set_active_tab(tab_id) is True
        tab = sidebar.tabs.currentWidget()
        assert tab is not None
        assert tab.objectName() != SIDEKICK_PLACEHOLDER_OBJECT_NAME


def test_sidekick_terminal_tab_uses_scoped_inherited_theme(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SidekickDesignTokens,
        UnifiedToolsSidebar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        design_tokens=SidekickDesignTokens(
            {
                "color.text": "#111111",
                "color.surface": "#222222",
                "color.accent": "#333333",
                "color.selection": "#444444",
            },
        ),
    )

    assert sidebar.set_active_tab("terminal") is True
    terminal = sidebar.tabs.currentWidget()

    assert "QWidget#SidekickTerminalTab QPlainTextEdit" in terminal.styleSheet()
    assert "color: #111111" in terminal.styleSheet()
    assert "background: #222222" in terminal.styleSheet()
    assert "border: 1px solid #333333" in terminal.styleSheet()


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
    result = calculator.findChild(QtWidgets.QLabel, "SidekickCalculatorResult")
    assert expression is not None
    assert predictive is not None
    assert evaluate is not None
    assert result is not None
    for widget in (expression, predictive, evaluate, result):
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
    assert sidebar.registry.get("calculator_result") == "4"

    assert sidebar.set_active_tab("terminal") is True
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
    assert editor is not None
    assert save is not None

    editor.setPlainText("persistent note")
    save.click()

    reloaded = UnifiedToolsSidebar(project_root=tmp_path)
    assert reloaded.set_active_tab("notes") is True
    reloaded_editor = reloaded.tabs.currentWidget().findChild(
        QtWidgets.QPlainTextEdit,
        "SidekickNotesEditor",
    )
    assert reloaded_editor is not None
    assert reloaded_editor.toPlainText() == "persistent note"


def test_sidekick_calculator_predictive_preference_survives_state_round_trip(
    tmp_path: Path,
) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    state_path = tmp_path / "sidekick-state.json"
    SidebarState(calculator_predictive_text_enabled=True).save_json(state_path)

    state = SidebarState.load_json(state_path)
    sidebar = UnifiedToolsSidebar(project_root=tmp_path, state=state)

    assert sidebar.set_active_tab("calculator") is True
    calculator = sidebar.tabs.currentWidget()
    predictive = calculator.findChild(
        QtWidgets.QCheckBox,
        "SidekickCalculatorPredictiveText",
    )

    assert predictive is not None
    assert predictive.isChecked() is True
    assert sidebar.snapshot_state().calculator_predictive_text_enabled is True
    assert "solve(" in calculator.suggestions_for("sol")
