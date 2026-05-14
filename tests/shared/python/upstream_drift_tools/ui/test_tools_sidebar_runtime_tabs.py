"""Runtime tab contract tests for the unified tools sidebar."""

from __future__ import annotations

from pathlib import Path

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
    assert calculator._registry.get("calculator_result") == [[1, 2], [3, 4]]  # noqa: SLF001
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


def test_sidekick_terminal_and_notes_controls_have_tooltips(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_active_tab("terminal") is True
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
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesSave"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesClear"),
        notes.findChild(QtWidgets.QPushButton, "SidekickNotesRestore"),
    )
    for widget in note_widgets:
        assert widget is not None
        assert widget.toolTip()
