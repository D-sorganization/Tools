"""Integration and unit tests for MainWindowFileCommandsMixin."""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QAction, QCloseEvent  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from rate_of_closure.application.commands import AppCommandId  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def window(qtbot, tmp_path):  # type: ignore[no-untyped-def]
    settings = QSettings(str(tmp_path / "test_nav.ini"), QSettings.Format.IniFormat)
    win = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(win)
    yield win
    win._workspace_baseline = win._fingerprint(win._capture_workspace_state())
    win.close()


def _action(window: RateOfClosureMainWindow, command_id: str) -> QAction:
    action = window.findChild(QAction, command_id)
    assert action is not None, command_id
    return cast(QAction, action)


def test_initial_workspace_is_clean(window) -> None:  # type: ignore[no-untyped-def]
    assert not window.workspace_is_dirty()
    assert window._workspace_path is None
    recent_action = _action(window, AppCommandId.FILE_OPEN_RECENT_WORKSPACE.value)
    assert not recent_action.isEnabled()


def test_dirty_tracking_detects_scenario_change(window) -> None:  # type: ignore[no-untyped-def]
    assert not window.workspace_is_dirty()
    window._controls._spins["clubhead_speed_mph"].setValue(135.0)
    assert window.workspace_is_dirty()

    # Reverting to initial 120.0 clears dirty
    window._controls._spins["clubhead_speed_mph"].setValue(120.0)
    assert not window.workspace_is_dirty()


def test_save_workspace_as_and_save_workspace(window, tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_file = tmp_path / "test_session.roc-workspace.json"
    window._controls._spins["clubhead_speed_mph"].setValue(130.0)
    assert window.workspace_is_dirty()

    with patch(
        "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getSaveFileName",
        return_value=(str(save_file), "ROC Workspace (*.roc-workspace.json)"),
    ):
        window.save_workspace_as()

    assert save_file.exists()
    assert not window.workspace_is_dirty()
    assert window._workspace_path == save_file
    title = window.windowTitle()
    assert "test_session" in title or "Rate of Closure" in title

    # Verify JSON structure
    content = json.loads(save_file.read_text(encoding="utf-8"))
    assert content["schema"] == "rate_of_closure.workspace"
    assert content["schema_version"] == 3

    # Modifying again marks dirty
    window._controls._spins["clubhead_speed_mph"].setValue(132.0)
    assert window.workspace_is_dirty()

    # Direct save without dialog
    window.save_workspace()
    assert not window.workspace_is_dirty()

    updated = json.loads(save_file.read_text(encoding="utf-8"))
    delivery = updated["model_session"]["data"]["scenario"]
    assert delivery["clubhead_speed_mph"] == 132.0


def test_open_workspace_restores_saved_state(window, tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_file = tmp_path / "saved_for_open.roc-workspace.json"
    window._controls._spins["clubhead_speed_mph"].setValue(128.5)

    with patch(
        "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getSaveFileName",
        return_value=(str(save_file), "ROC Workspace (*.roc-workspace.json)"),
    ):
        window.save_workspace_as()

    # Change to a different value
    window._controls._spins["clubhead_speed_mph"].setValue(110.0)
    assert window._controls.scenario().clubhead_speed_mph == 110.0

    # Open the saved file, confirming discard of unsaved changes
    with (
        patch(
            "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getOpenFileName",
            return_value=(str(save_file), "ROC Workspace (*.roc-workspace.json)"),
        ),
        patch.object(
            QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard
        ),
    ):
        window.open_workspace()

    assert not window.workspace_is_dirty()
    assert window._controls.scenario().clubhead_speed_mph == 128.5
    assert window._workspace_path == save_file


def test_open_recent_workspace(window, tmp_path) -> None:  # type: ignore[no-untyped-def]
    save_file = tmp_path / "recent_session.roc-workspace.json"
    window._controls._spins["clubhead_speed_mph"].setValue(125.0)

    with patch(
        "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getSaveFileName",
        return_value=(str(save_file), "ROC Workspace (*.roc-workspace.json)"),
    ):
        window.save_workspace_as()

    recent_action = _action(window, AppCommandId.FILE_OPEN_RECENT_WORKSPACE.value)
    assert recent_action.isEnabled()
    assert "recent_session.roc-workspace.json" in recent_action.text()

    window._controls._spins["clubhead_speed_mph"].setValue(115.0)

    # Trigger open recent, confirming discard of unsaved changes
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard
    ):
        recent_action.trigger()
    assert not window.workspace_is_dirty()
    assert window._controls.scenario().clubhead_speed_mph == 125.0


def test_new_workspace_and_destructive_prompts(window) -> None:  # type: ignore[no-untyped-def]
    window._controls._spins["clubhead_speed_mph"].setValue(135.0)
    assert window.workspace_is_dirty()

    # User cancels new workspace prompt
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel
    ):
        window.new_workspace()
    assert window.workspace_is_dirty()
    assert window._controls.scenario().clubhead_speed_mph == 135.0

    # User confirms discard
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard
    ):
        window.new_workspace()
    assert not window.workspace_is_dirty()
    assert window._controls.scenario().clubhead_speed_mph == 120.0
    assert window._workspace_path is None


def test_close_workspace_prompts_and_resets(window) -> None:  # type: ignore[no-untyped-def]
    window._controls._spins["clubhead_speed_mph"].setValue(133.0)
    assert window.workspace_is_dirty()

    # Cancel close
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel
    ):
        window.close_workspace()
    assert window.workspace_is_dirty()

    # Confirm close
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard
    ):
        window.close_workspace()
    assert not window.workspace_is_dirty()
    assert window._controls.scenario().clubhead_speed_mph == 120.0


def test_close_event_prompt_cancellation(window) -> None:  # type: ignore[no-untyped-def]
    window._controls._spins["clubhead_speed_mph"].setValue(131.0)
    assert window.workspace_is_dirty()
    window.confirm_on_close = True

    # Cancel close event
    event = QCloseEvent()
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel
    ):
        window.closeEvent(event)
    assert not event.isAccepted()

    # Accept close event
    event = QCloseEvent()
    with patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard
    ):
        window.closeEvent(event)
    window._controls._spins["clubhead_speed_mph"].setValue(120.0)
    window._workspace_baseline = window._fingerprint(window._capture_workspace_state())


def test_close_event_unconfirmed_when_confirm_on_close_false(window) -> None:  # type: ignore[no-untyped-def]
    window._controls._spins["clubhead_speed_mph"].setValue(131.0)
    assert window.workspace_is_dirty()
    window.confirm_on_close = False

    event = QCloseEvent()
    with patch.object(QMessageBox, "warning") as mock_warning:
        window.closeEvent(event)
        mock_warning.assert_not_called()
    assert event.isAccepted()
    window.confirm_on_close = True
    window._controls._spins["clubhead_speed_mph"].setValue(120.0)
    window._workspace_baseline = window._fingerprint(window._capture_workspace_state())


def test_import_and_export_workspace_view_layout(window, tmp_path) -> None:  # type: ignore[no-untyped-def]
    layout_file = tmp_path / "layout.roc-view.json"

    with patch(
        "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getSaveFileName",
        return_value=(str(layout_file), "View Layout (*.roc-view.json)"),
    ):
        window.export_workspace()

    assert layout_file.exists()
    layout_data = json.loads(layout_file.read_text(encoding="utf-8"))
    assert layout_data["format"] == "rate_of_closure.view_workspace/2"

    with patch(
        "rate_of_closure.ui.pyqt6.main_window_file_commands.QFileDialog.getOpenFileName",
        return_value=(str(layout_file), "View Layout (*.roc-view.json)"),
    ):
        window.import_workspace()
