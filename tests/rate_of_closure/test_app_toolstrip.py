"""Application toolstrip command and workspace-management contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QAction  # noqa: E402
from PyQt6.QtWidgets import QMenu, QToolBar, QToolButton  # noqa: E402

from rate_of_closure.application.commands import (  # noqa: E402
    APP_COMMAND_IDS,
    AppCommandId,
)
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.view_workspace import ViewKind  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def window(qtbot, tmp_path):  # type: ignore[no-untyped-def]
    from PyQt6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "toolstrip.ini"), QSettings.Format.IniFormat)
    widget = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def _action(window: RateOfClosureMainWindow, command_id: str) -> QAction:
    action = window.findChild(QAction, command_id)
    assert action is not None, command_id
    return action


def test_real_top_toolstrip_exposes_stable_file_view_and_tools_surfaces(window) -> None:  # type: ignore[no-untyped-def]
    toolbar = window.findChild(QToolBar, "applicationToolstrip")
    assert toolbar is not None
    assert toolbar.accessibleName() == "Application Commands"
    assert not toolbar.isMovable()
    for object_name in ("fileMenuButton", "viewMenuButton", "toolsMenuButton"):
        button = toolbar.findChild(QToolButton, object_name)
        assert button is not None
        assert button.menu() is not None


@pytest.mark.parametrize(
    "command_id",
    (
        AppCommandId.FILE_NEW_WORKSPACE,
        AppCommandId.FILE_OPEN_WORKSPACE,
        AppCommandId.FILE_OPEN_RECENT_WORKSPACE,
        AppCommandId.FILE_SAVE_WORKSPACE,
        AppCommandId.FILE_SAVE_WORKSPACE_AS,
        AppCommandId.FILE_IMPORT_WORKSPACE,
        AppCommandId.FILE_EXPORT_WORKSPACE,
        AppCommandId.FILE_CLOSE_WORKSPACE,
    ),
)
def test_file_commands_are_truthfully_disabled_until_project_contract_exists(
    window, command_id: AppCommandId
) -> None:  # type: ignore[no-untyped-def]
    action = _action(window, command_id)
    assert not action.isEnabled()
    assert "project document contract" in action.toolTip().lower()
    assert action.statusTip() == action.toolTip()


def test_glossary_is_first_class_and_recovers_a_hidden_module(window) -> None:  # type: ignore[no-untyped-def]
    assert window.set_primary_module_visible("glossary", False)
    glossary = _action(window, AppCommandId.GLOBAL_OPEN_GLOSSARY.value)
    assert glossary.shortcut().toString()

    glossary.trigger()

    assert "glossary" in window.visible_primary_tab_ids()
    assert window.current_primary_module_id() == "glossary"


def test_theme_control_has_accessible_unavailable_and_bound_states(window) -> None:  # type: ignore[no-untyped-def]
    button = window.findChild(QToolButton, "themeMenuButton")
    assert button is not None
    assert not button.isEnabled()
    assert "launcher" in button.toolTip().lower()

    theme_menu = QMenu("Theme", window)
    theme_menu.addAction("Dark")
    window.bind_theme_menu(theme_menu)

    assert button.isEnabled()
    assert button.menu() is theme_menu
    assert button.accessibleName() == "Theme"


def test_shortcut_help_is_discoverable_from_tools_and_toolbar(window) -> None:  # type: ignore[no-untyped-def]
    action = _action(window, AppCommandId.GLOBAL_SHOW_SHORTCUTS.value)
    assert action.shortcut().toString()
    action.trigger()
    dialog = window.shortcut_help_dialog()
    assert dialog is not None
    assert dialog.isVisible()
    assert "Glossary" in dialog.findChild(type(window._explanation)).toPlainText()


def test_module_manager_exposes_required_state_and_reorder_controls(window) -> None:  # type: ignore[no-untyped-def]
    _action(window, AppCommandId.VIEW_MANAGE_MODULES.value).trigger()
    dialog = window.module_manager_dialog()
    assert dialog is not None
    assert dialog.isVisible()
    required = dialog.module_item("clubhead")
    optional = dialog.module_item("plots")
    assert required.checkState() == Qt.CheckState.Checked
    assert not required.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert optional.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert "cannot be hidden" in required.toolTip().lower()
    assert dialog.findChild(QToolButton, "moveModuleUpButton") is not None
    assert dialog.findChild(QToolButton, "moveModuleDownButton") is not None
    assert _action(
        window, AppCommandId.VIEW_RESTORE_DEFAULT_WORKSPACE.value
    ).isEnabled()


def test_module_manager_applies_visibility_and_order_changes(window) -> None:  # type: ignore[no-untyped-def]
    _action(window, AppCommandId.VIEW_MANAGE_MODULES.value).trigger()
    dialog = window.module_manager_dialog()
    assert dialog is not None
    plots = dialog.module_item("plots")
    plots.setCheckState(Qt.CheckState.Unchecked)
    assert "plots" not in window.visible_primary_tab_ids()

    simulation = dialog.module_item("simulation")
    dialog._list.setCurrentItem(simulation)
    before = window.primary_tab_ids().index("simulation")
    dialog.findChild(QToolButton, "moveModuleUpButton").click()
    assert window.primary_tab_ids().index("simulation") == before - 1


@pytest.mark.parametrize(
    ("command_id", "view_id"),
    (
        (AppCommandId.VIEW_SHOW_IMPACT, "impact"),
        (AppCommandId.VIEW_SHOW_SWING, "swing"),
        (AppCommandId.VIEW_SHOW_FLIGHT, "flight"),
    ),
)
def test_multi_view_commands_open_real_single_view_hosts(
    window, command_id: AppCommandId, view_id: str
) -> None:  # type: ignore[no-untyped-def]
    action = _action(window, command_id.value)
    assert action.isEnabled()

    action.trigger()

    assert window.current_primary_module_id() == "simulation"
    compositor = window._simulation_tab.compositor()
    assert compositor.workspace().active_slot_id == view_id
    assert compositor.visible_view_ids() == (view_id,)


def test_multi_view_hosts_share_run_and_time_but_keep_distinct_view_instances(
    window,
) -> None:  # type: ignore[no-untyped-def]
    _action(window, AppCommandId.VIEW_SHOW_SWING.value).trigger()
    compositor = window._simulation_tab.compositor()
    run = window._simulation_tab.last_run()
    assert run is not None and run.impact_time_s is not None

    swing = compositor.view(ViewKind.SWING)
    impact = compositor.view(ViewKind.IMPACT)
    flight = compositor.view(ViewKind.FLIGHT)
    assert len({id(swing), id(impact), id(flight)}) == 3
    assert swing.run() is run
    assert impact.run() is run

    swing.set_playback_time(run.impact_time_s + 0.2)
    assert flight.playback_time_s() == pytest.approx(0.2)


def test_every_ui_neutral_command_id_is_registered_exactly_once(window) -> None:  # type: ignore[no-untyped-def]
    registered = [
        action.objectName()
        for action in window.findChildren(QAction)
        if action.objectName() in {command.value for command in APP_COMMAND_IDS}
    ]
    assert sorted(registered) == sorted(command.value for command in APP_COMMAND_IDS)
