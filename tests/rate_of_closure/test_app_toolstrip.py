"""Application toolstrip command and workspace-management contracts."""

from __future__ import annotations

from pathlib import Path
import json

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
from rate_of_closure.application.regional_ground_variation_request import (  # noqa: E402
    regional_ground_variation_request_from_json,
from rate_of_closure.application.workspace_session import (  # noqa: E402
    document_from_state,
)
from rate_of_closure.application.workspace_simulation_session import (  # noqa: E402
    SimulationWorkspaceState,
)
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.view_workspace import ViewKind  # noqa: E402
from shared.python.swing_sim.ball_setup import (  # noqa: E402
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.solver import (  # noqa: E402
    BoxTolerance,
    SpatialTarget,
    SurfaceCircleTolerance,
    TargetPoint,
)

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


def test_implemented_file_commands_are_enabled_and_recent_remains_honest(
    window,
) -> None:  # type: ignore[no-untyped-def]
    supported = (
        AppCommandId.FILE_NEW_WORKSPACE,
        AppCommandId.FILE_OPEN_WORKSPACE,
        AppCommandId.FILE_SAVE_WORKSPACE,
        AppCommandId.FILE_SAVE_WORKSPACE_AS,
        AppCommandId.FILE_IMPORT_WORKSPACE,
        AppCommandId.FILE_EXPORT_WORKSPACE,
        AppCommandId.FILE_CLOSE_WORKSPACE,
    )
    assert all(_action(window, command_id).isEnabled() for command_id in supported)
    assert all(_action(window, command_id).toolTip() for command_id in supported)
    recent = _action(window, AppCommandId.FILE_OPEN_RECENT_WORKSPACE)
    assert not recent.isEnabled()
    assert "no recent workspace" in recent.toolTip().lower()
    assert recent.statusTip() == recent.toolTip()


def test_save_as_and_open_restore_supported_state_atomically(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    target = tmp_path / "session.roc-workspace.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    saved_target = SpatialTarget(
        label="Saved apex gate",
        kind="aerial_waypoint",
        point=TargetPoint.from_frame((142.0, 3.0, 25.0), "flight"),
        tolerance=BoxTolerance((5.0, 2.0, 4.0)),
        elevation_source="absolute",
    )
    window._simulation_tab.apply_simulation_workspace_state(
        SimulationWorkspaceState(
            ball_setup=BallSetup(BallSupportMode.TEE, 0.052),
            ball_setup_user_overridden=True,
            spatial_target=saved_target,
        )
    )
    _action(window, AppCommandId.FILE_SAVE_WORKSPACE_AS).trigger()
    saved = window._capture_workspace_state()
    assert target.is_file()
    assert not window.workspace_is_dirty()
    recent = _action(window, AppCommandId.FILE_OPEN_RECENT_WORKSPACE)
    assert recent.isEnabled()
    assert str(target) in recent.toolTip()

    window._controls.apply_preset("Zero rotation (control)")
    window._simulation_tab.apply_simulation_workspace_state(
        SimulationWorkspaceState(
            ball_setup=BallSetup(BallSupportMode.GROUND, 0.0),
            ball_setup_user_overridden=True,
            spatial_target=SpatialTarget(
                label="Temporary landing",
                kind="landing_area",
                point=TargetPoint(80.0, 0.0, 5.0),
                tolerance=SurfaceCircleTolerance(8.0),
                elevation_source="course_surface",
                ground_source="course.surface/test",
            ),
        )
    )
    assert window.workspace_is_dirty()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Discard,
    )
    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == saved
    assert not window.workspace_is_dirty()

    window._controls.apply_preset("Zero rotation (control)")
    recent.trigger()

    assert window._capture_workspace_state() == saved
    assert not window.workspace_is_dirty()


def test_cancelled_dirty_new_preserves_live_state(window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QMessageBox

    window._controls.apply_preset("Zero rotation (control)")
    dirty = window._controls.scenario()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Cancel,
    )

    _action(window, AppCommandId.FILE_NEW_WORKSPACE).trigger()

    assert window._controls.scenario() == dirty
    assert window.workspace_is_dirty()


def test_invalid_open_reports_error_without_partial_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    target = tmp_path / "invalid.roc-workspace.json"
    target.write_text('{"schema":"wrong"}', encoding="utf-8")
    before = window._capture_workspace_state()
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and warnings[0][0] == "Open Failed"


def test_invalid_nested_target_is_rejected_before_native_ui_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = document_from_state(before, window._workspace_metadata).to_json_dict()
    raw["model_session"]["data"]["simulation_setup"]["data"]["spatial_target"][
        "source_frame"
    ] = "camera"
    target = tmp_path / "invalid-target.roc-workspace.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and "source_frame" in warnings[0][1]


def test_invalid_torque_selection_is_rejected_before_native_ui_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = document_from_state(before, window._workspace_metadata).to_json_dict()
    provenance = raw["model_session"]["data"]["torque_selection"]["data"][
        "selection_provenance"
    ]
    provenance["profile_source"] = "drawn"
    target = tmp_path / "invalid-torque-selection.roc-workspace.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and "provenance" in warnings[0][1]


def test_invalid_variation_selection_is_rejected_before_native_ui_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = document_from_state(before, window._workspace_metadata).to_json_dict()
    selection = raw["model_session"]["data"]["variation_study"]["data"]
    selection["selected_output_metrics"] = ["unknown_metric"]
    target = tmp_path / "invalid-variation-selection.roc-workspace.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and "metric" in warnings[0][1]


def test_invalid_capability_request_is_rejected_before_native_ui_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = document_from_state(before, window._workspace_metadata).to_json_dict()
    raw["model_session"]["data"]["capability_request"]["computed_result"] = {}
    target = tmp_path / "invalid-capability-request.roc-workspace.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and "capability workflow" in warnings[0][1]


@pytest.mark.parametrize(
    ("kind", "message"), [("mph", "unit"), ("covariance", "correlation")]
)
def test_noncanonical_capability_basis_reports_open_error_without_mutation(
    window, tmp_path, monkeypatch, kind: str, message: str
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = document_from_state(before, window._workspace_metadata).to_json_dict()
    club = raw["model_session"]["data"]["capability_request"]["profile"]["clubs"][0]
    if kind == "mph":
        club["parameters"][0]["unit"] = "mph"
    else:
        club["matrix_kind"] = "covariance"
    target = tmp_path / f"invalid-capability-{kind}.roc-workspace.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *_args: (str(target), "")
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text, *_args: warnings.append((title, text)),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and message in warnings[0][1]


def test_oversized_capability_number_reports_open_error_without_mutation(
    window, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    before = window._capture_workspace_state()
    raw = json.dumps(
        document_from_state(before, window._workspace_metadata).to_json_dict()
    )
    raw = raw.replace('"candidate_budget": 8', '"candidate_budget": ' + "9" * 4000)
    target = tmp_path / "oversized-capability-number.roc-workspace.json"
    target.write_text(raw, encoding="utf-8")
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: warnings.append(
            (title, message)
        ),
    )

    _action(window, AppCommandId.FILE_OPEN_WORKSPACE).trigger()

    assert window._capture_workspace_state() == before
    assert warnings and "finite" in warnings[0][1]


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


def test_combined_request_commands_follow_relevant_module_context(window) -> None:  # type: ignore[no-untyped-def]
    command_ids = (
        AppCommandId.FILE_OPEN_REGIONAL_GROUND_VARIATION_REQUEST,
        AppCommandId.FILE_SAVE_REGIONAL_GROUND_VARIATION_REQUEST_AS,
    )
    for command_id in command_ids:
        action = _action(window, command_id.value)
        assert not action.isEnabled()
        assert "ground surfaces and variation" in action.toolTip().lower()

    window.set_primary_module_active("variation")
    assert all(_action(window, item.value).isEnabled() for item in command_ids)

    window.set_primary_module_active("regional_surfaces")
    assert all(_action(window, item.value).isEnabled() for item in command_ids)

    window.set_primary_module_active("plots")
    assert all(not _action(window, item.value).isEnabled() for item in command_ids)


def test_combined_request_save_rejects_unvalidated_illustrative_surface(window) -> None:  # type: ignore[no-untyped-def]
    window.set_primary_module_active("variation")

    _action(
        window, AppCommandId.FILE_SAVE_REGIONAL_GROUND_VARIATION_REQUEST_AS.value
    ).trigger()

    assert "explicitly validated" in window.statusBar().currentMessage().lower()
    assert "error" in window.statusBar().currentMessage().lower()


def test_combined_request_apply_is_exact_until_an_owner_changes(window) -> None:  # type: ignore[no-untyped-def]
    fixture = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "regional_ground_variation_request_golden_v1.json"
    )
    request = regional_ground_variation_request_from_json(
        fixture.read_text(encoding="utf-8")
    )

    window.apply_regional_ground_variation_request(request)

    assert window.current_regional_ground_variation_request() is request
    window._variation_tab._seed_spin.setValue(request.plan.seed + 1)
    changed = window.current_regional_ground_variation_request()
    assert changed is not request
    assert changed.plan.seed == request.plan.seed + 1
