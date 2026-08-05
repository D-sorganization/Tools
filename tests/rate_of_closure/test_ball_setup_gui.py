"""PyQt interaction and rendering contracts for physical ball setup."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from rate_of_closure.simulation import (
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
    ContactMode,
    ImpactStatus,
)
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    yield widget
    widget.stop()


def test_driver_defaults_to_editable_tee_setup_with_accessible_controls(tab) -> None:  # type: ignore[no-untyped-def]
    control = tab.ball_setup_control()

    assert tab.config().ball_setup == BallSetup(
        BallSupportMode.TEE, DEFAULT_DRIVER_TEE_HEIGHT_M
    )
    assert control.mode_combo().currentText() == "Tee"
    assert control.tee_height_spin().value() == pytest.approx(38.1)
    assert control.tee_height_spin().isEnabled()
    assert control.use_club_default_check().isChecked()
    for widget in control.interactive_widgets():
        assert widget.accessibleName()
        assert "Source:" in widget.toolTip()


def test_club_defaults_follow_selection_until_user_overrides(tab) -> None:  # type: ignore[no-untyped-def]
    control = tab.ball_setup_control()
    tab._club_combo.setCurrentText("7-Iron")

    assert tab.config().ball_setup == BallSetup(BallSupportMode.GROUND, 0.0)
    assert not control.tee_height_spin().isEnabled()
    assert "Ground mode" in control.tee_height_spin().toolTip()

    tab._club_combo.setCurrentText("Driver 10.5°")
    control.tee_height_spin().setValue(25.4)
    assert not control.use_club_default_check().isChecked()

    tab._club_combo.setCurrentText("7-Iron")
    assert tab.config().ball_setup == BallSetup(BallSupportMode.TEE, 0.0254)

    control.use_club_default_check().setChecked(True)
    assert tab.config().ball_setup == BallSetup(BallSupportMode.GROUND, 0.0)


def test_ground_mode_zeros_and_disables_height_then_restores_tee_editing(tab) -> None:  # type: ignore[no-untyped-def]
    control = tab.ball_setup_control()
    control.tee_height_spin().setValue(30.0)
    control.mode_combo().setCurrentIndex(
        control.mode_combo().findData(BallSupportMode.GROUND)
    )

    assert tab.config().ball_setup == BallSetup(BallSupportMode.GROUND, 0.0)
    assert control.tee_height_spin().value() == pytest.approx(0.0)
    assert not control.tee_height_spin().isEnabled()
    assert "bottom of the ball" in control.status_text().lower()

    control.mode_combo().setCurrentIndex(
        control.mode_combo().findData(BallSupportMode.TEE)
    )
    assert control.tee_height_spin().isEnabled()
    assert tab.config().ball_setup == BallSetup(BallSupportMode.TEE, 0.03)


def test_tee_height_supports_whole_field_keyboard_replacement(tab, qtbot) -> None:  # type: ignore[no-untyped-def]
    spin = tab.ball_setup_control().tee_height_spin()
    line_edit = spin.lineEdit()
    assert line_edit is not None
    qtbot.mouseClick(line_edit, Qt.MouseButton.LeftButton)
    qtbot.wait(1)
    assert line_edit.selectedText() == spin.text()

    spin.setFocus()
    qtbot.keyClick(spin, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(spin, "31.75")
    qtbot.keyClick(spin, Qt.Key.Key_Enter)

    assert spin.value() == pytest.approx(31.75)
    assert tab.config().ball_setup.tee_height_m == pytest.approx(0.03175)


def test_tee_height_guidance_does_not_reject_larger_finite_values(tab) -> None:  # type: ignore[no-untyped-def]
    spin = tab.ball_setup_control().tee_height_spin()

    spin.setValue(250.0)

    assert spin.value() == pytest.approx(250.0)
    assert tab.config().ball_setup == BallSetup(BallSupportMode.TEE, 0.25)


def test_canonical_setup_reload_does_not_create_a_second_schema(tab) -> None:  # type: ignore[no-untyped-def]
    saved = BallSetup(BallSupportMode.TEE, 0.032).to_json_dict()
    tab.set_ball_setup(BallSetup.from_json_dict(saved))

    assert tab.config().ball_setup.to_json_dict() == saved
    assert not tab.ball_setup_control().use_club_default_check().isChecked()
    tab._club_combo.setCurrentText("7-Iron")
    assert tab.config().ball_setup == BallSetup(BallSupportMode.TEE, 0.032)


def test_swing_scene_uses_run_ball_center_and_draws_tee_only_for_tee(tab) -> None:  # type: ignore[no-untyped-def]
    tee_run = tab.run_now()
    assert tee_run is not None
    assert np.allclose(
        tab.view().rendered_ball_center_m(), tee_run.config.ball_position_m
    )
    assert tab.view().tee_visible()

    tab._club_combo.setCurrentText("7-Iron")
    ground_run = tab.run_now()
    assert ground_run is not None
    assert tab.view().rendered_ball_center_m()[1] == pytest.approx(GOLF_BALL_RADIUS_M)
    assert not tab.view().tee_visible()


def test_no_impact_scene_preserves_configured_tee_geometry(tab) -> None:  # type: ignore[no-untyped-def]
    tab._source_combo.setCurrentIndex(1)
    tab._contact_combo.setCurrentIndex(
        tab._contact_combo.findData(ContactMode.FIXED_BALL_CONTACT)
    )
    run = tab.run_now()

    assert run is not None
    assert run.impact_outcome.status is ImpactStatus.MISS
    assert run.config.ball_setup.support_mode is BallSupportMode.TEE
    assert np.allclose(tab.view().rendered_ball_center_m(), run.config.ball_position_m)
    assert tab.view().tee_visible()
