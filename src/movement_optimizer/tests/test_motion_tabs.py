# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""GUI tests for swingset and chain analysis tabs."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
)

from movement_optimizer.gui.app_icon import movement_optimizer_icon, movement_optimizer_icon_path
from movement_optimizer.gui.main_window import MainWindow
from movement_optimizer.gui.motion_tabs import (
    ChainDynamicsTab,
    MotionCanvas,
    NumericControl,
    SwingsetTab,
)


def test_main_window_preserves_barbell_tabs_and_adds_motion_tabs(qapp) -> None:
    window = MainWindow()

    tab_names = [window.tabs.tabText(index).strip() for index in range(window.tabs.count())]

    assert tab_names[:7] == [
        "Bottoms Up Squat",
        "Full Squat",
        "Deadlift",
        "Bench Press",
        "Clean",
        "Jerk",
        "Snatch",
    ]
    assert tab_names[-2:] == ["Swingset Model", "Chain Dynamics"]


def test_main_window_uses_packaged_launcher_icon(qapp) -> None:
    qapp.setWindowIcon(movement_optimizer_icon())
    window = MainWindow()

    assert movement_optimizer_icon_path().name == "project_map.svg"
    assert not qapp.windowIcon().isNull()
    assert not window.windowIcon().isNull()


def test_analysis_tabs_disable_barbell_only_controls(qapp) -> None:
    window = MainWindow()
    window.tabs.setCurrentIndex(0)
    window._sync_motion_tab_controls()
    window.sidebar.opt_btn.setEnabled(True)
    window.sidebar.export_btn.setEnabled(True)

    window.tabs.setCurrentIndex(window.tabs.count() - 1)

    assert not window.sidebar.opt_btn.isEnabled()
    assert not window.sidebar.export_btn.isEnabled()
    assert window.controls.isEnabled()

    window.tabs.setCurrentIndex(0)

    assert window.sidebar.opt_btn.isEnabled()
    assert window.sidebar.export_btn.isEnabled()
    assert window.controls.isEnabled()


def test_layout_header_status_and_splitter_use_full_height(qapp) -> None:
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    qapp.processEvents()

    title = next(
        label
        for label in window.centralWidget().findChildren(QLabel)
        if label.text() == "Movement Optimizer"
    )
    splitter = window.centralWidget().findChild(QSplitter)

    assert title.geometry().y() <= 12
    assert window.status_label.geometry().height() <= 24
    assert splitter is not None
    assert splitter.geometry().height() > 700


def test_main_window_top_toolstrip_controls_left_and_right_sidebars(qapp) -> None:
    window = MainWindow()
    window.show()
    window.tabs.setCurrentIndex(0)
    qapp.processEvents()

    buttons = window.centralWidget().findChildren(QPushButton)
    assert all(button.text() != "Hide sidebar" for button in buttons)
    assert window._left_sidebar_toggle_btn.text() == "Hide left"
    assert window._right_sidebar_toggle_btn.text() == "Right panel"
    assert not window._right_sidebar_toggle_btn.isEnabled()

    window._left_sidebar_toggle_btn.click()
    assert not window.sidebar.isVisible()
    assert window._left_sidebar_toggle_btn.text() == "Show left"

    window.tabs.setCurrentIndex(window.tabs.count() - 2)
    swing_tab = window.tabs.currentWidget()
    assert window._right_sidebar_toggle_btn.isEnabled()
    assert window._right_sidebar_toggle_btn.text() == "Hide right"

    window._right_sidebar_toggle_btn.click()
    assert not swing_tab.control_panel_visible()
    assert window._right_sidebar_toggle_btn.text() == "Show right"


def test_swingset_and_chain_tabs_run_local_simulations(qapp) -> None:
    swingset = SwingsetTab()
    chain = ChainDynamicsTab()

    # Keep the iterative optimizer cheap and deterministic for CI: the default
    # production budget (600 evaluations x ~130-220 simulated steps) takes ~30s
    # and tips past the 60s pytest-timeout on loaded shared runners. Note that
    # "policy_steps" has no effect on the iterative path here because "cycles"
    # drives the rollout length via _steps_for_candidate; constrain budget and
    # cycles instead. This still exercises the full
    # _optimize_policy -> optimize_cyclic_policy_iterative -> simulate_swingset
    # path and produces the asserted "Best height" metric.
    swingset._controls["budget"].set_value(60)
    swingset._controls["cycles"].set_value(1)
    swingset._optimize_policy()
    chain._simulate()

    assert "Best height" in swingset.metric_label.text()
    assert "peak tip speed" in chain.metric_label.text()


def test_swingset_tab_exposes_policy_tuning_and_progress(qapp) -> None:
    swingset = SwingsetTab()
    for key in (
        "cycles",
        "freq_min",
        "freq_max",
        "freq_samples",
        "hip_rate_min",
        "hip_rate_max",
        "hip_samples",
        "torso_rate_min",
        "torso_rate_max",
        "torso_samples",
        "knee_ratio_min",
        "knee_ratio_max",
        "knee_samples",
        "phase_samples",
    ):
        assert key in swingset._controls
    swingset.iterative_checkbox.setChecked(False)  # exercise the grid-search fallback path.
    swingset._controls["cycles"].set_value(1)
    swingset._controls["freq_samples"].set_value(2)
    swingset._controls["hip_samples"].set_value(1)
    swingset._controls["torso_samples"].set_value(1)
    swingset._controls["knee_samples"].set_value(1)
    swingset._controls["phase_samples"].set_value(2)

    swingset._optimize_policy()

    progress = swingset.findChild(QProgressBar)
    assert progress is not None
    assert progress.value() == progress.maximum() == 4
    assert "4 candidates" in swingset.metric_label.text()
    assert swingset.policy_trace_canvas.sample_count() == 4
    assert "Peak torque" in swingset.policy_detail_label.text()
    assert "frequency" in swingset.policy_detail_label.text()


def test_swingset_policy_terminology_is_not_walking(qapp) -> None:
    swingset = SwingsetTab()

    visible_text = " ".join(
        widget.text() for widget in swingset.findChildren((QLabel, QPushButton)) if widget.text()
    )

    assert "walking" not in visible_text.lower()
    assert "swing cycles" in visible_text.lower()
    assert "Optimize Swing Policy" in visible_text


def test_motion_tab_parameter_panels_are_scrollable_and_not_compressed(qapp) -> None:
    for tab in (SwingsetTab(), ChainDynamicsTab()):
        scroll_area = tab.findChild(QScrollArea)

        assert scroll_area is not None
        assert scroll_area.widgetResizable()
        assert tab.control_panel_visible()
        assert all(line_edit.minimumHeight() >= 28 for line_edit in tab.findChildren(QLineEdit))

        tab.set_control_panel_visible(False)
        assert not tab.control_panel_visible()
        tab.set_control_panel_visible(True)
        assert tab.control_panel_visible()


def test_swingset_optimize_policy_action_is_sticky_above_scroll_area(qapp) -> None:
    swingset = SwingsetTab()
    scroll_area = swingset.findChild(QScrollArea)

    assert scroll_area is not None
    assert swingset.optimize_button.text() == "Optimize Swing Policy"
    assert swingset.optimize_button not in scroll_area.widget().findChildren(QPushButton)


def test_swingset_policy_trace_canvas_accepts_optimization_samples(qapp) -> None:
    swingset = SwingsetTab()
    swingset.iterative_checkbox.setChecked(False)  # exercise the grid-search fallback path.
    swingset._controls["cycles"].set_value(1)
    swingset._controls["freq_samples"].set_value(2)
    swingset._controls["hip_samples"].set_value(1)
    swingset._controls["torso_samples"].set_value(1)
    swingset._controls["knee_samples"].set_value(1)
    swingset._controls["phase_samples"].set_value(2)

    swingset._optimize_policy()

    assert swingset.policy_trace_canvas.sample_count() == 4
    assert swingset.policy_trace_canvas.has_parameter_series("frequency_hz")
    swingset.policy_trace_canvas.resize(360, 180)
    rendered = swingset.policy_trace_canvas.grab()
    assert not rendered.isNull()
    assert "knee ratio" in swingset.policy_detail_label.text()


def test_swingset_policy_trace_canvas_handles_sparse_series(qapp) -> None:
    swingset = SwingsetTab()

    swingset.policy_trace_canvas.resize(240, 120)
    empty_render = swingset.policy_trace_canvas.grab()
    pixmap = QPixmap(120, 80)
    painter = QPainter(pixmap)
    try:
        swingset.policy_trace_canvas._draw_normalized_series(painter, "missing", QColor("white"), 1)
    finally:
        painter.end()

    assert not empty_render.isNull()


def test_swingset_tab_configures_seat_placement_percent(qapp) -> None:
    swingset = SwingsetTab()

    swingset._controls["seat_placement"].set_value(62.5)
    config = swingset._config()

    assert config.seat_placement_thigh_fraction == pytest.approx(0.625)


def test_bottom_playback_controls_drive_analysis_tabs(qapp) -> None:
    window = MainWindow()
    window.tabs.setCurrentIndex(window.tabs.count() - 2)
    swing_tab = window.tabs.currentWidget()
    swing_tab._controls["policy_steps"].set_value(30)

    window._toggle_play()

    assert swing_tab.playback_status()[2]
    assert window.controls.btn_play.text() == "Pause"

    window._step_fwd()

    assert swing_tab.playback_status()[2] is False
    assert swing_tab.playback_status()[0] == 2

    window.tabs.setCurrentIndex(window.tabs.count() - 1)
    window._toggle_play()
    chain_tab = window.tabs.currentWidget()

    assert chain_tab.playback_status()[2]
    window._jump_to_end()
    assert chain_tab.playback_status()[2] is False
    assert chain_tab.playback_status()[0] == chain_tab.playback_status()[1]


def test_motion_tabs_use_slider_text_controls_without_spinbox_arrows(qapp) -> None:
    swingset = SwingsetTab()
    chain = ChainDynamicsTab()

    assert not swingset.findChildren(QAbstractSpinBox)
    assert not chain.findChildren(QAbstractSpinBox)
    assert swingset.findChildren(QSlider)
    assert swingset.findChildren(QLineEdit)
    assert chain.findChildren(QSlider)
    assert chain.findChildren(QLineEdit)


def test_numeric_control_accepts_typed_values(qapp) -> None:
    control = NumericControl(0.0, 10.0, 1.0)

    control.edit.setText("4.25")
    control.edit.editingFinished.emit()

    assert control.value() == 4.25


def test_numeric_control_validates_ranges_and_recovers_bad_text(qapp) -> None:
    with pytest.raises(ValueError, match="upper must be greater"):
        NumericControl(2.0, 2.0, 2.0)

    control = NumericControl(0.0, 10.0, 1.0)
    observed: list[float] = []
    control.valueChanged.connect(observed.append)

    control.slider.setValue(control.slider.maximum())
    assert control.value() == 10.0
    assert observed[-1] == 10.0

    control.edit.setText("not numeric")
    control.edit.editingFinished.emit()
    assert control.edit.text() == "10.000"


def test_motion_canvas_handles_empty_and_bodyless_paints(qapp) -> None:
    canvas = MotionCanvas()
    canvas.resize(320, 240)

    canvas.grab()

    canvas.set_scene([(0.0, 0.0), (0.0, 1.0)])
    image = QPixmap(64, 64)
    painter = QPainter(image)
    try:
        canvas._draw_body(painter, canvas._projector())
    finally:
        painter.end()


def test_swingset_playback_controls_cover_policy_rollout_branches(qapp) -> None:
    swingset = SwingsetTab()
    swingset._controls["cycles"].set_value(1)
    swingset._controls["policy_steps"].set_value(30)
    swingset._controls["freq_samples"].set_value(1)
    swingset._controls["hip_samples"].set_value(1)
    swingset._controls["torso_samples"].set_value(1)
    swingset._controls["knee_samples"].set_value(1)
    swingset._controls["phase_samples"].set_value(1)

    swingset._ensure_rollout()
    assert swingset._rollout is not None

    swingset._toggle_playback()
    assert swingset.playback_status()[2]
    swingset.set_playback_speed(2.0)
    swingset._advance_frame()
    swingset._toggle_playback()
    assert not swingset.playback_status()[2]

    swingset.playback_step_back()
    swingset.playback_rewind()
    swingset.playback_jump_to_end()
    assert swingset.playback_status()[0] == swingset.playback_status()[1]

    swingset._rollout = None
    swingset._advance_frame()
    swingset._control_panel_widget = None
    with pytest.raises(RuntimeError, match="Swingset controls"):
        swingset.set_control_panel_visible(True)


def test_swingset_playback_methods_return_without_rollout(qapp, monkeypatch) -> None:
    swingset = SwingsetTab()
    swingset._rollout = None
    monkeypatch.setattr(swingset, "_optimize_policy", lambda: None)

    swingset.playback_step_forward()
    swingset.playback_step_back()
    swingset.playback_rewind()
    swingset.playback_jump_to_end()

    assert swingset.playback_status() == (0, 0, False)


def test_chain_tab_supports_free_segment_angles_and_realtime_speed(qapp) -> None:
    chain = ChainDynamicsTab()
    chain.tie_segments.setChecked(False)
    chain._controls["segments"].set_value(3)
    chain.angle_edit.setText("9999.0, 0.1, -9999.0")
    chain._controls["dt"].set_value(0.02)
    chain._controls["duration"].set_value(0.6)
    chain._controls["speed"].set_value(2.0)

    chain._simulate()

    assert chain._rollout is not None
    assert chain._playback_interval_ms() == 10


def test_chain_tab_converts_typed_degrees(qapp) -> None:
    chain = ChainDynamicsTab()
    chain.tie_segments.setChecked(False)
    chain.use_degrees.setChecked(True)
    chain._controls["segments"].set_value(2)
    chain.angle_edit.setText("180, -90")

    state = chain._state()

    assert state.angles_rad[0] == pytest.approx(np.pi)
    assert state.angles_rad[1] == pytest.approx(-np.pi / 2.0)
    assert "degrees" in chain.angle_edit.placeholderText()


def test_chain_tab_exposes_damping_duration_and_random_wadded_start(qapp) -> None:
    chain = ChainDynamicsTab()
    chain._controls["segments"].set_value(5)
    chain._controls["damping"].set_value(0.42)
    chain._controls["bend_damping"].set_value(1.4)
    chain._controls["coupling"].set_value(22.0)
    chain._controls["duration"].set_value(1.2)
    chain._controls["dt"].set_value(0.2)
    chain._controls["random_seed"].set_value(11)

    config = chain._config()
    chain._randomize_wadded_start()
    chain._simulate()

    assert config.damping == pytest.approx(0.42)
    assert config.bend_damping == pytest.approx(1.4)
    assert config.coupling == pytest.approx(22.0)
    assert not chain.tie_segments.isChecked()
    assert len(chain.angle_edit.text().split(",")) == 5
    assert chain._rollout is not None
    assert len(chain._rollout.states) == 7
    assert "real time 1.20 s" in chain.metric_label.text()


def test_canvas_keeps_anchor_projection_fixed(qapp) -> None:
    from movement_optimizer.gui.motion_tabs import MotionCanvas

    canvas = MotionCanvas()
    canvas.resize(500, 400)
    canvas.set_scene([(0.0, 0.0), (0.2, 1.0)])
    first_anchor = canvas._projector()((0.0, 0.0))
    canvas.set_scene([(0.0, 0.0), (-0.8, 2.0), (0.7, 3.0)])
    second_anchor = canvas._projector()((0.0, 0.0))

    assert second_anchor.x() == pytest.approx(first_anchor.x())
    assert second_anchor.y() == pytest.approx(first_anchor.y())


def test_canvas_keeps_rigid_link_scale_across_chain_poses(qapp) -> None:
    from movement_optimizer.gui.motion_tabs import MotionCanvas

    canvas = MotionCanvas()
    canvas.resize(500, 400)
    canvas.set_scene([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
    projector = canvas._projector()
    straight = projector((0.0, 1.0)).y() - projector((0.0, 0.0)).y()

    canvas.set_scene([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    projector = canvas._projector()
    curled = projector((1.0, 0.0)).x() - projector((0.0, 0.0)).x()

    assert abs(curled) == pytest.approx(abs(straight))


def test_chain_rollout_keeps_physical_anchor_fixed(qapp) -> None:
    chain = ChainDynamicsTab()
    chain._simulate()

    assert chain._rollout is not None
    np.testing.assert_allclose(chain._rollout.positions[:, 0, :], 0.0)


def test_chain_tab_reports_invalid_inputs_and_covers_playback_branches(qapp, monkeypatch) -> None:
    chain = ChainDynamicsTab()
    chain.tie_segments.setChecked(False)
    chain._controls["segments"].set_value(3)
    chain.angle_edit.setText("0.0, 1.0")

    chain._refresh()
    assert "Expected 3 segment angles" in chain.metric_label.text()
    chain._simulate()
    assert "Expected 3 segment angles" in chain.metric_label.text()

    chain.angle_edit.setText("0.0, 0.5, 1.0")
    chain._simulate()
    assert chain._rollout is not None

    chain._toggle_playback()
    assert chain.playback_status()[2]
    chain.set_playback_speed(2.0)
    chain._advance_frame()
    chain._toggle_playback()
    assert not chain.playback_status()[2]

    chain.playback_step_forward()
    chain.playback_step_back()
    chain.playback_rewind()
    chain.playback_jump_to_end()
    assert chain.playback_status()[0] == chain.playback_status()[1]

    chain._rollout = None
    monkeypatch.setattr(chain, "_simulate", lambda: None)
    chain._toggle_playback()
    chain.playback_step_forward()
    chain.playback_step_back()
    chain.playback_rewind()
    chain.playback_jump_to_end()
    chain._advance_frame()
    chain._render_chain_frame()

    chain._control_scroll = None
    with pytest.raises(RuntimeError, match="Chain controls"):
        chain.set_control_panel_visible(True)


# ---------------------------------------------------------------------------
# Force overlays, analysis panel, iterative optimizer wiring, tooltips
# ---------------------------------------------------------------------------


def test_swingset_iterative_optimize_populates_panel_and_overlays(qapp) -> None:
    swingset = SwingsetTab()
    assert swingset.iterative_checkbox.isChecked()  # iterative is the default
    swingset._controls["budget"].set_value(50)
    swingset._controls["cycles"].set_value(1)

    swingset._optimize_policy()

    assert swingset._rollout is not None
    progress = swingset.findChild(QProgressBar)
    assert progress.maximum() == 50
    assert 0 < swingset.policy_trace_canvas.sample_count() <= 50
    # Analysis plots populated.
    assert swingset.analysis_panel.axes["torques"].get_lines()
    # Force overlay drawn (all toggles default-on).
    assert swingset.canvas._overlay.arrows or swingset.canvas._overlay.com_markers


def test_swingset_force_toggle_does_not_recompute(qapp) -> None:
    swingset = SwingsetTab()
    swingset._controls["budget"].set_value(50)
    swingset._controls["cycles"].set_value(1)
    swingset._optimize_policy()

    rollout_id = id(swingset._rollout)
    overlay_before = swingset.canvas._overlay
    swingset._force_toggles["gravity"].setChecked(False)

    assert id(swingset._rollout) == rollout_id  # no resimulation
    assert swingset.canvas._overlay is not overlay_before  # overlay rebuilt


def test_swingset_force_toggle_before_optimize_is_safe(qapp) -> None:
    swingset = SwingsetTab()
    swingset._force_toggles["com"].setChecked(False)  # must not raise without a rollout
    assert not swingset.canvas._overlay.arrows


def test_chain_simulate_populates_panel_and_overlays(qapp) -> None:
    chain = ChainDynamicsTab()
    chain._controls["segments"].set_value(6)
    chain._controls["duration"].set_value(0.2)
    chain._controls["dt"].set_value(0.02)

    chain._simulate()

    assert chain._rollout is not None
    assert chain.analysis_panel.axes["tension"].get_lines()
    assert chain.canvas._overlay.arrows


def test_chain_force_toggle_does_not_recompute(qapp) -> None:
    chain = ChainDynamicsTab()
    chain._controls["segments"].set_value(6)
    chain._controls["duration"].set_value(0.2)
    chain._controls["dt"].set_value(0.02)
    chain._simulate()

    rollout_id = id(chain._rollout)
    chain._force_toggles["net"].setChecked(False)
    assert id(chain._rollout) == rollout_id


def test_motion_tab_buttons_and_controls_have_tooltips(qapp) -> None:
    swingset = SwingsetTab()
    assert swingset.optimize_button.toolTip()
    assert swingset.play_button.toolTip()
    assert swingset._controls["budget"].toolTip()
    assert swingset._force_toggles["gravity"].toolTip()

    chain = ChainDynamicsTab()
    assert chain._controls["segments"].toolTip()
    assert chain._force_toggles["tension"].toolTip()


def test_motion_tab_analysis_helpers_are_safe_without_rollout(qapp) -> None:
    swingset = SwingsetTab()
    swingset._rollout = None
    swingset._populate_analysis_panel()  # None-guard, no error

    chain = ChainDynamicsTab()
    chain._rollout = None
    chain._populate_analysis_panel()  # None-guard
    chain._refresh_overlays()  # None-guard clears overlays
    assert not chain.canvas._overlay.arrows
