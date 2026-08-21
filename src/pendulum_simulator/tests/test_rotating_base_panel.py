"""PyQt contracts for the qualified rotating-base companion surface."""

from __future__ import annotations

import numpy as np

from shared.python.swing_sim.rotating_base import (
    RotatingBaseRunRequest,
    RotatingBaseRunResult,
    RotatingBaseRunTrace,
    load_embedded_qualified_study,
)


def _adverse_result() -> RotatingBaseRunResult:
    authority = load_embedded_qualified_study()
    case = next(case for case in authority.study.cases if not case.valid)
    request = RotatingBaseRunRequest(
        torso_profile=case.torso_profile,
        matching_rule=case.matching_rule,
        initial_torso_rate_rad_s=case.initial_torso_rate_rad_s,
    )
    trace = RotatingBaseRunTrace(
        time_s=np.array([0.0, 0.001]),
        torso_rate_rad_s=np.array([1.5, 1.4]),
        club_rate_rad_s=np.array([2.5, 2.6]),
        clubhead_speed_m_s=np.array([1.0, case.metrics.impact_speed_m_s]),
        contact_power_on_club_w=np.array([0.0, 1.0]),
        force_generated_couple_nm=np.array([0.0, 2.0]),
        force_on_club_n=np.zeros((2, 2, 2)),
        distal_segment_kinetic_energy_j=np.array([1.0, 2.0]),
    )
    return RotatingBaseRunResult(request=request, case=case, trace=trace)


def test_panel_exposes_only_registered_design_and_boundaries(qapp) -> None:
    from double_pendulum_golf.gui.rotating_base_panel import RotatingBasePanel

    panel = RotatingBasePanel()

    assert [panel._profile_combo.itemData(i) for i in range(3)] == [
        "accelerate",
        "constant_rate",
        "decelerate",
    ]
    assert [panel._matching_combo.itemData(i) for i in range(2)] == [
        "relative_club_rate",
        "absolute_club_rate",
    ]
    assert [panel._rate_combo.itemData(i) for i in range(3)] == [1.5, 3.5, 5.5]
    boundary = panel._boundary.text().lower()
    assert "nonanatomical" in boundary
    assert "no governed human validation" in boundary
    assert "no coaching recommendation" in boundary
    assert "torso" in panel._killswitches.text().lower()
    assert "bilateral arm" in panel._killswitches.text().lower()
    assert "bilateral wrist" in panel._killswitches.text().lower()


def test_panel_retains_adverse_row_and_all_required_diagnostics(qapp) -> None:
    from double_pendulum_golf.gui.rotating_base_panel import RotatingBasePanel

    panel = RotatingBasePanel()
    result = _adverse_result()

    panel._accept_result(result)

    status = panel._status.text().lower()
    assert "invalid/adverse retained" in status
    assert result.case.exclusion_reasons[0] in status
    assert set(panel._metric_labels) == {
        "impact_speed_m_s",
        "contact_work_on_club_j",
        "braking_grip_work_j",
        "force_couple_work_j",
        "negative_along_path_impulse_ns",
        "bilateral_wrist_work_j",
        "total_control_work_j",
        "distal_energy_gain_j",
        "peak_grip_force_n",
        "maximum_constraint_residual_m",
        "maximum_velocity_constraint_residual_m_s",
        "maximum_contact_power_identity_residual_w",
        "work_energy_closure_j",
    }
    assert (
        panel._metric_labels["impact_speed_m_s"]
        .text()
        .startswith(f"{result.case.metrics.impact_speed_m_s:.6g}")
    )
    assert panel._export_button.isEnabled()
    if panel._figure is not None:
        assert panel._axis_couple.get_ylabel() == "Force Couple (N·m)"
        assert len(panel._axis_couple.lines) == 2
        assert panel._axis_force.get_ylabel() == "Grip Force (N)"
        assert len(panel._axis_force.lines) == 2


def test_panel_runs_registered_provider_off_gui_thread(qapp, qtbot, monkeypatch) -> None:
    import double_pendulum_golf.gui.rotating_base_panel as panel_module

    result = _adverse_result()
    monkeypatch.setattr(panel_module, "run_registered_case", lambda _request: result)
    panel = panel_module.RotatingBasePanel()

    panel._start_run()

    assert not panel._run_button.isEnabled()
    qtbot.waitUntil(lambda: panel._result is result, timeout=2_000)
    qtbot.waitUntil(lambda: panel._thread is None, timeout=2_000)
    assert panel._run_button.isEnabled()
    assert panel._worker is None


def test_analysis_tab_exposes_separate_rotating_base_study(qapp) -> None:
    from double_pendulum_golf.gui.analysis_tab import AnalysisTab

    tab = AnalysisTab()
    labels = [tab._plot_tabs.tabText(index) for index in range(tab._plot_tabs.count())]

    assert "Drift Transfer" in labels
    assert "Rotating-Base Study" in labels
