"""Tests for panel builders."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtWidgets import QMainWindow, QTabWidget
from shared.python.theme.integration import ThemedWindowMixin
from double_pendulum_golf.gui.panel_builders import (
    build_double_panel,
    build_triple_panel,
    build_golfer_panel,
    wire_toolstrip,
)
from double_pendulum_golf.gui.toolstrip_widget import ToolStrip


class MockResult:
    def __init__(self, steps=2):
        self.n_steps = steps
        self.t = np.linspace(0, 1, steps)

    def joint_velocities_at(self, idx) -> Any:
        return {"tip": (1.0, 2.0), "club_tip": (3.0, 4.0)}

    def positions_at(self, idx) -> Any:
        return {"tip": (10.0 + idx, 20.0 + idx), "club_tip": (30.0 + idx, 40.0 + idx)}


@patch("double_pendulum_golf.gui.panel_builders.SimulationPanel.set_perturbation_panel")
@patch("double_pendulum_golf.gui.panel_builders.run_simulation")
def test_build_double_panel(mock_run, mock_set_perturb, qapp) -> Any:
    mw = MagicMock()
    panel = build_double_panel(mw)

    # Test param builders
    p_dict = {
        "m1": 1,
        "m2": 2,
        "L1": 1,
        "L2": 1,
        "theta1_rad": 0,
        "phi_rad": 0,
        "dtheta1": 0,
        "dphi": 0,
        "shoulder_coeffs": [0],
        "wrist_coeffs": [0],
        "enable_limits": True,
        "enable_clamp": True,
        "torque_limits": [50, 20],
        "tilt_deg": 10,
        "t_end": 1.0,
    }

    params = (
        panel._build_params(p_dict)
        if hasattr(panel, "_build_params")
        else panel._params_builder(p_dict)
    )
    (
        panel._build_state(p_dict)
        if hasattr(panel, "_build_state")
        else panel._state_builder(p_dict)
    )
    (
        panel._build_torque(p_dict)
        if hasattr(panel, "_build_torque")
        else panel._torque_builder(p_dict)
    )

    # limits_builder / clamp_builder
    limits = panel._limits_builder(p_dict)
    clamp = panel._clamp_builder(p_dict)

    assert params is not None
    assert limits is not None
    assert clamp is not None

    # Test limits off
    p_dict_off = p_dict.copy()
    p_dict_off["enable_limits"] = False
    p_dict_off["enable_clamp"] = False
    assert panel._limits_builder(p_dict_off) is None
    assert panel._clamp_builder(p_dict_off) is None

    # Test objective builder
    p_dict["t_end"] = 1.0
    obj_fn = panel.objective_builder(p_dict)

    # Test valid simulation return
    mock_run.return_value = MockResult()
    assert obj_fn(np.array([0.0, 0.0])) < 0

    # Test exception fallback
    mock_run.side_effect = ValueError("boom")
    assert obj_fn(np.array([0.0, 0.0])) == 0.0

    # Test perturbation callbacks
    mock_run.side_effect = None
    mock_run.return_value = MockResult()

    # Extract real perturbation panel
    real_perturb = mock_set_perturb.call_args[0][0]
    simulate_fn = real_perturb._simulate_fn
    extract_fn = real_perturb._extract_fn

    with patch.object(panel.controls, "get_params", return_value=p_dict):
        res = simulate_fn([[1], [1]])
        assert res is not None

    ex = extract_fn(MockResult())
    assert "tip_speed_final" in ex

    # Pre-sets
    presets = real_perturb._get_coeffs_for_preset_fn("Default")
    assert isinstance(presets, list)
    panel.controls.PRESETS = {"Default": ["", "", "", "", "1.0", ""]}
    parsed = real_perturb._get_coeffs_for_preset_fn("Default")
    assert len(parsed) == 2


@patch("double_pendulum_golf.gui.panel_builders.SimulationPanel.set_perturbation_panel")
@patch("double_pendulum_golf.gui.panel_builders.run_simulation_triple")
def test_build_triple_panel(mock_run, mock_set_perturb, qapp) -> Any:
    mw = MagicMock()
    panel = build_triple_panel(mw)

    # Test param builders
    p_dict = {
        "m1": 1,
        "m2": 2,
        "m3": 3,
        "L1": 1,
        "L2": 1,
        "L3": 1,
        "theta1_rad": 0,
        "phi1_rad": 0,
        "phi2_rad": 0,
        "dtheta1": 0,
        "dphi1": 0,
        "dphi2": 0,
        "shoulder_coeffs": [0],
        "elbow_coeffs": [0],
        "wrist_coeffs": [0],
        "enable_limits": True,
        "enable_clamp": True,
        "limit_mins_rad": [0, 0, 0],
        "limit_maxs_rad": [1, 1, 1],
        "torque_limits": [50, 20, 10],
        "t_end": 1.0,
    }

    limits = panel._limits_builder(p_dict)
    clamp = panel._clamp_builder(p_dict)
    assert limits is not None
    assert clamp is not None

    # Objective builder
    obj_fn = panel.objective_builder(p_dict)
    mock_run.return_value = MockResult()
    assert obj_fn(np.array([0.0, 0.0, 0.0])) < 0

    mock_run.side_effect = ValueError()
    assert obj_fn(np.array([0.0, 0.0, 0.0])) == 0.0

    # Perturbation
    mock_run.side_effect = None
    real_perturb = mock_set_perturb.call_args[0][0]
    simulate_fn = real_perturb._simulate_fn
    extract_fn = real_perturb._extract_fn

    with patch.object(panel.controls, "get_params", return_value=p_dict):
        simulate_fn([[1], [1], [1]])

    # Single frame triple result
    ex1 = extract_fn(MockResult(steps=1))
    assert ex1["tip_speed_final"] == 0.0

    ex2 = extract_fn(MockResult(steps=3))
    assert ex2["tip_speed_final"] > 0.0

    real_perturb._get_coeffs_for_preset_fn("Default")

    panel.controls.PRESETS = {
        "Default": ["0", "0", "0", "0", "0", "0", "1.0, 2.0", "3.0", ""]
    }
    parsed = real_perturb._get_coeffs_for_preset_fn("Default")
    assert len(parsed) == 3


@patch("double_pendulum_golf.gui.panel_builders.SimulationPanel.set_perturbation_panel")
@patch("double_pendulum_golf.gui.panel_builders.run_simulation_golfer")
def test_build_golfer_panel(mock_run, mock_set_perturb, qapp) -> Any:
    mw = MagicMock()
    panel = build_golfer_panel(mw)

    p_dict = {
        "m_hub": 1,
        "m_r_upper": 1,
        "m_r_fore": 1,
        "m_l_upper": 1,
        "m_l_fore": 1,
        "m_club": 1,
        "L_hub": 1,
        "L_r_upper": 1,
        "L_r_fore": 1,
        "L_l_upper": 1,
        "L_l_fore": 1,
        "L_club": 1,
        "d_rs": 1,
        "d_ls": 1,
        "grip_right": 1,
        "grip_left": 1,
        "theta_hub_rad": 0,
        "alpha_rs_rad": 0,
        "alpha_re_rad": 0,
        "alpha_rh_rad": 0,
        "alpha_ls_rad": 0,
        "alpha_le_rad": 0,
        "alpha_lh_rad": 0,
        "hub_coeffs": [0],
        "rs_coeffs": [0],
        "re_coeffs": [0],
        "rh_coeffs": [0],
        "ls_coeffs": [0],
        "le_coeffs": [0],
        "lh_coeffs": [0],
        "enable_limits": True,
        "enable_clamp": True,
        "limit_mins_rad": [0] * 7,
        "limit_maxs_rad": [1] * 7,
        "torque_limits": [50] * 7,
        "t_end": 1.0,
    }

    limits = panel._limits_builder(p_dict)
    assert limits is not None

    obj_fn = panel.objective_builder(p_dict)
    mock_run.return_value = MockResult()
    assert obj_fn(np.zeros(7)) < 0

    mock_run.side_effect = RuntimeError()
    assert obj_fn(np.zeros(7)) == 0.0

    # Perturbation
    mock_run.side_effect = None
    real_perturb = mock_set_perturb.call_args[0][0]
    simulate_fn = real_perturb._simulate_fn
    extract_fn = real_perturb._extract_fn

    with patch.object(panel.controls, "get_params", return_value=p_dict):
        simulate_fn([[0]] * 7)
        # hit _golfer_coeffs_fn
        coeffs = real_perturb._get_coeffs_fn()
        assert len(coeffs) == 7

    ex = extract_fn(MockResult(steps=2))
    assert ex["tip_speed_final"] > 0.0

    ex_one = extract_fn(MockResult(steps=1))
    assert ex_one["tip_speed_final"] == 0.0

    real_perturb._get_coeffs_for_preset_fn("Default")
    panel.controls.PRESETS = {"Default": {"tau_hub": "1.0, 2.0", "tau_rs": ""}}
    parsed = real_perturb._get_coeffs_for_preset_fn("Default")
    assert len(parsed) == 7


def test_wire_toolstrip(qapp) -> Any:
    class DummyMainWindow(ThemedWindowMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.setup_theme_support()
            self._toolstrip = ToolStrip()
            self._tabs = QTabWidget()
            self._double_panel = MagicMock()
            self._double_panel.pendulum = MagicMock()
            self._double_panel.pendulum.reset_view = MagicMock()
            self._double_panel.pendulum.set_visible_segments = MagicMock()

            self._triple_panel = MagicMock()
            self._golfer_panel = MagicMock()

        def _active_panel(self) -> Any:
            return self._double_panel

        def _on_popout_chart(self) -> Any:
            pass

        def _on_tab_changed(self, idx) -> Any:
            pass

    mw = DummyMainWindow()
    wire_toolstrip(mw)

    # Trigger signals on toolstrip
    ts = mw._toolstrip
    ts.run_requested.emit()
    ts.reset_requested.emit()
    ts.play_toggled.emit(True)
    ts.speed_changed.emit(2.0)
    ts.frame_scrubbed.emit(10)
    ts.export_data_requested.emit()
    ts.export_video_requested.emit()
    ts.popout_chart_requested.emit()

    mw._double_panel.controls.run_requested.emit.assert_called_once()

    # Check simple overlay signal
    ts.force_scale_changed.emit(5)

    # Check reset view
    ts.reset_view_requested.emit()

    # Check segment visibility signal
    # need string list
    ts.segment_visibility_changed.emit(["one", "two"])

    # Tabs changing
    ts.model_changed.emit(1)
    mw._tabs.currentChanged.emit(2)

    # Simulate started on active panel
    mw._double_panel.sim_started.emit()
    mw._double_panel.sim_finished.emit()
    mw._double_panel.frame_changed.emit(5, 1.0)
