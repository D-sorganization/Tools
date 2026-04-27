from typing import Any

"""Tests for GolferPendulumWidget."""


import numpy as np
from unittest.mock import MagicMock
from PyQt6.QtGui import QPainter
from double_pendulum_golf.gui.golfer_pendulum_widget import GolferPendulumWidget
from double_pendulum_golf.physics_golfer import GolferParams


def create_mock_result() -> Any:
    res = MagicMock()
    res.n_steps = 2
    res.params = GolferParams(
        m_hub=40.0,
        m_r_upper=3.0,
        m_r_fore=1.5,
        m_l_upper=3.0,
        m_l_fore=1.5,
        m_club=0.5,
        L_hub=0.5,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.3,
        grip_left=0.3,
    )
    res.states = [np.zeros(8), np.zeros(8)]
    res.t = [0.0, 1.0]

    pos_dict = {
        "hub": (0.0, 0.0),
        "rs": (0.1, 0.1),
        "ls": (-0.1, 0.1),
        "re": (0.2, 0.2),
        "le": (-0.2, 0.2),
        "rh": (0.3, 0.3),
        "lh": (-0.3, 0.3),
        "club_base": (0.0, 0.3),
        "club_tip": (0.0, 1.3),
        "grip_right": (0.1, 0.4),
        "grip_left": (-0.1, 0.4),
        "rscap": (0.05, 0.05),
        "lscap": (-0.05, 0.05),
    }

    def mock_pos_at(*args) -> Any:
        return pos_dict

    res.positions_at.side_effect = mock_pos_at

    res.joint_forces_at.return_value = {
        "hub": (1.0, 1.0),
        "re": (0.0, 0.0),
        "unknown": (0.0, 0.0),
    }

    res.torques_at.return_value = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return res


def test_golfer_widget_init(qapp) -> Any:
    w = GolferPendulumWidget()
    assert w._get_total_length() == 2.5
    assert not w._has_result()


def test_set_simulation_and_frame(qapp, monkeypatch) -> Any:
    w = GolferPendulumWidget()
    res = create_mock_result()

    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual_golfer.zero_torque_joint_forces",
        lambda *args: {"hub": (1.0, 1.0)},
    )

    w.set_simulation(res)
    assert w._has_result()
    assert w._get_total_length() > 2.0

    w.set_frame(1)
    assert w._current_idx == 1

    w.clear()
    assert not w._has_result()


def test_painting(qapp, monkeypatch) -> Any:
    w = GolferPendulumWidget()
    res = create_mock_result()

    # Mock zero_torque
    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual_golfer.zero_torque_joint_forces",
        lambda *args: {"hub": (1.0, 1.0)},
    )
    # Mock ellipsoids
    ell_data = {
        "hub": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": np.array([1.0, 1.0]),
        },
        "club_tip": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": None,  # coverage branch
        },
    }
    monkeypatch.setattr(
        "double_pendulum_golf.jacobians_golfer.ellipsoids_golfer",
        lambda *args: ell_data,
    )

    w.set_simulation(res)

    # Enable all feature toggles
    w.set_show_forces(True)
    w.set_show_zero_torque_forces(True)
    w.set_show_torque_vectors(True)
    w.set_show_mob_ellipsoids(True)
    w.set_show_force_ellipsoids(True)
    w.set_show_com(True)
    w.set_gravity_on(False)

    painter = MagicMock(spec=QPainter)
    w.resize(400, 400)

    # Exception branch for zero_torque_joint_forces
    def broken_zero(*args) -> Any:
        raise ValueError()

    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual_golfer.zero_torque_joint_forces",
        broken_zero,
    )
    w.set_simulation(res)
    w.paintEvent(MagicMock())

    # Draw without result
    w.clear()
    w.paintEvent(MagicMock())

    # Draw with actual everything
    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual_golfer.zero_torque_joint_forces",
        lambda *args: {"hub": (1.0, 1.0)},
    )
    w.set_simulation(res)
    w.paintEvent(MagicMock())

    # Test visible_segments filter
    w.set_visible_segments({"hub"})
    w.paintEvent(MagicMock())

    # Exception branches inside drawing
    res.joint_forces_at.side_effect = AttributeError
    w._draw_force_vectors(painter)

    monkeypatch.setattr(
        "double_pendulum_golf.jacobians_golfer.ellipsoids_golfer",
        MagicMock(side_effect=ValueError),
    )
    w._draw_ellipsoids_at_frame(painter)

    res.torques_at.side_effect = AttributeError
    w._draw_torque_vectors(painter)
