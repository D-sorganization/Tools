import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from double_pendulum_golf.gui.pendulum_widget import PendulumWidget


def create_mock_result(is_triple=False):
    res = MagicMock()
    res.n_steps = 2

    if is_triple:
        res.params = SimpleNamespace(L1=1.0, L2=1.0, L3=1.0, m1=1.0, m2=1.0, m3=1.0)
        res.states = np.zeros((2, 6))
        pos_dict = {
            "shoulder": (0.0, 0.0),
            "wrist1": (1.0, 0.0),
            "wrist2": (2.0, 0.0),
            "tip": (3.0, 0.0),
        }
    else:
        res.params = SimpleNamespace(L1=1.0, L2=1.0, m1=1.0, m2=1.0)
        res.states = np.zeros((2, 4))
        pos_dict = {
            "shoulder": (0.0, 0.0),
            "wrist": (1.0, 0.0),
            "tip": (2.0, 0.0),
        }

    res.t = [0.0, 1.0]

    def mock_pos_at(idx):
        return pos_dict

    res.positions_at.side_effect = mock_pos_at

    res.joint_forces_at.return_value = {
        "shoulder": (1.0, 1.0),
        "wrist": (0.0, 0.0),
        "wrist1": (0.0, 0.0),
        "wrist2": (0.0, 0.0),
        "unknown": (0.0, 0.0),
    }

    res.torques_at.return_value = [1.0, -1.0, 0.0] if is_triple else [1.0, -1.0]
    return res


def test_pendulum_widget_init(qapp):
    w = PendulumWidget()
    assert w._get_total_length() == 2.0
    assert not w._has_result()


@patch(
    "double_pendulum_golf.counterfactual.zero_torque_joint_forces_double",
    lambda *args: {"shoulder": (1.0, 1.0)},
)
@patch(
    "double_pendulum_golf.counterfactual.zero_torque_joint_forces_triple",
    lambda *args: {"shoulder": (1.0, 1.0)},
)
def test_set_simulation_and_frame(qapp):
    w = PendulumWidget()

    # Test double
    res = create_mock_result(is_triple=False)

    w.set_simulation(res)
    assert w._has_result()
    assert w._get_total_length() == 2.0

    # Boundary tests for idx
    w.set_frame(-1)
    w.set_frame(100)
    w.set_frame(1)
    assert w._current_idx == 1

    # Test tip cache missing fallback
    w._tip_positions_cache = None
    w.set_frame(1)

    # Dead code _draw_model
    w._draw_model(MagicMock())

    w.clear()
    assert not w._has_result()

    # Test triple
    res3 = create_mock_result(is_triple=True)
    w.set_simulation(res3)
    assert w._get_total_length() == 3.0


def test_painting(qapp, monkeypatch):
    w = PendulumWidget()
    res = create_mock_result(is_triple=False)

    # Mock zero_torque
    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual.zero_torque_joint_forces_double",
        lambda *args: {"shoulder": (1.0, 1.0)},
    )

    # Mock ellipsoids
    ell_data = {
        "shoulder": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": np.array([1.0, 1.0]),
        },
        "tip": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": None,  # coverage branch
        },
    }
    monkeypatch.setattr(
        "double_pendulum_golf.jacobians.ellipsoids_double", lambda *args: ell_data
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

    # Extra toggles
    w.set_show_sum_moments(True)
    w.set_show_moment_of_force(True)

    painter = MagicMock()
    w.resize(400, 400)

    # Draw with actual everything (double)
    w.paintEvent(MagicMock())

    # Handle exception in zero_torque
    def broken_zero(*args):
        raise ValueError()

    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual.zero_torque_joint_forces_double",
        broken_zero,
    )
    w.set_simulation(res)
    # also toggle 3D mode for coverage
    w._3d_mode = True
    w.paintEvent(MagicMock())

    # Missing attribute branch and empty lists in drawing
    del res.joint_forces_at
    w._draw_force_vectors(painter, {})

    # torques_at Exception branch cover for _draw_torque_vectors and _draw_moment_of_force
    res.torques_at.side_effect = AttributeError("test error")
    w._draw_torque_vectors(painter)
    w._draw_moment_of_force(painter)

    # Draw without result
    w.clear()
    w.paintEvent(MagicMock())


def test_painting_triple(qapp, monkeypatch):
    w = PendulumWidget()
    res = create_mock_result(is_triple=True)

    monkeypatch.setattr(
        "double_pendulum_golf.counterfactual.zero_torque_joint_forces_triple",
        lambda *args: {"shoulder": (1.0, 1.0)},
    )

    ell_data = {
        "shoulder": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": np.array([1.0, 1.0]),
        },
        "tip": {
            "directions": np.eye(2),
            "mob_semi_axes": np.array([1.0, 1.0]),
            "force_semi_axes": None,
        },
    }
    monkeypatch.setattr(
        "double_pendulum_golf.jacobians.ellipsoids_triple", lambda *args: ell_data
    )

    w.set_simulation(res)
    w.set_show_forces(True)
    w.set_show_zero_torque_forces(True)
    w.set_show_torque_vectors(True)
    w.set_show_mob_ellipsoids(True)
    w.set_show_force_ellipsoids(True)
    w.set_show_com(True)
    w.set_gravity_on(False)
    w.set_show_sum_moments(True)
    w.set_show_moment_of_force(True)

    # Enable 3d_mode rendering for triple
    w._3d_mode = True

    painter = MagicMock()
    w.resize(400, 400)
    w.paintEvent(painter)

    # Exception branch for moments and ellipsoids
    monkeypatch.setattr(
        "double_pendulum_golf.gui.pendulum_widget.triple_pendulum_moments",
        MagicMock(side_effect=ValueError),
    )
    w._draw_moment_of_force(painter)

    monkeypatch.setattr(
        "double_pendulum_golf.jacobians.ellipsoids_triple",
        MagicMock(side_effect=ValueError),
    )
    w._draw_ellipsoids_at_frame(painter)


@patch(
    "double_pendulum_golf.counterfactual.zero_torque_joint_forces_double",
    lambda *args: {"shoulder": (1.0, 1.0)},
)
def test_visible_segments(qapp, monkeypatch):
    w = PendulumWidget()
    res = create_mock_result(is_triple=False)
    w.set_simulation(res)
    w.set_visible_segments({"shoulder"})
    w.paintEvent(MagicMock())


@patch(
    "double_pendulum_golf.counterfactual.zero_torque_joint_forces_double",
    lambda *args: {"shoulder": (1.0, 1.0)},
)
def test_zoom_controls(qapp):
    w = PendulumWidget()
    res = create_mock_result()
    w.set_simulation(res)

    painter = MagicMock()
    w._draw_zoom_controls(painter)

    import PyQt6.QtCore as QtCore

    # zoom in
    w._handle_zoom_button_click(
        QtCore.QPoint(w._zoom_btn_rects[0].x() + 5, w._zoom_btn_rects[0].y() + 5)
    )
    assert w._zoom > 1.0

    # zoom out
    w._handle_zoom_button_click(
        QtCore.QPoint(w._zoom_btn_rects[1].x() + 5, w._zoom_btn_rects[1].y() + 5)
    )

    # reset
    w._handle_zoom_button_click(
        QtCore.QPoint(w._zoom_btn_rects[2].x() + 5, w._zoom_btn_rects[2].y() + 5)
    )
    assert w._zoom == 1.0
