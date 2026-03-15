"""Tests for simulation_panel.py"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QByteArray, QSettings, pyqtSignal
from PyQt6.QtWidgets import QWidget

from double_pendulum_golf.gui.simulation_panel import SimulationPanel, _SimWorker

from double_pendulum_golf.gui.controls_widget import ControlsWidget
from double_pendulum_golf.gui.controls_widget_triple import ControlsWidgetTriple
from double_pendulum_golf.gui.controls_widget_golfer import ControlsWidgetGolfer


class MockControls(QWidget):
    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_changed = pyqtSignal(int)
    export_data_requested = pyqtSignal()
    export_video_requested = pyqtSignal()
    export_image_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.btn_run = MagicMock()
        self.btn_reset = MagicMock()
        self.btn_play = MagicMock()

    def get_params(self):
        return {"t_end": 1.0, "dt": 0.005}

    def set_slider_range(self, val):
        pass

    def set_slider_value(self, val):
        pass

    def stop_playback(self):
        pass


class MockControlsDouble(ControlsWidget, MockControls):
    def __init__(self):
        MockControls.__init__(self)


class MockControlsTriple(ControlsWidgetTriple, MockControls):
    def __init__(self):
        MockControls.__init__(self)


class MockControlsGolfer(ControlsWidgetGolfer, MockControls):
    def __init__(self):
        MockControls.__init__(self)


class MockViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.set_simulation = MagicMock()
        self.set_frame = MagicMock()
        self.clear = MagicMock()
        self.grab = MagicMock()


def create_mock_result(n_steps=10, is_triple=False, is_golfer=False):
    res = MagicMock()
    res.n_steps = n_steps
    res.t = np.linspace(0, 1, n_steps)
    if is_triple:
        res.states = np.zeros((n_steps, 6))
        res.torques_at.return_value = [0, 0, 0]
    elif is_golfer:
        res.states = np.zeros((n_steps, 8))
        res.torques_at.return_value = [0, 0, 0, 0, 0, 0, 0]
    else:
        res.states = np.zeros((n_steps, 4))
        res.torques_at.return_value = [0, 0]

    res.joint_forces_at.return_value = {
        "shoulder": (0, 0),
        "wrist": (0, 0),
        "wrist1": (0, 0),
        "wrist2": (0, 0),
    }
    res.friction_torques_at.return_value = [0, 0]
    res.total_torques_at.return_value = [0, 0]
    res.positions_at.return_value = {"tip": (1, 1)}
    return res


@pytest.fixture
def mock_sim_kwargs():
    controls = MockControls()
    pendulum = MockViewer()
    matrix = MockViewer()
    torque_history = MockViewer()
    MagicMock()

    # We define the signal on the optimizer mock so we can emit it
    class MockOpt(QWidget):
        optimized_coefficients = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self._btn_run = MagicMock()
            self._log = MagicMock()
            self.set_objective_function = MagicMock()
            self._on_run = MagicMock()

    opt = MockOpt()

    return {
        "controls": controls,
        "pendulum": pendulum,
        "matrix": matrix,
        "torque_history": torque_history,
        "optimizer": opt,
        "params_builder": MagicMock(return_value={}),
        "torque_builder": MagicMock(return_value={}),
        "state_builder": MagicMock(return_value=np.zeros(4)),
        "run_simulation": MagicMock(return_value=create_mock_result()),
        "objective_builder": MagicMock(return_value=lambda x: 0.0),
        "limits_builder": MagicMock(return_value={}),
        "clamp_builder": MagicMock(return_value={}),
    }


def test_panel_init_and_signals(qapp, mock_sim_kwargs):
    QSettings().setValue("splitter_double", QByteArray(b"dummy"))
    panel = SimulationPanel(**mock_sim_kwargs)

    # Add perturbation
    panel.set_perturbation_panel(QWidget())
    assert panel.perturbation_panel is not None

    # Test layout saving
    panel.save_layout()
    assert (
        QSettings("D-sorganization", "PendulumSimulator").value(panel._settings_key)
        is not None
    )


@patch("PyQt6.QtCore.QThread.start")
def test_on_run_success(mock_start, qapp, mock_sim_kwargs):
    panel = SimulationPanel(**mock_sim_kwargs)

    # normal run
    panel._on_run()
    assert panel._sim_thread is not None
    assert isinstance(panel._sim_worker, _SimWorker)

    # Test sim done
    res = create_mock_result()
    panel._on_sim_done(res)
    assert panel._result == res
    assert panel._anim_idx == 0


def test_on_run_failures(qapp, mock_sim_kwargs):
    panel = SimulationPanel(**mock_sim_kwargs)

    # param validation fail
    panel.controls.get_params = MagicMock(side_effect=ValueError("test"))
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.warning") as msg:
        panel._on_run()
        msg.assert_called_once()

    # param builder fail
    panel.controls.get_params = MagicMock(return_value={"t_end": 1.0, "dt": 0.005})
    panel._params_builder.side_effect = AssertionError("test")
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.warning") as msg:
        panel._on_run()
        msg.assert_called_once()

    # t_end negative
    panel._params_builder.side_effect = None
    panel.controls.get_params = MagicMock(return_value={"t_end": -1.0, "dt": 0.005})
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.warning") as msg:
        panel._on_run()
        msg.assert_called_once()

    # state builder fail
    panel.controls.get_params = MagicMock(return_value={"t_end": 1.0, "dt": 0.005})
    panel._state_builder.side_effect = ValueError("test")
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.warning") as msg:
        panel._on_run()
        msg.assert_called_once()


def test_sim_error(qapp, mock_sim_kwargs):
    panel = SimulationPanel(**mock_sim_kwargs)
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.critical") as msg:
        panel._on_sim_error("test error")
        msg.assert_called_once()


def test_playback(qapp, mock_sim_kwargs):
    panel = SimulationPanel(**mock_sim_kwargs)
    res = create_mock_result(n_steps=5)
    panel._on_sim_done(res)

    # on play toggle without result
    panel._result = None
    panel._on_play_toggle(True)

    # with result
    panel._result = res
    panel._anim_idx = 4  # end

    # test reset
    panel._on_reset()
    assert panel._anim_idx == 0
    assert panel._result is None

    # Restore result
    panel._result = res
    panel._anim_idx = 4
    panel.pendulum._trail = []
    panel._on_play_toggle(True)
    assert panel._anim_idx == 0  # reset to start

    panel._on_play_toggle(False)  # stop timer

    panel._on_speed_change(2.0)
    assert panel._playback_speed == 2.0

    # scrub
    panel.scrub_to_frame(3)
    assert panel._anim_idx == 3

    # _advance_frame
    panel._playback_speed = 5.0  # fast, skip frames
    panel._sim_dt = 0.005
    panel._advance_frame()
    assert panel._anim_idx > 3

    # loop logic
    panel._loop_playback = True
    panel.scrub_to_frame(4)
    panel._advance_frame()
    assert panel._anim_idx == 0

    # frame change logic missing cache
    panel.pendulum._tip_positions_cache = None
    panel._on_frame_change(2)

    # return early if result is None
    panel._result = None
    panel._on_frame_change(2)

    # restore result
    panel._result = res
    panel.pendulum._tip_positions_cache = [np.array([0, 0])] * 10
    panel._on_frame_change(2)

    panel.current_n_steps()


def test_export_data(qapp, mock_sim_kwargs, tmp_path):
    panel = SimulationPanel(**mock_sim_kwargs)

    # show message if no result
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QMessageBox.information"
    ) as info:
        panel._on_export_data()
        info.assert_called_once()

    res = create_mock_result()
    panel._on_sim_done(res)

    out_file = tmp_path / "test.csv"
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_file), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel._on_export_data()

    assert out_file.exists()

    # test triple
    res3 = create_mock_result(is_triple=True)
    panel._on_sim_done(res3)
    out_file3 = tmp_path / "test3.csv"
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_file3), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel._on_export_data()

    assert out_file3.exists()


def test_export_image(qapp, mock_sim_kwargs, tmp_path):
    panel = SimulationPanel(**mock_sim_kwargs)
    res = create_mock_result()
    panel._on_sim_done(res)

    out_png = tmp_path / "test.png"
    panel.pendulum.grab.return_value.save.return_value = True
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_png), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel.export_image()

    out_svg = tmp_path / "test.svg"
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_svg), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel.export_image()

    out_pdf = tmp_path / "test.pdf"
    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_pdf), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel.export_image()


@patch("double_pendulum_golf.gui.simulation_panel.subprocess.run")
@patch("shutil.which", return_value="/bin/ffmpeg")
def test_export_video(mock_which, mock_run, qapp, mock_sim_kwargs, tmp_path):
    panel = SimulationPanel(**mock_sim_kwargs)
    res = create_mock_result(n_steps=2)
    panel._on_sim_done(res)

    out_mp4 = tmp_path / "video.mp4"
    mock_run.return_value = MagicMock(returncode=0)

    with patch(
        "double_pendulum_golf.gui.simulation_panel.QFileDialog.getSaveFileName",
        return_value=(str(out_mp4), ""),
    ):
        with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information"):
            panel._on_export_video()


def test_apply_optimized_coefficients(qapp, mock_sim_kwargs):
    # Test double
    mock_sim_kwargs["controls"] = MockControlsDouble()
    panel_double = SimulationPanel(**mock_sim_kwargs)
    panel_double.controls.inp_tau_shoulder = MagicMock()
    panel_double.controls.inp_tau_wrist = MagicMock()
    panel_double._apply_optimized_coefficients({"coeffs": [1.0, 2.0, 3.0, 4.0]})

    # Test triple
    mock_sim_kwargs["controls"] = MockControlsTriple()
    panel_triple = SimulationPanel(**mock_sim_kwargs)
    panel_triple.controls.inp_tau_shoulder = MagicMock()
    panel_triple.controls.inp_tau_elbow = MagicMock()
    panel_triple.controls.inp_tau_wrist = MagicMock()
    panel_triple._apply_optimized_coefficients(
        {"coeffs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )

    # Test golfer
    mock_sim_kwargs["controls"] = MockControlsGolfer()
    panel_golfer = SimulationPanel(**mock_sim_kwargs)
    panel_golfer.controls.inp_tau_hub = MagicMock()
    panel_golfer._apply_optimized_coefficients({"coeffs": np.zeros(14)})

    # Test missing coeffs
    panel_golfer._apply_optimized_coefficients({})


def test_patched_on_run_optimizer(qapp, mock_sim_kwargs):
    panel = SimulationPanel(**mock_sim_kwargs)

    # Access the patched on run method to cover lines 294-301
    panel.optimizer._btn_run.clicked.emit()

    # Test ValueError
    panel.controls.get_params = MagicMock(side_effect=ValueError("test val"))
    panel.optimizer._btn_run.clicked.emit()

    # Test AssertionError
    panel.controls.get_params = MagicMock(side_effect=AssertionError("test err"))
    panel.optimizer._btn_run.clicked.emit()


def test_sim_worker():
    run_fn = MagicMock(return_value="obj")
    worker = _SimWorker(run_fn, {"a": 1})

    worker.finished = MagicMock()
    worker.progress = MagicMock()
    worker.error = MagicMock()

    worker.run()
    worker.finished.emit.assert_called_with("obj")

    run_fn.side_effect = RuntimeError("test")
    worker.run()
    worker.error.emit.assert_called_with("test")
