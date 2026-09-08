from typing import Any

"""Tests for simulation_panel.py"""

from typing import Any
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import QByteArray, QSettings, pyqtSignal
from PyQt6.QtWidgets import QWidget

from double_pendulum_golf.gui.controls_widget import ControlsWidget
from double_pendulum_golf.gui.controls_widget_golfer import ControlsWidgetGolfer
from double_pendulum_golf.gui.controls_widget_triple import ControlsWidgetTriple
from double_pendulum_golf.gui.simulation_panel import SimulationPanel, _SimWorker


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

    def get_params(self) -> Any:
        return {"t_end": 1.0, "dt": 0.005}

    def set_slider_range(self, val) -> Any:
        pass

    def set_slider_value(self, val) -> Any:
        pass

    def stop_playback(self) -> Any:
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


def create_mock_result(n_steps=10, is_triple=False, is_golfer=False) -> Any:
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
def mock_sim_kwargs() -> Any:
    controls = MockControls()
    pendulum = MockViewer()
    matrix = MockViewer()
    torque_history = MockViewer()

    # We define the signal on the optimizer mock so we can emit it
    class MockOpt(QWidget):
        optimized_coefficients = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.bind_objective_builder = MagicMock()
            self.set_objective_function = MagicMock()

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


def test_panel_init_and_signals(qapp, mock_sim_kwargs) -> Any:
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
def test_on_run_success(mock_start, qapp, mock_sim_kwargs) -> Any:
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


def test_on_run_failures(qapp, mock_sim_kwargs) -> Any:
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


def test_sim_error(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.critical") as msg:
        panel._on_sim_error("test error")
        msg.assert_called_once()


def test_playback(qapp, mock_sim_kwargs) -> Any:
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


def test_export_data(qapp, mock_sim_kwargs, tmp_path) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)

    # show message if no result
    with patch("double_pendulum_golf.gui.simulation_panel.QMessageBox.information") as info:
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


def test_export_image(qapp, mock_sim_kwargs, tmp_path) -> Any:
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
def test_export_video(mock_which, mock_run, qapp, mock_sim_kwargs, tmp_path) -> Any:
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


def test_apply_optimized_coefficients(qapp, mock_sim_kwargs) -> Any:
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
    panel_triple._apply_optimized_coefficients({"coeffs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    # Test golfer
    mock_sim_kwargs["controls"] = MockControlsGolfer()
    panel_golfer = SimulationPanel(**mock_sim_kwargs)
    panel_golfer.controls.inp_tau_hub = MagicMock()
    panel_golfer._apply_optimized_coefficients({"coeffs": np.zeros(14)})

    # Test missing coeffs
    panel_golfer._apply_optimized_coefficients({})


def test_patched_on_run_optimizer(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)

    panel.optimizer.bind_objective_builder.assert_called_once()
    params_getter, objective_builder = panel.optimizer.bind_objective_builder.call_args[0]

    assert params_getter == panel.controls.get_params
    assert objective_builder == panel.objective_builder


def test_sim_worker() -> Any:
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


# Regression tests for the tabbed-side-panel layout refactor


def test_panel_uses_side_panel_tabs(qapp, mock_sim_kwargs) -> Any:
    """SimulationPanel hosts a SidePanelTabs as the right-hand container."""
    from double_pendulum_golf.gui.side_panel_tabs import SidePanelTabs

    panel = SimulationPanel(**mock_sim_kwargs)
    assert hasattr(panel, "_side_tabs")
    assert isinstance(panel._side_tabs, SidePanelTabs)


def test_setup_tab_is_first(qapp, mock_sim_kwargs) -> Any:
    """The Setup tab (controls widget) is always first so users see it on open."""
    panel = SimulationPanel(**mock_sim_kwargs)
    labels = panel._side_tabs.panel_labels()
    assert labels[0] == SimulationPanel.TAB_SETUP
    assert panel._side_tabs.panel_widget(SimulationPanel.TAB_SETUP) is panel.controls


def test_mass_matrix_tab_present(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)
    labels = panel._side_tabs.panel_labels()
    assert SimulationPanel.TAB_MASS_MATRIX in labels


def test_plots_tab_present_when_torque_history_supplied(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)
    labels = panel._side_tabs.panel_labels()
    assert SimulationPanel.TAB_PLOTS in labels
    assert panel._side_tabs.panel_widget(SimulationPanel.TAB_PLOTS) is panel.torque_history


def test_plots_tab_absent_when_torque_history_omitted(qapp, mock_sim_kwargs) -> Any:
    kwargs = {**mock_sim_kwargs, "torque_history": None}
    panel = SimulationPanel(**kwargs)
    assert SimulationPanel.TAB_PLOTS not in panel._side_tabs.panel_labels()


def test_optimizer_tab_present_when_optimizer_supplied(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)
    assert SimulationPanel.TAB_OPTIMIZER in panel._side_tabs.panel_labels()


def test_optimizer_tab_absent_when_optimizer_omitted(qapp, mock_sim_kwargs) -> Any:
    kwargs = {**mock_sim_kwargs, "optimizer": None, "objective_builder": None}
    panel = SimulationPanel(**kwargs)
    assert SimulationPanel.TAB_OPTIMIZER not in panel._side_tabs.panel_labels()


def test_set_perturbation_panel_creates_noise_tab(qapp, mock_sim_kwargs) -> Any:
    """set_perturbation_panel adds a Noise tab as the last entry."""
    from PyQt6.QtWidgets import QLabel

    panel = SimulationPanel(**mock_sim_kwargs)
    perturb = QLabel("perturbation panel")
    panel.set_perturbation_panel(perturb)
    labels = panel._side_tabs.panel_labels()
    assert labels[-1] == SimulationPanel.TAB_NOISE
    assert panel.perturbation_panel is perturb
    assert panel._side_tabs.panel_widget(SimulationPanel.TAB_NOISE) is perturb


def test_set_perturbation_panel_rejects_none(qapp, mock_sim_kwargs) -> Any:
    panel = SimulationPanel(**mock_sim_kwargs)
    with pytest.raises((AssertionError, ValueError), match="None"):
        panel.set_perturbation_panel(None)  # type: ignore[arg-type]


def test_pendulum_widget_is_left_of_tabs(qapp, mock_sim_kwargs) -> Any:
    """The pendulum graphic occupies splitter index 0; tabs are index 1.

    Locks the "graphic always visible on the left" contract.
    """
    panel = SimulationPanel(**mock_sim_kwargs)
    splitter = panel._splitter
    assert splitter.count() == 2
    assert splitter.widget(0) is panel.pendulum
    assert splitter.widget(1) is panel._side_tabs


def test_pendulum_widget_is_not_collapsible(qapp, mock_sim_kwargs) -> Any:
    """The user cannot accidentally hide the graphic by dragging the splitter."""
    panel = SimulationPanel(**mock_sim_kwargs)
    splitter = panel._splitter
    assert splitter.isCollapsible(0) is False
    assert splitter.isCollapsible(1) is False


def test_save_layout_persists_active_tab(qapp, mock_sim_kwargs) -> Any:
    """save_layout writes both splitter state AND active tab to QSettings."""
    panel = SimulationPanel(**mock_sim_kwargs)
    panel._side_tabs.set_active_tab(SimulationPanel.TAB_OPTIMIZER)
    panel.save_layout()
    settings = QSettings("D-sorganization", "PendulumSimulator")
    saved_tab = settings.value(f"{panel._settings_key}/active_tab")
    assert saved_tab == SimulationPanel.TAB_OPTIMIZER
