import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.launch_pyqt6 import ControlPanel
from lower_body_model.simulator import LowerBodySimulator

_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    _APP = app
    return app


@pytest.fixture
def test_sim():
    xml = build_lower_body_xml()
    return LowerBodySimulator(xml)


@pytest.fixture
def mock_viewer():
    viewer = MagicMock()
    viewer.is_running.return_value = True
    return viewer


def test_control_panel_init_and_properties(test_sim, mock_viewer):
    """Test that the PyQt6 control panel initializes correctly without crashing."""
    _qapp()

    panel = ControlPanel(test_sim, mock_viewer)
    assert panel is not None
    assert panel.windowTitle() == "Lower Body Control Panel"

    # Assert play state is safe
    assert not panel.is_playing

    # Safely close to clean up background threads
    test_sim.reset()
    mock_viewer.is_running.return_value = False
    panel.close()


def test_full_reset_keeps_loaded_hip_target_and_resets_playback(test_sim, mock_viewer):
    """Full reset clears playback history and reapplies the target at t=0."""
    _qapp()

    panel = ControlPanel(test_sim, mock_viewer)
    test_sim.configure_hip_rotation_target(duration_sec=1.0)
    panel.is_playing = True
    panel.play_btn.setText("Pause")
    panel.timeline_slider.setEnabled(False)

    for _ in range(3):
        test_sim.step()
    assert test_sim.data.time > 0
    assert test_sim.history

    panel.full_reset_simulation()

    assert not panel.is_playing
    assert panel.play_btn.text() == "Play"
    assert panel.timeline_slider.isEnabled()
    assert panel.timeline_slider.maximum() == 0
    assert panel.timeline_slider.value() == 0
    assert test_sim.data.time == 0
    assert test_sim.history == []
    assert test_sim.hip_rotation_target is not None
    assert test_sim.hip_rotation_target.rotation_degrees_at(0.0) == 0.0
    for side in ("r", "l"):
        assert test_sim.data.qpos[test_sim.jnt_qpos_idx[f"{side}_hip_z"]] == 0.0
        assert test_sim.data.qpos[
            test_sim.jnt_qpos_idx[f"{side}_hip_x"]
        ] == pytest.approx(np.radians(test_sim.hip_rotation_target.incline_degrees))
    mock_viewer.sync.assert_called()

    mock_viewer.is_running.return_value = False
    panel.close()
