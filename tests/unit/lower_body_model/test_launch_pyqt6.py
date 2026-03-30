import sys
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.launch_pyqt6 import ControlPanel
from lower_body_model.simulator import LowerBodySimulator


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
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    panel = ControlPanel(test_sim, mock_viewer)
    assert panel is not None
    assert panel.windowTitle() == "Lower Body Control Panel"

    # Assert play state is safe
    assert not panel.is_playing

    # Safely close to clean up background threads
    test_sim.reset()
    mock_viewer.is_running.return_value = False
    panel.close()
