from typing import Any

"""Tests for torque history and preview widgets."""

import pytest
from unittest.mock import MagicMock
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QColor, QPaintEvent, QRegion

from double_pendulum_golf.gui.torque_history_widget import (
    TorqueHistoryWidget,
    _HAS_PYQTGRAPH,
)
from double_pendulum_golf.gui.torque_preview_widget import TorquePreviewWidget


@pytest.fixture
def mock_double_sim() -> Any:
    sim = MagicMock()
    sim.n_steps = 10
    sim.t = np.linspace(0, 1, 10)
    sim.torques_at.side_effect = lambda i: [0.1 * i, 0.2 * i]
    sim.friction_torques_at.side_effect = lambda i: [-0.01 * i, -0.02 * i]
    sim.total_torques_at.side_effect = lambda i: [0.09 * i, 0.18 * i]
    return sim


@pytest.fixture
def mock_golfer_sim() -> Any:
    sim = MagicMock()
    sim.n_steps = 10
    sim.t = np.linspace(0, 1, 10)
    sim.torques_at.side_effect = lambda i: np.ones(8) * 0.1 * i
    sim.friction_torques_at.side_effect = lambda i: np.ones(8) * -0.01 * i
    sim.total_torques_at.side_effect = lambda i: np.ones(8) * 0.09 * i
    return sim


@pytest.fixture(autouse=True)
def mock_pyqtgraph(monkeypatch) -> Any:
    import sys

    class DummyPlotWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._plot_item = MagicMock()

        def getPlotItem(self) -> Any:
            return self._plot_item

        def setBackground(self, *args) -> Any:
            pass

        def addItem(self, *args) -> Any:
            pass

        def plot(self, *args, **kwargs) -> Any:
            return MagicMock()

    class DummyInfiniteLine(MagicMock):
        pass

    pg_mock = MagicMock()
    pg_mock.mkPen.return_value = "pen"
    pg_mock.PlotWidget = DummyPlotWidget
    pg_mock.InfiniteLine = DummyInfiniteLine

    # We must patch sys.modules before importing the widget
    monkeypatch.setitem(sys.modules, "pyqtgraph", pg_mock)

    # Force _HAS_PYQTGRAPH to True in the module
    import double_pendulum_golf.gui.torque_history_widget as thw

    monkeypatch.setattr(thw, "_HAS_PYQTGRAPH", True)
    monkeypatch.setattr(thw, "pg", pg_mock, raising=False)


class TestTorqueHistoryWidget:
    def test_init_and_clear(self, qapp) -> Any:
        w = TorqueHistoryWidget()
        w.clear()

    def test_set_simulation_double(self, qapp, mock_double_sim) -> Any:
        w = TorqueHistoryWidget()
        w.set_simulation(mock_double_sim)
        assert w._n_joints == 2

        w.set_frame(5)
        w.clear()
        assert w._result is None

    def test_set_simulation_golfer(self, qapp, mock_golfer_sim) -> Any:
        w = TorqueHistoryWidget()
        w.set_simulation(mock_golfer_sim)
        assert w._n_joints == 8

        w.set_frame(5)

    def test_theme_changed(self, qapp) -> Any:
        w = TorqueHistoryWidget()
        mock_theme = MagicMock()
        mock_theme.axes_facecolor = "#111111"
        mock_theme.text_color = "#222222"
        mock_theme.grid_color = "#333333"
        w._on_plot_theme_changed(mock_theme)

        if _HAS_PYQTGRAPH:
            assert w._bg_color == "#111111"

    def test_no_pyqtgraph(self, qapp, monkeypatch, mock_double_sim) -> Any:
        import double_pendulum_golf.gui.torque_history_widget as thw

        monkeypatch.setattr(thw, "_HAS_PYQTGRAPH", False)

        # Test widget creation without pyqtgraph
        w = thw.TorqueHistoryWidget()

        # Test clear
        w.clear()

        # Test set_simulation
        w.set_simulation(mock_double_sim)
        assert w._result == mock_double_sim

        # Test set_frame
        w.set_frame(5)

        # Test theme changed returns early
        w._on_plot_theme_changed(object())

        # Test try_attach_theme exception natively if doing it again
        # We'll just leave this out since the method is not found directly.

    def test_theme_exceptions(self, qapp) -> Any:
        w = TorqueHistoryWidget()
        # Pass an object missing axes_facecolor to trigger AttributeError
        w._on_plot_theme_changed(object())


class TestTorquePreviewWidget:
    def test_basic_methods(self, qapp) -> Any:
        w = TorquePreviewWidget()

        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)  # Empty state

        w.set_duration(1.5)
        assert w._t_end == 1.5

        # Unclamped
        w.set_profiles(
            [
                ("Joint 1", [1.0, 0.0], QColor(255, 0, 0)),
                ("Joint 2", [], QColor(0, 255, 0)),  # Empty coeffs
            ]
        )

        w.paintEvent(pe)

        # Clamped
        w.set_profiles(
            [
                ("Joint 1", [100.0, 0.0], QColor(255, 0, 0)),
            ],
            clamp_limits=[50.0],
        )

        w.paintEvent(pe)
