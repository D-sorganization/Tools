"""Tests for matrix display widgets."""

import pytest
from PyQt6.QtGui import QPainter, QPaintEvent, QRegion
from unittest.mock import MagicMock

from double_pendulum_golf.gui.matrix_widget import MatrixWidget
from double_pendulum_golf.gui.matrix_widget_triple import TripleMatrixWidget
from double_pendulum_golf.gui.matrix_widget_golfer import GolferMatrixWidget


@pytest.fixture
def mock_double_sim():
    sim = MagicMock()
    sim.n_steps = 100
    sim.mass_matrix_at.return_value = {"M11": 1.0, "M12": 0.5, "M21": 0.5, "M22": 2.0}
    sim.torques_at.return_value = [0.0, 0.0]
    sim.coriolis_at.return_value = [0.0, 0.0]
    sim.gravity_at.return_value = [0.0, 0.0]
    sim.energy_at.return_value = {"total": 0.0, "kinetic": 0.0, "potential": 0.0}
    return sim


@pytest.fixture
def mock_triple_sim():
    sim = MagicMock()
    sim.n_steps = 100
    sim.mass_matrix_at.return_value = {
        "M11": 1.0,
        "M12": 0.1,
        "M13": 0.2,
        "M21": 0.1,
        "M22": 2.0,
        "M23": 0.3,
        "M31": 0.2,
        "M32": 0.3,
        "M33": 3.0,
    }
    sim.torques_at.return_value = [0.0, 0.0, 0.0]
    sim.coriolis_at.return_value = [0.0, 0.0, 0.0]
    sim.gravity_at.return_value = [0.0, 0.0, 0.0]
    sim.energy_at.return_value = {"total": 0.0, "kinetic": 0.0, "potential": 0.0}
    return sim


@pytest.fixture
def mock_golfer_sim():
    import numpy as np

    sim = MagicMock()  # no spec to avoid SimulationResult constraints if any
    sim.n_steps = 100
    sim.mass_matrix_at.return_value = np.zeros((8, 8))
    sim.mass_matrix_at.return_value[0, 0] = 1.0
    sim.torques_at.return_value = [0.0] * 8
    sim.coriolis_at.return_value = [0.0] * 8
    sim.gravity_at.return_value = [0.0] * 8
    sim.qdot_at.return_value = [0.0] * 8
    sim.accelerations_at.return_value = [0.0] * 8
    sim.constraint_forces_at.return_value = [0.0] * 8
    sim.constraint_violation_at.return_value = 0.0
    sim.energy_at.return_value = {"total": 0.0, "kinetic": 0.0, "potential": 0.0}
    return sim


class TestMatrixWidgets:
    def test_double_matrix_methods(self, qapp, mock_double_sim):
        w = MatrixWidget()

        # Test no simulation state
        w.set_frame(50)  # ignores gracefully
        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)  # paint "No simulation loaded" message

        w.set_simulation(mock_double_sim)
        w.set_frame(50)

        assert w.get_matrix_size() == (2, 2)
        assert len(w.get_column_labels()) == 2
        entries = w.get_matrix_entries(mock_double_sim.mass_matrix_at(0))
        assert len(entries) == 4

        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)

        w.clear()
        assert w._result is None
        w.paintEvent(pe)

    def test_triple_matrix_methods(self, qapp, mock_triple_sim):
        w = TripleMatrixWidget()
        w.set_simulation(mock_triple_sim)
        w.set_frame(50)

        assert w.get_matrix_size() == (3, 3)
        assert len(w.get_column_labels()) == 3
        entries = w.get_matrix_entries(mock_triple_sim.mass_matrix_at(0))
        assert len(entries) == 9

        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)

        w.clear()

    def test_golfer_matrix_methods(self, qapp, mock_golfer_sim):
        w = GolferMatrixWidget()

        # specific to golfer to cover lines 65-73 and 182
        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)

        assert w._draw_coupling_ratio(QPainter(), {}, 100) == 100

        w.set_simulation(mock_golfer_sim)
        w.set_frame(50)

        assert w.get_matrix_size() == (8, 8)
        assert len(w.get_column_labels()) == 8
        entries = w.get_matrix_entries(mock_golfer_sim.mass_matrix_at(0))
        assert len(entries) == 64

        pe = QPaintEvent(QRegion(w.rect()))
        w.paintEvent(pe)
