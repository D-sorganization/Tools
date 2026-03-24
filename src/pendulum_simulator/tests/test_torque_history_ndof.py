"""Tests for N-DOF TorqueHistoryWidget.

TDD: These tests verify the TorqueHistoryWidget dynamically supports
2, 3, and 7 joints (double, triple, and golfer models).
Tests will fail if N-DOF support is not implemented.

Pure-logic tests (labels, colors) import from the Qt-free constants module,
so they run in any environment including headless CI.
"""

from __future__ import annotations


import pytest

# Import from the Qt-free constants module — no display server needed
from double_pendulum_golf.gui.torque_history_constants import (
    _DRIVE_COLORS,
    _FRICTION_COLORS,
    _TOTAL_COLORS,
    _joint_labels_for_ndof,
)


class TestJointLabelsForNdof:
    """Test the joint label lookup function."""

    def test_double_labels(self):
        labels = _joint_labels_for_ndof(2)
        assert labels == ["Shoulder", "Wrist"]

    def test_triple_labels(self):
        labels = _joint_labels_for_ndof(3)
        assert labels == ["Shoulder", "Elbow", "Wrist"]

    def test_golfer_labels(self):
        labels = _joint_labels_for_ndof(7)
        assert len(labels) == 7
        assert labels[0] == "Hub"
        assert "R Shoulder" in labels
        assert "L Wrist" in labels

    def test_arbitrary_ndof_fallback(self):
        labels = _joint_labels_for_ndof(5)
        assert len(labels) == 5
        assert labels[0] == "Joint 1"
        assert labels[4] == "Joint 5"

    def test_returns_new_list_each_call(self):
        """Ensure we get a fresh copy, not the same mutable list."""
        a = _joint_labels_for_ndof(2)
        b = _joint_labels_for_ndof(2)
        assert a == b
        assert a is not b  # distinct objects

    def test_zero_joints_raises(self):
        """n_joints must be positive."""
        with pytest.raises(AssertionError):
            _joint_labels_for_ndof(0)


class TestTorqueHistoryWidgetContract:
    """Contract tests for the TorqueHistoryWidget public interface.

    These tests require a display server and PyQt6, so they are skipped
    in headless environments.
    """

    def test_clear_without_simulation_does_not_crash(self):
        """clear() on a fresh widget should not raise."""
        pytest.importorskip("pyqtgraph")
        try:
            from PyQt6.QtWidgets import QApplication

            _app = QApplication.instance() or QApplication([])
        except Exception:  # noqa: BLE001
            pytest.skip("Qt not available")

        from double_pendulum_golf.gui.torque_history_widget import (
            TorqueHistoryWidget,
        )

        widget = TorqueHistoryWidget()
        widget.clear()  # should not raise

    def test_set_frame_without_simulation_noop(self):
        """set_frame() before set_simulation should be a no-op."""
        pytest.importorskip("pyqtgraph")
        try:
            from PyQt6.QtWidgets import QApplication

            _app = QApplication.instance() or QApplication([])
        except Exception:  # noqa: BLE001
            pytest.skip("Qt not available")

        from double_pendulum_golf.gui.torque_history_widget import (
            TorqueHistoryWidget,
        )

        widget = TorqueHistoryWidget()
        widget.set_frame(0)  # should not raise


class TestColorPaletteCoverage:
    """Verify color palettes cover at least 7 joints."""

    def test_drive_colors_length(self):
        assert len(_DRIVE_COLORS) >= 7

    def test_friction_colors_length(self):
        assert len(_FRICTION_COLORS) >= 7

    def test_total_colors_length(self):
        assert len(_TOTAL_COLORS) >= 7

    def test_colors_are_rgb_tuples(self):
        """Each color should be a 3-tuple of ints in [0, 255]."""
        for palette in [_DRIVE_COLORS, _FRICTION_COLORS, _TOTAL_COLORS]:
            for color in palette:
                assert len(color) == 3
                assert all(0 <= c <= 255 for c in color)
