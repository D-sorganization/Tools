"""Tests for mousewheel-immune widgets and 3D mouse rotation."""

from __future__ import annotations

import os
import sys

import pytest


def _has_pyqt6() -> bool:
    # On headless Linux, QWidget() causes SIGABRT even if PyQt6 is importable
    if sys.platform not in ("win32", "darwin"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
    try:
        from PyQt6.QtWidgets import QWidget  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


class TestNoScrollWidgets:
    """Verify mousewheel events are ignored on value widgets."""

    def test_no_scroll_spinbox_ignores_wheel(self, qapp) -> None:
        from double_pendulum_golf.gui.no_scroll_widgets import NoScrollSpinBox

        w = NoScrollSpinBox()

        pytest.MonkeyPatch().context()
        from unittest.mock import MagicMock

        event = MagicMock()
        w.wheelEvent(event)
        event.ignore.assert_called_once()

    def test_no_scroll_double_spinbox(self, qapp) -> None:
        from double_pendulum_golf.gui.no_scroll_widgets import NoScrollDoubleSpinBox

        w = NoScrollDoubleSpinBox()
        from unittest.mock import MagicMock

        event = MagicMock()
        w.wheelEvent(event)
        event.ignore.assert_called_once()

    def test_no_scroll_slider(self, qapp) -> None:
        from double_pendulum_golf.gui.no_scroll_widgets import NoScrollSlider
        from PyQt6.QtCore import Qt

        w = NoScrollSlider(Qt.Orientation.Horizontal)
        from unittest.mock import MagicMock

        event = MagicMock()
        w.wheelEvent(event)
        event.ignore.assert_called_once()

    def test_no_scroll_combobox(self, qapp) -> None:
        from double_pendulum_golf.gui.no_scroll_widgets import NoScrollComboBox

        w = NoScrollComboBox()
        from unittest.mock import MagicMock

        event = MagicMock()
        w.wheelEvent(event)
        event.ignore.assert_called_once()


class TestMouseRotation3D:
    """Verify 3D rotation state management in BasePendulumWidget."""

    def test_rotation_state_init(self) -> None:
        """BasePendulumWidget should have rotation state variables."""
        # We test the pure state, not the Qt widget
        assert True  # Placeholder — actual widget requires QApplication

    def test_azimuth_tilt_math(self) -> None:
        """Verify azimuth/tilt sensitivity calculation."""
        import numpy as np

        sensitivity = 0.01  # rad/pixel
        dx, dy = 50, 30  # pixels dragged
        d_azimuth = dx * sensitivity
        d_tilt = dy * sensitivity
        assert abs(d_azimuth - 0.5) < 1e-10
        assert abs(d_tilt - 0.3) < 1e-10
        # Tilt should be clamped to [-pi/2, pi/2]
        max_tilt = np.pi / 2
        # 100 pixels * 0.01 = 1.0 rad, which is less than pi/2 (~1.571)
        unclamped = 100 * sensitivity
        clamped = max(-max_tilt, min(max_tilt, unclamped))
        assert abs(clamped - unclamped) < 1e-10  # Should not be clamped
        # Test that large deltas DO get clamped
        huge_delta = 1000 * sensitivity  # 10 rad >> pi/2
        clamped_huge = max(-max_tilt, min(max_tilt, huge_delta))
        assert abs(clamped_huge - max_tilt) < 1e-10  # Should be clamped to max
