"""Tests for mousewheel-immune widgets and 3D mouse rotation."""

from __future__ import annotations

import os
import sys

import pytest


def _has_pyqt6() -> bool:
    # On headless Linux, QWidget() causes SIGABRT without a platform backend.
    # QT_QPA_PLATFORM=offscreen is safe; DISPLAY/WAYLAND_DISPLAY mean a real server.
    if sys.platform not in ("win32", "darwin"):
        has_platform = (
            os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or bool(os.environ.get("DISPLAY"))
            or bool(os.environ.get("WAYLAND_DISPLAY"))
        )
        if not has_platform:
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

    def test_right_drag_updates_rotation_state(self, qapp) -> None:
        """Right-dragging the canvas should rotate and clamp the 3D view."""
        import numpy as np
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

        class TestPendulumWidget(BasePendulumWidget):
            def _get_total_length(self) -> float:
                return 1.0

            def _draw_model(self, painter) -> None:
                return None

            def _draw_info(self, painter) -> None:
                return None

            def _draw_placeholder(self, painter) -> None:
                return None

            def _has_result(self) -> bool:
                return True

        widget = TestPendulumWidget()
        right_button = Qt.MouseButton.RightButton
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            right_button,
            right_button,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(60, 40),
            Qt.MouseButton.NoButton,
            right_button,
            Qt.KeyboardModifier.NoModifier,
        )
        huge_tilt_move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(60, 1000),
            Qt.MouseButton.NoButton,
            right_button,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(60, 40),
            right_button,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

        widget.mousePressEvent(press)
        widget.mouseMoveEvent(move)

        assert widget._rotate_start is not None
        assert widget._view_azimuth == pytest.approx(0.5)
        assert widget._tilt_angle == pytest.approx(0.3)
        assert widget.is_view_auto_fit() is False

        widget.mouseMoveEvent(huge_tilt_move)
        assert widget._tilt_angle == pytest.approx(float(np.pi / 2))

        widget.mouseReleaseEvent(release)
        assert widget._rotate_start is None

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
