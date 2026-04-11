from typing import Any

"""Tests for BasePendulumWidget."""


from unittest.mock import MagicMock
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainter, QMouseEvent, QWheelEvent
from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget


class DummyPendulumWidget(BasePendulumWidget):
    def _get_total_length(self) -> Any:
        return 1.0

    def _draw_model(self, painter) -> Any:
        pass

    def _draw_info(self, painter) -> Any:
        pass

    def _draw_placeholder(self, painter) -> Any:
        pass

    def _has_result(self) -> Any:
        return True


def test_feature_toggles_and_view(qapp) -> Any:
    w = DummyPendulumWidget()
    w.set_gravity_on(False)
    assert not w._gravity_on
    w.set_tilt_angle(0.5)
    assert w._tilt_angle == 0.5
    w.set_view_azimuth(0.3)
    assert w._view_azimuth == 0.3
    w.set_show_torque_vectors(True)
    assert w._show_torque_vectors
    w.set_show_moment_of_force(True)
    assert w._show_moment_of_force
    w.set_show_sum_moments(True)
    assert w._show_sum_moments
    w.set_3d_mode(True)
    assert w._3d_mode
    w.set_show_forces(True)
    assert w._show_forces
    w.set_show_zero_torque_forces(True)
    assert w._show_zero_torque_forces
    w.set_force_scale(2.0)
    assert w._force_scale == 2.0
    w.set_show_mob_ellipsoids(True)
    assert w._show_mob_ellipsoids
    w.set_show_force_ellipsoids(True)
    assert w._show_force_ellipsoids
    w.set_mob_ellipsoid_scale(2.0)
    assert w._mob_ellipsoid_scale == 2.0
    w.set_force_ellipsoid_scale(2.0)
    assert w._force_ellipsoid_scale == 2.0
    w.set_show_com(True)
    assert w._show_com
    w.set_visible_segments({"seg1"})
    assert "seg1" in w._visible_segments

    # reset view
    w._zoom = 2.0
    w._pan_x = 5.0
    w.reset_view()
    assert w._zoom == 1.0
    assert w._pan_x == 0.0


def test_compute_base_scale(qapp) -> Any:
    w = DummyPendulumWidget()
    w.resize(800, 600)
    w._get_total_length = lambda: 1.0
    scale = w._compute_base_scale()
    assert scale >= 30.0


def test_mouse_events(qapp) -> Any:
    w = DummyPendulumWidget()

    # Non-events early exit
    w.wheelEvent(object())
    w.mousePressEvent(object())
    w.mouseMoveEvent(object())
    w.mouseReleaseEvent(object())

    # Wheel event
    we = MagicMock(spec=QWheelEvent)
    we.angleDelta().y.return_value = 120
    we.position().x.return_value = 100
    we.position().y.return_value = 100
    w.wheelEvent(we)
    assert w._zoom > 1.0

    we.angleDelta().y.return_value = -120
    w.wheelEvent(we)

    # Mouse press event (pan)
    btn_l = Qt.MouseButton.LeftButton
    btn_r = Qt.MouseButton.RightButton

    mp_left = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10, 10),
        btn_l,
        btn_l,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(mp_left)
    assert w._drag_start is not None

    # Mouse press event (zoom bypass)
    w._drag_start = None
    w._handle_zoom_button_click = lambda pos: True
    w.mousePressEvent(mp_left)
    assert w._drag_start is None
    del w._handle_zoom_button_click
    w.mousePressEvent(mp_left)  # set it again!

    # Mouse move event (pan)
    mm_left = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(20, 20),
        Qt.MouseButton.NoButton,
        btn_l,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(mm_left)
    assert w._pan_x != 0.0

    # Mouse release event (pan)
    mr_left = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(20, 20),
        btn_l,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseReleaseEvent(mr_left)
    assert w._drag_start is None

    # Mouse press event (orbit)
    mp_right = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10, 10),
        btn_r,
        btn_r,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(mp_right)
    assert w._rotate_start is not None

    # Mouse move event (orbit)
    mm_right = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(30, 40),
        Qt.MouseButton.NoButton,
        btn_r,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(mm_right)
    assert w._view_azimuth != 0.0

    # Mouse release event (orbit)
    mr_right = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(30, 40),
        btn_r,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseReleaseEvent(mr_right)
    assert w._rotate_start is None

    # Double click
    mdc = MagicMock()
    w.mouseDoubleClickEvent(mdc)
    assert w._pan_x == 0.0


def test_drawing_methods(qapp) -> Any:
    w = DummyPendulumWidget()
    painter = MagicMock(spec=QPainter)

    # Set minimum physical size so coordinate functions don't crash
    w.resize(400, 300)

    # Draw Grid
    w._draw_grid(painter)

    # Draw Ground
    w._draw_ground_line(painter, -1.0)
    w._draw_ground_plane(painter, -1.0)

    # Draw Tilt plane
    w._tilt_angle = 0.5
    w._draw_tilt_plane(painter)
    w._tilt_angle = 0.0
    w._draw_tilt_plane(painter)  # branch 0.0

    # Draw Ball
    w._draw_ball(painter, 1.0, 1.0)

    # Draw Joint
    w._draw_joint(painter, QPointF(0, 0), 10.0, Qt.GlobalColor.red)

    # Draw Gravity Badge
    w._draw_no_gravity_badge(painter)

    # Draw 3D Segment
    w._draw_3d_segment(painter, QPointF(0, 0), QPointF(10, 10), 2.0, 1.0, w.COLOR_TRAIL)
    w._draw_3d_segment(
        painter, QPointF(0, 0), QPointF(0, 0), 2.0, 1.0, w.COLOR_TRAIL
    )  # short branch


def test_draw_trail(qapp) -> Any:
    w = DummyPendulumWidget()
    painter = MagicMock(spec=QPainter)

    w._draw_trail(painter)  # < 2

    w._trail.append((0.0, 0.0))
    w._trail.append((0.1, 0.1))
    w._draw_trail(painter)  # < 4

    w._trail.clear()
    w._trail.extend([(float(i), float(i)) for i in range(10)])
    w._draw_trail(painter)  # >= 4


def test_catmull_rom_smooth() -> Any:
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    smoothed = BasePendulumWidget._catmull_rom_smooth(pts, n_sub=2)
    assert len(smoothed) > len(pts)


def test_3d_joint_cap(qapp):
    """Test the 3D joint cap rendering method."""
    w = DummyPendulumWidget()
    painter = MagicMock(spec=QPainter)
    w.resize(400, 300)

    from PyQt6.QtGui import QColor

    # Normal operation
    w._draw_3d_joint_cap(painter, QPointF(100, 100), 8.0, QColor(255, 0, 0))
    assert painter.drawEllipse.called

    # DbC: radius must be positive
    import pytest

    with pytest.raises(ValueError, match="positive"):
        w._draw_3d_joint_cap(painter, QPointF(100, 100), 0, QColor(255, 0, 0))
    with pytest.raises(ValueError, match="positive"):
        w._draw_3d_joint_cap(painter, QPointF(100, 100), -5, QColor(255, 0, 0))


def test_shadow_projection(qapp):
    """Test shadow projection onto ground plane."""
    w = DummyPendulumWidget()
    painter = MagicMock(spec=QPainter)
    w.resize(400, 300)

    # Too few points (no-op)
    w._draw_shadow_projection(painter, [(0.0, 0.0)], -1.0)
    painter.drawLine.assert_not_called()

    # Normal shadow projection
    points = [(0.0, 0.0), (0.5, -0.5), (1.0, -1.0)]
    w._draw_shadow_projection(painter, points, -2.0)
    assert painter.drawLine.called


def test_export_image_dbc(qapp):
    """Test DbC on export_image dimensions."""
    w = DummyPendulumWidget()
    w.resize(400, 300)

    import pytest

    with pytest.raises(ValueError, match="positive"):
        w.export_image("test.png", width=0, height=100)
    with pytest.raises(ValueError, match="positive"):
        w.export_image("test.png", width=100, height=-1)


def test_export_image_png(qapp, tmp_path):
    """Test PNG image export creates a file."""
    w = DummyPendulumWidget()
    w.resize(400, 300)

    out_path = str(tmp_path / "test_export.png")
    w.export_image(out_path, width=800, height=600)

    import os

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


# ----------------------------------------------------------------------
# Bulletproof view-fitting regression tests
# ----------------------------------------------------------------------

def test_auto_fit_default_state(qapp):
    """Fresh widget with no trajectory still produces a valid view."""
    w = DummyPendulumWidget()
    w.resize(800, 600)

    assert w._trajectory_bbox is None
    assert w.is_view_auto_fit() is True

    # auto_fit_view on an empty widget must not crash and must center
    w.auto_fit_view()
    assert w._view_center_world == (0.0, 0.0)
    assert w._zoom == 1.0
    assert w._pan_x == 0.0
    assert w._pan_y == 0.0


def test_compute_and_store_trajectory_bbox(qapp):
    """Bbox computation centers the view on the trajectory midpoint."""
    w = DummyPendulumWidget()
    w.resize(800, 600)

    samples = [
        {"a": (1.0, 2.0), "b": (3.0, 4.0)},
        {"a": (1.5, 2.5), "b": (4.0, 5.0)},
        {"a": (2.0, 3.0), "b": (5.0, 6.0)},
    ]
    w.compute_and_store_trajectory_bbox(samples)

    bbox = w._trajectory_bbox
    assert bbox is not None
    xmin, xmax, ymin, ymax = bbox
    # Origin (0,0) is always included as the standoff anchor
    assert xmin == 0.0
    assert xmax == 5.0
    assert ymin == 0.0
    assert ymax == 6.0
    # Center is the bbox midpoint
    assert w._view_center_world == (2.5, 3.0)
    # Auto-fit lock is engaged
    assert w.is_view_auto_fit() is True


def test_auto_fit_handles_degenerate_bbox(qapp):
    """Single-point trajectory is padded so divide-by-zero never happens."""
    w = DummyPendulumWidget()
    w.resize(800, 600)

    w.compute_and_store_trajectory_bbox([{"a": (1.0, 1.0)}])
    bbox = w._trajectory_bbox
    assert bbox is not None
    xmin, xmax, ymin, ymax = bbox
    # Origin is always included → xmin/ymin = 0
    assert xmin == 0.0
    assert ymin == 0.0
    # The bbox (0..1) for both dimensions is non-degenerate so no padding kicks in
    assert xmax >= 1.0
    assert ymax >= 1.0


def test_auto_fit_filters_non_finite_coords(qapp):
    """NaN/inf joint positions are skipped, not propagated into the bbox."""
    import math

    w = DummyPendulumWidget()
    w.resize(800, 600)

    samples = [
        {"a": (1.0, 2.0), "bad": (math.nan, 0.0)},
        {"a": (3.0, 4.0), "worse": (0.0, math.inf)},
    ]
    w.compute_and_store_trajectory_bbox(samples)
    bbox = w._trajectory_bbox
    assert bbox is not None
    xmin, xmax, ymin, ymax = bbox
    # All four should be finite
    for v in (xmin, xmax, ymin, ymax):
        assert math.isfinite(v)


def test_user_pan_releases_auto_fit_lock(qapp):
    """Manually panning drops the auto-fit flag so the user's intent wins."""
    w = DummyPendulumWidget()
    w.resize(800, 600)
    w.compute_and_store_trajectory_bbox([{"a": (1.0, 1.0), "b": (2.0, 2.0)}])
    assert w.is_view_auto_fit() is True

    # Simulate a wheel event
    from PyQt6.QtCore import QPointF as _QPF, QPoint as _QP
    from PyQt6.QtGui import QWheelEvent as _QWE

    wheel = _QWE(
        _QPF(100, 100),
        _QPF(100, 100),
        _QP(0, 120),
        _QP(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    w.wheelEvent(wheel)
    assert w.is_view_auto_fit() is False


def test_reset_view_restores_auto_fit(qapp):
    """reset_view() always brings back the canonical in-view state."""
    w = DummyPendulumWidget()
    w.resize(800, 600)
    w.compute_and_store_trajectory_bbox([{"a": (1.0, 1.0), "b": (2.0, 2.0)}])

    # Force the view into a wildly off state
    w._zoom = 0.1
    w._pan_x = 9999.0
    w._pan_y = -9999.0
    w._tilt_angle = 1.0
    w._view_azimuth = 1.0
    w._auto_fit_locked = False

    w.reset_view()

    assert w.is_view_auto_fit() is True
    assert w._zoom == 1.0
    assert w._pan_x == 0.0
    assert w._pan_y == 0.0
    assert w._tilt_angle == 0.0
    assert w._view_azimuth == 0.0


def test_world_points_in_view_detects_offscreen(qapp):
    """The off-screen detector flips True/False at the widget edge."""
    w = DummyPendulumWidget()
    w.resize(800, 600)
    w.compute_and_store_trajectory_bbox([{"a": (0.0, 0.0)}])

    in_view, _ = w._world_points_in_view([(0.0, 0.0)])
    assert in_view is True

    # Pan the world far away from the widget
    w._auto_fit_locked = False
    w._pan_x = 5000.0
    w._pan_y = 5000.0
    in_view2, _ = w._world_points_in_view([(0.0, 0.0)])
    assert in_view2 is False


def test_compute_base_scale_uses_bbox_when_available(qapp):
    """Base scale picks up the bbox so the trajectory fits with margin."""
    w = DummyPendulumWidget()
    w.resize(800, 600)
    # Without bbox: legacy fallback path
    w._trajectory_bbox = None
    legacy = w._compute_base_scale()
    assert legacy >= 30.0

    # With a small bbox: should produce a larger scale (more pixels per meter)
    w.compute_and_store_trajectory_bbox(
        [{"a": (-0.1, -0.1), "b": (0.1, 0.1)}]
    )
    bbox_scale = w._compute_base_scale()
    assert bbox_scale > legacy
