"""Tests for BasePendulumWidget."""

import pytest
from unittest.mock import MagicMock
from PyQt6.QtCore import QPointF, QPoint, Qt
from PyQt6.QtGui import QPainter, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QWidget
from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

class DummyPendulumWidget(BasePendulumWidget):
    def _get_total_length(self): return 1.0
    def _draw_model(self, painter): pass
    def _draw_info(self, painter): pass
    def _draw_placeholder(self, painter): pass
    def _has_result(self): return True

def test_feature_toggles_and_view(qapp):
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

def test_compute_base_scale(qapp):
    w = DummyPendulumWidget()
    w.resize(800, 600)
    w._get_total_length = lambda: 1.0
    scale = w._compute_base_scale()
    assert scale >= 30.0

def test_mouse_events(qapp):
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
    
    mp_left = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(10, 10), btn_l, btn_l, Qt.KeyboardModifier.NoModifier)
    w.mousePressEvent(mp_left)
    assert w._drag_start is not None
    
    # Mouse press event (zoom bypass)
    w._drag_start = None
    w._handle_zoom_button_click = lambda pos: True
    w.mousePressEvent(mp_left)
    assert w._drag_start is None
    del w._handle_zoom_button_click
    w.mousePressEvent(mp_left) # set it again!
    
    # Mouse move event (pan)
    mm_left = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(20, 20), Qt.MouseButton.NoButton, btn_l, Qt.KeyboardModifier.NoModifier)
    w.mouseMoveEvent(mm_left)
    assert w._pan_x != 0.0
    
    # Mouse release event (pan)
    mr_left = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(20, 20), btn_l, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    w.mouseReleaseEvent(mr_left)
    assert w._drag_start is None
    
    # Mouse press event (orbit)
    mp_right = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(10, 10), btn_r, btn_r, Qt.KeyboardModifier.NoModifier)
    w.mousePressEvent(mp_right)
    assert w._rotate_start is not None
    
    # Mouse move event (orbit)
    mm_right = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(30, 40), Qt.MouseButton.NoButton, btn_r, Qt.KeyboardModifier.NoModifier)
    w.mouseMoveEvent(mm_right)
    assert w._view_azimuth != 0.0
    
    # Mouse release event (orbit)
    mr_right = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(30, 40), btn_r, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    w.mouseReleaseEvent(mr_right)
    assert w._rotate_start is None
    
    # Double click
    mdc = MagicMock()
    w.mouseDoubleClickEvent(mdc)
    assert w._pan_x == 0.0

def test_drawing_methods(qapp):
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
    w._draw_tilt_plane(painter) # branch 0.0
    
    # Draw Ball
    w._draw_ball(painter, 1.0, 1.0)
    
    # Draw Joint
    w._draw_joint(painter, QPointF(0, 0), 10.0, Qt.GlobalColor.red)
    
    # Draw Gravity Badge
    w._draw_no_gravity_badge(painter)
    
    # Draw 3D Segment
    w._draw_3d_segment(painter, QPointF(0, 0), QPointF(10, 10), 2.0, 1.0, w.COLOR_TRAIL)
    w._draw_3d_segment(painter, QPointF(0, 0), QPointF(0, 0), 2.0, 1.0, w.COLOR_TRAIL) # short branch

def test_draw_trail(qapp):
    w = DummyPendulumWidget()
    painter = MagicMock(spec=QPainter)
    
    w._draw_trail(painter) # < 2
    
    w._trail.append((0.0, 0.0))
    w._trail.append((0.1, 0.1))
    w._draw_trail(painter) # < 4
    
    w._trail.clear()
    for i in range(10):
        w._trail.append((float(i), float(i)))
    w._draw_trail(painter) # >= 4

def test_catmull_rom_smooth():
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    smoothed = BasePendulumWidget._catmull_rom_smooth(pts, n_sub=2)
    assert len(smoothed) > len(pts)
