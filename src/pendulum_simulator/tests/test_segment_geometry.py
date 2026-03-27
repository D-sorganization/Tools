from typing import Any

"""
Tests for segment_geometry module — 3D segment rendering geometry.

Covers:
- SegmentStyle enum
- Cylinder cross-section generation
- Ellipsoid cross-section generation
- Tapered cylinder generation
- Depth sorting
- Projection to 2D for QPainter rendering
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from double_pendulum_golf.segment_geometry import (
    SegmentStyle,
    cylinder_cross_section,
    depth_sort_segments,
    ellipsoid_cross_section,
    project_3d_to_2d,
    tapered_cylinder_cross_section,
)

# ---------------------------------------------------------------------------
# SegmentStyle enum
# ---------------------------------------------------------------------------


class TestSegmentStyle:
    def test_line_exists(self) -> Any:
        assert SegmentStyle.LINE.value == "line"

    def test_cylinder_exists(self) -> Any:
        assert SegmentStyle.CYLINDER.value == "cylinder"

    def test_ellipsoid_exists(self) -> Any:
        assert SegmentStyle.ELLIPSOID.value == "ellipsoid"

    def test_tapered_exists(self) -> Any:
        assert SegmentStyle.TAPERED.value == "tapered"


# ---------------------------------------------------------------------------
# Cylinder cross-section
# ---------------------------------------------------------------------------


class TestCylinderCrossSection:
    def test_returns_polygon(self) -> Any:
        pts = cylinder_cross_section(
            start=np.array([0.0, 0.0]),
            end=np.array([1.0, 0.0]),
            radius=0.05,
        )
        assert pts.shape[0] == 4  # rectangle: 4 corners
        assert pts.shape[1] == 2

    def test_width_matches_radius(self) -> Any:
        pts = cylinder_cross_section(
            start=np.array([0.0, 0.0]),
            end=np.array([1.0, 0.0]),
            radius=0.1,
        )
        # Width perpendicular to segment should be 2*radius
        widths = np.abs(pts[:, 1])
        assert np.max(widths) == pytest.approx(0.1, abs=1e-10)

    def test_length_matches_segment(self) -> Any:
        pts = cylinder_cross_section(
            start=np.array([0.0, 0.0]),
            end=np.array([2.0, 0.0]),
            radius=0.05,
        )
        # Length along x should span [0, 2]
        assert np.min(pts[:, 0]) == pytest.approx(0.0, abs=1e-10)
        assert np.max(pts[:, 0]) == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Ellipsoid cross-section
# ---------------------------------------------------------------------------


class TestEllipsoidCrossSection:
    def test_returns_polygon(self) -> Any:
        pts = ellipsoid_cross_section(
            centre=np.array([0.5, 0.0]),
            semi_a=0.5,
            semi_b=0.1,
            angle=0.0,
            n_points=32,
        )
        assert pts.shape == (32, 2)

    def test_extents_match_semi_axes(self) -> Any:
        pts = ellipsoid_cross_section(
            centre=np.array([0.0, 0.0]),
            semi_a=1.0,
            semi_b=0.5,
            angle=0.0,
            n_points=64,
        )
        assert np.max(np.abs(pts[:, 0])) == pytest.approx(1.0, abs=0.02)
        assert np.max(np.abs(pts[:, 1])) == pytest.approx(0.5, abs=0.02)

    def test_rotation(self) -> Any:
        """Rotated 90° should swap axes."""
        pts = ellipsoid_cross_section(
            centre=np.array([0.0, 0.0]),
            semi_a=1.0,
            semi_b=0.3,
            angle=np.pi / 2,
            n_points=64,
        )
        assert np.max(np.abs(pts[:, 0])) == pytest.approx(0.3, abs=0.02)
        assert np.max(np.abs(pts[:, 1])) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Tapered cylinder
# ---------------------------------------------------------------------------


class TestTaperedCylinderCrossSection:
    def test_returns_polygon(self) -> Any:
        pts = tapered_cylinder_cross_section(
            start=np.array([0.0, 0.0]),
            end=np.array([1.0, 0.0]),
            radius_start=0.1,
            radius_end=0.05,
        )
        assert pts.shape[0] == 4
        assert pts.shape[1] == 2

    def test_taper(self) -> Any:
        """Start end should be wider than the tip end."""
        pts = tapered_cylinder_cross_section(
            start=np.array([0.0, 0.0]),
            end=np.array([1.0, 0.0]),
            radius_start=0.2,
            radius_end=0.05,
        )
        # Points near x=0 should have larger |y| than points near x=1
        start_widths = np.abs(pts[pts[:, 0] < 0.5, 1])
        end_widths = np.abs(pts[pts[:, 0] > 0.5, 1])
        assert np.max(start_widths) > np.max(end_widths)


# ---------------------------------------------------------------------------
# Depth sorting
# ---------------------------------------------------------------------------


class TestDepthSortSegments:
    def test_sorts_by_depth(self) -> Any:
        segments = [
            {"id": "far", "depth": 10.0},
            {"id": "near", "depth": 2.0},
            {"id": "mid", "depth": 5.0},
        ]
        sorted_segs = depth_sort_segments(segments)
        assert sorted_segs[0]["id"] == "far"
        assert sorted_segs[1]["id"] == "mid"
        assert sorted_segs[2]["id"] == "near"

    def test_empty_list(self) -> Any:
        assert depth_sort_segments([]) == []


# ---------------------------------------------------------------------------
# 3D to 2D projection
# ---------------------------------------------------------------------------


class TestProject3dTo2d:
    def test_identity_at_zero_angles(self) -> Any:
        """With zero tilt and azimuth, x stays x, y stays y."""
        pt = project_3d_to_2d(np.array([1.0, 2.0, 0.0]), tilt=0.0, azimuth=0.0)
        assert pt[0] == pytest.approx(1.0, abs=1e-10)
        assert pt[1] == pytest.approx(2.0, abs=1e-10)

    def test_returns_2d(self) -> Any:
        pt = project_3d_to_2d(np.array([1.0, 2.0, 3.0]), tilt=0.3, azimuth=0.5)
        assert pt.shape == (2,)

    def test_depth_returned(self) -> Any:
        pt, depth = project_3d_to_2d(
            np.array([0.0, 0.0, 5.0]), tilt=0.0, azimuth=0.0, return_depth=True
        )
        assert pt.shape == (2,)
        assert isinstance(depth, float)
