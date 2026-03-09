"""TDD tests for asteroid_jumper.asteroid_shape."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from asteroid_jumper.asteroid_shape import (
    ShapeKind,
    _angle_diff,
    make_circle,
    make_ellipse,
    make_random,
    surface_point_at_angle,
)


class TestMakeCircle:
    def test_returns_circle_kind(self) -> None:
        s = make_circle(5.0)
        assert s.kind == ShapeKind.CIRCLE

    def test_vertices_on_radius(self) -> None:
        r = 5.0
        s = make_circle(r)
        for x, y in s.vertices:
            assert math.hypot(x, y) == pytest.approx(r, rel=1e-9)

    def test_semi_axes_equal_radius(self) -> None:
        s = make_circle(7.0)
        assert s.semi_a == pytest.approx(7.0)
        assert s.semi_b == pytest.approx(7.0)

    def test_zero_radius_raises(self) -> None:
        with pytest.raises(AssertionError):
            make_circle(0.0)

    def test_vertex_count(self) -> None:
        s = make_circle(5.0, n_pts=16)
        assert len(s.vertices) == 16


class TestMakeEllipse:
    def test_returns_ellipse_kind(self) -> None:
        s = make_ellipse(10.0, 6.0)
        assert s.kind == ShapeKind.ELLIPSE

    def test_semi_axes_stored(self) -> None:
        s = make_ellipse(10.0, 6.0)
        assert s.semi_a == pytest.approx(10.0)
        assert s.semi_b == pytest.approx(6.0)

    def test_vertices_on_ellipse(self) -> None:
        a, b = 10.0, 6.0
        s = make_ellipse(a, b, n_pts=32)
        for x, y in s.vertices:
            # Point (x, y) on ellipse: (x/a)^2 + (y/b)^2 ≈ 1
            assert (x / a) ** 2 + (y / b) ** 2 == pytest.approx(1.0, rel=1e-9)


class TestMakeRandom:
    def test_returns_random_kind(self) -> None:
        s = make_random(10.0, seed=123)
        assert s.kind == ShapeKind.RANDOM

    def test_deterministic_with_seed(self) -> None:
        s1 = make_random(10.0, seed=42)
        s2 = make_random(10.0, seed=42)
        assert s1.vertices == s2.vertices

    def test_different_seeds_different_shapes(self) -> None:
        s1 = make_random(10.0, seed=1)
        s2 = make_random(10.0, seed=2)
        assert s1.vertices != s2.vertices

    def test_positive_semi_axes(self) -> None:
        s = make_random(10.0, seed=99)
        assert s.semi_a > 0
        assert s.semi_b > 0

    def test_roughness_zero_is_circular(self) -> None:
        s = make_random(10.0, roughness=0.0, seed=0)
        for x, y in s.vertices:
            assert math.hypot(x, y) == pytest.approx(10.0, rel=1e-9)

    def test_minimum_vertices(self) -> None:
        s = make_random(5.0, n_pts=6, seed=0)
        assert len(s.vertices) == 6


class TestSurfacePoint:
    def test_origin_direction_on_circle(self) -> None:
        s = make_circle(5.0)
        x, y = surface_point_at_angle(s, 0.0)  # right (+x)
        assert math.hypot(x, y) == pytest.approx(5.0, abs=0.1)

    def test_top_direction_on_circle(self) -> None:
        s = make_circle(5.0, n_pts=64)
        x, y = surface_point_at_angle(s, math.pi / 2)  # up (+y)
        assert y == pytest.approx(5.0, abs=0.15)


class TestAngleDiff:
    def test_zero_diff(self) -> None:
        assert _angle_diff(1.0, 1.0) == pytest.approx(0.0)

    def test_positive_diff(self) -> None:
        d = _angle_diff(0.5, 0.2)
        assert d == pytest.approx(0.3)

    def test_wraps_over_pi(self) -> None:
        d = _angle_diff(math.pi + 0.1, -math.pi + 0.1)
        assert -math.pi <= d <= math.pi
