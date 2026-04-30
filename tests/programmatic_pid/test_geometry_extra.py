"""Edge-case tests for ``programmatic_pid.geometry``.

Complements :mod:`test_geometry.py` with:

* ``to_float`` with NaN/Inf inputs (returns the value as-is — finite check
  is the *caller's* responsibility, but float coercion must succeed).
* ``clamp`` with degenerate ranges and exact endpoints.
* ``closest_point_on_rect`` for points inside, on edges, and at corners.
* ``rects_overlap`` with negative/zero pad and exactly-touching rects.
* ``text_box`` with every alignment combination and weird inputs.
* ``distance`` with negative coordinates and zero distance.
* ``dedupe_points`` tolerance behaviour at the 1e-9 boundary.
* ``find_free_region`` with empty occupied list and crowded grid.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("ezdxf")
from programmatic_pid.geometry import (
    clamp,
    closest_point_on_rect,
    dedupe_points,
    distance,
    find_free_region,
    rects_overlap,
    text_box,
    to_float,
)
from programmatic_pid.types import BBox, Point


# ── to_float ─────────────────────────────────────────────────────────────


class TestToFloat:
    def test_int_returns_float(self) -> None:
        assert to_float(7) == 7.0
        assert isinstance(to_float(7), float)

    def test_bool_is_treated_as_int(self) -> None:
        # bool is an int subclass in Python.
        assert to_float(True) == 1.0
        assert to_float(False) == 0.0

    def test_object_returns_default(self) -> None:
        assert to_float(object(), default=42.0) == 42.0

    def test_list_returns_default(self) -> None:
        assert to_float([1, 2, 3], default=-1.0) == -1.0

    def test_default_is_float_coerced(self) -> None:
        # Default is itself float()-ed so int is fine.
        assert to_float("nonsense", default=5) == 5.0

    def test_inf_passes_through(self) -> None:
        # math.inf is a valid float.
        assert math.isinf(to_float(math.inf))


# ── clamp ────────────────────────────────────────────────────────────────


class TestClamp:
    @pytest.mark.parametrize(
        ("value", "lo", "hi", "expected"),
        [
            (5.0, 0.0, 10.0, 5.0),
            (0.0, 0.0, 10.0, 0.0),  # exact lo
            (10.0, 0.0, 10.0, 10.0),  # exact hi
            (-5.0, 0.0, 10.0, 0.0),
            (15.0, 0.0, 10.0, 10.0),
            (5.0, 5.0, 5.0, 5.0),  # degenerate lo == hi
            (-1.0, -10.0, -5.0, -5.0),  # all negative
        ],
    )
    def test_clamp_values(
        self, value: float, lo: float, hi: float, expected: float
    ) -> None:
        assert clamp(value, lo, hi) == expected


# ── closest_point_on_rect ────────────────────────────────────────────────


class TestClosestPointOnRect:
    def test_point_inside_rect_returns_point(self) -> None:
        # Point inside is its own closest point (clamp is a no-op).
        p = closest_point_on_rect((5.0, 5.0), (0.0, 0.0, 10.0, 10.0))
        assert p == Point(5.0, 5.0)

    def test_point_left_of_rect(self) -> None:
        p = closest_point_on_rect((-5.0, 5.0), (0.0, 0.0, 10.0, 10.0))
        assert p == Point(0.0, 5.0)

    def test_point_at_corner(self) -> None:
        p = closest_point_on_rect((100.0, 100.0), (0.0, 0.0, 10.0, 10.0))
        assert p == Point(10.0, 10.0)

    def test_point_below_rect(self) -> None:
        p = closest_point_on_rect((5.0, -50.0), (0.0, 0.0, 10.0, 10.0))
        assert p == Point(5.0, 0.0)

    def test_none_point_raises(self) -> None:
        with pytest.raises(ValueError, match="point"):
            closest_point_on_rect(None, (0, 0, 10, 10))  # type: ignore[arg-type]

    def test_unparseable_coords_use_fallback(self) -> None:
        # to_float is called on x/y -> non-coercible value falls back to 0.
        p = closest_point_on_rect(("bad", "also bad"), (-5.0, -5.0, 5.0, 5.0))
        assert p == Point(0.0, 0.0)


# ── rects_overlap ────────────────────────────────────────────────────────


class TestRectsOverlap:
    def test_touching_edges_treated_as_no_overlap(self) -> None:
        # When ax2 == bx1 with pad=0, the function returns False per its
        # `<=` boundary check — touching is *not* overlap.
        assert not rects_overlap((0, 0, 10, 10), (10, 0, 20, 10))

    def test_touching_with_positive_pad_becomes_overlap(self) -> None:
        # Adding pad pushes the boundary into the neighbour.
        assert rects_overlap((0, 0, 10, 10), (10, 0, 20, 10), pad=1.0)

    def test_negative_pad_separates_overlapping_rects(self) -> None:
        # Negative pad shrinks rects: heavy overlap stays overlap, narrow
        # overlap should disappear.
        assert not rects_overlap((0, 0, 10, 10), (9, 0, 19, 10), pad=-2.0)

    def test_one_inside_the_other(self) -> None:
        assert rects_overlap((0, 0, 100, 100), (10, 10, 20, 20))

    def test_none_first_rect_raises(self) -> None:
        with pytest.raises(ValueError, match="a"):
            rects_overlap(None, (0, 0, 10, 10))  # type: ignore[arg-type]


# ── text_box ─────────────────────────────────────────────────────────────


class TestTextBox:
    @pytest.mark.parametrize(
        "align",
        [
            "MIDDLE_CENTER",
            "MIDDLE_LEFT",
            "MIDDLE_RIGHT",
            "TOP_CENTER",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_CENTER",
            "BOTTOM_LEFT",
            "BOTTOM_RIGHT",
        ],
    )
    def test_alignment_invariants(self, align: str) -> None:
        x1, y1, x2, y2 = text_box("hello", 5.0, 5.0, 1.0, align=align)
        assert x1 < x2 and y1 < y2  # always non-empty
        # Width approximately len(text)*h*0.55, height h*1.2.
        assert (x2 - x1) == pytest.approx(5 * 0.55, rel=1e-6)
        assert (y2 - y1) == pytest.approx(1.2, rel=1e-6)

    def test_top_align_has_y_above_anchor(self) -> None:
        _, y1, _, y2 = text_box("x", 0.0, 0.0, 1.0, align="TOP_LEFT")
        assert y2 == 0.0
        assert y1 == -1.2

    def test_bottom_align_has_y_below_anchor(self) -> None:
        _, y1, _, y2 = text_box("x", 0.0, 0.0, 1.0, align="BOTTOM_LEFT")
        assert y1 == 0.0
        assert y2 == 1.2

    def test_right_align_has_x_to_left_of_anchor(self) -> None:
        x1, _, x2, _ = text_box("hi", 10.0, 0.0, 1.0, align="MIDDLE_RIGHT")
        assert x2 == 10.0
        assert x1 < 10.0

    def test_empty_string_treated_as_unit_width(self) -> None:
        # max(len(text), 1) ensures a non-zero width.
        x1, _, x2, _ = text_box("", 0.0, 0.0, 1.0, align="MIDDLE_LEFT")
        assert (x2 - x1) > 0

    def test_zero_height_clamped_to_minimum(self) -> None:
        # h is max(to_float(h, 1.0), 0.1)
        x1, y1, x2, y2 = text_box("a", 0.0, 0.0, 0.0)
        assert (y2 - y1) == pytest.approx(0.1 * 1.2, rel=1e-6)

    def test_none_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            text_box(None, 0.0, 0.0, 1.0)  # type: ignore[arg-type]

    def test_none_align_uses_default(self) -> None:
        # `str(align or "MIDDLE_CENTER")` handles None.
        x1, _, x2, _ = text_box("hi", 0.0, 0.0, 1.0, align=None)  # type: ignore[arg-type]
        assert x1 < 0.0 < x2


# ── distance ─────────────────────────────────────────────────────────────


class TestDistance:
    def test_zero_distance(self) -> None:
        assert distance((1.5, 2.5), (1.5, 2.5)) == 0.0

    def test_negative_coordinates(self) -> None:
        assert distance((-3, -4), (0, 0)) == pytest.approx(5.0)

    def test_symmetric(self) -> None:
        assert distance((1, 2), (4, 6)) == distance((4, 6), (1, 2))


# ── dedupe_points ────────────────────────────────────────────────────────


class TestDedupePoints:
    def test_empty_input(self) -> None:
        assert dedupe_points([]) == []

    def test_single_point(self) -> None:
        assert dedupe_points([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_unparseable_coords_become_zero(self) -> None:
        # to_float falls back to 0 for non-numeric values.
        result = dedupe_points([("bad", 5.0), ("bad", 5.0)])
        assert result == [(0.0, 5.0)]

    def test_within_tolerance_treated_as_duplicate(self) -> None:
        # 1e-10 < 1e-9 tolerance; should dedup.
        result = dedupe_points([(1.0, 1.0), (1.0 + 1e-10, 1.0 + 1e-10)])
        assert len(result) == 1

    def test_above_tolerance_not_treated_as_duplicate(self) -> None:
        # 1e-8 > 1e-9 tolerance; should keep both.
        result = dedupe_points([(1.0, 1.0), (1.0 + 1e-8, 1.0 + 1e-8)])
        assert len(result) == 2

    def test_only_consecutive_duplicates_removed(self) -> None:
        result = dedupe_points([(0, 0), (1, 1), (0, 0), (1, 1), (1, 1)])
        # Pattern 0,1,0,1 has no consecutive dup until the trailing 1,1.
        assert result == [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)]


# ── find_free_region ─────────────────────────────────────────────────────


class TestFindFreeRegion:
    def test_empty_occupied_returns_origin_region(self) -> None:
        result = find_free_region([], width=5, height=5)
        assert result is not None
        # First candidate at radius=0, dx=0, dy=0 from origin.
        assert result.x_min == 0.0 and result.y_min == 0.0

    def test_origin_blocked_returns_offset(self) -> None:
        occupied = [BBox(-50, -50, 50, 50)]
        result = find_free_region(occupied, width=5, height=5)
        # Should find a free spot somewhere outside the blocking rect.
        assert result is not None
        assert not result.overlaps(occupied[0], pad=2.0)

    def test_huge_blocker_exceeds_search_radius_returns_none(self) -> None:
        # Search radius is capped at 500; a >1000-wide block has no escape.
        occupied = [BBox(-1000, -1000, 1000, 1000)]
        result = find_free_region(occupied, width=5, height=5)
        assert result is None

    def test_search_origin_respected(self) -> None:
        # Block origin only, ensure the algorithm searches outward and
        # finds a region somewhere away from the origin.
        occupied = [BBox(-3, -3, 3, 3)]
        result = find_free_region(
            occupied, width=2, height=2, search_origin=Point(50, 50)
        )
        assert result is not None
        assert not result.overlaps(occupied[0], pad=2.0)

    def test_none_occupied_raises(self) -> None:
        with pytest.raises(ValueError, match="occupied"):
            find_free_region(None, 5, 5)  # type: ignore[arg-type]

    def test_returned_region_has_requested_dimensions(self) -> None:
        result = find_free_region([], width=7.5, height=3.25)
        assert result is not None
        assert result.x_max - result.x_min == pytest.approx(7.5)
        assert result.y_max - result.y_min == pytest.approx(3.25)
