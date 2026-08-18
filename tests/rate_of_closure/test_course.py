"""Course scene palette + layout (epic #4125, H7a)."""

from __future__ import annotations

import pytest

from rate_of_closure.ui.course import (
    CourseLayout,
    blend,
    course_colors,
)

pytestmark = [pytest.mark.unit]


class TestBlend:
    def test_endpoints_and_midpoint(self) -> None:
        assert blend("#000000", "#ffffff", 0.0) == "#000000"
        assert blend("#000000", "#ffffff", 1.0) == "#ffffff"
        assert blend("#000000", "#ffffff", 0.5) == "#808080"

    def test_fraction_is_clamped(self) -> None:
        assert blend("#102030", "#ffffff", 2.0) == "#ffffff"
        assert blend("#102030", "#ffffff", -1.0) == "#102030"

    def test_non_hex_palette_entries_pass_through(self) -> None:
        # The matplotlib fallback cycle ("C1") must degrade gracefully.
        assert blend("C1", "#000000", 0.5) == "C1"


class TestCourseColors:
    def test_all_tones_are_hex_and_distinct(self) -> None:
        tones = course_colors()
        values = [
            tones.rough,
            tones.fairway,
            tones.green,
            tones.hole,
            tones.flag,
            tones.tee,
        ]
        for value in values:
            assert value.startswith("#") and len(value) == 7, value
        assert len(set(values)) == len(values)

    def test_grass_family_is_ordered_dark_to_light(self) -> None:
        """Rough < fairway < green in luminance — one grass family."""
        tones = course_colors()

        def luminance(color: str) -> float:
            r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
            return 0.299 * r + 0.587 * g + 0.114 * b

        assert luminance(tones.hole) < luminance(tones.rough)
        assert luminance(tones.rough) < luminance(tones.fairway)
        assert luminance(tones.fairway) < luminance(tones.green)

    def test_tones_derive_from_the_palette_green(self) -> None:
        """Every grass tone is an exact blend of the chart palette green."""
        from shared.python.theme.matplotlib_style import get_chart_color

        grass = get_chart_color(1)
        tones = course_colors()
        assert tones.rough == blend(grass, "#000000", 0.45)
        assert tones.fairway == blend(grass, "#000000", 0.15)
        assert tones.green == blend(grass, "#ffffff", 0.20)
        assert tones.flag == get_chart_color(3)
        assert tones.tee == get_chart_color(6)


class TestCourseLayout:
    def test_defaults_are_sane(self) -> None:
        layout = CourseLayout()
        assert layout.green_distance_m > layout.green_radius_m
        assert layout.fairway_half_width_m > 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"green_distance_m": 0.0},
            {"green_distance_m": -10.0},
            {"green_radius_m": 0.0},
            {"fairway_half_width_m": -1.0},
        ],
    )
    def test_rejects_non_positive_geometry(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CourseLayout(**kwargs)
