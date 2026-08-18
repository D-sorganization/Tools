"""Golf-course scene palette + layout (epic #4125, H7a).

The simulation/flight displays render as a course: the ground plane is
grass, a lighter fairway strip runs along the target line, and a
distinct putting green with a hole/flag marker sits at a configurable
distance. Every scene tone here is DERIVED from the shared UpstreamDrift
chart palette (``get_chart_color`` — themes.json ``chartColors``) by
blending the palette green toward black/white; no scene color is
hard-coded in any widget. The web mirror is
``web/src/model/course.ts`` (same blend math, same fractions).
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


__all__ = ["CourseColors", "CourseLayout", "blend", "course_colors"]

#: Palette indices (themes.json chartColors): green / red / yellow.
_GRASS_INDEX = 1
_FLAG_INDEX = 3
_TEE_INDEX = 6


def _parse_hex(color: str) -> tuple[int, int, int] | None:
    """``#rrggbb`` -> (r, g, b), or None for non-hex palette entries."""
    value = color.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def blend(color: str, other: str, fraction: float) -> str:
    """Linear RGB blend of two hex colors (``fraction`` toward ``other``).

    Non-hex inputs (e.g. the matplotlib ``C1`` fallback cycle names)
    pass through unchanged so the module degrades gracefully without
    the theme package.
    """
    a, b = _parse_hex(color), _parse_hex(other)
    if a is None or b is None:
        return color
    t = min(max(fraction, 0.0), 1.0)
    mixed = tuple(round(ca + (cb - ca) * t) for ca, cb in zip(a, b, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _shade(color: str, fraction: float) -> str:
    """Blend toward black (darker grass tones)."""
    return blend(color, "#000000", fraction)


def _tint(color: str, fraction: float) -> str:
    """Blend toward white (lighter grass tones)."""
    return blend(color, "#ffffff", fraction)


@dataclass(frozen=True)
class CourseColors:
    """Scene tones, all derived from the theme chart palette."""

    rough: str  #: general grass ground plane (darkest tone)
    fairway: str  #: lighter mown strip along the target line
    green: str  #: putting surface (lightest, cleanest tone)
    hole: str  #: hole cup (near-black shade of the grass hue)
    flag: str  #: flagstick + flag (palette red)
    tee: str  #: tee marker at the origin (palette yellow)


def course_colors() -> CourseColors:
    """Derive the course tones from the active chart palette.

    The palette green anchors every grass tone: rough is a 45% shade,
    the fairway a 15% shade, and the putting green a 20% tint of the
    same hue, so all three read as one grass family in any theme.
    """
    grass = get_chart_color(_GRASS_INDEX)
    return CourseColors(
        rough=_shade(grass, 0.45),
        fairway=_shade(grass, 0.15),
        green=_tint(grass, 0.20),
        hole=_shade(grass, 0.85),
        flag=get_chart_color(_FLAG_INDEX),
        tee=get_chart_color(_TEE_INDEX),
    )


@dataclass(frozen=True)
class CourseLayout:
    """Where the course furniture sits, app frame (x downrange [m]).

    The green distance is configurable — the flight views expose it and
    the H7b target region drives it — with sensible driver-hole
    defaults; the tee is always the origin.
    """

    green_distance_m: float = 230.0  #: tee -> hole along the target line
    green_radius_m: float = 10.0  #: putting-surface radius
    fairway_half_width_m: float = 16.0  #: fairway strip half-width

    def __post_init__(self) -> None:
        if self.green_distance_m <= 0.0:
            raise ValueError("green distance must be positive")
        if self.green_radius_m <= 0.0:
            raise ValueError("green radius must be positive")
        if self.fairway_half_width_m <= 0.0:
            raise ValueError("fairway half-width must be positive")
