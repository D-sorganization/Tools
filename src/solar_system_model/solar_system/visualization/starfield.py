"""Build accurate starfield geometry from catalog data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..data.star_catalog import StarEntry, equatorial_to_cartesian


@dataclass(frozen=True)
class StarVertex:
    """Renderable star with position and display color."""

    name: str
    position: np.ndarray
    color: Sequence[float]
    magnitude: float


def _spectral_color(bv_index: float) -> list[float]:
    """Map a B-V index to an RGB tuple using a simple black-body gradient."""

    # Clamp to a sensible stellar range
    bv_clamped = float(np.clip(bv_index, -0.4, 2.0))

    # Convert to a temperature-like value and then to RGB using a simple piecewise fit
    temperature = 4600.0 * (
        (1.0 / (0.92 * bv_clamped + 1.7)) + (1.0 / (0.92 * bv_clamped + 0.62))
    )

    # Normalize temperature to 3000-12000 K range
    normalized = np.clip((temperature - 3000.0) / 9000.0, 0.0, 1.0)

    # Interpolate between warm (reddish) and cool (bluish) colors
    red = float(np.interp(normalized, [0.0, 0.5, 1.0], [1.0, 1.0, 0.6]))
    green = float(np.interp(normalized, [0.0, 0.5, 1.0], [0.6, 1.0, 1.0]))
    blue = float(np.interp(normalized, [0.0, 0.5, 1.0], [0.2, 0.8, 1.0]))

    return [red, green, blue]


def _magnitude_to_intensity(magnitude: float) -> float:
    """Convert a visual magnitude to a relative brightness multiplier."""

    base_mag = -1.46  # Sirius reference point
    intensity = 10 ** (-0.4 * (magnitude - base_mag))
    return float(np.clip(intensity, 0.05, 4.5))


def point_size_from_magnitude(
    magnitude: float, min_size: float = 1.0, max_size: float = 6.0
) -> float:
    """Map apparent magnitude to an OpenGL point size for crisp rendering.

    The mapping keeps bright stars bold without letting dim catalog entries disappear.
    A logarithmic scale better matches human perception and prevents over-sizing Sirius.
    """

    # Use Pogson scale relative to Sirius to keep the brightest at ``max_size``
    if not (magnitude is not None):
        raise ValueError("magnitude must be provided")
    relative_brightness = 10.0 ** (-0.4 * (magnitude + 1.46))
    size = min_size + (max_size - min_size) * np.clip(relative_brightness, 0.0, 1.0)
    return float(np.clip(size, min_size, max_size))


def build_star_vertices(
    catalog: Iterable[StarEntry],
    radius: float = 1200.0,
) -> list[StarVertex]:
    """Generate star vertices positioned on a celestial sphere.

    Args:
        catalog: Iterable of :class:`StarEntry` rows.
        radius: Distance to place the sky dome.
    """

    if not (catalog is not None):
        raise ValueError("catalog must be provided")
    vertices: list[StarVertex] = []

    for entry in catalog:
        direction = np.array(equatorial_to_cartesian(entry.ra_hours, entry.dec_degrees))
        position = direction * radius
        intensity = _magnitude_to_intensity(entry.magnitude)
        base_color = _spectral_color(entry.bv_index)
        color = [channel * intensity for channel in base_color]
        vertices.append(
            StarVertex(
                name=entry.name,
                position=position,
                color=color,
                magnitude=entry.magnitude,
            )
        )

    # Brighter stars first ensures the renderer keeps vivid ones crisp
    # when GL_POINT_SIZE is limited.
    vertices.sort(key=lambda star: star.magnitude)
    return vertices
