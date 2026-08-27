"""UpstreamDrift ``putting_green`` topography adapter (#4800 P9).

Runtime-free format adapter between UpstreamDrift's serialized green
topography and the ``swing_sim.green_surface/1`` heightfield forms in
:mod:`.surface` — the same posture as the engine adapters in
:mod:`shared.python.swing_sim.delivery_interchange`: UpstreamDrift is
never imported; the *documented file format* is parsed, and the
adapter is fixture-tested against a sample authored to match UD's
schema exactly (the fixture in ``tests/fixtures/`` is synthesized
field-for-field from ``_load_json_topography`` — UD ships no canned
topography JSON to copy). Delivery follows the C8/H5 cross-repo
pattern: this Tools-side adapter lands first; the UD pin bump plus a
UD-side consumer test is the tracked follow-up.

Format mapping
--------------
UpstreamDrift reads JSON green topography in
``src/engines/physics_engines/putting_green/python/_surface_io.py``
(``SurfaceIOMixin._load_json_topography``), keys ``contours``,
``slopes``, and ``hole_position``:

=================================  ====================================
UD field (``_surface_io`` JSON)    ``swing_sim.green_surface/1`` side
=================================  ====================================
``contours[k].x`` [m]              grid node ``origin_m[0] + i*spacing``
``contours[k].y`` [m]              grid node ``origin_m[1] + j*spacing``
``contours[k].elevation`` [m]      ``heights_m[j][i]``
``slopes`` (SlopeRegion objects)   **refused** — see below
``hole_position`` [m, m]           import metadata
                                   (:attr:`UdGreenTopography.hole_position_m`),
                                   not surface geometry
=================================  ====================================

Import requires the contour points to form a **complete regular square
grid** (one spacing, shared by both axes — the v1 wire carries a
single ``spacing_m``). UD interpolates *scattered* contours with a
SciPy thin-plate-spline RBF (smoothing 0.01); a runtime-free adapter
cannot reproduce that surface, so scattered input is refused instead
of silently diverging — author interchange greens on a grid. On grid
input the two engines agree at every node, and agree *everywhere* for
planar fields (bilinear interpolation reproduces planes exactly);
off-node values of genuinely curved surfaces differ (UD cubic/RBF vs
wire bilinear) — a documented difference, not reconciled.

``slopes`` are refused because UD applies slope regions directly as a
weighted *slope* field (``_surface_geometry.get_slope_at``) with no
elevation function behind it; the summed radial-falloff field is
generally non-conservative, so no heightfield exists whose gradient
reproduces it. Bake regions into contour elevations on the UD side
before interchange.

Out of scope in v1 (documented): UD's ``.npy``/GeoTIFF heightmap paths
(binary formats, and UD gaussian-smooths heightmaps on load by default
— runtime behaviour a format adapter must not replicate); the
``_green_loader`` green-config wrapper (turf and green dimensions —
the wire is geometry-only, same posture as the P2 wire: stimp and
friction stay in the simulation call); and coordinate-frame alignment
(UD green coordinates are green-local ``[0, width] x [0, height]``
with the hole wherever ``hole_position`` says; the Tools putt frame
puts the ball at the origin with x along the putt line — placing and
rotating the surface into the putt frame is the caller's job, exactly
as for any other imported surface).

Physics reconciliation (documented, per #4800 Amendment 1)
----------------------------------------------------------
Where the engines model the same phenomenon they share the same laws:
in-plane gravity is ``-g * grad h`` in both (small-slope; UD
``get_gravitational_acceleration`` = ``-g * slope``), and flat-green
pure roll is a constant deceleration ``mu * g`` in both, giving the
shared roll-out form ``d = v^2 / (2 mu g)``. The cross-engine
consistency gates in ``tests/test_ud_adapter.py`` therefore gate what
both models share — signs (uphill/downhill roll-out asymmetry, break
toward the cross-slope downhill side), monotonicity (roll-out in
launch speed and in stimp), the flat-green straight line, and the
bitwise-identical gravity law on imported planes. Magnitudes are NOT
forced to agree, because the mu laws legitimately differ:

* UD (``turf_properties.rolling_friction_coefficient``):
  ``mu = 0.196 / stimp`` (times height-of-cut, condition, and grain
  factors), which back-solves to an assumed stimpmeter release speed
  of ~1.08 m/s.
* Tools (:func:`.roll.stimp_to_rolling_mu`):
  ``mu = v_release^2 / (2 g S)`` with the USGA-geometry release speed
  ~1.83 m/s, i.e. ``mu ~= 0.559 / stimp``.

Same ``1/stimp`` form, so the ratio is a stimp-independent constant
``mu_tools / mu_ud ~= 2.854`` (gate-pinned): at equal rolling speed a
UD green rolls out ~2.85x farther than the same stimp in Tools.
Sliding also differs: UD scales its rolling mu by 1.8; Tools uses the
turf constant ``mu_slide = 0.40``. Hole capture differs: UD
``is_in_hole`` (radius test plus a 1.5 m/s edge lip-out heuristic) vs
the published Holmes/Penner effective-radius model in
:mod:`.capture`. Neither pair is gated against the other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from shared.python.contracts import require, require_finite

from .surface import (
    _MAX_GRID_NODES,
    GreenSurface,
    GridGreenSurface,
    PlanarGreenSurface,
    _finite_number,
)

__all__ = [
    "UdGreenTopography",
    "green_surface_from_ud_json",
    "green_surface_to_ud_json",
]

_TOP_LEVEL_FIELDS = frozenset({"contours", "hole_position"})
_CONTOUR_FIELDS = frozenset({"x", "y", "elevation"})

#: Relative tolerance for grid-regularity checks (node coordinates are
#: compared against the ideal ``first + i * spacing`` lattice).
_GRID_REL_TOL = 1e-9


@dataclass(frozen=True)
class UdGreenTopography:
    """A parsed UpstreamDrift topography document.

    Attributes:
        surface: The contour grid as a wire-form heightfield.
        hole_position_m: UD's ``hole_position`` [m, m] in the same
            (UD green-local) coordinates as the surface, or None when
            the document does not carry one. Metadata, not geometry —
            the ``swing_sim.green_surface/1`` wire is geometry-only.
    """

    surface: GridGreenSurface
    hole_position_m: tuple[float, float] | None


def _regular_axis_spacing(values: list[float], name: str) -> float:
    """Spacing of a sorted axis that must be a regular lattice."""
    count = len(values)
    require(count >= 2, f"contours must span at least 2 distinct {name} values")
    require(count <= _MAX_GRID_NODES, f"too many distinct {name} values", count)
    span = values[-1] - values[0]
    spacing = span / (count - 1)
    require(spacing > 0.0, f"{name} spacing must be positive", spacing)
    tolerance = _GRID_REL_TOL * max(1.0, abs(span))
    for index, value in enumerate(values):
        ideal = values[0] + index * spacing
        require(
            abs(value - ideal) <= tolerance,
            f"contour {name} values must be evenly spaced",
            value,
        )
    return spacing


def _parse_contour_points(rows: Any) -> dict[tuple[float, float], float]:
    """Contour objects -> ``{(x, y): elevation}``; duplicates refused."""
    require(isinstance(rows, list), "contours must be a list")
    points: dict[tuple[float, float], float] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), "each contour must be an object", index)
        require(
            set(row) == _CONTOUR_FIELDS,
            f"contour fields must be exactly {sorted(_CONTOUR_FIELDS)}",
            index,
        )
        node = (
            _finite_number(row["x"], f"contours[{index}].x"),
            _finite_number(row["y"], f"contours[{index}].y"),
        )
        require(node not in points, "duplicate contour node", node)
        points[node] = _finite_number(row["elevation"], f"contours[{index}].elevation")
    return points


def _parse_hole_position(value: Any) -> tuple[float, float]:
    require(
        isinstance(value, list) and len(value) == 2,
        "hole_position must be an [x, y] pair",
    )
    return (
        _finite_number(value[0], "hole_position[0]"),
        _finite_number(value[1], "hole_position[1]"),
    )


def green_surface_from_ud_json(text: str) -> UdGreenTopography:
    """Parse an UpstreamDrift ``_surface_io`` topography document.

    The contour points must form a complete regular square grid (see
    the module docstring for why scattered contours and ``slopes`` are
    refused). Unknown fields are refused fail-closed.

    Args:
        text: The JSON topography document.

    Returns:
        The parsed :class:`UdGreenTopography` — surface plus optional
        hole position metadata.

    Raises:
        ValueError: If the document is not a representable topography.
        TypeError: If ``text`` or a field has the wrong type.
    """
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "topography must be an object")
    require(
        "slopes" not in data,
        "slopes are refused: UD's slope-region field is a non-conservative "
        "slope (no heightfield reproduces it) — bake regions into contour "
        "elevations for interchange",
    )
    unknown = set(data) - _TOP_LEVEL_FIELDS
    require(not unknown, f"unknown topography fields: {sorted(unknown)}")
    require("contours" in data, "topography must carry contours")

    points = _parse_contour_points(data["contours"])
    xs = sorted({x for x, _y in points})
    ys = sorted({y for _x, y in points})
    require(
        len(points) == len(xs) * len(ys),
        "contours must cover every node of a complete regular grid",
        (len(points), len(xs), len(ys)),
    )
    spacing_x = _regular_axis_spacing(xs, "x")
    spacing_y = _regular_axis_spacing(ys, "y")
    require(
        math.isclose(spacing_x, spacing_y, rel_tol=_GRID_REL_TOL),
        "the green_surface/1 wire carries one spacing: x and y must match",
        (spacing_x, spacing_y),
    )
    heights = tuple(tuple(points[(x, y)] for x in xs) for y in ys)
    surface = GridGreenSurface(
        origin_m=(xs[0], ys[0]),
        spacing_m=spacing_x,
        heights_m=heights,
    )
    hole = (
        _parse_hole_position(data["hole_position"]) if "hole_position" in data else None
    )
    return UdGreenTopography(surface=surface, hole_position_m=hole)


def _planar_axis_nodes(extent_m: float, spacing_m: float, name: str) -> int:
    """Node count for a sampled planar axis; extent must fit the lattice."""
    require_finite(extent_m, name)
    require(extent_m > 0.0, f"{name} must be positive", extent_m)
    cells = extent_m / spacing_m
    require(
        abs(cells - round(cells)) <= _GRID_REL_TOL * max(1.0, cells),
        f"{name} must be an integer multiple of spacing_m",
        (extent_m, spacing_m),
    )
    nodes = round(cells) + 1
    require(2 <= nodes <= _MAX_GRID_NODES, f"{name} node count out of range", nodes)
    return nodes


def _planar_nodes(
    extent_m: tuple[float, float],
    spacing_m: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sample lattice for a planar export over ``[0, extent] x [0, extent]``."""
    require(
        isinstance(extent_m, tuple) and len(extent_m) == 2,
        "extent_m must be a (width, height) tuple",
    )
    require_finite(spacing_m, "spacing_m")
    require(
        0.01 <= spacing_m <= 100.0,
        "spacing must be in [0.01, 100] m",
        spacing_m,
    )
    nx = _planar_axis_nodes(extent_m[0], spacing_m, "extent_m[0]")
    ny = _planar_axis_nodes(extent_m[1], spacing_m, "extent_m[1]")
    xs = tuple(i * spacing_m for i in range(nx))
    ys = tuple(j * spacing_m for j in range(ny))
    return xs, ys


def green_surface_to_ud_json(
    surface: GreenSurface,
    *,
    hole_position_m: tuple[float, float] | None = None,
    extent_m: tuple[float, float] | None = None,
    spacing_m: float | None = None,
) -> str:
    """Serialize a green surface as UD ``_surface_io`` topography JSON.

    The emitted document loads directly through UD's
    ``_load_json_topography`` (``contours`` plus optional
    ``hole_position``) and round-trips byte-identically through
    :func:`green_surface_from_ud_json` for canonical (binary-exact)
    lattices. Serialization is deterministic: row-major node order,
    sorted keys, compact separators, non-finite values refused.

    Args:
        surface: The surface to export. A grid exports its own nodes;
            a plane is unbounded, so it is sampled over
            ``[0, extent_m[0]] x [0, extent_m[1]]`` (bilinear grids
            reproduce planes exactly, so nothing is lost).
        hole_position_m: Optional UD ``hole_position`` metadata [m, m].
        extent_m: Sampled extent for planar surfaces (required for
            planes, refused for grids).
        spacing_m: Sample spacing for planar surfaces (required for
            planes, refused for grids).

    Returns:
        The topography JSON document.

    Raises:
        ValueError: If inputs are out of range.
        TypeError: If ``surface`` is not a green surface.
    """
    if isinstance(surface, GridGreenSurface):
        require(
            extent_m is None and spacing_m is None,
            "grid surfaces carry their own nodes: omit extent_m/spacing_m",
        )
        xs = tuple(
            surface.origin_m[0] + i * surface.spacing_m
            for i in range(len(surface.heights_m[0]))
        )
        ys = tuple(
            surface.origin_m[1] + j * surface.spacing_m
            for j in range(len(surface.heights_m))
        )
        heights = surface.heights_m
    elif isinstance(surface, PlanarGreenSurface):
        if extent_m is None or spacing_m is None:
            raise ValueError(
                "planar surfaces are unbounded: pass extent_m and spacing_m"
            )
        xs, ys = _planar_nodes(extent_m, spacing_m)
        heights = tuple(tuple(surface.height_m(x, y) for x in xs) for y in ys)
    else:
        raise TypeError("surface must be a GreenSurface")

    contours = [
        {"x": x, "y": y, "elevation": heights[j][i]}
        for j, y in enumerate(ys)
        for i, x in enumerate(xs)
    ]
    payload: dict[str, object] = {"contours": contours}
    if hole_position_m is not None:
        require(
            isinstance(hole_position_m, tuple) and len(hole_position_m) == 2,
            "hole_position_m must be an (x, y) tuple",
        )
        require_finite(hole_position_m[0], "hole_position_m[0]")
        require_finite(hole_position_m[1], "hole_position_m[1]")
        payload["hole_position"] = list(hole_position_m)
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
