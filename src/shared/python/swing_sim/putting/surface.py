"""Green surface heightfield: parametric plane + grid form (#4800 P2).

Two surface forms, one contract
-------------------------------
A green surface answers two queries in the putt frame (x = initial
putt line, y = left of the putt line, both in metres):

* ``height_m(x, y)`` — surface elevation [m] relative to the origin.
* ``gravity_inplane_mps2(x, y)`` — the in-plane gravity acceleration
  ``-g * grad h`` [m/s^2] felt by a ball at ``(x, y)`` (small-slope
  approximation, the same one the planar model has always used; green
  grades are a few percent, so ``cos`` factors are 1 to first order).

:class:`PlanarGreenSurface` is the parametric (degenerate) case — the
uniform grade + aspect plane the putting vertical has modeled since
#4125 H3. Its in-plane gravity is computed with the exact expression
:func:`~.green.simulate_putt` historically used, so the planar limit
reproduces the legacy integrator bit-for-bit (regression-gated).

:class:`GridGreenSurface` is a regular square-grid heightfield with
bilinear interpolation inside each cell (bilinear reproduces any
plane exactly, so a grid sampled from a plane matches the parametric
form to floating-point rounding). Outside the grid hull the surface
continues **flat** (zero in-plane gravity, edge-clamped height) —
documented so a putt that leaves the modeled patch still terminates
by friction.

Wire ``swing_sim.green_surface/1``
----------------------------------
Versioned, fail-closed JSON with the same posture as
:mod:`shared.python.swing_sim.delivery_interchange`: sorted keys,
compact separators, ``allow_nan=False``, unknown fields refused,
missing fields refused, non-finite values refused, and byte-identical
round-trips (``to_json(from_json(text)) == text`` for canonical
input). The payload is geometry only — green speed (stimp) and
friction stay in the simulation call, so the same surface serves any
condition set. Float formatting is runtime-local (Python ``repr`` vs
JS shortest-round-trip differ on integral floats); cross-runtime
interchange is by JSON value, with byte determinism guaranteed within
each runtime. The wire is the seam UpstreamDrift's ``putting_green``
``_surface_io`` adapter targets (epic #4800 Amendment 1, P9).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from shared.python.contracts import require, require_finite

from .roll import GRAVITY_M_S2

GREEN_SURFACE_FORMAT = "swing_sim.green_surface/1"

__all__ = [
    "GREEN_SURFACE_FORMAT",
    "GreenSurface",
    "GridGreenSurface",
    "PlanarGreenSurface",
    "green_surface_from_json",
    "green_surface_to_json",
]

#: Maximum grid nodes per axis (keeps hostile wires bounded).
_MAX_GRID_NODES = 2048

#: Maximum |dh| / spacing between adjacent nodes — 25 % local grade,
#: comfortably beyond real greens (planar caps at 10 %) while keeping
#: the small-slope model honest.
_MAX_LOCAL_GRADE = 0.25

_PLANAR_FIELDS = frozenset({"format", "kind", "aspect_deg", "grade_percent"})
_GRID_FIELDS = frozenset({"format", "kind", "heights_m", "origin_m", "spacing_m"})


def _finite_number(value: object, name: str) -> float:
    """A strict JSON number: int or float, never bool, always finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite", value)
    return result


@dataclass(frozen=True)
class PlanarGreenSurface:
    """Uniform-slope plane — the parametric heightfield.

    Attributes:
        grade_percent: Uniform slope grade [%]; 0-10 covers greens.
        aspect_deg: Downhill direction, CCW from the putt line [deg];
            0 = downhill ahead, +90 = downhill to the left.
    """

    grade_percent: float
    aspect_deg: float
    #: In-plane gravity, precomputed with the exact legacy expression
    #: so the planar limit is bit-identical to the historic integrator.
    _gravity: tuple[float, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_finite(self.grade_percent, "grade_percent")
        require(
            0.0 <= self.grade_percent <= 10.0,
            "grade must be in [0, 10] percent",
            self.grade_percent,
        )
        require_finite(self.aspect_deg, "aspect_deg")
        require(
            -360.0 <= self.aspect_deg <= 360.0,
            "aspect must be in [-360, 360] deg",
            self.aspect_deg,
        )
        aspect = math.radians(self.aspect_deg)
        grade = self.grade_percent / 100.0
        object.__setattr__(
            self,
            "_gravity",
            (
                GRAVITY_M_S2 * grade * math.cos(aspect),
                GRAVITY_M_S2 * grade * math.sin(aspect),
            ),
        )

    def height_m(self, x_m: float, y_m: float) -> float:
        """Plane elevation ``-(grade)(x cos a + y sin a)`` [m]."""
        aspect = math.radians(self.aspect_deg)
        grade = self.grade_percent / 100.0
        return -grade * (x_m * math.cos(aspect) + y_m * math.sin(aspect))

    def gravity_inplane_mps2(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Constant in-plane gravity ``-g grad h`` (position-free)."""
        return self._gravity


@dataclass(frozen=True)
class GridGreenSurface:
    """Regular square-grid heightfield with bilinear interpolation.

    ``heights_m[j][i]`` is the elevation at
    ``(origin_m[0] + i * spacing_m, origin_m[1] + j * spacing_m)`` —
    rows index y, columns index x. Outside the grid hull the surface
    continues flat (see module docstring).

    Attributes:
        origin_m: Grid origin ``(x0, y0)`` in the putt frame [m].
        spacing_m: Node spacing [m], uniform in x and y.
        heights_m: Row-major node elevations [m], at least 2 x 2.
    """

    origin_m: tuple[float, float]
    spacing_m: float
    heights_m: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        require(
            isinstance(self.origin_m, tuple) and len(self.origin_m) == 2,
            "origin_m must be an (x, y) tuple",
        )
        require_finite(self.origin_m[0], "origin_m[0]")
        require_finite(self.origin_m[1], "origin_m[1]")
        require_finite(self.spacing_m, "spacing_m")
        require(
            0.01 <= self.spacing_m <= 100.0,
            "spacing must be in [0.01, 100] m",
            self.spacing_m,
        )
        rows = self.heights_m
        require(isinstance(rows, tuple) and len(rows) >= 2, "need at least 2 rows")
        require(len(rows) <= _MAX_GRID_NODES, "too many grid rows", len(rows))
        width = len(rows[0]) if isinstance(rows[0], tuple) else -1
        require(2 <= width <= _MAX_GRID_NODES, "need 2..2048 columns", width)
        for j, row in enumerate(rows):
            require(
                isinstance(row, tuple) and len(row) == width,
                "heights_m must be rectangular",
                j,
            )
            for i, height in enumerate(row):
                require(
                    isinstance(height, float) and math.isfinite(height),
                    "heights must be finite floats",
                    (j, i),
                )
        self._require_plausible_grades()

    def _require_plausible_grades(self) -> None:
        """Adjacent-node grade bound (small-slope model validity)."""
        limit = _MAX_LOCAL_GRADE * self.spacing_m
        for j, row in enumerate(self.heights_m):
            for i, height in enumerate(row):
                if i + 1 < len(row):
                    require(
                        abs(row[i + 1] - height) <= limit,
                        "local grade exceeds 25 percent",
                        (j, i),
                    )
                if j + 1 < len(self.heights_m):
                    require(
                        abs(self.heights_m[j + 1][i] - height) <= limit,
                        "local grade exceeds 25 percent",
                        (j, i),
                    )

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int, float, float]:
        """Clamped cell index and in-cell fractions for a query point."""
        nx = len(self.heights_m[0])
        ny = len(self.heights_m)
        u = (x_m - self.origin_m[0]) / self.spacing_m
        v = (y_m - self.origin_m[1]) / self.spacing_m
        u = min(max(u, 0.0), float(nx - 1))
        v = min(max(v, 0.0), float(ny - 1))
        i = min(int(u), nx - 2)
        j = min(int(v), ny - 2)
        return i, j, u - i, v - j

    def _inside(self, x_m: float, y_m: float) -> bool:
        nx = len(self.heights_m[0])
        ny = len(self.heights_m)
        return (
            self.origin_m[0] <= x_m <= self.origin_m[0] + (nx - 1) * self.spacing_m
            and self.origin_m[1] <= y_m <= self.origin_m[1] + (ny - 1) * self.spacing_m
        )

    def height_m(self, x_m: float, y_m: float) -> float:
        """Bilinear elevation [m]; edge-clamped outside the hull."""
        i, j, tx, ty = self._cell(x_m, y_m)
        h00 = self.heights_m[j][i]
        h10 = self.heights_m[j][i + 1]
        h01 = self.heights_m[j + 1][i]
        h11 = self.heights_m[j + 1][i + 1]
        top = h00 * (1.0 - tx) + h10 * tx
        bottom = h01 * (1.0 - tx) + h11 * tx
        return top * (1.0 - ty) + bottom * ty

    def gravity_inplane_mps2(self, x_m: float, y_m: float) -> tuple[float, float]:
        """``-g grad h`` from the bilinear cell; zero outside the hull."""
        if not self._inside(x_m, y_m):
            return (0.0, 0.0)
        i, j, tx, ty = self._cell(x_m, y_m)
        h00 = self.heights_m[j][i]
        h10 = self.heights_m[j][i + 1]
        h01 = self.heights_m[j + 1][i]
        h11 = self.heights_m[j + 1][i + 1]
        dhdx = ((h10 - h00) * (1.0 - ty) + (h11 - h01) * ty) / self.spacing_m
        dhdy = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / self.spacing_m
        return (-GRAVITY_M_S2 * dhdx, -GRAVITY_M_S2 * dhdy)


#: Either surface form — both answer height and in-plane gravity.
GreenSurface = PlanarGreenSurface | GridGreenSurface


def green_surface_to_json(surface: GreenSurface) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
    payload: dict[str, Any]
    if isinstance(surface, PlanarGreenSurface):
        payload = {
            "format": GREEN_SURFACE_FORMAT,
            "kind": "planar",
            "grade_percent": surface.grade_percent,
            "aspect_deg": surface.aspect_deg,
        }
    elif isinstance(surface, GridGreenSurface):
        payload = {
            "format": GREEN_SURFACE_FORMAT,
            "kind": "grid",
            "origin_m": list(surface.origin_m),
            "spacing_m": surface.spacing_m,
            "heights_m": [list(row) for row in surface.heights_m],
        }
    else:
        raise TypeError("surface must be a GreenSurface")
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def green_surface_from_json(text: str) -> GreenSurface:
    """Parse and validate; unknown fields and wrong formats are refused."""
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "green surface must be an object")
    require(
        data.get("format") == GREEN_SURFACE_FORMAT,
        f"format must be {GREEN_SURFACE_FORMAT!r}",
    )
    kind = data.get("kind")
    if kind == "planar":
        require(
            set(data) == _PLANAR_FIELDS,
            f"planar surface fields must be exactly {sorted(_PLANAR_FIELDS)}",
        )
        return PlanarGreenSurface(
            grade_percent=_finite_number(data["grade_percent"], "grade_percent"),
            aspect_deg=_finite_number(data["aspect_deg"], "aspect_deg"),
        )
    if kind == "grid":
        require(
            set(data) == _GRID_FIELDS,
            f"grid surface fields must be exactly {sorted(_GRID_FIELDS)}",
        )
        origin = data["origin_m"]
        require(
            isinstance(origin, list) and len(origin) == 2,
            "origin_m must be an [x, y] pair",
        )
        rows = data["heights_m"]
        require(isinstance(rows, list), "heights_m must be a list of rows")
        heights = []
        for j, row in enumerate(rows):
            require(isinstance(row, list), "each heights_m row must be a list", j)
            heights.append(
                tuple(
                    _finite_number(item, f"heights_m[{j}][{i}]")
                    for i, item in enumerate(row)
                )
            )
        return GridGreenSurface(
            origin_m=(
                _finite_number(origin[0], "origin_m[0]"),
                _finite_number(origin[1], "origin_m[1]"),
            ),
            spacing_m=_finite_number(data["spacing_m"], "spacing_m"),
            heights_m=tuple(heights),
        )
    raise ValueError("kind must be 'planar' or 'grid'")
