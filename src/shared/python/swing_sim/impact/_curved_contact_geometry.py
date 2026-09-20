"""Curved face and non-spherical contact geometry.

Provides:
- CurvedFaceGeometry: 3D face model with roll and bulge curvature.
- NonSphericalContactGeometry: contact projection finding closest point /
  Center of Pressure (COP) and local outward normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...golf_club._grip_contracts import finite_array
from ...golf_club._validation import (
    Vector3,
    require_finite_float,
    require_vector3,
)

_NORMAL_TOLERANCE = 1e-12


def _vector3(val: object, name: str) -> Vector3:
    return require_vector3(finite_array(val, (3,), name), name)


@dataclass(frozen=True)
class CurvedFaceGeometry:
    """Parametric golf club face with bulge (horizontal) and roll (vertical) curvature.

    Coordinates in face frame:
    - x: heel (-) to toe (+)
    - y: sole (-) to crown (+)
    - z: outward along face loft normal (+)

    Surface elevation: z(x, y) = -0.5 * (x^2 / R_bulge + y^2 / R_roll).
    """

    bulge_radius_m: float
    roll_radius_m: float
    face_half_width_m: float
    face_half_height_m: float
    face_origin_m: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bulge_radius_m",
            require_finite_float(self.bulge_radius_m, "bulge_radius_m", positive=True),
        )
        object.__setattr__(
            self,
            "roll_radius_m",
            require_finite_float(self.roll_radius_m, "roll_radius_m", positive=True),
        )
        object.__setattr__(
            self,
            "face_half_width_m",
            require_finite_float(
                self.face_half_width_m, "face_half_width_m", positive=True
            ),
        )
        object.__setattr__(
            self,
            "face_half_height_m",
            require_finite_float(
                self.face_half_height_m, "face_half_height_m", positive=True
            ),
        )
        object.__setattr__(
            self,
            "face_origin_m",
            _vector3(self.face_origin_m, "face_origin_m"),
        )

    def surface_elevation_m(self, x: float, y: float) -> float:
        """Compute surface z-coordinate in face frame."""
        return float(-0.5 * (x**2 / self.bulge_radius_m + y**2 / self.roll_radius_m))

    def surface_normal(self, x: float, y: float) -> Vector3:
        """Compute outward unit normal vector at face coordinates (x, y)."""
        nx = x / self.bulge_radius_m
        ny = y / self.roll_radius_m
        nz = 1.0
        norm = float(np.sqrt(nx**2 + ny**2 + nz**2))
        return _vector3((nx / norm, ny / norm, nz / norm), "surface normal")


@dataclass(frozen=True)
class NonSphericalContactGeometry:
    """Pairing of curved face geometry with spherical or ellipsoidal ball geometry."""

    face: CurvedFaceGeometry
    ball_radii_m: tuple[float, float, float] = (0.02135, 0.02135, 0.02135)

    def __post_init__(self) -> None:
        if not isinstance(self.face, CurvedFaceGeometry):
            raise TypeError("face must be CurvedFaceGeometry")
        radii = [
            require_finite_float(r, f"ball_radius_{i}", positive=True)
            for i, r in enumerate(self.ball_radii_m)
        ]
        if len(radii) != 3:
            raise ValueError("ball_radii_m must have length 3")
        object.__setattr__(self, "ball_radii_m", tuple(radii))

    @property
    def nominal_ball_radius_m(self) -> float:
        """Nominal spherical radius."""
        return float(np.mean(self.ball_radii_m))

    def project_cop(
        self,
        ball_x: float,
        ball_y: float,
        ball_z: float,
        max_iterations: int = 15,
        tol: float = 1e-13,
    ) -> tuple[float, float, float, Vector3]:
        """Project ball center onto curved face to find COP and surface normal.

        Solves for (x, y) where normal aligns with line connecting surface to ball:
        F(x, y) = (x - x_b, y - y_b) + (z(x, y) - z_b) * grad_z(x, y) = 0.
        """
        rb = self.face.bulge_radius_m
        rr = self.face.roll_radius_m

        # Initial guess with parabaloid scaling
        denom_x = max(0.1, 1.0 + ball_z / rb)
        denom_y = max(0.1, 1.0 + ball_z / rr)
        x = ball_x / denom_x
        y = ball_y / denom_y

        for _ in range(max_iterations):
            z = -0.5 * (x**2 / rb + y**2 / rr)

            # Collinearity condition between surface normal (x/rb, y/rr, 1)
            # and line from COP to ball center (ball_x - x, ball_y - y, ball_z - z):
            # (x - ball_x) + (ball_z - z) * (x / rb) = 0
            # (y - ball_y) + (ball_z - z) * (y / rr) = 0
            fx = (x - ball_x) + (ball_z - z) * (x / rb)
            fy = (y - ball_y) + (ball_z - z) * (y / rr)

            if abs(fx) < tol and abs(fy) < tol:
                break

            # Jacobian of F:
            # d(fx)/dx = 1 + (ball_z - z)/rb + (x^2)/(rb^2)
            # d(fx)/dy = (x * y) / (rb * rr)
            j11 = 1.0 + (ball_z - z) / rb + (x**2) / (rb**2)
            j12 = (x * y) / (rb * rr)
            j21 = (x * y) / (rb * rr)
            j22 = 1.0 + (ball_z - z) / rr + (y**2) / (rr**2)

            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-15:
                break

            dx = (j22 * fx - j12 * fy) / det
            dy = (-j21 * fx + j11 * fy) / det

            x -= dx
            y -= dy

        z = self.face.surface_elevation_m(x, y)
        normal = self.face.surface_normal(x, y)
        return float(x), float(y), float(z), normal


__all__ = (
    "CurvedFaceGeometry",
    "NonSphericalContactGeometry",
)
