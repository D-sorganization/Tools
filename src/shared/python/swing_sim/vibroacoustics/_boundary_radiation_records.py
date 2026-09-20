"""Discretized boundary surface mesh and acoustic medium records (IA-T5, #5074).

Defines acoustic fluid medium properties, discrete boundary surface elements,
and planar/curved mesh collections for transient vibroacoustic radiation solving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._waveform_contracts import owned_real_samples

logger = logging.getLogger(__name__)


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    num = float(value)
    if not np.isfinite(num):
        raise ValueError(f"{name} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return num


@dataclass(frozen=True)
class AcousticMedium:
    """Acoustic fluid medium properties.

    Attributes:
        density_kg_per_m3: Ambient fluid density in kg/m^3 (default: 1.204).
        sound_speed_mps: Small-signal speed of sound in m/s (default: 343.2).
    """

    density_kg_per_m3: float = 1.204
    sound_speed_mps: float = 343.2

    def __post_init__(self) -> None:
        density = _finite_scalar(self.density_kg_per_m3, "density", positive=True)
        speed = _finite_scalar(self.sound_speed_mps, "sound speed", positive=True)
        object.__setattr__(self, "density_kg_per_m3", density)
        object.__setattr__(self, "sound_speed_mps", speed)

    @property
    def characteristic_impedance(self) -> float:
        """Characteristic acoustic specific impedance z0 = rho0 * c0 (Pa*s/m)."""
        return self.density_kg_per_m3 * self.sound_speed_mps


@dataclass(frozen=True)
class RadiatingElement:
    """A single planar boundary surface element.

    Attributes:
        element_id: Nonnegative unique identifier for the element.
        area_m2: Surface area of the element in m^2 (> 0).
        centroid_m: (3,) Cartesian position vector of the element centroid in meters.
        normal: (3,) Outward unit normal vector.
    """

    element_id: int
    area_m2: float
    centroid_m: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.element_id, bool)
            or not isinstance(self.element_id, int)
            or self.element_id < 0
        ):
            raise ValueError("element_id must be a nonnegative integer")
        area = _finite_scalar(self.area_m2, "area", positive=True)
        object.__setattr__(self, "area_m2", area)

        centroid = owned_real_samples(self.centroid_m)
        if centroid.shape != (3,):
            raise ValueError("centroid_m must have shape (3,)")
        object.__setattr__(self, "centroid_m", centroid)

        n = owned_real_samples(self.normal)
        if n.shape != (3,):
            raise ValueError("normal must have shape (3,)")
        norm_len = float(np.linalg.norm(n))
        if norm_len < 1e-6 or not np.isfinite(norm_len):
            raise ValueError("normal must be a nonzero vector")
        object.__setattr__(self, "normal", owned_real_samples(n / norm_len))


@dataclass(frozen=True)
class RadiatingSurfaceMesh:
    """Discretized vibrating boundary surface composed of radiating elements."""

    elements: tuple[RadiatingElement, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("RadiatingSurfaceMesh requires at least one element")
        for el in self.elements:
            if not isinstance(el, RadiatingElement):
                raise TypeError("elements must contain RadiatingElement instances")

    @property
    def total_area_m2(self) -> float:
        """Sum of all element surface areas."""
        return sum(el.area_m2 for el in self.elements)

    @property
    def centroid_m(self) -> np.ndarray:
        """Area-weighted centroid of the surface mesh."""
        total_a = self.total_area_m2
        weighted = sum(el.centroid_m * el.area_m2 for el in self.elements)
        return owned_real_samples(weighted / total_a)

    @property
    def max_element_diameter_m(self) -> float:
        """Characteristic maximum element diameter estimated from element areas."""
        return float(max(2.0 * np.sqrt(el.area_m2 / np.pi) for el in self.elements))

    def validate_mesh_convergence(
        self,
        max_frequency_hz: float,
        sound_speed_mps: float = 343.2,
    ) -> bool:
        """Validate boundary element spatial convergence criterion.

        Rule of thumb for boundary element / Rayleigh integral discretization:
        h <= lambda_min / 6, where lambda_min = c0 / f_max.

        Raises:
            ValueError: If the maximum element size exceeds lambda_min / 6.
        """
        f_max = _finite_scalar(max_frequency_hz, "max_frequency_hz", positive=True)
        c0 = _finite_scalar(sound_speed_mps, "sound_speed_mps", positive=True)
        lambda_min = c0 / f_max
        h_max_allowed = lambda_min / 6.0
        h_actual = self.max_element_diameter_m

        if h_actual > h_max_allowed:
            raise ValueError(
                f"Mesh is too coarse for max frequency {f_max:.1f} Hz: "
                f"element diameter h={h_actual * 1000:.2f} mm exceeds "
                f"lambda/6 limit {h_max_allowed * 1000:.2f} mm"
            )
        return True

    @classmethod
    def planar_rectangular_plate(
        cls,
        length_x_m: float,
        width_y_m: float,
        nx: int,
        ny: int,
        z_m: float = 0.0,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> RadiatingSurfaceMesh:
        """Construct a uniform planar rectangular boundary mesh in the xy-plane."""
        lx = _finite_scalar(length_x_m, "length_x_m", positive=True)
        wy = _finite_scalar(width_y_m, "width_y_m", positive=True)
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers")

        dx = lx / nx
        dy = wy / ny
        area_el = dx * dy

        n_vec = np.array(normal, dtype=np.float64)

        elements: list[RadiatingElement] = []
        elem_id = 0
        x_starts = np.linspace(-lx / 2.0 + dx / 2.0, lx / 2.0 - dx / 2.0, nx)
        y_starts = np.linspace(-wy / 2.0 + dy / 2.0, wy / 2.0 - dy / 2.0, ny)

        for x in x_starts:
            for y in y_starts:
                centroid = np.array([x, y, z_m], dtype=np.float64)
                elements.append(
                    RadiatingElement(
                        element_id=elem_id,
                        area_m2=area_el,
                        centroid_m=centroid,
                        normal=n_vec,
                    )
                )
                elem_id += 1

        return cls(elements=tuple(elements))


__all__ = [
    "AcousticMedium",
    "RadiatingElement",
    "RadiatingSurfaceMesh",
]
