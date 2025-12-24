"""Asteroid belt and minor body definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import OrbitalElements, PhysicalProperties


@dataclass(frozen=True)
class AsteroidDescriptor:
    name: str
    elements: OrbitalElements
    color: tuple
    properties: PhysicalProperties


MAJOR_ASTEROIDS: list[AsteroidDescriptor] = [
    AsteroidDescriptor(
        "Ceres",
        OrbitalElements(
            semi_major_axis=2.7675,
            eccentricity=0.0758,
            inclination=10.593,
            longitude_ascending=80.305,
            longitude_perihelion=73.597,
            mean_longitude=95.989,
        ),
        (0.82, 0.72, 0.62),
        PhysicalProperties(
            mass=9.393e20,
            radius=473.0,
            density=2160.0,
            surface_gravity=0.27,
            escape_velocity=0.51,
            rotation_period=9.074,
            axial_tilt=4.0,
            albedo=0.09,
            temperature=167.0,
            color=(0.82, 0.72, 0.62),
        ),
    ),
    AsteroidDescriptor(
        "Vesta",
        OrbitalElements(
            semi_major_axis=2.3615,
            eccentricity=0.0887,
            inclination=7.141,
            longitude_ascending=103.851,
            longitude_perihelion=150.986,
            mean_longitude=150.977,
        ),
        (0.86, 0.76, 0.67),
        PhysicalProperties(
            mass=2.59076e20,
            radius=262.7,
            density=3456.0,
            surface_gravity=0.25,
            escape_velocity=0.36,
            rotation_period=5.342,
            axial_tilt=29.0,
            albedo=0.42,
            temperature=150.0,
            color=(0.86, 0.76, 0.67),
        ),
    ),
    AsteroidDescriptor(
        "Pallas",
        OrbitalElements(
            semi_major_axis=2.7718,
            eccentricity=0.2308,
            inclination=34.837,
            longitude_ascending=173.090,
            longitude_perihelion=310.172,
            mean_longitude=33.470,
        ),
        (0.74, 0.70, 0.68),
        PhysicalProperties(
            mass=2.11e20,
            radius=256.0,
            density=2900.0,
            surface_gravity=0.23,
            escape_velocity=0.33,
            rotation_period=7.813,
            axial_tilt=84.0,
            albedo=0.12,
            temperature=165.0,
            color=(0.74, 0.70, 0.68),
        ),
    ),
    AsteroidDescriptor(
        "Hygiea",
        OrbitalElements(
            semi_major_axis=3.1420,
            eccentricity=0.1125,
            inclination=3.842,
            longitude_ascending=283.196,
            longitude_perihelion=312.346,
            mean_longitude=114.163,
        ),
        (0.65, 0.63, 0.62),
        PhysicalProperties(
            mass=8.67e19,
            radius=215.0,
            density=1940.0,
            surface_gravity=0.09,
            escape_velocity=0.21,
            rotation_period=13.825,
            axial_tilt=3.0,
            albedo=0.07,
            temperature=164.0,
            color=(0.65, 0.63, 0.62),
        ),
    ),
]


def generate_belt_particles(count: int = 720) -> np.ndarray:
    """Generate deterministic asteroid belt ring positions in AU."""

    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    semi_major = np.linspace(2.1, 3.2, count)
    positions = np.zeros((count, 3), dtype=float)

    for idx, angle in enumerate(angles):
        radius = semi_major[idx]
        positions[idx] = [radius * np.cos(angle), 0.0, radius * np.sin(angle)]

    return positions
