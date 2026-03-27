"""Curated moon system data for gas giants and Earth."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.constants import OrbitalElements, PhysicalProperties


@dataclass(frozen=True)
class MoonDescriptor:
    name: str
    parent: str
    elements: OrbitalElements
    properties: PhysicalProperties


MOONS: list[MoonDescriptor] = [
    MoonDescriptor(
        "Moon",
        "Earth",
        OrbitalElements(
            semi_major_axis=384400.0 / 149_597_870.7,
            eccentricity=0.0549,
            inclination=5.145,
            longitude_ascending=125.08,
            longitude_perihelion=318.15,
            mean_longitude=135.27,
            mean_longitude_rate=13.176358,
        ),
        PhysicalProperties(
            mass=7.34767309e22,
            radius=1737.4,
            density=3344.0,
            surface_gravity=1.62,
            escape_velocity=2.38,
            rotation_period=655.728,
            axial_tilt=6.687,
            albedo=0.12,
            temperature=220.0,
            color=(0.8, 0.8, 0.78),
        ),
    ),
    MoonDescriptor(
        "Io",
        "Jupiter",
        OrbitalElements(
            semi_major_axis=421_700.0 / 149_597_870.7,
            eccentricity=0.0041,
            inclination=0.036,
            longitude_ascending=43.977,
            longitude_perihelion=84.129,
            mean_longitude=171.016,
            mean_longitude_rate=203.4889538,
        ),
        PhysicalProperties(
            mass=8.9319e22,
            radius=1821.6,
            density=3528.0,
            surface_gravity=1.80,
            escape_velocity=2.56,
            rotation_period=152.853,
            axial_tilt=0.05,
            albedo=0.62,
            temperature=110.0,
            color=(0.92, 0.79, 0.58),
        ),
    ),
    MoonDescriptor(
        "Europa",
        "Jupiter",
        OrbitalElements(
            semi_major_axis=671_100.0 / 149_597_870.7,
            eccentricity=0.0094,
            inclination=0.466,
            longitude_ascending=219.106,
            longitude_perihelion=88.970,
            mean_longitude=133.540,
            mean_longitude_rate=101.3747235,
        ),
        PhysicalProperties(
            mass=4.7998e22,
            radius=1560.8,
            density=3013.0,
            surface_gravity=1.31,
            escape_velocity=2.02,
            rotation_period=306.919,
            axial_tilt=0.10,
            albedo=0.68,
            temperature=102.0,
            color=(0.85, 0.87, 0.89),
        ),
    ),
    MoonDescriptor(
        "Ganymede",
        "Jupiter",
        OrbitalElements(
            semi_major_axis=1_070_400.0 / 149_597_870.7,
            eccentricity=0.0013,
            inclination=0.177,
            longitude_ascending=63.552,
            longitude_perihelion=192.417,
            mean_longitude=30.237,
            mean_longitude_rate=50.3176081,
        ),
        PhysicalProperties(
            mass=1.4819e23,
            radius=2634.1,
            density=1942.0,
            surface_gravity=1.43,
            escape_velocity=2.74,
            rotation_period=607.243,
            axial_tilt=0.33,
            albedo=0.43,
            temperature=110.0,
            color=(0.76, 0.74, 0.71),
        ),
    ),
    MoonDescriptor(
        "Callisto",
        "Jupiter",
        OrbitalElements(
            semi_major_axis=1_882_700.0 / 149_597_870.7,
            eccentricity=0.0074,
            inclination=0.192,
            longitude_ascending=298.848,
            longitude_perihelion=52.643,
            mean_longitude=120.303,
            mean_longitude_rate=21.5710715,
        ),
        PhysicalProperties(
            mass=1.0759e23,
            radius=2410.3,
            density=1834.0,
            surface_gravity=1.24,
            escape_velocity=2.44,
            rotation_period=1440.0,
            axial_tilt=0.40,
            albedo=0.17,
            temperature=134.0,
            color=(0.58, 0.55, 0.52),
        ),
    ),
    MoonDescriptor(
        "Titan",
        "Saturn",
        OrbitalElements(
            semi_major_axis=1_221_870.0 / 149_597_870.7,
            eccentricity=0.0288,
            inclination=0.348,
            longitude_ascending=28.060,
            longitude_perihelion=172.670,
            mean_longitude=261.158,
            mean_longitude_rate=22.5769768,
        ),
        PhysicalProperties(
            mass=1.3452e23,
            radius=2574.7,
            density=1880.0,
            surface_gravity=1.35,
            escape_velocity=2.64,
            rotation_period=382.68,
            axial_tilt=0.31,
            albedo=0.21,
            temperature=94.0,
            color=(0.89, 0.76, 0.56),
        ),
    ),
    MoonDescriptor(
        "Enceladus",
        "Saturn",
        OrbitalElements(
            semi_major_axis=238_020.0 / 149_597_870.7,
            eccentricity=0.0047,
            inclination=0.009,
            longitude_ascending=338.985,
            longitude_perihelion=115.365,
            mean_longitude=57.219,
            mean_longitude_rate=262.731900,
        ),
        PhysicalProperties(
            mass=1.0802e20,
            radius=252.1,
            density=1610.0,
            surface_gravity=0.11,
            escape_velocity=0.24,
            rotation_period=32.884,
            axial_tilt=0.00,
            albedo=0.99,
            temperature=75.0,
            color=(0.88, 0.92, 0.96),
        ),
    ),
    MoonDescriptor(
        "Triton",
        "Neptune",
        OrbitalElements(
            semi_major_axis=354_759.0 / 149_597_870.7,
            eccentricity=0.000016,
            inclination=156.865,
            longitude_ascending=214.730,
            longitude_perihelion=37.114,
            mean_longitude=177.777,
            mean_longitude_rate=-61.2572637,
        ),
        PhysicalProperties(
            mass=2.139e22,
            radius=1353.4,
            density=2059.0,
            surface_gravity=0.78,
            escape_velocity=1.45,
            rotation_period=141.0,
            axial_tilt=0.0,
            albedo=0.76,
            temperature=38.0,
            color=(0.78, 0.80, 0.86),
        ),
    ),
]


def moons_by_parent() -> dict[str, list[MoonDescriptor]]:
    """Return a mapping of parent body name to moon descriptors."""

    grouped: dict[str, list[MoonDescriptor]] = {}
    grouped.setdefault(moon.parent, []).extend([moon for moon in MOONS])  # noqa: F821
    return grouped
