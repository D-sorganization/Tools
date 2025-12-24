"""Comet definitions for long-period and periodic visitors."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.constants import OrbitalElements, PhysicalProperties


@dataclass(frozen=True)
class CometDescriptor:
    name: str
    elements: OrbitalElements
    color: tuple
    properties: PhysicalProperties


COMETS: list[CometDescriptor] = [
    CometDescriptor(
        "1P/Halley",
        OrbitalElements(
            semi_major_axis=17.834,
            eccentricity=0.96714,
            inclination=162.26,
            longitude_ascending=58.420,
            longitude_perihelion=111.332,
            mean_longitude=38.384,
        ),
        (0.75, 0.92, 1.0),
        PhysicalProperties(
            mass=2.2e14,
            radius=11.0,
            density=600.0,
            surface_gravity=0.0003,
            escape_velocity=0.0007,
            rotation_period=52.0,
            axial_tilt=0.0,
            albedo=0.04,
            temperature=50.0,
            color=(0.75, 0.92, 1.0),
        ),
    ),
    CometDescriptor(
        "C/1995 O1 Hale-Bopp",
        OrbitalElements(
            semi_major_axis=186.0,
            eccentricity=0.9951,
            inclination=89.4,
            longitude_ascending=282.47,
            longitude_perihelion=130.59,
            mean_longitude=0.0,
        ),
        (0.82, 0.96, 1.0),
        PhysicalProperties(
            mass=2.2e14,
            radius=30.0,
            density=600.0,
            surface_gravity=0.001,
            escape_velocity=0.002,
            rotation_period=11.3,
            axial_tilt=0.0,
            albedo=0.04,
            temperature=40.0,
            color=(0.82, 0.96, 1.0),
        ),
    ),
    CometDescriptor(
        "2P/Encke",
        OrbitalElements(
            semi_major_axis=2.215,
            eccentricity=0.8483,
            inclination=11.78,
            longitude_ascending=334.57,
            longitude_perihelion=159.00,
            mean_longitude=186.54,
        ),
        (0.70, 0.85, 0.95),
        PhysicalProperties(
            mass=1.7e13,
            radius=2.4,
            density=500.0,
            surface_gravity=0.0001,
            escape_velocity=0.0003,
            rotation_period=11.1,
            axial_tilt=0.0,
            albedo=0.05,
            temperature=60.0,
            color=(0.70, 0.85, 0.95),
        ),
    ),
]
