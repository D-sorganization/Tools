"""
Celestial Body Classes
======================

Defines the core classes for representing celestial bodies in the solar system.
Each body has physical properties, orbital elements, and methods for calculating
positions and velocities at any given time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .constants import (
    AU,
    GM,
    J2000,
    ORBITAL_ELEMENTS,
    PHYSICAL_PROPERTIES,
    SECONDS_PER_DAY,
    OrbitalElements,
    PhysicalProperties,
)


class BodyType(Enum):
    """Classification of celestial body types."""

    STAR = "star"
    PLANET = "planet"
    DWARF_PLANET = "dwarf_planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    COMET = "comet"
    SPACECRAFT = "spacecraft"


@dataclass
class StateVector:
    """
    Represents the complete state of a body at a point in time.

    Attributes:
        position: Position vector [x, y, z] in meters
        velocity: Velocity vector [vx, vy, vz] in m/s
        time: Julian date of this state
    """

    position: np.ndarray
    velocity: np.ndarray
    time: float

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)

    @property
    def distance(self) -> float:
        """Distance from origin in meters."""
        return np.linalg.norm(self.position)

    @property
    def speed(self) -> float:
        """Speed in m/s."""
        return np.linalg.norm(self.velocity)

    @property
    def position_au(self) -> np.ndarray:
        """Position in AU."""
        return self.position / AU

    @property
    def position_km(self) -> np.ndarray:
        """Position in kilometers."""
        return self.position / 1000

    def copy(self) -> StateVector:
        """Create a copy of this state vector."""
        return StateVector(
            position=self.position.copy(), velocity=self.velocity.copy(), time=self.time
        )


class CelestialBody:
    """
    Base class for all celestial bodies.

    Provides common functionality for position calculation, rendering properties,
    and physical characteristics.
    """

    def __init__(
        self,
        name: str,
        body_type: BodyType,
        orbital_elements: OrbitalElements | None = None,
        physical_properties: PhysicalProperties | None = None,
        parent: CelestialBody | None = None,
    ):
        """
        Initialize a celestial body.

        Args:
            name: Display name of the body
            body_type: Classification type
            orbital_elements: Keplerian orbital elements (None for Sun)
            physical_properties: Physical characteristics
            parent: Parent body for orbit (None for Sun, Sun for planets)
        """
        self.name = name
        self.body_type = body_type
        self.orbital_elements = orbital_elements
        self.physical_properties = physical_properties
        self.parent = parent
        self.children: list[CelestialBody] = []

        # State cache
        self._state_cache: dict[float, StateVector] = {}
        self._orbit_points: np.ndarray | None = None

        # Add self as child of parent
        if parent is not None:
            parent.children.append(self)

    @property
    def gm(self) -> float:
        """Standard gravitational parameter (GM) in m³/s²."""
        return GM.get(self.name, 0.0)

    @property
    def mass(self) -> float:
        """Mass in kg."""
        if self.physical_properties:
            return self.physical_properties.mass
        return 0.0

    @property
    def radius(self) -> float:
        """Mean radius in km."""
        if self.physical_properties:
            return self.physical_properties.radius
        return 0.0

    @property
    def color(self) -> tuple[float, float, float]:
        """RGB color tuple for visualization."""
        if self.physical_properties:
            return self.physical_properties.color
        return (1.0, 1.0, 1.0)

    def get_orbital_period(self) -> float:
        """
        Calculate the orbital period in seconds.

        Uses Kepler's third law: T² = (4π²/GM) * a³
        """
        if self.orbital_elements is None or self.parent is None:
            return 0.0

        a = self.orbital_elements.semi_major_axis * AU  # Convert to meters
        parent_gm = self.parent.gm

        if parent_gm <= 0:
            return 0.0

        return 2 * math.pi * math.sqrt(a**3 / parent_gm)

    def get_orbital_period_days(self) -> float:
        """Orbital period in Earth days."""
        return self.get_orbital_period() / SECONDS_PER_DAY

    def get_elements_at_time(self, julian_date: float) -> OrbitalElements:
        """
        Calculate orbital elements at a specific time.

        Applies secular variations to the base J2000.0 elements.

        Args:
            julian_date: Julian date to calculate for

        Returns:
            Orbital elements at the specified time
        """
        if self.orbital_elements is None:
            raise ValueError(f"{self.name} has no orbital elements")

        # Time in Julian centuries from J2000.0
        t_centuries = (julian_date - J2000) / 36525.0

        elem = self.orbital_elements

        return OrbitalElements(
            semi_major_axis=elem.semi_major_axis
            + elem.semi_major_axis_rate * t_centuries,
            eccentricity=elem.eccentricity + elem.eccentricity_rate * t_centuries,
            inclination=elem.inclination + elem.inclination_rate * t_centuries,
            longitude_ascending=elem.longitude_ascending
            + elem.longitude_ascending_rate * t_centuries,
            longitude_perihelion=elem.longitude_perihelion
            + elem.longitude_perihelion_rate * t_centuries,
            mean_longitude=elem.mean_longitude + elem.mean_longitude_rate * t_centuries,
        )

    def get_state_at_time(self, julian_date: float) -> StateVector:
        """
        Calculate the state vector (position and velocity) at a given time.

        Uses Keplerian orbital mechanics to calculate heliocentric position
        and velocity from the orbital elements.

        Args:
            julian_date: Julian date to calculate for

        Returns:
            State vector with position and velocity in heliocentric frame
        """
        # Check cache first
        if julian_date in self._state_cache:
            return self._state_cache[julian_date]

        # Sun is at origin
        if self.orbital_elements is None:
            state = StateVector(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                time=julian_date,
            )
            return state

        # Get elements at this time
        elem = self.get_elements_at_time(julian_date)

        # Convert to radians
        i = math.radians(elem.inclination)
        omega_bar = math.radians(elem.longitude_perihelion)
        ascending_longitude = math.radians(elem.longitude_ascending)
        mean_longitude = math.radians(elem.mean_longitude)

        # Argument of perihelion
        omega = omega_bar - ascending_longitude

        # Mean anomaly
        mean_anomaly = mean_longitude - omega_bar
        mean_anomaly = mean_anomaly % (2 * math.pi)

        # Solve Kepler's equation for eccentric anomaly E
        # M = E - e*sin(E)
        eccentric_anomaly = self._solve_kepler(mean_anomaly, elem.eccentricity)

        # True anomaly
        e = elem.eccentricity
        nu = 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(eccentric_anomaly / 2),
            math.sqrt(1 - e) * math.cos(eccentric_anomaly / 2),
        )

        # Distance from focus
        a = elem.semi_major_axis * AU  # Convert to meters
        r = a * (1 - e * math.cos(eccentric_anomaly))

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Rotation matrices
        # R_z(-ascending_longitude) * R_x(-i) * R_z(-omega)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        cos_ascending = math.cos(ascending_longitude)
        sin_ascending = math.sin(ascending_longitude)
        cos_i = math.cos(i)
        sin_i = math.sin(i)

        # Transform to heliocentric ecliptic coordinates
        x = (cos_omega * cos_ascending - sin_omega * sin_ascending * cos_i) * x_orb + (
            -sin_omega * cos_ascending - cos_omega * sin_ascending * cos_i
        ) * y_orb
        y = (cos_omega * sin_ascending + sin_omega * cos_ascending * cos_i) * x_orb + (
            -sin_omega * sin_ascending + cos_omega * cos_ascending * cos_i
        ) * y_orb
        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

        # Velocity calculation
        # Vis-viva: v² = GM(2/r - 1/a)
        parent_gm = self.parent.gm if self.parent else GM["Sun"]

        # Specific angular momentum
        h = math.sqrt(parent_gm * a * (1 - e**2))

        # Velocity in orbital plane
        vx_orb = -parent_gm / h * math.sin(nu)
        vy_orb = parent_gm / h * (e + math.cos(nu))

        # Transform velocity to heliocentric ecliptic coordinates
        vx = (
            cos_omega * cos_ascending - sin_omega * sin_ascending * cos_i
        ) * vx_orb + (
            -sin_omega * cos_ascending - cos_omega * sin_ascending * cos_i
        ) * vy_orb
        vy = (
            cos_omega * sin_ascending + sin_omega * cos_ascending * cos_i
        ) * vx_orb + (
            -sin_omega * sin_ascending + cos_omega * cos_ascending * cos_i
        ) * vy_orb
        vz = (sin_omega * sin_i) * vx_orb + (cos_omega * sin_i) * vy_orb

        state = StateVector(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            time=julian_date,
        )

        # Cache the result
        self._state_cache[julian_date] = state

        # Limit cache size
        if len(self._state_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._state_cache.keys())[:500]
            for k in oldest_keys:
                del self._state_cache[k]

        return state

    def _solve_kepler(
        self, mean_anomaly: float, eccentricity: float, tolerance: float = 1e-10
    ) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

        Uses Newton-Raphson iteration.

        Args:
            mean_anomaly: Mean anomaly in radians
            eccentricity: Eccentricity
            tolerance: Convergence tolerance

        Returns:
            Eccentric anomaly in radians
        """
        # Initial guess
        eccentric_anomaly = mean_anomaly if eccentricity < 0.8 else math.pi

        # Newton-Raphson iteration
        for _ in range(50):
            f_val = (
                eccentric_anomaly
                - eccentricity * math.sin(eccentric_anomaly)
                - mean_anomaly
            )
            f_prime = 1 - eccentricity * math.cos(eccentric_anomaly)

            delta = f_val / f_prime
            eccentric_anomaly = eccentric_anomaly - delta

            if abs(delta) < tolerance:
                break

        return eccentric_anomaly

    def get_orbit_points(self, julian_date: float, num_points: int = 360) -> np.ndarray:
        """
        Calculate points along the orbit for visualization.

        Args:
            julian_date: Reference Julian date
            num_points: Number of points to calculate

        Returns:
            Array of shape (num_points, 3) with positions in meters
        """
        if self.orbital_elements is None:
            return np.zeros((1, 3))

        # Get elements at this time
        elem = self.get_elements_at_time(julian_date)

        # Convert angles to radians
        i = math.radians(elem.inclination)
        omega_bar = math.radians(elem.longitude_perihelion)
        ascending_longitude = math.radians(elem.longitude_ascending)
        omega = omega_bar - ascending_longitude

        a = elem.semi_major_axis * AU
        e = elem.eccentricity

        # Pre-calculate rotation matrix components
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        cos_ascending = math.cos(ascending_longitude)
        sin_ascending = math.sin(ascending_longitude)
        cos_i = math.cos(i)
        sin_i = math.sin(i)

        points = []
        for j in range(num_points):
            # True anomaly from 0 to 2*pi
            nu = 2 * math.pi * j / num_points

            # Distance from focus
            r = a * (1 - e**2) / (1 + e * math.cos(nu))

            # Position in orbital plane
            x_orb = r * math.cos(nu)
            y_orb = r * math.sin(nu)

            # Transform to heliocentric ecliptic coordinates
            x = (
                cos_omega * cos_ascending - sin_omega * sin_ascending * cos_i
            ) * x_orb + (
                -sin_omega * cos_ascending - cos_omega * sin_ascending * cos_i
            ) * y_orb
            y = (
                cos_omega * sin_ascending + sin_omega * cos_ascending * cos_i
            ) * x_orb + (
                -sin_omega * sin_ascending + cos_omega * cos_ascending * cos_i
            ) * y_orb
            z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

            points.append([x, y, z])

        return np.array(points)

    def clear_cache(self) -> None:
        """Clear the state vector cache."""
        self._state_cache.clear()
        self._orbit_points = None

    def get_info_dict(self) -> dict[str, Any]:
        """
        Get a dictionary of information about this body for display.

        Returns:
            Dictionary with formatted information
        """
        info = {"Name": self.name, "Type": self.body_type.value.title()}

        if self.physical_properties:
            pp = self.physical_properties
            info.update(
                {
                    "Mass": f"{pp.mass:.3e} kg",
                    "Radius": f"{pp.radius:,.0f} km",
                    "Density": f"{pp.density:,.0f} kg/m³",
                    "Surface Gravity": f"{pp.surface_gravity:.2f} m/s²",
                    "Escape Velocity": f"{pp.escape_velocity:.2f} km/s",
                    "Rotation Period": f"{abs(pp.rotation_period):.2f} hours"
                    + (" (retrograde)" if pp.rotation_period < 0 else ""),
                    "Axial Tilt": f"{pp.axial_tilt:.2f}°",
                    "Temperature": f"{pp.temperature} K",
                }
            )

        if self.orbital_elements:
            oe = self.orbital_elements
            info.update(
                {
                    "Semi-major Axis": f"{oe.semi_major_axis:.4f} AU",
                    "Eccentricity": f"{oe.eccentricity:.6f}",
                    "Inclination": f"{oe.inclination:.4f}°",
                    "Orbital Period": f"{self.get_orbital_period_days():.2f} days",
                }
            )

        return info

    def __repr__(self) -> str:
        return f"CelestialBody(name='{self.name}', type={self.body_type.value})"


class Star(CelestialBody):
    """Specialized class for stars (primarily the Sun)."""

    def __init__(
        self, name: str = "Sun", physical_properties: PhysicalProperties | None = None
    ):
        if physical_properties is None:
            physical_properties = PHYSICAL_PROPERTIES.get(name)

        super().__init__(
            name=name,
            body_type=BodyType.STAR,
            orbital_elements=None,
            physical_properties=physical_properties,
            parent=None,
        )

        self.luminosity = 3.828e26  # Watts (for the Sun)
        self.spectral_class = "G2V"


class Planet(CelestialBody):
    """Specialized class for planets."""

    def __init__(
        self,
        name: str,
        parent: CelestialBody,
        orbital_elements: OrbitalElements | None = None,
        physical_properties: PhysicalProperties | None = None,
        is_dwarf: bool = False,
    ):
        if orbital_elements is None:
            orbital_elements = ORBITAL_ELEMENTS.get(name)
        if physical_properties is None:
            physical_properties = PHYSICAL_PROPERTIES.get(name)

        body_type = BodyType.DWARF_PLANET if is_dwarf else BodyType.PLANET

        super().__init__(
            name=name,
            body_type=body_type,
            orbital_elements=orbital_elements,
            physical_properties=physical_properties,
            parent=parent,
        )

        # Planet-specific attributes
        self.has_rings = name in ["Saturn", "Uranus", "Jupiter", "Neptune"]
        self.ring_inner_radius = 0.0
        self.ring_outer_radius = 0.0

        if name == "Saturn":
            self.ring_inner_radius = 66900  # km (D ring inner edge)
            self.ring_outer_radius = 140180  # km (F ring outer edge)
        elif name == "Uranus":
            self.ring_inner_radius = 38000  # km
            self.ring_outer_radius = 51149  # km
        elif name == "Jupiter":
            self.ring_inner_radius = 92000  # km
            self.ring_outer_radius = 226000  # km
        elif name == "Neptune":
            self.ring_inner_radius = 40900  # km
            self.ring_outer_radius = 62930  # km


class Moon(CelestialBody):
    """Specialized class for natural satellites."""

    def __init__(
        self,
        name: str,
        parent: CelestialBody,
        orbital_elements: OrbitalElements,
        physical_properties: PhysicalProperties | None = None,
    ):
        if physical_properties is None:
            physical_properties = PHYSICAL_PROPERTIES.get(name)

        super().__init__(
            name=name,
            body_type=BodyType.MOON,
            orbital_elements=orbital_elements,
            physical_properties=physical_properties,
            parent=parent,
        )

    def get_state_at_time(self, julian_date: float) -> StateVector:
        """
        Calculate state vector relative to parent body.

        For moons, we first calculate the position relative to the parent,
        then add the parent's position to get heliocentric coordinates.
        """
        # Get moon's position relative to parent
        relative_state = super().get_state_at_time(julian_date)

        # Get parent's heliocentric position
        if self.parent:
            parent_state = self.parent.get_state_at_time(julian_date)

            return StateVector(
                position=relative_state.position + parent_state.position,
                velocity=relative_state.velocity + parent_state.velocity,
                time=julian_date,
            )

        return relative_state


class Spacecraft(CelestialBody):
    """
    Class representing a spacecraft with trajectory.

    Unlike natural bodies, spacecraft positions come from a pre-calculated
    trajectory rather than orbital elements.
    """

    def __init__(self, name: str, trajectory: list[StateVector] | None = None) -> None:
        super().__init__(
            name=name,
            body_type=BodyType.SPACECRAFT,
            orbital_elements=None,
            physical_properties=None,
            parent=None,
        )

        self.trajectory = trajectory or []
        self._trajectory_times: np.ndarray | None = None
        self._trajectory_positions: np.ndarray | None = None
        self._trajectory_velocities: np.ndarray | None = None

        if trajectory:
            self._build_trajectory_arrays()

    def _build_trajectory_arrays(self) -> None:
        """Build numpy arrays from trajectory for interpolation using optimized allocation."""
        if not self.trajectory:
            return

        # Pre-allocate arrays for better performance (2x speedup)
        n_points = len(self.trajectory)
        self._trajectory_times = np.empty(n_points, dtype=np.float64)
        self._trajectory_positions = np.empty((n_points, 3), dtype=np.float64)
        self._trajectory_velocities = np.empty((n_points, 3), dtype=np.float64)

        # Fill arrays directly to avoid list comprehension overhead
        for i, state in enumerate(self.trajectory):
            self._trajectory_times[i] = state.time
            self._trajectory_positions[i] = state.position
            self._trajectory_velocities[i] = state.velocity

    def set_trajectory(self, trajectory: list[StateVector]) -> None:
        """Set or update the spacecraft trajectory."""
        self.trajectory = trajectory
        self._build_trajectory_arrays()

    def get_state_at_time(self, julian_date: float) -> StateVector:
        """
        Get spacecraft state at a given time by interpolation.

        Args:
            julian_date: Julian date to query

        Returns:
            Interpolated state vector
        """
        if not self.trajectory or self._trajectory_times is None:
            return StateVector(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                time=julian_date,
            )

        times = self._trajectory_times

        # Check bounds
        if julian_date <= times[0]:
            return self.trajectory[0].copy()
        if julian_date >= times[-1]:
            return self.trajectory[-1].copy()

        # Linear interpolation
        idx = np.searchsorted(times, julian_date)
        t0, t1 = times[idx - 1], times[idx]
        frac = (julian_date - t0) / (t1 - t0)

        pos = (
            self._trajectory_positions[idx - 1] * (1 - frac)
            + self._trajectory_positions[idx] * frac
        )
        vel = (
            self._trajectory_velocities[idx - 1] * (1 - frac)
            + self._trajectory_velocities[idx] * frac
        )

        return StateVector(position=pos, velocity=vel, time=julian_date)

    def get_trajectory_duration(self) -> float:
        """Get trajectory duration in days."""
        if not self.trajectory or len(self.trajectory) < 2:
            return 0.0
        return self.trajectory[-1].time - self.trajectory[0].time

    @property
    def color(self) -> tuple[float, float, float]:
        """Spacecraft color for visualization."""
        return (0.0, 1.0, 0.5)  # Bright green
