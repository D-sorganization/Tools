"""
Trajectory Planner
==================

Calculates interplanetary transfer trajectories including:
- Hohmann transfer orbits
- Bi-elliptic transfers
- Launch windows and phase angles
- Delta-v requirements
- Time of flight calculations
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..core.celestial_body import CelestialBody, Spacecraft, StateVector
from ..core.constants import AU, GM, SECONDS_PER_DAY
from .orbital_mechanics import OrbitalMechanics


class TransferType(Enum):
    """Types of orbital transfers."""

    HOHMANN = "hohmann"
    BI_ELLIPTIC = "bi_elliptic"
    FAST_TRANSFER = "fast"
    GRAVITY_ASSIST = "gravity_assist"


@dataclass
class ManeuverNode:
    """
    Represents a single maneuver in a trajectory.

    Attributes:
        time: Julian date of the maneuver
        position: Position at maneuver (meters)
        delta_v: Change in velocity (m/s) as a vector
        delta_v_magnitude: Magnitude of delta-v (m/s)
        description: Human-readable description
    """

    time: float
    position: np.ndarray
    delta_v: np.ndarray
    delta_v_magnitude: float
    description: str

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.delta_v = np.array(self.delta_v, dtype=np.float64)


@dataclass
class TransferTrajectory:
    """
    Complete transfer trajectory between two bodies.

    Attributes:
        origin: Starting celestial body
        destination: Target celestial body
        transfer_type: Type of transfer orbit
        departure_time: Julian date of departure
        arrival_time: Julian date of arrival
        time_of_flight: Duration in days
        total_delta_v: Total delta-v requirement (m/s)
        maneuvers: List of maneuver nodes
        trajectory_points: List of state vectors along the path
        phase_angle: Required phase angle at departure (degrees)
    """

    origin: str
    destination: str
    transfer_type: TransferType
    departure_time: float
    arrival_time: float
    time_of_flight: float
    total_delta_v: float
    maneuvers: list[ManeuverNode]
    trajectory_points: list[StateVector] = field(default_factory=list)
    phase_angle: float = 0.0

    def get_info_dict(self) -> dict[str, Any]:
        """Get formatted information about the transfer."""
        return {
            "Route": f"{self.origin} → {self.destination}",
            "Transfer Type": self.transfer_type.value.replace("_", " ").title(),
            "Time of Flight": (
                f"{self.time_of_flight:.1f} days "
                f"({self.time_of_flight/365.25:.2f} years)"
            ),
            "Total Δv": (
                f"{self.total_delta_v:.1f} m/s " f"({self.total_delta_v/1000:.2f} km/s)"
            ),
            "Phase Angle": f"{self.phase_angle:.1f}°",
            "Maneuvers": len(self.maneuvers),
        }


@dataclass
class LaunchWindow:
    """
    Information about a launch window opportunity.

    Attributes:
        departure_date: Julian date of optimal departure
        arrival_date: Julian date of arrival
        phase_angle: Phase angle at departure (degrees)
        delta_v: Total delta-v requirement (m/s)
        time_of_flight: Transfer duration (days)
    """

    departure_date: float
    arrival_date: float
    phase_angle: float
    delta_v: float
    time_of_flight: float


class TrajectoryPlanner:
    """
    Plans interplanetary trajectories using patched conics approximation.

    This class provides methods to calculate transfer orbits between
    planets, find optimal launch windows, and generate trajectory data
    for visualization.
    """

    def __init__(self, central_body_mu: float = None):
        """
        Initialize the trajectory planner.

        Args:
            central_body_mu: Standard gravitational parameter of the central body.
                           Defaults to Sun's GM.
        """
        self.mu = central_body_mu if central_body_mu is not None else GM["Sun"]

    def hohmann_transfer(
        self, r1: float, r2: float
    ) -> tuple[float, float, float, float]:
        """
        Calculate Hohmann transfer parameters between circular orbits.

        Args:
            r1: Radius of initial orbit (meters)
            r2: Radius of final orbit (meters)

        Returns:
            Tuple of (delta_v1, delta_v2, time_of_flight, transfer_semi_major_axis)
            - delta_v1: Departure burn (m/s)
            - delta_v2: Arrival burn (m/s)
            - time_of_flight: Transfer time (seconds)
            - a_transfer: Semi-major axis of transfer orbit (meters)
        """
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Velocities in circular orbits
        v1_circular = OrbitalMechanics.circular_velocity(r1, self.mu)
        v2_circular = OrbitalMechanics.circular_velocity(r2, self.mu)

        # Velocities in transfer orbit at r1 and r2
        v1_transfer = OrbitalMechanics.vis_viva(r1, a_transfer, self.mu)
        v2_transfer = OrbitalMechanics.vis_viva(r2, a_transfer, self.mu)

        # Delta-v for each burn
        delta_v1 = abs(v1_transfer - v1_circular)
        delta_v2 = abs(v2_circular - v2_transfer)

        # Time of flight (half the transfer orbit period)
        tof = OrbitalMechanics.orbital_period(a_transfer, self.mu) / 2

        return delta_v1, delta_v2, tof, a_transfer

    def hohmann_phase_angle(self, r1: float, r2: float) -> float:
        """
        Calculate the required phase angle for a Hohmann transfer.

        The target should be at this angle ahead of the spacecraft
        at departure time.

        Args:
            r1: Radius of initial orbit (meters)
            r2: Radius of final orbit (meters)

        Returns:
            Phase angle in degrees
        """
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Time of flight
        tof = OrbitalMechanics.orbital_period(a_transfer, self.mu) / 2

        # Angular velocity of target body (assuming circular)
        omega_target = 2 * math.pi / OrbitalMechanics.orbital_period(r2, self.mu)

        # Angle swept by target during transfer
        theta_swept = omega_target * tof

        # Phase angle: target needs to be at π - theta_swept ahead
        phase_angle = math.pi - theta_swept

        return math.degrees(phase_angle)

    def bi_elliptic_transfer(
        self, r1: float, r2: float, r_intermediate: float
    ) -> tuple[float, float, float, float]:
        """
        Calculate bi-elliptic transfer parameters.

        More efficient than Hohmann for large radius ratios (r2/r1 > 11.94).

        Args:
            r1: Radius of initial orbit (meters)
            r2: Radius of final orbit (meters)
            r_intermediate: Apoapsis of first transfer ellipse (meters)

        Returns:
            Tuple of (delta_v1, delta_v2, delta_v3, time_of_flight)
        """
        # First transfer ellipse (r1 to r_intermediate)
        a1 = (r1 + r_intermediate) / 2

        # Second transfer ellipse (r_intermediate to r2)
        a2 = (r_intermediate + r2) / 2

        # Velocities
        v1_circular = OrbitalMechanics.circular_velocity(r1, self.mu)
        v2_circular = OrbitalMechanics.circular_velocity(r2, self.mu)

        # At r1
        v1_transfer = OrbitalMechanics.vis_viva(r1, a1, self.mu)

        # At r_intermediate (from first ellipse)
        v_int_1 = OrbitalMechanics.vis_viva(r_intermediate, a1, self.mu)

        # At r_intermediate (from second ellipse)
        v_int_2 = OrbitalMechanics.vis_viva(r_intermediate, a2, self.mu)

        # At r2
        v2_transfer = OrbitalMechanics.vis_viva(r2, a2, self.mu)

        # Delta-v for each burn
        delta_v1 = abs(v1_transfer - v1_circular)
        delta_v2 = abs(v_int_2 - v_int_1)
        delta_v3 = abs(v2_circular - v2_transfer)

        # Time of flight
        tof1 = OrbitalMechanics.orbital_period(a1, self.mu) / 2
        tof2 = OrbitalMechanics.orbital_period(a2, self.mu) / 2
        total_tof = tof1 + tof2

        return delta_v1, delta_v2, delta_v3, total_tof

    def synodic_period_planets(
        self, origin: CelestialBody, destination: CelestialBody
    ) -> float:
        """
        Calculate the synodic period between two planets.

        Args:
            origin: Origin planet
            destination: Destination planet

        Returns:
            Synodic period in days
        """
        t1 = origin.get_orbital_period()
        t2 = destination.get_orbital_period()

        return OrbitalMechanics.synodic_period(t1, t2) / SECONDS_PER_DAY

    def find_launch_windows(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        start_date: float,
        search_duration_days: float = 1000,
        window_tolerance_deg: float = 5.0,
    ) -> list[LaunchWindow]:
        """
        Find optimal launch windows between two bodies.

        Args:
            origin: Origin celestial body
            destination: Destination celestial body
            start_date: Julian date to start search from
            search_duration_days: Number of days to search
            window_tolerance_deg: Tolerance for phase angle matching (degrees)

        Returns:
            List of launch window opportunities
        """
        windows = []

        # Get approximate orbital radii
        r1 = origin.orbital_elements.semi_major_axis * AU
        r2 = destination.orbital_elements.semi_major_axis * AU

        # Calculate ideal phase angle for Hohmann transfer
        ideal_phase = self.hohmann_phase_angle(r1, r2)

        # Hohmann transfer parameters
        dv1, dv2, tof, _ = self.hohmann_transfer(r1, r2)
        total_dv = dv1 + dv2
        tof_days = tof / SECONDS_PER_DAY

        # Search for windows
        step_days = 1  # Check every day
        current_date = start_date

        while current_date < start_date + search_duration_days:
            # Get positions
            origin_state = origin.get_state_at_time(current_date)
            dest_state = destination.get_state_at_time(current_date)

            # Calculate current phase angle
            phase = OrbitalMechanics.phase_angle(
                origin_state.position, dest_state.position
            )
            phase_deg = math.degrees(phase)

            # Check if phase angle is close to ideal
            angle_diff = abs(phase_deg - ideal_phase)
            # Account for wrap-around
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            if angle_diff < window_tolerance_deg:
                windows.append(
                    LaunchWindow(
                        departure_date=current_date,
                        arrival_date=current_date + tof_days,
                        phase_angle=phase_deg,
                        delta_v=total_dv,
                        time_of_flight=tof_days,
                    )
                )
                # Skip ahead to avoid duplicate windows
                current_date += 30

            current_date += step_days

        return windows

    def calculate_transfer(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        transfer_type: TransferType = TransferType.HOHMANN,
    ) -> TransferTrajectory:
        """
        Calculate a complete transfer trajectory.

        Args:
            origin: Origin celestial body
            destination: Destination celestial body
            departure_date: Julian date of departure
            transfer_type: Type of transfer to calculate

        Returns:
            TransferTrajectory with complete trajectory information
        """
        # Get orbital radii at departure
        origin_state = origin.get_state_at_time(departure_date)
        r1 = np.linalg.norm(origin_state.position)

        # Estimate destination radius (use semi-major axis for planning)
        r2 = destination.orbital_elements.semi_major_axis * AU

        if transfer_type == TransferType.HOHMANN:
            return self._calculate_hohmann(origin, destination, departure_date, r1, r2)
        elif transfer_type == TransferType.BI_ELLIPTIC:
            return self._calculate_bi_elliptic(
                origin, destination, departure_date, r1, r2
            )
        elif transfer_type == TransferType.GRAVITY_ASSIST:
            raise ValueError("Use calculate_gravity_assist to specify an assist body")
        else:
            # Default to Hohmann
            return self._calculate_hohmann(origin, destination, departure_date, r1, r2)

    def _calculate_hohmann(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        r1: float,
        r2: float,
    ) -> TransferTrajectory:
        """Calculate a Hohmann transfer trajectory."""
        # Get transfer parameters
        dv1, dv2, tof, a_transfer = self.hohmann_transfer(r1, r2)
        tof_days = tof / SECONDS_PER_DAY
        arrival_date = departure_date + tof_days

        # Get states
        origin_state = origin.get_state_at_time(departure_date)
        dest_state = destination.get_state_at_time(arrival_date)

        # Calculate phase angle
        dest_state_at_departure = destination.get_state_at_time(departure_date)
        phase = OrbitalMechanics.phase_angle(
            origin_state.position, dest_state_at_departure.position
        )

        # Calculate delta-v direction (prograde at departure)
        v_unit = origin_state.velocity / np.linalg.norm(origin_state.velocity)
        delta_v1_vec = v_unit * dv1 if r2 > r1 else -v_unit * dv1

        # Create maneuver nodes
        maneuvers = [
            ManeuverNode(
                time=departure_date,
                position=origin_state.position,
                delta_v=delta_v1_vec,
                delta_v_magnitude=dv1,
                description=f"Trans-{destination.name} Injection burn",
            ),
            ManeuverNode(
                time=arrival_date,
                position=dest_state.position,
                delta_v=-v_unit * dv2,  # Retrograde at arrival
                delta_v_magnitude=dv2,
                description=f"{destination.name} Orbit Insertion burn",
            ),
        ]

        # Generate trajectory points
        trajectory_points = self._generate_trajectory_points(
            origin_state, a_transfer, r1, r2, departure_date, tof_days
        )

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.HOHMANN,
            departure_time=departure_date,
            arrival_time=arrival_date,
            time_of_flight=tof_days,
            total_delta_v=dv1 + dv2,
            maneuvers=maneuvers,
            trajectory_points=trajectory_points,
            phase_angle=math.degrees(phase),
        )

    def _calculate_bi_elliptic(
        self,
        origin: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        r1: float,
        r2: float,
    ) -> TransferTrajectory:
        """Calculate a bi-elliptic transfer trajectory."""
        # Use intermediate radius 1.5x the larger orbit
        r_intermediate = max(r1, r2) * 1.5

        dv1, dv2, dv3, tof = self.bi_elliptic_transfer(r1, r2, r_intermediate)
        tof_days = tof / SECONDS_PER_DAY
        arrival_date = departure_date + tof_days

        origin_state = origin.get_state_at_time(departure_date)

        # Calculate intermediate maneuver time
        a1 = (r1 + r_intermediate) / 2
        t_intermediate = (
            OrbitalMechanics.orbital_period(a1, self.mu) / 2 / SECONDS_PER_DAY
        )
        intermediate_date = departure_date + t_intermediate

        v_unit = origin_state.velocity / np.linalg.norm(origin_state.velocity)

        maneuvers = [
            ManeuverNode(
                time=departure_date,
                position=origin_state.position,
                delta_v=v_unit * dv1,
                delta_v_magnitude=dv1,
                description="First transfer burn",
            ),
            ManeuverNode(
                time=intermediate_date,
                position=np.array([r_intermediate, 0, 0]),  # Approximate
                delta_v=v_unit * dv2,
                delta_v_magnitude=dv2,
                description="Intermediate plane change",
            ),
            ManeuverNode(
                time=arrival_date,
                position=np.array([r2, 0, 0]),  # Approximate
                delta_v=-v_unit * dv3,
                delta_v_magnitude=dv3,
                description="Orbit insertion",
            ),
        ]

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.BI_ELLIPTIC,
            departure_time=departure_date,
            arrival_time=arrival_date,
            time_of_flight=tof_days,
            total_delta_v=dv1 + dv2 + dv3,
            maneuvers=maneuvers,
            trajectory_points=[],
            phase_angle=0.0,
        )

    def calculate_gravity_assist(
        self,
        origin: CelestialBody,
        assist_body: CelestialBody,
        destination: CelestialBody,
        departure_date: float,
        periapsis_altitude_km: float = 300.0,
    ) -> TransferTrajectory:
        """Plan a patched-conic gravity assist sequence."""

        first_leg = self.calculate_transfer(
            origin, assist_body, departure_date, TransferType.HOHMANN
        )
        assist_arrival = first_leg.arrival_time

        flyby_radius = (assist_body.radius + periapsis_altitude_km) * 1000.0
        flyby_speed = (
            math.sqrt(max(assist_body.gm, 0.0) / flyby_radius)
            if assist_body.gm > 0
            else 0.0
        )
        assist_heliocentric_speed = np.linalg.norm(
            assist_body.get_state_at_time(assist_arrival).velocity
        )

        second_leg = self.calculate_transfer(
            assist_body,
            destination,
            assist_arrival + 0.5,
            TransferType.HOHMANN,
        )

        assist_bonus = flyby_speed + assist_heliocentric_speed * 0.3
        total_delta_v = max(
            first_leg.total_delta_v + second_leg.total_delta_v - assist_bonus, 0.0
        )

        maneuvers = first_leg.maneuvers + second_leg.maneuvers
        trajectory_points = first_leg.trajectory_points + second_leg.trajectory_points

        return TransferTrajectory(
            origin=origin.name,
            destination=destination.name,
            transfer_type=TransferType.GRAVITY_ASSIST,
            departure_time=departure_date,
            arrival_time=second_leg.arrival_time,
            time_of_flight=second_leg.arrival_time - departure_date,
            total_delta_v=total_delta_v,
            maneuvers=maneuvers,
            trajectory_points=trajectory_points,
            phase_angle=second_leg.phase_angle,
        )

    def _generate_trajectory_points(
        self,
        initial_state: StateVector,
        a_transfer: float,
        r_start: float,
        r_end: float,
        start_date: float,
        duration_days: float,
        num_points: int = 100,
    ) -> list[StateVector]:
        """Generate points along a transfer trajectory."""
        points = []

        # Determine orbit parameters
        r_min = min(r_start, r_end)
        r_max = max(r_start, r_end)
        e = OrbitalMechanics.eccentricity_from_apsides(r_min, r_max)

        # Determine anomaly range
        if r_start < r_end:
            # Outward: Periapsis -> Apoapsis
            nu_start = 0.0
            nu_end = math.pi
        else:
            # Inward: Apoapsis -> Periapsis
            nu_start = math.pi
            nu_end = 2 * math.pi

        # Get initial position angle
        pos = initial_state.position
        initial_angle = math.atan2(pos[1], pos[0])

        # Optimization: Precalculate constants
        mu = self.mu
        h = math.sqrt(mu * a_transfer * (1 - e**2))
        n = math.sqrt(mu / a_transfer**3)
        sqrt_1_minus_e = math.sqrt(1 - e)
        sqrt_1_plus_e = math.sqrt(1 + e)
        parameter = a_transfer * (1 - e**2)  # Semi-latus rectum

        # Calculate initial Mean Anomaly
        sin_nu_start_2 = math.sin(nu_start / 2)
        cos_nu_start_2 = math.cos(nu_start / 2)
        E_start = 2 * math.atan2(
            sqrt_1_minus_e * sin_nu_start_2, sqrt_1_plus_e * cos_nu_start_2
        )
        M_start = E_start - e * math.sin(E_start)

        # Precalculate angle offset constants
        # angle = nu + (initial_angle - nu_start)
        phi = initial_angle - nu_start
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        nu_start_2 = nu_start / 2
        nu_range_2 = (nu_end - nu_start) / 2

        # Generate points
        for _ in range(num_points + 1):
            fraction = i / num_points

            # Use half-angle for iteration
            nu_2 = nu_start_2 + nu_range_2 * fraction
            sin_nu_2 = math.sin(nu_2)
            cos_nu_2 = math.cos(nu_2)

            # Double angle formulas for nu
            sin_nu = 2 * sin_nu_2 * cos_nu_2
            cos_nu = cos_nu_2**2 - sin_nu_2**2

            # 1. Radius
            r = parameter / (1 + e * cos_nu)

            # 2. Velocity components
            v_r = mu * e * sin_nu / h
            v_t = h / r

            # 3. Position and Velocity in inertial frame
            # angle = nu + phi
            cos_angle = cos_nu * cos_phi - sin_nu * sin_phi
            sin_angle = sin_nu * cos_phi + cos_nu * sin_phi

            x = r * cos_angle
            y = r * sin_angle
            z = 0.0  # Assuming coplanar

            vx = v_r * cos_angle - v_t * sin_angle
            vy = v_r * sin_angle + v_t * cos_angle
            vz = 0.0

            # 4. Time
            E = 2 * math.atan2(sqrt_1_minus_e * sin_nu_2, sqrt_1_plus_e * cos_nu_2)
            M = E - e * math.sin(E)

            delta_M = M - M_start
            if delta_M < 0:
                delta_M += 2 * math.pi

            dt = delta_M / n
            time = start_date + dt / SECONDS_PER_DAY

            points.append(
                StateVector(
                    position=np.array([x, y, z]),
                    velocity=np.array([vx, vy, vz]),
                    time=time,
                )
            )

        return points

    def create_spacecraft_from_transfer(
        self, trajectory: TransferTrajectory, name: str = "Spacecraft"
    ) -> Spacecraft:
        """
        Create a Spacecraft object from a transfer trajectory.

        Args:
            trajectory: The calculated transfer trajectory
            name: Name for the spacecraft

        Returns:
            Spacecraft with the trajectory set
        """
        spacecraft = Spacecraft(name=name, trajectory=trajectory.trajectory_points)
        return spacecraft

    def get_transfer_summary(
        self, origin: CelestialBody, destination: CelestialBody
    ) -> dict[str, Any]:
        """
        Get a summary of transfer options between two bodies.

        Args:
            origin: Origin body
            destination: Destination body

        Returns:
            Dictionary with transfer information
        """
        r1 = origin.orbital_elements.semi_major_axis * AU
        r2 = destination.orbital_elements.semi_major_axis * AU

        # Hohmann transfer
        dv1, dv2, tof, a_transfer = self.hohmann_transfer(r1, r2)
        phase = self.hohmann_phase_angle(r1, r2)

        # Synodic period
        synodic = self.synodic_period_planets(origin, destination)

        summary = {
            "Route": f"{origin.name} → {destination.name}",
            "Distance": f"{abs(r2-r1)/AU:.2f} AU",
            "Hohmann Transfer": {
                "Departure Δv": f"{dv1:.1f} m/s",
                "Arrival Δv": f"{dv2:.1f} m/s",
                "Total Δv": f"{dv1+dv2:.1f} m/s",
                "Time of Flight": f"{tof/SECONDS_PER_DAY:.1f} days",
                "Phase Angle": f"{phase:.1f}°",
            },
            "Synodic Period": f"{synodic:.1f} days ({synodic/365.25:.2f} years)",
        }

        return summary
