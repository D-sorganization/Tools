"""
Orbital Mechanics Engine
========================

Provides calculations for orbital mechanics including:
- Orbital element conversions
- Vis-viva equation
- Sphere of influence calculations
- Synodic periods
- Phase angles
- Energy and angular momentum
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalParameters:
    """Computed orbital parameters for a body or transfer."""

    semi_major_axis: float  # meters
    eccentricity: float
    periapsis: float  # meters
    apoapsis: float  # meters
    period: float  # seconds
    specific_energy: float  # J/kg
    angular_momentum: float  # m²/s
    velocity_periapsis: float  # m/s
    velocity_apoapsis: float  # m/s


class OrbitalMechanics:
    """
    Static utility class for orbital mechanics calculations.

    All methods are designed to be physically accurate and use SI units
    unless otherwise specified.
    """

    @staticmethod
    def vis_viva(r: float, a: float, mu: float) -> float:
        """
        Calculate orbital velocity using the vis-viva equation.

        v² = μ(2/r - 1/a)

        Args:
            r: Current distance from central body (meters)
            a: Semi-major axis (meters)
            mu: Standard gravitational parameter of central body (m³/s²)

        Returns:
            Orbital velocity in m/s
        """
        return float(math.sqrt(mu * (2.0 / r - 1.0 / a)))

    @staticmethod
    def orbital_period(a: float, mu: float) -> float:
        """
        Calculate orbital period from semi-major axis.

        T = 2π√(a³/μ)

        Args:
            a: Semi-major axis (meters)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Orbital period in seconds
        """
        return 2 * math.pi * math.sqrt(a**3 / mu)

    @staticmethod
    def semi_major_axis_from_period(period_seconds: float, mu: float) -> float:
        """
        Calculate semi-major axis from orbital period.

        a = (μT²/4π²)^(1/3)

        Args:
            period_seconds: Orbital period (seconds)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Semi-major axis in meters
        """
        return float((mu * period_seconds**2 / (4 * math.pi**2)) ** (1 / 3))

    @staticmethod
    def specific_orbital_energy(a: float, mu: float) -> float:
        """
        Calculate specific orbital energy.

        ε = -μ/2a

        Args:
            a: Semi-major axis (meters)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Specific orbital energy in J/kg
        """
        return -mu / (2 * a)

    @staticmethod
    def specific_angular_momentum(a: float, e: float, mu: float) -> float:
        """
        Calculate specific angular momentum.

        h = √(μa(1-e²))

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Specific angular momentum in m²/s
        """
        return math.sqrt(mu * a * (1 - e**2))

    @staticmethod
    def escape_velocity(r: float, mu: float) -> float:
        """
        Calculate escape velocity at a given distance.

        v_esc = √(2μ/r)

        Args:
            r: Distance from central body (meters)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Escape velocity in m/s
        """
        return math.sqrt(2 * mu / r)

    @staticmethod
    def circular_velocity(r: float, mu: float) -> float:
        """
        Calculate circular orbital velocity at a given distance.

        v_circ = √(μ/r)

        Args:
            r: Orbital radius (meters)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Circular velocity in m/s
        """
        return math.sqrt(mu / r)

    @staticmethod
    def periapsis_apoapsis(a: float, e: float) -> tuple[float, float]:
        """
        Calculate periapsis and apoapsis distances.

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity

        Returns:
            Tuple of (periapsis, apoapsis) in meters
        """
        if not (a is not None):
            raise ValueError("a must be provided")
        periapsis = a * (1 - e)
        apoapsis = a * (1 + e)
        return periapsis, apoapsis

    @staticmethod
    def eccentricity_from_apsides(periapsis: float, apoapsis: float) -> float:
        """
        Calculate eccentricity from periapsis and apoapsis.

        e = (ra - rp) / (ra + rp)

        Args:
            periapsis: Periapsis distance (meters)
            apoapsis: Apoapsis distance (meters)

        Returns:
            Eccentricity
        """
        return (apoapsis - periapsis) / (apoapsis + periapsis)

    @staticmethod
    def semi_major_axis_from_apsides(periapsis: float, apoapsis: float) -> float:
        """
        Calculate semi-major axis from periapsis and apoapsis.

        a = (ra + rp) / 2

        Args:
            periapsis: Periapsis distance (meters)
            apoapsis: Apoapsis distance (meters)

        Returns:
            Semi-major axis in meters
        """
        return (apoapsis + periapsis) / 2

    @staticmethod
    def sphere_of_influence(a: float, m_body: float, m_central: float) -> float:
        """
        Calculate the sphere of influence radius.

        r_SOI ≈ a(m_body/m_central)^(2/5)

        Args:
            a: Semi-major axis of body's orbit around central body (meters)
            m_body: Mass of the body (kg)
            m_central: Mass of the central body (kg)

        Returns:
            Sphere of influence radius in meters
        """
        return float(a * (m_body / m_central) ** 0.4)

    @staticmethod
    def synodic_period(period_one: float, period_two: float) -> float:
        """
        Calculate synodic period between two bodies.

        1/P_syn = |1/T1 - 1/T2|

        Args:
            period_one: Orbital period of first body (any time unit)
            period_two: Orbital period of second body (same time unit)

        Returns:
            Synodic period in the same time unit
        """
        if not (period_one is not None):
            raise ValueError("period_one must be provided")
        if period_one == period_two:
            return float("inf")
        return abs(1 / (1 / period_one - 1 / period_two))

    @staticmethod
    def phase_angle(
        r1: np.ndarray, r2: np.ndarray, reference_up: np.ndarray | None = None
    ) -> float:
        """
        Calculate the phase angle between two bodies as seen from the Sun.

        Args:
            r1: Position vector of first body (meters)
            r2: Position vector of second body (meters)
            reference_up: Up vector for determining sign (default: +Z)

        Returns:
            Phase angle in radians (-π to π)
        """
        if not (r1 is not None):
            raise ValueError("r1 must be provided")
        if reference_up is None:
            reference_up = np.array([0, 0, 1])

        # Normalize position vectors
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)

        # Calculate angle using dot product
        dot = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        angle = math.acos(dot)

        # Determine sign using cross product
        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, reference_up) < 0:
            angle = -angle

        return angle

    @staticmethod
    def true_anomaly_from_eccentric(eccentric_anomaly: float, e: float) -> float:
        """
        Convert eccentric anomaly to true anomaly.

        tan(ν/2) = √((1+e)/(1-e)) * tan(E/2)

        Args:
            eccentric_anomaly: Eccentric anomaly (radians)
            e: Eccentricity

        Returns:
            True anomaly in radians
        """
        return 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(eccentric_anomaly / 2),
            math.sqrt(1 - e) * math.cos(eccentric_anomaly / 2),
        )

    @staticmethod
    def eccentric_anomaly_from_true(nu: float, e: float) -> float:
        """
        Convert true anomaly to eccentric anomaly.

        tan(E/2) = √((1-e)/(1+e)) * tan(ν/2)

        Args:
            nu: True anomaly (radians)
            e: Eccentricity

        Returns:
            Eccentric anomaly in radians
        """
        return 2 * math.atan2(
            math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2)
        )

    @staticmethod
    def mean_anomaly_from_eccentric(eccentric_anomaly: float, e: float) -> float:
        """
        Convert eccentric anomaly to mean anomaly (Kepler's equation).

        M = E - e*sin(E)

        Args:
            eccentric_anomaly: Eccentric anomaly (radians)
            e: Eccentricity

        Returns:
            Mean anomaly in radians
        """
        return eccentric_anomaly - e * math.sin(eccentric_anomaly)

    @staticmethod
    def time_of_flight(nu1: float, nu2: float, a: float, e: float, mu: float) -> float:
        """
        Calculate time of flight between two true anomalies.

        Args:
            nu1: Initial true anomaly (radians)
            nu2: Final true anomaly (radians)
            a: Semi-major axis (meters)
            e: Eccentricity
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Time of flight in seconds
        """
        # Convert to eccentric anomalies
        if not (nu1 is not None):
            raise ValueError("nu1 must be provided")
        eccentric_anomaly_start = OrbitalMechanics.eccentric_anomaly_from_true(nu1, e)
        eccentric_anomaly_end = OrbitalMechanics.eccentric_anomaly_from_true(nu2, e)

        # Convert to mean anomalies
        mean_anomaly_start = OrbitalMechanics.mean_anomaly_from_eccentric(
            eccentric_anomaly_start, e
        )
        mean_anomaly_end = OrbitalMechanics.mean_anomaly_from_eccentric(
            eccentric_anomaly_end, e
        )

        # Handle wrap-around
        delta_mean_anomaly = mean_anomaly_end - mean_anomaly_start
        if delta_mean_anomaly < 0:
            delta_mean_anomaly += 2 * math.pi

        # Mean motion
        n = math.sqrt(mu / a**3)

        return delta_mean_anomaly / n

    @staticmethod
    def radius_at_true_anomaly(a: float, e: float, nu: float) -> float:
        """
        Calculate orbital radius at a given true anomaly.

        r = a(1-e²) / (1 + e*cos(ν))

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity
            nu: True anomaly (radians)

        Returns:
            Orbital radius in meters
        """
        return a * (1 - e**2) / (1 + e * math.cos(nu))

    @staticmethod
    def velocity_at_true_anomaly(
        a: float, e: float, nu: float, mu: float
    ) -> tuple[float, float]:
        """
        Calculate radial and tangential velocity components at true anomaly.

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity
            nu: True anomaly (radians)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Tuple of (radial velocity, tangential velocity) in m/s
        """
        if not (a is not None):
            raise ValueError("a must be provided")
        h = OrbitalMechanics.specific_angular_momentum(a, e, mu)
        r = OrbitalMechanics.radius_at_true_anomaly(a, e, nu)

        v_r = mu * e * math.sin(nu) / h  # Radial
        v_t = h / r  # Tangential

        return v_r, v_t

    @staticmethod
    def orbital_parameters(a: float, e: float, mu: float) -> OrbitalParameters:
        """
        Calculate comprehensive orbital parameters.

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            OrbitalParameters dataclass with all computed values
        """
        if not (a is not None):
            raise ValueError("a must be provided")
        periapsis, apoapsis = OrbitalMechanics.periapsis_apoapsis(a, e)

        return OrbitalParameters(
            semi_major_axis=a,
            eccentricity=e,
            periapsis=periapsis,
            apoapsis=apoapsis,
            period=OrbitalMechanics.orbital_period(a, mu),
            specific_energy=OrbitalMechanics.specific_orbital_energy(a, mu),
            angular_momentum=OrbitalMechanics.specific_angular_momentum(a, e, mu),
            velocity_periapsis=OrbitalMechanics.vis_viva(periapsis, a, mu),
            velocity_apoapsis=OrbitalMechanics.vis_viva(apoapsis, a, mu),
        )

    @staticmethod
    def state_to_elements(
        position: np.ndarray, velocity: np.ndarray, mu: float
    ) -> dict:
        """
        Convert state vectors to orbital elements.

        Args:
            position: Position vector [x, y, z] in meters
            velocity: Velocity vector [vx, vy, vz] in m/s
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Dictionary with orbital elements
        """
        if not (position is not None):
            raise ValueError("position must be provided")
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)

        # Specific angular momentum
        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)

        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)

        # Eccentricity vector
        e_vec = (
            (v**2 - mu / r) * position - np.dot(position, velocity) * velocity
        ) / mu
        e = np.linalg.norm(e_vec)

        # Specific energy
        energy = v**2 / 2 - mu / r

        # Semi-major axis
        a = float("inf") if abs(e - 1.0) < 1e-10 else -mu / (2 * energy)

        # Inclination
        i = math.acos(h_vec[2] / h)

        # Right ascension of ascending node
        if n > 1e-10:
            ascending_node = math.acos(n_vec[0] / n)
            if n_vec[1] < 0:
                ascending_node = 2 * math.pi - ascending_node
        else:
            ascending_node = 0.0

        # Argument of periapsis
        if n > 1e-10 and e > 1e-10:
            omega = math.acos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                omega = 2 * math.pi - omega
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-10:
            nu = math.acos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * math.pi - nu
        else:
            nu = 0.0

        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": math.degrees(i),
            "longitude_ascending": math.degrees(ascending_node),
            "argument_periapsis": math.degrees(omega),
            "true_anomaly": math.degrees(nu),
            "specific_energy": energy,
            "angular_momentum": h,
        }

    @staticmethod
    def elements_to_state(
        a: float,
        e: float,
        i: float,
        ascending_node: float,
        omega: float,
        nu: float,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to state vectors.

        Args:
            a: Semi-major axis (meters)
            e: Eccentricity
            i: Inclination (radians)
            ascending_node: Longitude of ascending node (radians)
            omega: Argument of periapsis (radians)
            nu: True anomaly (radians)
            mu: Standard gravitational parameter (m³/s²)

        Returns:
            Tuple of (position, velocity) vectors in meters and m/s
        """
        # Distance
        if not (a is not None):
            raise ValueError("a must be provided")
        r = a * (1 - e**2) / (1 + e * math.cos(nu))

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Velocity in orbital plane
        h = math.sqrt(mu * a * (1 - e**2))
        vx_orb = -mu / h * math.sin(nu)
        vy_orb = mu / h * (e + math.cos(nu))

        # Rotation matrices
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        cos_ascending = math.cos(ascending_node)
        sin_ascending = math.sin(ascending_node)
        cos_i = math.cos(i)
        sin_i = math.sin(i)

        # Transform to inertial frame
        x = (cos_omega * cos_ascending - sin_omega * sin_ascending * cos_i) * x_orb + (
            -sin_omega * cos_ascending - cos_omega * sin_ascending * cos_i
        ) * y_orb
        y = (cos_omega * sin_ascending + sin_omega * cos_ascending * cos_i) * x_orb + (
            -sin_omega * sin_ascending + cos_omega * cos_ascending * cos_i
        ) * y_orb
        z = (sin_omega * sin_i) * x_orb + (cos_omega * sin_i) * y_orb

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

        return np.array([x, y, z]), np.array([vx, vy, vz])
