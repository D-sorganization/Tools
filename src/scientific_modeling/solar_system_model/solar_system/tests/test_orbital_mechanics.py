"""
Tests for orbital mechanics calculations.

These tests verify the scientific accuracy of the simulation
by comparing calculated values against known astronomical data.
"""

import math
import unittest

import numpy as np
from solar_system.core.celestial_body import Planet, Star, StateVector
from solar_system.core.constants import AU, GM, J2000, PHYSICAL_PROPERTIES
from solar_system.physics.orbital_mechanics import OrbitalMechanics
from solar_system.physics.trajectory_planner import TrajectoryPlanner


class TestOrbitalMechanics(unittest.TestCase):
    """Test orbital mechanics calculations."""

    def test_circular_velocity_earth(self) -> None:
        """Test circular velocity at Earth's orbit."""
        # Earth's average orbital velocity is about 29.78 km/s
        r = 1.0 * AU  # 1 AU
        v = OrbitalMechanics.circular_velocity(r, GM["Sun"])

        # Should be approximately 29.78 km/s
        v_kms = v / 1000
        self.assertAlmostEqual(v_kms, 29.78, delta=0.5)

    def test_escape_velocity_earth_orbit(self) -> None:
        """Test escape velocity at Earth's orbit from the Sun."""
        r = 1.0 * AU
        v_esc = OrbitalMechanics.escape_velocity(r, GM["Sun"])

        # Should be sqrt(2) times circular velocity
        v_circ = OrbitalMechanics.circular_velocity(r, GM["Sun"])
        self.assertAlmostEqual(v_esc, v_circ * math.sqrt(2), delta=1)

    def test_orbital_period_earth(self) -> None:
        """Test Earth's orbital period calculation."""
        a = 1.0 * AU
        t = OrbitalMechanics.orbital_period(a, GM["Sun"])

        # Earth's orbital period is about 365.25 days
        t_days = t / 86400
        self.assertAlmostEqual(t_days, 365.25, delta=1)

    def test_vis_viva_at_perihelion(self) -> None:
        """Test vis-viva equation at perihelion."""
        # For an orbit with a=1 AU and e=0.5
        a = 1.0 * AU
        e = 0.5
        r_peri = a * (1 - e)

        v = OrbitalMechanics.vis_viva(r_peri, a, GM["Sun"])

        # Velocity at perihelion should be higher than circular
        v_circ = OrbitalMechanics.circular_velocity(r_peri, GM["Sun"])
        self.assertGreater(v, v_circ)

    def test_kepler_third_law(self) -> None:
        """Verify Kepler's third law: T² ∝ a³."""
        # Compare Earth and Mars
        a_earth = 1.0 * AU
        a_mars = 1.524 * AU

        t_earth = OrbitalMechanics.orbital_period(a_earth, GM["Sun"])
        t_mars = OrbitalMechanics.orbital_period(a_mars, GM["Sun"])

        # T²/a³ should be constant
        ratio_earth = t_earth**2 / a_earth**3
        ratio_mars = t_mars**2 / a_mars**3

        self.assertAlmostEqual(ratio_earth, ratio_mars, places=10)

    def test_eccentricity_from_apsides(self) -> None:
        """Test eccentricity calculation from apsides."""
        # Earth's orbit: perihelion ~0.983 AU, aphelion ~1.017 AU
        r_peri = 0.983 * AU
        r_aph = 1.017 * AU

        e = OrbitalMechanics.eccentricity_from_apsides(r_peri, r_aph)

        # Earth's eccentricity is about 0.0167
        self.assertAlmostEqual(e, 0.017, delta=0.002)

    def test_sphere_of_influence_earth(self) -> None:
        """Test Earth's sphere of influence calculation."""
        a_earth = 1.0 * AU
        m_earth = PHYSICAL_PROPERTIES["Earth"].mass
        m_sun = PHYSICAL_PROPERTIES["Sun"].mass

        soi = OrbitalMechanics.sphere_of_influence(a_earth, m_earth, m_sun)

        # Earth's SOI is about 925,000 km
        soi_km = soi / 1000
        self.assertAlmostEqual(soi_km, 925000, delta=50000)

    def test_synodic_period_earth_mars(self) -> None:
        """Test synodic period calculation for Earth-Mars."""
        t_earth = 365.25  # days
        t_mars = 687.0  # days

        synodic = OrbitalMechanics.synodic_period(t_earth, t_mars)

        # Earth-Mars synodic period is about 780 days
        self.assertAlmostEqual(synodic, 780, delta=10)

    def test_velocity_at_true_anomaly(self) -> None:
        """Test velocity vector components at true anomaly."""
        # For circular orbit, v_r=0, v_t = v_circ
        r = 1.0 * AU
        a = r
        e = 0.0
        nu = math.pi / 2
        v_r, v_t = OrbitalMechanics.velocity_at_true_anomaly(a, e, nu, GM["Sun"])

        v_circ = OrbitalMechanics.circular_velocity(r, GM["Sun"])
        self.assertAlmostEqual(v_r, 0.0)
        self.assertAlmostEqual(v_t, v_circ)

    def test_time_of_flight_half_orbit(self) -> None:
        """Test time of flight for half orbit."""
        a = 1.0 * AU
        e = 0.0
        nu1 = 0.0
        nu2 = math.pi
        tof = OrbitalMechanics.time_of_flight(nu1, nu2, a, e, GM["Sun"])
        period = OrbitalMechanics.orbital_period(a, GM["Sun"])
        self.assertAlmostEqual(tof, period / 2)


class TestCelestialBodies(unittest.TestCase):
    """Test celestial body classes."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sun = Star("Sun")
        self.earth = Planet("Earth", parent=self.sun)
        self.mars = Planet("Mars", parent=self.sun)

    def test_planet_creation(self) -> None:
        """Test planet is created with correct properties."""
        self.assertEqual(self.earth.name, "Earth")
        self.assertIsNotNone(self.earth.orbital_elements)
        self.assertIsNotNone(self.earth.physical_properties)

    def test_earth_orbital_period(self) -> None:
        """Test Earth's orbital period from orbital elements."""
        period_days = self.earth.get_orbital_period_days()

        # Should be approximately 365.25 days
        self.assertAlmostEqual(period_days, 365.25, delta=1)

    def test_mars_orbital_period(self) -> None:
        """Test Mars's orbital period."""
        period_days = self.mars.get_orbital_period_days()

        # Mars orbital period is about 687 days
        self.assertAlmostEqual(period_days, 687, delta=5)

    def test_earth_position_at_j2000(self) -> None:
        """Test Earth's position at J2000 epoch."""
        state = self.earth.get_state_at_time(J2000)

        # Earth should be about 1 AU from Sun
        distance_au = np.linalg.norm(state.position) / AU
        self.assertAlmostEqual(distance_au, 1.0, delta=0.02)

    def test_planet_position_continuity(self) -> None:
        """Test that planet positions change smoothly over time."""
        jd1 = J2000
        jd2 = J2000 + 1  # 1 day later

        state1 = self.earth.get_state_at_time(jd1)
        state2 = self.earth.get_state_at_time(jd2)

        # Position should change by a small amount (Earth moves ~2.57 million km/day)
        distance_change = np.linalg.norm(state2.position - state1.position)
        distance_change_km = distance_change / 1000

        self.assertAlmostEqual(distance_change_km, 2.57e6, delta=0.5e6)


class TestTrajectoryPlanner(unittest.TestCase):
    """Test trajectory planning calculations."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.planner = TrajectoryPlanner()
        self.sun = Star("Sun")
        self.earth = Planet("Earth", parent=self.sun)
        self.mars = Planet("Mars", parent=self.sun)

    def test_hohmann_earth_mars(self) -> None:
        """Test Hohmann transfer from Earth to Mars."""
        r1 = 1.0 * AU  # Earth orbit
        r2 = 1.524 * AU  # Mars orbit

        dv1, dv2, tof, _ = self.planner.hohmann_transfer(r1, r2)

        # Delta-v should be reasonable values
        # Departure ~2.9 km/s, arrival ~2.6 km/s
        self.assertAlmostEqual(dv1 / 1000, 2.9, delta=0.3)
        self.assertAlmostEqual(dv2 / 1000, 2.6, delta=0.3)

        # Time of flight should be about 259 days
        tof_days = tof / 86400
        self.assertAlmostEqual(tof_days, 259, delta=10)

    def test_hohmann_phase_angle(self) -> None:
        """Test phase angle calculation for Hohmann transfer."""
        r1 = 1.0 * AU
        r2 = 1.524 * AU

        phase = self.planner.hohmann_phase_angle(r1, r2)

        # Phase angle for Earth-Mars should be about 44 degrees
        self.assertAlmostEqual(phase, 44, delta=5)

    def test_synodic_period(self) -> None:
        """Test synodic period between planets."""
        synodic = self.planner.synodic_period_planets(self.earth, self.mars)

        # Earth-Mars synodic period is about 780 days
        self.assertAlmostEqual(synodic, 780, delta=20)


class TestStateVector(unittest.TestCase):
    """Test state vector operations."""

    def test_state_vector_creation(self) -> None:
        """Test state vector initialization."""
        state = StateVector(position=[1e11, 0, 0], velocity=[0, 30000, 0], time=J2000)

        self.assertEqual(state.position[0], 1e11)
        self.assertEqual(state.velocity[1], 30000)

    def test_state_vector_distance(self) -> None:
        """Test distance calculation."""
        state = StateVector(position=[3e8, 4e8, 0], velocity=[0, 0, 0], time=J2000)

        self.assertAlmostEqual(state.distance, 5e8)

    def test_state_vector_copy(self) -> None:
        """Test state vector copy."""
        original = StateVector(position=[1, 2, 3], velocity=[4, 5, 6], time=J2000)

        copy = original.copy()

        # Modify original
        original.position[0] = 100

        # Copy should be unchanged
        self.assertEqual(copy.position[0], 1)


if __name__ == "__main__":
    unittest.main()
