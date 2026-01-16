import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solar_system.core.celestial_body import CelestialBody, BodyType, OrbitalElements, PhysicalProperties
from solar_system.core.constants import AU
from solar_system.physics.orbital_mechanics import OrbitalMechanics

class TestCelestialBody(unittest.TestCase):
    def test_initialization(self):
        body = CelestialBody(
            name="TestBody",
            body_type=BodyType.PLANET
        )
        self.assertEqual(body.name, "TestBody")
        self.assertEqual(body.body_type, BodyType.PLANET)

    def test_orbital_period(self):
        # Create a mock parent (Sun)
        sun = CelestialBody(name="Sun", body_type=BodyType.STAR)

        # Earth-like elements
        elements = OrbitalElements(
            semi_major_axis=1.0,
            eccentricity=0.0167,
            inclination=0.0,
            longitude_ascending=0.0,
            longitude_perihelion=102.9,
            mean_longitude=100.4
        )

        earth = CelestialBody(
            name="Earth",
            body_type=BodyType.PLANET,
            orbital_elements=elements,
            parent=sun
        )

        # Period should be ~1 year (365.25 days)
        period_days = earth.get_orbital_period_days()
        self.assertAlmostEqual(period_days, 365.25, delta=1.0)

class TestOrbitalMechanics(unittest.TestCase):
    def test_circular_velocity(self):
        # V = sqrt(GM/r)
        mu = 1.0e14
        r = 1.0e7
        v = OrbitalMechanics.circular_velocity(r, mu)
        expected = np.sqrt(mu/r)
        self.assertAlmostEqual(v, expected)

    def test_vis_viva(self):
        # v^2 = GM(2/r - 1/a)
        mu = 1.0e14
        r = 1.0e7
        a = 2.0e7
        v = OrbitalMechanics.vis_viva(r, a, mu)
        expected = np.sqrt(mu * (2/r - 1/a))
        self.assertAlmostEqual(v, expected)

if __name__ == "__main__":
    unittest.main()
