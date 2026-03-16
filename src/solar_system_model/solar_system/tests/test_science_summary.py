"""Focused tests for educational science summaries in the solar system model."""

from __future__ import annotations

import unittest

from solar_system.core.celestial_body import Planet, Star
from solar_system.core.constants import J2000


class TestScienceSummary(unittest.TestCase):
    """Verify educational overlays expose useful physical and orbital context."""

    def setUp(self) -> None:
        self.sun = Star("Sun")
        self.earth = Planet("Earth", parent=self.sun)

    def test_time_aware_info_includes_nerdy_metrics(self) -> None:
        """Educational info should include live orbital and signal-delay metrics."""
        info = self.earth.get_info_dict_at_time(J2000)

        self.assertEqual(info["Name"], "Earth")
        self.assertIn("Current Speed", info)
        self.assertIn("Light-Time to Sun", info)
        self.assertIn("Specific Orbital Energy", info)
        self.assertIn("Sphere of Influence", info)
        self.assertIn("Distance from Sun", info)


if __name__ == "__main__":
    unittest.main()
