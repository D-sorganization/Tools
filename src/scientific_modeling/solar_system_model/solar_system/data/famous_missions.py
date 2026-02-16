"""
Famous Space Mission Trajectories
=================================

Provides pre-calculated or approximate trajectories for significant
historical space missions.
"""

import numpy as np

from ..core.celestial_body import StateVector
from ..core.constants import AU


def get_voyager1_trajectory() -> list[StateVector]:
    """Simplified Voyager 1 trajectory waypoints."""
    # Approximate Julian dates
    # Launch: 1977-09-05 (JD 2443391.5)
    # Jupiter: 1979-03-05 (JD 2443937.5)
    # Saturn: 1980-11-12 (JD 2444555.5)
    # Interstellar: 2012-08-25 (JD 2456164.5)

    waypoints = [
        StateVector(
            np.array([1.0, 0.0, 0.0]) * AU, np.array([0, 30000, 0]), 2443391.5
        ),  # Earth
        StateVector(
            np.array([-3.0, 4.0, 0.1]) * AU, np.array([-15000, 10000, 500]), 2443937.5
        ),  # Jupiter
        StateVector(
            np.array([-8.0, 6.0, 0.5]) * AU, np.array([-10000, 5000, 1000]), 2444555.5
        ),  # Saturn
        StateVector(
            np.array([-100.0, 50.0, 20.0]) * AU, np.array([0, 0, 0]), 2456164.5
        ),  # Interstellar
    ]
    return waypoints


def get_voyager2_trajectory() -> list[StateVector]:
    """Simplified Voyager 2 trajectory waypoints."""
    # Uranus: 1986-01-24 (JD 2446454.5)
    # Neptune: 1989-08-25 (JD 2447763.5)
    waypoints = [
        StateVector(
            np.array([1.0, 0.0, 0.0]) * AU, np.array([0, 30000, 0]), 2443375.5
        ),  # Earth
        StateVector(
            np.array([-3.5, -4.2, 0.0]) * AU, np.array([0, 0, 0]), 2444063.5
        ),  # Jupiter
        StateVector(
            np.array([-9.0, -2.0, -0.2]) * AU, np.array([0, 0, 0]), 2444843.5
        ),  # Saturn
        StateVector(
            np.array([-18.0, 5.0, -0.5]) * AU, np.array([0, 0, 0]), 2446454.5
        ),  # Uranus
        StateVector(
            np.array([-25.0, 15.0, -1.0]) * AU, np.array([0, 0, 0]), 2447763.5
        ),  # Neptune
    ]
    return waypoints


FAMOUS_MISSIONS = {
    "Voyager 1": get_voyager1_trajectory,
    "Voyager 2": get_voyager2_trajectory,
}
