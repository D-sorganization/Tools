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


def get_apollo11_trajectory() -> list[StateVector]:
    """Simplified Apollo 11 trajectory: Earth to Moon and back.

    Approximate Julian dates:
        Launch: 1969-07-16 (JD 2440418.5)
        Lunar orbit insertion: 1969-07-19 (JD 2440421.5)
        Landing: 1969-07-20 (JD 2440422.5)
        Return: 1969-07-24 (JD 2440426.5)
    """
    # Moon is ~0.00257 AU from Earth; positions approximate
    earth_pos = np.array([0.35, 0.93, 0.0]) * AU  # Earth in July 1969
    moon_offset = np.array([0.002, 0.001, 0.0001]) * AU

    waypoints = [
        StateVector(
            earth_pos, np.array([0, 1000, 0]), 2440418.5
        ),  # Launch from Earth
        StateVector(
            earth_pos + moon_offset * 0.5,
            np.array([500, 500, 50]),
            2440420.0,
        ),  # Trans-lunar coast
        StateVector(
            earth_pos + moon_offset,
            np.array([200, 100, 10]),
            2440421.5,
        ),  # Lunar orbit insertion
        StateVector(
            earth_pos + moon_offset,
            np.array([-200, -100, -10]),
            2440423.5,
        ),  # Lunar departure
        StateVector(
            earth_pos + moon_offset * 0.3,
            np.array([-500, -500, -50]),
            2440425.0,
        ),  # Return coast
        StateVector(
            earth_pos, np.array([0, -1000, 0]), 2440426.5
        ),  # Splashdown
    ]
    return waypoints


def get_cassini_trajectory() -> list[StateVector]:
    """Simplified Cassini-Huygens trajectory with gravity assists.

    Approximate Julian dates:
        Launch: 1997-10-15 (JD 2450736.5)
        Venus 1 flyby: 1998-04-26 (JD 2450929.5)
        Venus 2 flyby: 1999-06-24 (JD 2451353.5)
        Earth flyby: 1999-08-18 (JD 2451408.5)
        Jupiter flyby: 2000-12-30 (JD 2451909.5)
        Saturn orbit insertion: 2004-07-01 (JD 2453187.5)
    """
    waypoints = [
        StateVector(
            np.array([1.0, 0.0, 0.0]) * AU, np.array([0, 30000, 0]), 2450736.5
        ),  # Earth launch
        StateVector(
            np.array([0.5, 0.5, 0.01]) * AU, np.array([25000, -15000, 0]), 2450929.5
        ),  # Venus 1
        StateVector(
            np.array([-0.3, 0.6, 0.02]) * AU,
            np.array([-20000, 20000, 0]),
            2451353.5,
        ),  # Venus 2
        StateVector(
            np.array([0.9, -0.4, 0.0]) * AU,
            np.array([10000, 25000, 500]),
            2451408.5,
        ),  # Earth flyby
        StateVector(
            np.array([-4.0, 3.0, 0.1]) * AU,
            np.array([-12000, 8000, 200]),
            2451909.5,
        ),  # Jupiter flyby
        StateVector(
            np.array([-7.0, -6.0, 0.3]) * AU,
            np.array([-5000, -3000, 100]),
            2453187.5,
        ),  # Saturn orbit insertion
    ]
    return waypoints


def get_new_horizons_trajectory() -> list[StateVector]:
    """Simplified New Horizons trajectory: Earth to Jupiter to Pluto.

    Approximate Julian dates:
        Launch: 2006-01-19 (JD 2453754.5)
        Jupiter flyby: 2007-02-28 (JD 2454159.5)
        Pluto flyby: 2015-07-14 (JD 2457217.5)
        Arrokoth flyby: 2019-01-01 (JD 2458484.5)
    """
    waypoints = [
        StateVector(
            np.array([0.6, 0.8, 0.0]) * AU, np.array([0, 35000, 0]), 2453754.5
        ),  # Earth launch
        StateVector(
            np.array([-4.5, -2.5, 0.05]) * AU,
            np.array([-15000, -8000, 300]),
            2454159.5,
        ),  # Jupiter gravity assist
        StateVector(
            np.array([-10.0, -30.0, 5.0]) * AU,
            np.array([-2000, -14000, 1000]),
            2457217.5,
        ),  # Pluto flyby
        StateVector(
            np.array([-12.0, -35.0, 6.0]) * AU,
            np.array([-1800, -13500, 900]),
            2458484.5,
        ),  # Arrokoth (Ultima Thule)
    ]
    return waypoints


def get_curiosity_trajectory() -> list[StateVector]:
    """Simplified Mars Curiosity rover trajectory: Earth to Mars.

    Approximate Julian dates:
        Launch: 2011-11-26 (JD 2455891.5)
        Mars arrival: 2012-08-06 (JD 2456145.5)
    """
    # Earth position in Nov 2011 and Mars position in Aug 2012
    earth_launch = np.array([0.5, -0.85, 0.0]) * AU
    mars_arrival = np.array([1.3, 0.6, 0.02]) * AU
    midpoint = (earth_launch + mars_arrival) / 2 + np.array([0.3, 0.1, 0.005]) * AU

    waypoints = [
        StateVector(
            earth_launch, np.array([5000, 25000, 0]), 2455891.5
        ),  # Earth launch
        StateVector(
            midpoint, np.array([15000, 15000, 100]), 2456018.0
        ),  # Hohmann transfer midpoint
        StateVector(
            mars_arrival, np.array([24000, 0, 0]), 2456145.5
        ),  # Mars arrival
    ]
    return waypoints


def get_pioneer10_trajectory() -> list[StateVector]:
    """Simplified Pioneer 10 trajectory: Earth to Jupiter to interstellar space.

    Approximate Julian dates:
        Launch: 1972-03-03 (JD 2441387.5)
        Jupiter flyby: 1973-12-03 (JD 2442027.5)
        Last contact: 2003-01-23 (JD 2452662.5)
    """
    waypoints = [
        StateVector(
            np.array([-0.9, 0.4, 0.0]) * AU, np.array([0, 30000, 0]), 2441387.5
        ),  # Earth launch
        StateVector(
            np.array([3.0, 4.0, 0.1]) * AU,
            np.array([10000, 8000, 200]),
            2442027.5,
        ),  # Jupiter flyby
        StateVector(
            np.array([30.0, 50.0, 5.0]) * AU,
            np.array([5000, 3000, 500]),
            2446000.0,
        ),  # Deep space (mid-1980s)
        StateVector(
            np.array([60.0, 80.0, 10.0]) * AU,
            np.array([3000, 2000, 300]),
            2452662.5,
        ),  # Last contact
    ]
    return waypoints


FAMOUS_MISSIONS = {
    "Voyager 1": get_voyager1_trajectory,
    "Voyager 2": get_voyager2_trajectory,
    "Apollo 11": get_apollo11_trajectory,
    "Cassini-Huygens": get_cassini_trajectory,
    "New Horizons": get_new_horizons_trajectory,
    "Curiosity": get_curiosity_trajectory,
    "Pioneer 10": get_pioneer10_trajectory,
}
