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
        StateVector(earth_pos, np.array([0, 1000, 0]), 2440418.5),  # Launch from Earth
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
        StateVector(earth_pos, np.array([0, -1000, 0]), 2440426.5),  # Splashdown
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
        StateVector(mars_arrival, np.array([24000, 0, 0]), 2456145.5),  # Mars arrival
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


def get_mariner2_trajectory() -> list[StateVector]:
    """Simplified Mariner 2 trajectory: Earth to Venus flyby.

    Approximate Julian dates:
        Launch: 1962-08-27 (JD 2437894.5)
        Venus flyby: 1962-12-14 (JD 2438012.5)
    """
    earth_launch = np.array([0.9, 0.45, 0.0]) * AU
    venus_flyby = np.array([0.65, -0.3, 0.01]) * AU

    waypoints = [
        StateVector(earth_launch, np.array([-15000, 25000, 0]), 2437894.5),
        StateVector(venus_flyby, np.array([-22000, -10000, 200]), 2438012.5),
        StateVector(venus_flyby * 1.5, np.array([-20000, -15000, 300]), 2438100.0),
    ]
    return waypoints


def get_mariner10_trajectory() -> list[StateVector]:
    """Simplified Mariner 10 trajectory: Earth to Venus to Mercury.

    Approximate Julian dates:
        Launch: 1973-11-03 (JD 2441989.5)
        Venus flyby: 1974-02-05 (JD 2442083.5)
        Mercury 1: 1974-03-29 (JD 2442135.5)
        Mercury 2: 1974-09-21 (JD 2442311.5)
        Mercury 3: 1975-03-16 (JD 2442487.5)
    """
    earth_launch = np.array([0.15, 0.98, 0.0]) * AU
    venus_flyby = np.array([-0.68, 0.25, 0.02]) * AU
    mercury_1 = np.array([-0.35, -0.15, 0.01]) * AU
    mercury_2 = np.array([0.38, 0.1, -0.01]) * AU
    mercury_3 = np.array([-0.3, 0.25, 0.02]) * AU

    waypoints = [
        StateVector(earth_launch, np.array([-30000, 5000, 0]), 2441989.5),
        StateVector(venus_flyby, np.array([-10000, -25000, 500]), 2442083.5),
        StateVector(mercury_1, np.array([30000, -15000, -200]), 2442135.5),
        StateVector(mercury_2, np.array([-25000, 30000, 300]), 2442311.5),
        StateVector(mercury_3, np.array([35000, -10000, -400]), 2442487.5),
    ]
    return waypoints


def get_galileo_trajectory() -> list[StateVector]:
    """Simplified Galileo trajectory: complex gravity assists.

    Approximate Julian dates:
        Launch: 1989-10-18 (JD 2447817.5)
        Venus: 1990-02-10 (JD 2447932.5)
        Earth 1: 1990-12-08 (JD 2448233.5)
        Earth 2: 1992-12-08 (JD 2448964.5)
        Jupiter: 1995-12-07 (JD 2450058.5)
    """
    waypoints = [
        StateVector(
            np.array([-0.9, -0.3, 0.0]) * AU, np.array([0, 30000, 0]), 2447817.5
        ),  # Earth launch
        StateVector(
            np.array([0.4, 0.6, 0.01]) * AU, np.array([30000, -10000, 0]), 2447932.5
        ),  # Venus
        StateVector(
            np.array([0.2, 0.95, 0.0]) * AU, np.array([-30000, 10000, 0]), 2448233.5
        ),  # Earth 1
        StateVector(
            np.array([1.0, 0.1, 0.0]) * AU, np.array([-5000, 30000, 0]), 2448964.5
        ),  # Earth 2
        StateVector(
            np.array([3.0, -4.0, 0.1]) * AU, np.array([10000, 5000, 100]), 2450058.5
        ),  # Jupiter
    ]
    return waypoints


FAMOUS_MISSIONS = {
    "Voyager 1": {
        "get_trajectory": get_voyager1_trajectory,
        "description": "Explored Jupiter and Saturn. First spacecraft to enter interstellar space.",
        "launch_date": "1977-09-05",
    },
    "Voyager 2": {
        "get_trajectory": get_voyager2_trajectory,
        "description": "Only spacecraft to visit Uranus and Neptune. Utilized a rare planetary alignment.",
        "launch_date": "1977-08-20",
    },
    "Apollo 11": {
        "get_trajectory": get_apollo11_trajectory,
        "description": "First crewed lunar landing. Neil Armstrong and Buzz Aldrin walked on the Moon.",
        "launch_date": "1969-07-16",
    },
    "Cassini-Huygens": {
        "get_trajectory": get_cassini_trajectory,
        "description": "Studied Saturn and its moons for 13 years. Deployed Huygens probe to Titan.",
        "launch_date": "1997-10-15",
    },
    "New Horizons": {
        "get_trajectory": get_new_horizons_trajectory,
        "description": "First mission to Pluto and the Kuiper Belt. Fastest spacecraft ever launched.",
        "launch_date": "2006-01-19",
    },
    "Curiosity": {
        "get_trajectory": get_curiosity_trajectory,
        "description": "Mars Science Laboratory rover. Searching for past habitable environments on Mars.",
        "launch_date": "2011-11-26",
    },
    "Pioneer 10": {
        "get_trajectory": get_pioneer10_trajectory,
        "description": "First spacecraft to travel through the asteroid belt and visit Jupiter.",
        "launch_date": "1972-03-03",
    },
    "Mariner 2": {
        "get_trajectory": get_mariner2_trajectory,
        "description": "First successful planetary flyby (Venus). Confirmed the planet's high surface temperature.",
        "launch_date": "1962-08-27",
    },
    "Mariner 10": {
        "get_trajectory": get_mariner10_trajectory,
        "description": "First to use gravity assist (Venus to Mercury). Visited Mercury three times.",
        "launch_date": "1973-11-03",
    },
    "Galileo": {
        "get_trajectory": get_galileo_trajectory,
        "description": "Complex mission to Jupiter using Venus and Earth gravity assists.",
        "launch_date": "1989-10-18",
    },
}
