"""Parametrized tests for solar system model components.

Converts repetitive unittest patterns into compact pytest.mark.parametrize
forms for better coverage and readability. (Issue #866)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from solar_system.core.celestial_body import BodyType, CelestialBody, StateVector
from solar_system.core.constants import AU, GM, J2000
from solar_system.data.famous_missions import FAMOUS_MISSIONS
from solar_system.data.historical_events import (
    SPACE_EVENTS,
    get_events_by_category,
    get_events_by_year,
)
from solar_system.physics.orbital_mechanics import OrbitalMechanics

# ---------------------------------------------------------------------------
# Orbital mechanics parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planet_name", "semi_major_au", "expected_period_days", "tolerance"),
    [
        ("Mercury", 0.387, 87.97, 1),
        ("Venus", 0.723, 224.7, 2),
        ("Earth", 1.000, 365.25, 1),
        ("Mars", 1.524, 687.0, 5),
        ("Jupiter", 5.203, 4332.6, 30),
        ("Saturn", 9.537, 10759, 60),
    ],
    ids=["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"],
)
def test_orbital_period_planets(
    planet_name: str,
    semi_major_au: float,
    expected_period_days: float,
    tolerance: float,
) -> None:
    """Verify Kepler's 3rd law orbital periods for multiple planets."""
    a = semi_major_au * AU
    t = OrbitalMechanics.orbital_period(a, GM["Sun"])
    t_days = t / 86400
    assert abs(t_days - expected_period_days) < tolerance, (
        f"{planet_name}: expected ~{expected_period_days}d, got {t_days:.1f}d"
    )


@pytest.mark.parametrize(
    ("r_au", "expected_v_kms", "tolerance"),
    [
        (0.387, 47.87, 1.0),  # Mercury
        (1.000, 29.78, 0.5),  # Earth
        (1.524, 24.13, 0.5),  # Mars
        (5.203, 13.07, 0.5),  # Jupiter
    ],
    ids=["Mercury", "Earth", "Mars", "Jupiter"],
)
def test_circular_velocity(
    r_au: float, expected_v_kms: float, tolerance: float
) -> None:
    """Circular velocity at various orbital radii."""
    r = r_au * AU
    v = OrbitalMechanics.circular_velocity(r, GM["Sun"])
    v_kms = v / 1000
    assert abs(v_kms - expected_v_kms) < tolerance


@pytest.mark.parametrize(
    ("r_au",),
    [(0.387,), (1.0,), (1.524,), (5.203,), (9.537,)],
    ids=["Mercury", "Earth", "Mars", "Jupiter", "Saturn"],
)
def test_escape_velocity_is_sqrt2_circular(r_au: float) -> None:
    """Escape velocity should be sqrt(2) * circular velocity."""
    r = r_au * AU
    v_esc = OrbitalMechanics.escape_velocity(r, GM["Sun"])
    v_circ = OrbitalMechanics.circular_velocity(r, GM["Sun"])
    assert abs(v_esc - v_circ * math.sqrt(2)) < 10  # within 10 m/s


@pytest.mark.parametrize(
    ("r_peri_au", "r_aph_au", "expected_e", "tolerance"),
    [
        (0.983, 1.017, 0.017, 0.002),  # Earth
        (1.381, 1.666, 0.093, 0.005),  # Mars
        (0.307, 0.467, 0.206, 0.005),  # Mercury
    ],
    ids=["Earth", "Mars", "Mercury"],
)
def test_eccentricity_from_apsides(
    r_peri_au: float, r_aph_au: float, expected_e: float, tolerance: float
) -> None:
    """Eccentricity from perihelion/aphelion distances."""
    e = OrbitalMechanics.eccentricity_from_apsides(r_peri_au * AU, r_aph_au * AU)
    assert abs(e - expected_e) < tolerance


# ---------------------------------------------------------------------------
# Kepler's third law ratio consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planet_a", "planet_b"),
    [
        (1.0, 1.524),  # Earth vs Mars
        (1.0, 5.203),  # Earth vs Jupiter
        (0.387, 1.0),  # Mercury vs Earth
        (1.524, 9.537),  # Mars vs Saturn
    ],
    ids=["Earth-Mars", "Earth-Jupiter", "Mercury-Earth", "Mars-Saturn"],
)
def test_kepler_third_law_ratio(planet_a: float, planet_b: float) -> None:
    """T^2/a^3 should be constant for any pair of planets."""
    a1, a2 = planet_a * AU, planet_b * AU
    t1 = OrbitalMechanics.orbital_period(a1, GM["Sun"])
    t2 = OrbitalMechanics.orbital_period(a2, GM["Sun"])
    ratio1 = t1**2 / a1**3
    ratio2 = t2**2 / a2**3
    np.testing.assert_allclose(ratio1, ratio2, rtol=1e-10)


# ---------------------------------------------------------------------------
# CelestialBody type tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body_type",),
    [
        (BodyType.STAR,),
        (BodyType.PLANET,),
        (BodyType.DWARF_PLANET,),
        (BodyType.MOON,),
        (BodyType.ASTEROID,),
        (BodyType.COMET,),
        (BodyType.SPACECRAFT,),
    ],
)
def test_celestial_body_types(body_type: BodyType) -> None:
    """All body types can be instantiated."""
    body = CelestialBody(name=f"Test-{body_type.value}", body_type=body_type)
    assert body.name == f"Test-{body_type.value}"
    assert body.body_type == body_type


# ---------------------------------------------------------------------------
# StateVector tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pos", "vel", "expected_dist"),
    [
        ([3e8, 4e8, 0], [0, 0, 0], 5e8),
        ([1e11, 0, 0], [0, 0, 0], 1e11),
        ([0, 0, 0], [100, 200, 300], 0.0),
        ([1e6, 1e6, 1e6], [0, 0, 0], math.sqrt(3) * 1e6),
    ],
    ids=["3-4-5", "x-axis", "origin", "diagonal"],
)
def test_state_vector_distance(
    pos: list[float], vel: list[float], expected_dist: float
) -> None:
    """State vector distance from origin."""
    state = StateVector(position=pos, velocity=vel, time=J2000)
    np.testing.assert_allclose(state.distance, expected_dist, rtol=1e-10)


# ---------------------------------------------------------------------------
# Historical events database tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category",
    ["mission", "discovery", "observation"],
)
def test_events_by_category(category: str) -> None:
    """Each category should have at least one event."""
    events = get_events_by_category(category)
    assert len(events) > 0, f"No events for category '{category}'"


@pytest.mark.parametrize(
    ("year", "min_count"),
    [
        (1969, 3),  # Apollo 11 year
        (2021, 2),  # Perseverance + Ingenuity + JWST
        (1977, 2),  # Voyager 1 + 2
    ],
    ids=["1969-Apollo", "2021-modern", "1977-Voyager"],
)
def test_events_by_year(year: int, min_count: int) -> None:
    """Years with known major events should have sufficient entries."""
    events = get_events_by_year(year)
    assert len(events) >= min_count


def test_all_events_have_required_fields() -> None:
    """Every event should have year, month, day, title, description, category."""
    required_keys = {"year", "month", "day", "title", "description", "category"}
    for i, event in enumerate(SPACE_EVENTS):
        missing = required_keys - set(event.keys())
        assert not missing, f"Event #{i} ({event.get('title', '?')}) missing: {missing}"


# ---------------------------------------------------------------------------
# Famous missions tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mission_name",
    list(FAMOUS_MISSIONS.keys()),
)
def test_famous_mission_trajectory(mission_name: str) -> None:
    """Each famous mission should produce at least 2 waypoints."""
    traj_func = FAMOUS_MISSIONS[mission_name]
    waypoints = traj_func()
    assert len(waypoints) >= 2, f"{mission_name} has too few waypoints"
    for wp in waypoints:
        assert isinstance(wp, StateVector)
        assert wp.position.shape == (3,)


@pytest.mark.parametrize(
    "mission_name",
    list(FAMOUS_MISSIONS.keys()),
)
def test_famous_mission_time_order(mission_name: str) -> None:
    """Waypoints should be in chronological order."""
    waypoints = FAMOUS_MISSIONS[mission_name]()
    times = [wp.time for wp in waypoints]
    assert times == sorted(times), f"{mission_name} waypoints not in time order"


@pytest.mark.parametrize(
    "mission_name",
    list(FAMOUS_MISSIONS.keys()),
)
def test_famous_mission_metadata_contract(mission_name: str) -> None:
    """Mission entries should remain callable and expose UI metadata."""
    mission = FAMOUS_MISSIONS[mission_name]
    assert callable(mission)
    assert mission.get("description")
    assert mission.get("launch_date")
    assert mission.get("science_highlights")
    assert mission.get("destinations")
