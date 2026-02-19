import numpy as np

from solar_system.data.asteroids import generate_belt_particles
from solar_system.data.star_catalog import equatorial_to_cartesian, star_count
from solar_system.visualization.renderer import RenderSettings
from solar_system.visualization.scene import SolarSystemScene


def test_star_catalog_generates_unit_vectors() -> None:
    vector = np.array(equatorial_to_cartesian(0.0, 0.0))
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert star_count() >= 80


def test_asteroid_belt_is_deterministic() -> None:
    belt_a = generate_belt_particles(10)
    belt_b = generate_belt_particles(10)
    assert np.allclose(belt_a, belt_b)
    assert belt_a.shape == (10, 3)


def test_scene_builds_moons_and_minors_without_renderer() -> None:
    scene = SolarSystemScene(RenderSettings())
    scene._create_solar_system()

    assert "Io" in scene.moons
    assert len(scene.asteroids) >= 4
    assert len(scene.comets) >= 3


def test_gravity_assist_reduces_delta_v() -> None:
    scene = SolarSystemScene(RenderSettings())
    scene._create_solar_system()

    earth = scene.planets["Earth"]
    mars = scene.planets["Mars"]
    venus = scene.planets["Venus"]
    planner = scene.trajectory_planner

    departure = scene.time_manager.julian_date
    direct = planner.calculate_transfer(earth, mars, departure)
    assist = planner.calculate_gravity_assist(earth, venus, mars, departure)

    assert assist.total_delta_v < direct.total_delta_v
    assert assist.time_of_flight > 0
