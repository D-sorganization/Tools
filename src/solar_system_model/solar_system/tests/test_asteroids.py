"""Behavioral contracts for asteroid catalog and belt particle generation."""

from __future__ import annotations

import math

import numpy as np
from solar_system.data.asteroids import MAJOR_ASTEROIDS, generate_belt_particles


def test_generate_belt_particles_default_ring_is_finite_and_bounded() -> None:
    positions = generate_belt_particles()

    assert positions.shape == (720, 3)
    assert np.isfinite(positions).all()
    np.testing.assert_array_equal(positions[:, 1], np.zeros(720))
    radii = np.hypot(positions[:, 0], positions[:, 2])
    np.testing.assert_allclose(radii, np.linspace(2.1, 3.2, 720))


def test_generate_belt_particles_small_counts_have_expected_positions() -> None:
    positions = generate_belt_particles(count=4)

    expected = np.array(
        [
            [2.1, 0.0, 0.0],
            [0.0, 0.0, 2.1 + 1.1 / 3],
            [-(2.1 + 2 * 1.1 / 3), 0.0, 0.0],
            [0.0, 0.0, -3.2],
        ]
    )
    np.testing.assert_allclose(positions, expected, atol=1e-12)
    np.testing.assert_array_equal(positions, generate_belt_particles(count=4))


def test_generate_belt_particles_handles_empty_and_single_particle_counts() -> None:
    assert generate_belt_particles(count=0).shape == (0, 3)
    np.testing.assert_allclose(
        generate_belt_particles(count=1), np.array([[2.1, 0.0, 0.0]])
    )


def test_major_asteroid_catalog_has_unique_valid_entries() -> None:
    names = [asteroid.name for asteroid in MAJOR_ASTEROIDS]
    assert len(names) == len(set(names))

    for asteroid in MAJOR_ASTEROIDS:
        assert asteroid.name.strip()
        assert math.isfinite(asteroid.elements.semi_major_axis)
        assert asteroid.elements.semi_major_axis > 0
        assert 0 <= asteroid.elements.eccentricity < 1
        assert asteroid.properties.mass > 0
        assert asteroid.properties.radius > 0
        assert asteroid.properties.density > 0
        assert all(0 <= component <= 1 for component in asteroid.color)
