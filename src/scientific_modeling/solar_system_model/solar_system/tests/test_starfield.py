"""Tests for starfield background rendering."""

import numpy as np

from solar_system.data.star_catalog import (
    equatorial_to_cartesian,
    iter_catalog,
    star_count,
)
from solar_system.visualization.starfield import (
    build_star_vertices,
    point_size_from_magnitude,
)


def test_catalog_density_and_unit_vectors() -> None:
    assert star_count() >= 80
    pole_vector = np.array(equatorial_to_cartesian(0.0, 90.0))
    assert np.isclose(np.linalg.norm(pole_vector), 1.0)


def test_vertices_sorted_and_colored() -> None:
    vertices = build_star_vertices(iter_catalog(), radius=10.0)
    magnitudes = [v.magnitude for v in vertices]
    assert magnitudes == sorted(magnitudes)
    assert all(len(v.color) == 3 for v in vertices)
    assert all(np.isclose(np.linalg.norm(v.position), 10.0) for v in vertices)


def test_point_size_scaling() -> None:
    bright = point_size_from_magnitude(-1.0)
    dim = point_size_from_magnitude(5.0)
    assert bright > dim
    assert 1.0 <= bright <= 6.0
    assert 1.0 <= dim <= 6.0
