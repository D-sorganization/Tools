"""Analytic gates for the shared divergence-theorem inertia tensor (C1, #4550).

The math authority is ``shared.python.golf_club.mesh_mass_properties``;
the driver-head case exercises it through the tool-local parametric head
exactly the way UpstreamDrift consumes it through ``vendor/ud-tools``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club.library import get_club
from rate_of_closure.club.parametric_head import build_parametric_head
from shared.python.contracts import PreconditionError
from shared.python.golf_club.mesh_mass_properties import mesh_inertia
from tests.rate_of_closure.test_club_heads import _sphere_mesh

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _box_mesh(extents: tuple[float, float, float], center: np.ndarray) -> np.ndarray:
    """A watertight outward-wound rectangular box."""
    hx, hy, hz = (e / 2.0 for e in extents)
    corners = center + np.array(
        [[sx, sy, sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-hz, hz)]
    )
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles = []
    for a, b, c, d in faces:
        triangles.append(corners[[a, b, c]])
        triangles.append(corners[[a, c, d]])
    return np.asarray(triangles, dtype=np.float64)


class TestAnalyticSolids:
    def test_cube_matches_m_l_squared_over_six_exactly(self) -> None:
        report = mesh_inertia(
            _box_mesh((0.4, 0.4, 0.4), np.zeros(3)), density_kg_m3=1234.5
        )
        expected = report.mass_kg * 0.4**2 / 6.0
        assert report.inertia_array() == pytest.approx(np.eye(3) * expected, rel=1e-12)

    def test_offset_box_matches_closed_form_and_centroid(self) -> None:
        extents = (0.3, 0.1, 0.05)
        center = np.array([1.7, -2.2, 0.9])
        report = mesh_inertia(_box_mesh(extents, center), mass_kg=2.5)
        ax, ay, az = extents
        expected = 2.5 / 12.0 * np.diag([ay**2 + az**2, ax**2 + az**2, ax**2 + ay**2])
        assert report.centroid_m == pytest.approx(tuple(center), abs=1e-12)
        assert report.inertia_array() == pytest.approx(expected, rel=1e-12)

    def test_sphere_matches_two_fifths_m_r_squared(self) -> None:
        radius = 0.11
        offset = np.array([0.3, 0.1, -0.2])
        report = mesh_inertia(_sphere_mesh(radius, bands=64) + offset, mass_kg=0.2)
        expected = 0.4 * 0.2 * radius**2
        for moment in report.principal_moments_kg_m2:
            # Tessellation slightly under-fills the ball; 64 bands ≈ 0.2%.
            assert moment == pytest.approx(expected, rel=5e-3)


class TestCovariance:
    def test_translation_moves_centroid_but_not_the_tensor(self) -> None:
        mesh = _box_mesh((0.2, 0.3, 0.4), np.zeros(3))
        base = mesh_inertia(mesh, density_kg_m3=1000.0)
        moved = mesh_inertia(mesh + np.array([5.0, -3.0, 2.0]), density_kg_m3=1000.0)
        assert moved.centroid_m == pytest.approx((5.0, -3.0, 2.0), abs=1e-10)
        assert moved.inertia_array() == pytest.approx(base.inertia_array(), rel=1e-9)

    def test_rotation_conjugates_the_tensor(self) -> None:
        mesh = _box_mesh((0.2, 0.3, 0.4), np.array([0.5, 0.0, 0.0]))
        angle = 0.7
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        base = mesh_inertia(mesh, mass_kg=1.0)
        rotated = mesh_inertia(mesh @ rot.T, mass_kg=1.0)
        expected = rot @ base.inertia_array() @ rot.T
        assert rotated.inertia_array() == pytest.approx(expected, abs=1e-12)

    def test_density_and_mass_paths_agree(self) -> None:
        mesh = _box_mesh((0.2, 0.2, 0.2), np.zeros(3))
        by_density = mesh_inertia(mesh, density_kg_m3=800.0)
        by_mass = mesh_inertia(mesh, mass_kg=by_density.mass_kg)
        assert by_mass.density_kg_m3 == pytest.approx(800.0, rel=1e-12)
        assert by_mass.inertia_array() == pytest.approx(
            by_density.inertia_array(), rel=1e-12
        )


class TestContracts:
    def test_requires_exactly_one_scale_selector(self) -> None:
        mesh = _box_mesh((0.1, 0.1, 0.1), np.zeros(3))
        with pytest.raises(PreconditionError):
            mesh_inertia(mesh)
        with pytest.raises(PreconditionError):
            mesh_inertia(mesh, density_kg_m3=1000.0, mass_kg=1.0)

    def test_rejects_nonpositive_scales(self) -> None:
        mesh = _box_mesh((0.1, 0.1, 0.1), np.zeros(3))
        with pytest.raises(PreconditionError):
            mesh_inertia(mesh, density_kg_m3=0.0)
        with pytest.raises(PreconditionError):
            mesh_inertia(mesh, mass_kg=float("nan"))

    def test_fails_closed_on_an_open_mesh(self) -> None:
        mesh = _box_mesh((0.1, 0.1, 0.1), np.zeros(3))[:-1]
        with pytest.raises(PreconditionError):
            mesh_inertia(mesh, mass_kg=0.2)


class TestGeneratedDriverHead:
    def test_uniform_driver_envelope_lands_in_the_plausible_band(self) -> None:
        """Solid 200 g driver envelope: below the hollow-shell ~4.5e-4 figure
        but within the same decade — the documented lower-bound proxy."""
        report = mesh_inertia(
            build_parametric_head(get_club("Driver 10.5°")), mass_kg=0.200
        )
        for moment in report.principal_moments_kg_m2:
            assert 5.0e-5 < moment < 1.0e-3
