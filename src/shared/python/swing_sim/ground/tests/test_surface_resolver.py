"""Contracts and exact-edge tests for the qualified static plane resolver."""

from __future__ import annotations

from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    PlanarSurfaceDomain,
    SurfaceKinematicSegment,
    SurfaceResolver,
)

from ._support import _surface, _surface_run_request


def test_resolver_requires_the_request_exact_surface_identity_and_geometry() -> None:
    surface = _surface()
    request = _surface_run_request(surface=surface)
    resolver = SurfaceResolver(PlanarSurfaceDomain(surface))

    resolver.validate_request(request)
    with pytest.raises(ValueError, match="provider"):
        SurfaceResolver(
            PlanarSurfaceDomain(replace(surface, provider_version="2.0.0"))
        ).validate_request(request)
    with pytest.raises(ValueError, match="normal"):
        SurfaceResolver(
            PlanarSurfaceDomain(replace(surface, normal_unit=(0.0, 0.8, 0.6)))
        ).validate_request(request)
    with pytest.raises(ValueError, match="material"):
        SurfaceResolver(
            PlanarSurfaceDomain(replace(surface, kinetic_friction=0.1))
        ).validate_request(request)


def test_quadratic_finite_boundary_crossing_is_exact() -> None:
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            _surface(),
            lower_coordinate_m=0.0,
            upper_coordinate_m=3.0,
        )
    )
    crossing = resolver.first_crossing(
        SurfaceKinematicSegment(
            start_position_m=(0.0, 0.02135, 0.0),
            start_velocity_m_s=(2.0, 0.0, 0.0),
            acceleration_m_s2=(2.0, 0.0, 0.0),
            duration_s=1.5,
        )
    )

    assert crossing is not None
    assert crossing.time_offset_s == pytest.approx(1.0, abs=1e-12)
    assert crossing.position_m[0] == pytest.approx(3.0, abs=1e-12)
    assert crossing.boundary_coordinate_m == 3.0


def test_unbounded_plane_has_no_boundary_and_start_must_be_inside_domain() -> None:
    resolver = SurfaceResolver(PlanarSurfaceDomain(_surface()))
    assert (
        resolver.first_crossing(
            SurfaceKinematicSegment(
                (1.0, 0.02135, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                1.0,
            )
        )
        is None
    )

    finite = SurfaceResolver(
        PlanarSurfaceDomain(
            _surface(),
            lower_coordinate_m=0.0,
            upper_coordinate_m=2.0,
        )
    )
    with pytest.raises(ValueError, match="inside"):
        finite.first_crossing(
            SurfaceKinematicSegment(
                (3.0, 0.02135, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                1.0,
            )
        )
