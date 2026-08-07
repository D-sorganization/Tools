"""Analytic contracts for three-dimensional target geometry (#4192)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
    TargetPoint,
    TargetRegion,
)

pytestmark = pytest.mark.physics


def test_target_point_round_trips_through_existing_flight_adapter() -> None:
    flight_point = (137.5, 3.25, 24.25)
    point = TargetPoint.from_frame(flight_point, source_frame="flight")

    assert point.app_coordinates_m == pytest.approx((137.5, 24.25, -3.25))
    assert point.coordinates_in("flight") == pytest.approx(flight_point)
    assert point.source_frame == "flight"


@pytest.mark.parametrize(
    ("source_frame", "coordinates", "error_type", "message"),
    [
        ("world", (1.0, 2.0, 3.0), ValueError, "source_frame"),
        ("app", (1.0, 2.0), ValueError, "three coordinates"),
        ("app", (1.0, math.nan, 3.0), ValueError, "finite"),
        ("app", "1,2,3", TypeError, "coordinates_m"),
    ],
)
def test_target_point_rejects_invalid_frame_or_coordinates(
    source_frame: str,
    coordinates: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        TargetPoint.from_frame(coordinates, source_frame=source_frame)  # type: ignore[arg-type]


def test_box_closest_point_has_signed_three_axis_miss_vector() -> None:
    target = SpatialTarget(
        label="Apex Gate",
        kind="aerial_waypoint",
        point=TargetPoint(100.0, 20.0, -5.0),
        tolerance=BoxTolerance((2.0, 3.0, 4.0)),
        elevation_source="absolute",
    )

    positive = target.miss((105.0, 25.0, 2.0))
    assert positive.closest_point_m == pytest.approx((102.0, 23.0, -1.0))
    assert positive.vector_m == pytest.approx((3.0, 2.0, 3.0))
    assert positive.downrange_m == pytest.approx(3.0)
    assert positive.elevation_m == pytest.approx(2.0)
    assert positive.right_m == pytest.approx(3.0)
    assert positive.distance_m == pytest.approx(math.sqrt(22.0))
    assert not positive.accepted

    negative = target.miss((95.0, 15.0, -12.0))
    assert negative.vector_m == pytest.approx((-3.0, -2.0, -3.0))


def test_sphere_acceptance_boundary_and_radial_closest_point() -> None:
    target = SpatialTarget(
        label="Waypoint",
        kind="aerial_waypoint",
        point=TargetPoint(10.0, 20.0, 30.0),
        tolerance=SphereTolerance(radius_m=2.0),
        elevation_source="absolute",
    )

    assert target.miss((11.2, 21.6, 30.0)).accepted
    outside = target.miss((13.0, 24.0, 30.0))
    assert outside.closest_point_m == pytest.approx((11.2, 21.6, 30.0))
    assert outside.vector_m == pytest.approx((1.8, 2.4, 0.0))
    assert outside.distance_m == pytest.approx(3.0)


def test_surface_circle_requires_surface_elevation_as_well_as_plan_containment() -> (
    None
):
    target = SpatialTarget(
        label="Raised Green",
        kind="landing_area",
        point=TargetPoint(100.0, 5.0, 10.0),
        tolerance=SurfaceCircleTolerance(radius_m=10.0),
        elevation_source="course_surface",
        ground_source="course.surface/raised-green",
    )

    assert target.miss((106.0, 5.0, 18.0)).accepted
    elevated = target.miss((106.0, 8.0, 18.0))
    assert not elevated.accepted
    assert elevated.vector_m == pytest.approx((0.0, 3.0, 0.0))
    assert elevated.distance_m == pytest.approx(3.0)


def test_surface_corridor_clamps_downrange_right_and_elevation() -> None:
    target = SpatialTarget(
        label="Fairway",
        kind="landing_area",
        point=TargetPoint(200.0, 2.0, 0.0),
        tolerance=SurfaceCorridorTolerance(half_length_m=20.0, half_width_m=15.0),
        elevation_source="course_surface",
        ground_source="course.surface/fairway-9",
    )

    miss = target.miss((225.0, 1.0, -19.0))
    assert miss.closest_point_m == pytest.approx((220.0, 2.0, -15.0))
    assert miss.vector_m == pytest.approx((5.0, -1.0, -4.0))
    assert miss.distance_m == pytest.approx(math.sqrt(42.0))


def test_landing_region_adapter_preserves_green_and_fairway_geometry() -> None:
    regions = (
        TargetRegion(kind="green", distance_m=180.0, radius_m=12.0, lateral_m=-4.0),
        TargetRegion(
            kind="fairway",
            distance_m=220.0,
            band_half_length_m=25.0,
            half_width_m=17.0,
        ),
    )

    for region in regions:
        spatial = SpatialTarget.from_target_region(
            region,
            surface_elevation_m=3.5,
            ground_source="course.surface/default",
        )
        assert spatial.to_target_region() == region
        assert spatial.point.elevation_m == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (
            lambda: SpatialTarget(
                label="bad",
                kind="landing_area",
                point=TargetPoint(1.0, 0.0, 0.0),
                tolerance=SphereTolerance(1.0),
                elevation_source="course_surface",
                ground_source="surface",
            ),
            "surface tolerance",
        ),
        (
            lambda: SpatialTarget(
                label="bad",
                kind="aerial_waypoint",
                point=TargetPoint(1.0, 2.0, 3.0),
                tolerance=BoxTolerance((1.0, 1.0, 1.0)),
                elevation_source="course_surface",
                ground_source="surface",
            ),
            "absolute elevation",
        ),
        (
            lambda: SpatialTarget(
                label="bad",
                kind="landing_area",
                point=TargetPoint(1.0, 0.0, 0.0),
                tolerance=SurfaceCircleTolerance(1.0),
                elevation_source="course_surface",
            ),
            "ground_source",
        ),
    ],
)
def test_target_kind_rejects_incompatible_geometry_or_elevation_source(
    target: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        target()  # type: ignore[operator]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: SphereTolerance(0.0),
        lambda: SphereTolerance(math.inf),
        lambda: BoxTolerance((1.0, -1.0, 1.0)),
        lambda: SurfaceCircleTolerance(math.nan),
        lambda: SurfaceCorridorTolerance(1.0, 0.0),
    ],
)
def test_tolerances_require_finite_positive_dimensions(constructor: object) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        constructor()  # type: ignore[operator]


def test_miss_from_flight_frame_uses_the_shared_adapter() -> None:
    target = SpatialTarget(
        label="Flight-frame gate",
        kind="aerial_waypoint",
        point=TargetPoint(100.0, 20.0, -5.0),
        tolerance=BoxTolerance((2.0, 3.0, 4.0)),
        elevation_source="absolute",
    )
    app_actual = np.array((105.0, 25.0, 2.0))
    flight_actual = np.array((105.0, -2.0, 25.0))

    assert target.miss_from_frame(flight_actual, frame="flight") == target.miss(
        app_actual
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("kind", 1, TypeError, "kind"),
        ("tolerance", object(), TypeError, "tolerance"),
        ("elevation_source", 1, TypeError, "elevation_source"),
        ("units", 1, TypeError, "units"),
        ("units", "ft", ValueError, "units"),
        ("frame", 1, TypeError, "frame"),
        ("frame", "flight", ValueError, "frame"),
    ],
)
def test_spatial_target_boundary_types_are_strict(
    field: str, value: object, error_type: type[Exception], message: str
) -> None:
    values: dict[str, object] = {
        "label": "Target",
        "kind": "aerial_waypoint",
        "point": TargetPoint(1.0, 2.0, 3.0),
        "tolerance": SphereTolerance(1.0),
        "elevation_source": "absolute",
    }
    values[field] = value

    with pytest.raises(error_type, match=message):
        SpatialTarget(**values)  # type: ignore[arg-type]
