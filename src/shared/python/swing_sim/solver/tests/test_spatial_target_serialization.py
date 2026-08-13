"""Versioned spatial-target persistence and migration tests (#4192)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    TargetPoint,
    spatial_target_from_json,
    spatial_target_from_json_dict,
    spatial_target_to_json,
)

pytestmark = pytest.mark.contract

_REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
_GOLDEN_PATH = (
    _REPOSITORY_ROOT
    / "src/rate_of_closure/web/src/model/__fixtures__/spatial_target_v1.json"
)


def _golden_target() -> SpatialTarget:
    return SpatialTarget(
        label="Apex Gate",
        kind="aerial_waypoint",
        point=TargetPoint.from_frame((137.5, 3.25, 24.25), source_frame="flight"),
        tolerance=BoxTolerance((4.5, 2.5, 3.5)),
        elevation_source="absolute",
    )


def test_versioned_json_matches_cross_language_golden_and_round_trips() -> None:
    golden = _GOLDEN_PATH.read_text(encoding="utf-8").strip()
    encoded = spatial_target_to_json(_golden_target())

    assert encoded == golden
    assert spatial_target_from_json(encoded) == _golden_target()
    assert spatial_target_to_json(spatial_target_from_json(encoded)) == encoded


@pytest.mark.parametrize(
    "legacy",
    [
        {
            "kind": "green",
            "distance_m": 230.0,
            "radius_m": 10.0,
            "lateral_m": 4.0,
            "band_half_length_m": 15.0,
            "half_width_m": 16.0,
        },
        {
            "kind": "green",
            "distanceM": 230.0,
            "radiusM": 10.0,
            "lateralM": 4.0,
            "bandHalfLengthM": 15.0,
            "halfWidthM": 16.0,
        },
    ],
)
def test_legacy_2d_green_migrates_without_changing_plan_geometry(
    legacy: dict[str, object],
) -> None:
    migrated = spatial_target_from_json_dict(legacy)

    assert migrated.kind == "landing_area"
    assert migrated.elevation_source == "course_surface"
    assert migrated.ground_source == "legacy.course_surface/default"
    assert migrated.point.app_coordinates_m == pytest.approx((230.0, 0.0, 4.0))
    assert migrated.to_target_region().radius_m == pytest.approx(10.0)
    assert migrated.to_target_region().lateral_m == pytest.approx(4.0)


def test_legacy_fairway_defaults_are_explicit_and_reversible() -> None:
    migrated = spatial_target_from_json(
        json.dumps({"kind": "fairway", "distance_m": 180.0})
    )

    region = migrated.to_target_region()
    assert region.kind == "fairway"
    assert region.band_half_length_m == pytest.approx(15.0)
    assert region.half_width_m == pytest.approx(16.0)


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (lambda data: data.update(schema_version=2), ValueError, "schema_version"),
        (lambda data: data.update(units="ft"), ValueError, "units"),
        (lambda data: data.update(frame="flight"), ValueError, "frame"),
        (lambda data: data.update(extra=True), ValueError, "unknown fields"),
        (
            lambda data: data["position_m"].update(x=float("nan")),
            ValueError,
            "finite",
        ),
    ],
)
def test_current_schema_rejects_invalid_version_units_frame_and_values(
    mutate: object, error_type: type[Exception], message: str
) -> None:
    data = json.loads(spatial_target_to_json(_golden_target()))
    mutate(data)  # type: ignore[operator]

    with pytest.raises(error_type, match=message):
        spatial_target_from_json_dict(data)


def test_serialization_rejects_wrong_boundary_types_and_invalid_json() -> None:
    with pytest.raises(TypeError, match="mapping"):
        spatial_target_from_json_dict([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text"):
        spatial_target_from_json(2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid JSON"):
        spatial_target_from_json("{")
    with pytest.raises(TypeError, match="real number"):
        spatial_target_from_json('{"kind":"green","distance_m":true}')
