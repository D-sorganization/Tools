"""Canonical spatial-target persistence across simulation run surfaces."""

from __future__ import annotations

import csv
import json

import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig, run_simulation
from rate_of_closure.simulation.export import (
    ball_setup_from_json_dict,
    manual_delivery_from_json_dict,
    run_to_json_dict,
    spatial_target_from_simulation_document,
    write_csv,
)
from rate_of_closure.simulation.manual_delivery import ManualDeliveryConfig
from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    TargetPoint,
    spatial_target_from_json_dict,
    spatial_target_to_json_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_CURRENT_V5_FORMATS = (
    "rate_of_closure.simulation_run/5",
    "rate_of_closure.simulation_run.web/5",
)


@pytest.fixture(scope="module")
def run():  # type: ignore[no-untyped-def]
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
        )
    )


@pytest.fixture
def aerial_target() -> SpatialTarget:
    return SpatialTarget(
        label="Apex gate",
        kind="aerial_waypoint",
        point=TargetPoint.from_frame((140.0, 3.0, 24.0), "flight"),
        tolerance=BoxTolerance((4.0, 2.0, 3.0)),
        elevation_source="absolute",
    )


def test_run_json_reuses_canonical_target_in_all_manifests(
    run, aerial_target: SpatialTarget
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run, spatial_target=aerial_target)
    canonical = spatial_target_to_json_dict(aerial_target)

    assert document["spatial_target"] == canonical
    assert document["solver_manifest"] == {
        "schema": "swing_sim.solver_manifest",
        "schema_version": 1,
        "target": canonical,
    }
    assert document["variation_manifest"]["target"] == canonical
    assert spatial_target_from_simulation_document(document) == aerial_target


def test_native_v5_export_without_an_explicit_target_writes_canonical_default(
    run,
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run)

    target = spatial_target_from_simulation_document(document)
    assert document["format"] == "rate_of_closure.simulation_run/5"
    assert document["spatial_target"] == spatial_target_to_json_dict(target)
    assert target.point.app_coordinates_m == pytest.approx((230.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("missing_block", "message"),
    (
        ("spatial_target", "requires spatial_target"),
        ("ball_setup", "requires parameters.ball_setup"),
        ("manual_delivery", "requires parameters.manual_delivery"),
    ),
)
@pytest.mark.parametrize("format_name", _CURRENT_V5_FORMATS)
def test_current_v5_rejects_missing_canonical_blocks(
    run,
    aerial_target: SpatialTarget,
    missing_block: str,
    message: str,
    format_name: str,
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run, spatial_target=aerial_target)
    document["format"] = format_name
    if missing_block == "spatial_target":
        document.pop(missing_block)
    else:
        document["parameters"].pop(missing_block)

    parser = {
        "spatial_target": spatial_target_from_simulation_document,
        "ball_setup": ball_setup_from_json_dict,
        "manual_delivery": manual_delivery_from_json_dict,
    }[missing_block]
    with pytest.raises(ValueError, match=message):
        parser(document)


@pytest.mark.parametrize(
    "parser", (ball_setup_from_json_dict, manual_delivery_from_json_dict)
)
@pytest.mark.parametrize("format_name", _CURRENT_V5_FORMATS)
def test_current_v5_requires_a_parameters_mapping(
    run,
    aerial_target: SpatialTarget,
    parser,
    format_name: str,
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run, spatial_target=aerial_target)
    document["format"] = format_name
    document.pop("parameters")

    with pytest.raises(ValueError, match="version 5 requires parameters"):
        parser(document)


@pytest.mark.parametrize(
    "missing_field",
    (
        "attack_angle_deg",
        "club_path_deg",
        "forward_shaft_lean_deg",
        "shaft_axis_datum",
    ),
)
@pytest.mark.parametrize("format_name", _CURRENT_V5_FORMATS)
def test_current_v5_rejects_incomplete_manual_delivery(
    run,
    aerial_target: SpatialTarget,
    missing_field: str,
    format_name: str,
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run, spatial_target=aerial_target)
    document["format"] = format_name
    document["parameters"]["manual_delivery"].pop(missing_field)

    with pytest.raises(ValueError, match=rf"manual_delivery requires.*{missing_field}"):
        manual_delivery_from_json_dict(document)


@pytest.mark.parametrize(
    "missing_field",
    ("support_mode", "tee_height_m", "height_reference", "ball_center_m"),
)
@pytest.mark.parametrize("format_name", _CURRENT_V5_FORMATS)
def test_current_v5_rejects_incomplete_ball_setup(
    run,
    aerial_target: SpatialTarget,
    missing_field: str,
    format_name: str,
) -> None:  # type: ignore[no-untyped-def]
    document = run_to_json_dict(run, spatial_target=aerial_target)
    document["format"] = format_name
    document["parameters"]["ball_setup"].pop(missing_field)

    with pytest.raises(ValueError, match=rf"ball_setup requires.*{missing_field}"):
        ball_setup_from_json_dict(document)


@pytest.mark.parametrize("version", [3, 4])
def test_unemitted_native_schema_versions_are_rejected(version: int) -> None:
    document = {
        "format": f"rate_of_closure.simulation_run/{version}",
        "parameters": {},
    }

    with pytest.raises(ValueError, match=rf"unsupported.*version {version}"):
        spatial_target_from_simulation_document(document)
    with pytest.raises(ValueError, match=rf"unsupported.*version {version}"):
        ball_setup_from_json_dict(document)
    with pytest.raises(ValueError, match=rf"unsupported.*version {version}"):
        manual_delivery_from_json_dict(document)


def test_web_v4_keeps_setup_and_delivery_migration(
    run,
    aerial_target: SpatialTarget,
) -> None:  # type: ignore[no-untyped-def]
    legacy_web = run_to_json_dict(run, spatial_target=aerial_target)
    legacy_web["format"] = "rate_of_closure.simulation_run.web/4"
    legacy_web["parameters"].pop("ball_setup")
    legacy_web["parameters"].pop("manual_delivery")
    assert ball_setup_from_json_dict(legacy_web) == BallSetup(
        BallSupportMode.GROUND, 0.0
    )
    assert manual_delivery_from_json_dict(legacy_web) == ManualDeliveryConfig()


def test_csv_repeats_exact_canonical_target_metadata(
    run, aerial_target: SpatialTarget, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "run-with-target.csv"
    write_csv(run, path, spatial_target=aerial_target)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    first = rows[0]
    assert first["target_schema"] == "swing_sim.spatial_target"
    assert first["target_schema_version"] == "1"
    assert first["target_label"] == "Apex gate"
    assert float(first["target_x_downrange_m"]) == pytest.approx(140.0)
    assert float(first["target_y_up_m"]) == pytest.approx(24.0)
    assert float(first["target_z_right_m"]) == pytest.approx(-3.0)
    assert json.loads(first["target_tolerance_json"]) == (
        spatial_target_to_json_dict(aerial_target)["tolerance"]
    )


def test_import_migrates_legacy_project_target_and_old_default() -> None:
    migrated = spatial_target_from_simulation_document(
        {
            "format": "rate_of_closure.simulation_run/1",
            "parameters": {
                "target": {
                    "kind": "green",
                    "distance_m": 205.0,
                    "radius_m": 12.0,
                    "lateral_m": -4.0,
                }
            },
        }
    )
    assert migrated.point.app_coordinates_m == pytest.approx((205.0, 0.0, -4.0))
    assert migrated.ground_source == "legacy.course_surface/default"

    default = spatial_target_from_simulation_document(
        {"format": "rate_of_closure.simulation_run/1", "parameters": {}}
    )
    assert default.point.app_coordinates_m == pytest.approx((230.0, 0.0, 0.0))


def test_import_reads_solver_manifest_and_rejects_incomplete_current_web_project(
    aerial_target: SpatialTarget,
) -> None:
    canonical = spatial_target_to_json_dict(aerial_target)
    from_manifest = spatial_target_from_simulation_document(
        {
            "format": "rate_of_closure.simulation_run/2",
            "solver_manifest": {
                "schema": "swing_sim.solver_manifest",
                "schema_version": 1,
                "target": canonical,
            },
        }
    )
    assert from_manifest == spatial_target_from_json_dict(canonical)

    with pytest.raises(ValueError, match="requires spatial_target"):
        spatial_target_from_simulation_document(
            {"format": "rate_of_closure.simulation_run.web/4"}
        )
    with pytest.raises(ValueError, match="solver_manifest schema"):
        spatial_target_from_simulation_document(
            {
                "format": "rate_of_closure.simulation_run/2",
                "solver_manifest": {"schema": "unknown", "schema_version": 1},
            }
        )
