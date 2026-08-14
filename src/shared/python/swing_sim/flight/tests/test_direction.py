"""Contracts for canonical launch-direction conventions and migration."""

from __future__ import annotations

import pytest

from shared.python.swing_sim.flight.direction import (
    DEFINITIONS,
    LaunchDirection,
    LaunchDirectionConvention,
    launch_direction_from_mapping,
    launch_direction_sign_labels,
    launch_direction_to_flight_azimuth,
    migrate_launch_direction_mapping,
)


@pytest.mark.parametrize("degrees", [0.0, 7.25, -7.25, 90.0, -90.0, 179.999, -179.999])
def test_round_trip_between_every_convention(degrees: float) -> None:
    for source in DEFINITIONS:
        direction = LaunchDirection(degrees, source)
        for target in DEFINITIONS:
            assert direction.to(target).to(source).degrees == pytest.approx(degrees)


def test_internal_flight_azimuth_has_the_opposite_sign() -> None:
    right = LaunchDirection(6.0, LaunchDirectionConvention.APP_NATIVE)
    assert launch_direction_to_flight_azimuth(right) == pytest.approx(-6.0)


def test_definitions_are_the_canonical_registry_records() -> None:
    trackman = DEFINITIONS[LaunchDirectionConvention.TRACKMAN_COMPARABLE]
    assert trackman.parameter_id.value == "launch_direction"
    assert trackman.sign_rule.value == "positive_right"
    assert trackman.retrieved_on == "2026-08-05"
    assert launch_direction_sign_labels(trackman.convention_id) == (
        "right of the target line",
        "left of the target line",
    )


def test_legacy_import_is_migrated_without_dropping_fields() -> None:
    source = {"launch_azimuth_deg": -3.5, "shot_name": "soft draw"}
    migrated = migrate_launch_direction_mapping(source)
    assert migrated == {
        "launch_azimuth_deg": -3.5,
        "shot_name": "soft draw",
        "launch_direction_deg": -3.5,
        "launch_direction_convention": "app_native",
        "launch_direction_schema_version": 1,
    }
    assert launch_direction_from_mapping(migrated).degrees == pytest.approx(-3.5)


def test_equivalent_canonical_and_legacy_values_are_accepted() -> None:
    migrated = migrate_launch_direction_mapping(
        {"launch_direction_deg": 2.0, "azimuth_deg": 2.0}
    )
    assert migrated["launch_direction_deg"] == pytest.approx(2.0)


def test_legacy_generic_convention_name_migrates_to_registry_id() -> None:
    migrated = migrate_launch_direction_mapping(
        {
            "launch_direction_deg": 2.0,
            "launch_direction_convention": "launch_monitor_comparable",
        }
    )
    assert migrated["launch_direction_convention"] == "trackman_comparable"


def test_unverified_foresight_sign_is_not_activated_by_this_adapter() -> None:
    with pytest.raises(ValueError, match="unsupported launch-direction convention"):
        LaunchDirection(2.0, LaunchDirectionConvention.FORESIGHT_COMPARABLE)


def test_conflicting_canonical_and_legacy_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting launch-direction"):
        migrate_launch_direction_mapping(
            {"launch_direction_deg": 2.0, "launch_azimuth_deg": -2.0}
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 181.0])
def test_invalid_values_are_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        migrate_launch_direction_mapping({"launch_direction_deg": value})
