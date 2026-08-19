"""Contract tests for reproducible variation plot definitions."""

from __future__ import annotations

import json

import pytest

from rate_of_closure.variation.plot_definition import (
    PLOT_DEFINITION_SCHEMA_VERSION,
    PlotDefinition,
    read_plot_definition,
    write_plot_definition,
)
from shared.python.contracts import ContractViolationError
from tests.rate_of_closure._variation_plot_definition_support import matrix_definition

pytestmark = pytest.mark.unit


def test_plot_definition_round_trips_complete_geometric_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    definition = PlotDefinition(
        result_id="ensemble-123",
        plot_type="swing_arc_overlay",
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_id="swing.clubhead.reference",
        position_unit="m",
        alignment_basis="common_simulation_time_s",
        dispersion_metric="confidence-ellipsoid-volume",
        dispersion_unit="m^3",
        quiet_threshold=1.25e-7,
        confidence_level=0.95,
        min_quiet_duration_s=0.02,
        min_quiet_samples=3,
        selected_trial_index=2,
        camera_yaw_deg=-37.0,
        camera_pitch_deg=22.0,
        camera_zoom=1.2,
        outcome_filter="evaluated_hit",
        phase_end_fraction=0.75,
        perturbation_source_key="swing_sim.swing.yaw_deg",
        perturbation_band="Upper Third",
        show_confidence_ellipsoids=True,
    )
    destination = tmp_path / "plot-definition.json"

    write_plot_definition(definition, destination)

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema_version"] == PLOT_DEFINITION_SCHEMA_VERSION == 3
    assert document["result_id"] == "ensemble-123"
    assert document["dispersion_metric"] == "confidence-ellipsoid-volume"
    assert document["dispersion_unit"] == "m^3"
    assert document["quiet_threshold"] == pytest.approx(1.25e-7)
    assert document["confidence_level"] == pytest.approx(0.95)
    assert document["min_quiet_duration_s"] == pytest.approx(0.02)
    assert document["min_quiet_samples"] == 3
    assert document["selected_trial_index"] == 2
    assert document["phase_end_fraction"] == pytest.approx(0.75)
    assert PlotDefinition.from_json_dict(document) == definition
    assert read_plot_definition(destination) == definition


def test_plot_definition_migrates_strict_v1_geometry_defaults() -> None:
    document = {
        "schema_version": 1,
        "result_id": "ensemble-v1",
        "plot_type": "geometric_variability",
        "coordinate_frame": "app_frame:x_target,y_up,z_right",
        "x_variable_key": None,
        "y_variable_key": None,
        "point_id": "swing.clubhead.reference",
        "position_unit": "m",
        "alignment_basis": "common_simulation_time_s",
        "quiet_threshold_m": None,
        "selected_trial_index": None,
        "camera_yaw_deg": None,
        "camera_pitch_deg": None,
        "camera_zoom": None,
        "outcome_filter": None,
        "phase_end_fraction": None,
        "perturbation_source_key": None,
        "perturbation_band": None,
        "variable_keys": None,
    }

    migrated = PlotDefinition.from_json_dict(document)

    assert migrated.dispersion_metric == "rms-radius"
    assert migrated.dispersion_unit == "m"
    assert migrated.quiet_threshold == pytest.approx(0.005)
    assert migrated.confidence_level is None
    assert migrated.min_quiet_duration_s == 0.0
    assert migrated.min_quiet_samples == 1
    assert migrated.to_json_dict()["schema_version"] == 3
    with pytest.raises(ContractViolationError, match="applicable"):
        PlotDefinition.from_json_dict(
            {**document, "variable_keys": ["input:a", "output:b"]}
        )


def test_plot_definition_migrates_exact_v2_with_surfaces_off() -> None:
    current = matrix_definition().to_json_dict()
    legacy = {
        key: value
        for key, value in current.items()
        if key != "show_confidence_ellipsoids"
    }
    legacy["schema_version"] = 2

    migrated = PlotDefinition.from_json_dict(legacy)

    assert migrated.show_confidence_ellipsoids is None
    assert migrated.to_json_dict()["schema_version"] == 3


@pytest.mark.parametrize(
    ("plot_type", "x_key", "y_key", "selected", "variable_keys"),
    [
        ("scalar_scatter", "input:speed", "output:carry_m", 2, None),
        (
            "distribution_matrix",
            None,
            None,
            None,
            ["input:speed", "output:carry_m"],
        ),
    ],
)
def test_plot_definition_migrates_authentic_v1_nongeometric_frame(
    plot_type: str,
    x_key: str | None,
    y_key: str | None,
    selected: int | None,
    variable_keys: list[str] | None,
) -> None:
    document = {
        "schema_version": 1,
        "result_id": "historical-v1",
        "plot_type": plot_type,
        "coordinate_frame": "app_frame:x_target,y_up,z_right",
        "x_variable_key": x_key,
        "y_variable_key": y_key,
        "point_id": None,
        "position_unit": None,
        "alignment_basis": None,
        "quiet_threshold_m": None,
        "selected_trial_index": selected,
        "camera_yaw_deg": None,
        "camera_pitch_deg": None,
        "camera_zoom": None,
        "outcome_filter": None,
        "phase_end_fraction": None,
        "perturbation_source_key": None,
        "perturbation_band": None,
        "variable_keys": variable_keys,
    }

    migrated = PlotDefinition.from_json_dict(document)

    assert migrated.coordinate_frame is None
    assert migrated.plot_type == plot_type
    with pytest.raises(ContractViolationError, match="legacy coordinate_frame"):
        PlotDefinition.from_json_dict({**document, "coordinate_frame": "other_frame"})


@pytest.mark.parametrize("schema_version", [True, 2.0, "2", 3])
def test_plot_definition_reader_rejects_coercive_or_unknown_versions(
    schema_version: object,
) -> None:
    with pytest.raises(ContractViolationError):
        PlotDefinition.from_json_dict({"schema_version": schema_version})


def test_plot_definition_reader_rejects_unknown_fields() -> None:
    definition = PlotDefinition(
        result_id="variation-17-3",
        plot_type="distribution_matrix",
        variable_keys=("input:speed", "output:carry_m"),
    ).to_json_dict()
    definition["unexpected"] = 1

    with pytest.raises(ContractViolationError, match="fields"):
        PlotDefinition.from_json_dict(definition)

    definition.pop("unexpected")
    definition.pop("camera_zoom")
    with pytest.raises(ContractViolationError, match="fields"):
        PlotDefinition.from_json_dict(definition)

    definition = PlotDefinition(
        result_id="variation-17-3",
        plot_type="distribution_matrix",
        variable_keys=("input:speed", "output:carry_m"),
    ).to_json_dict()
    definition["min_quiet_samples"] = "1"
    with pytest.raises(ContractViolationError, match="integer"):
        PlotDefinition.from_json_dict(definition)


def test_distribution_matrix_definition_requires_unique_selected_variables() -> None:
    definition = PlotDefinition(
        result_id="variation-17-3",
        plot_type="distribution_matrix",
        variable_keys=("input:speed", "output:carry_m", "output:lateral_m"),
    )

    assert definition.variable_keys == (
        "input:speed",
        "output:carry_m",
        "output:lateral_m",
    )

    with pytest.raises(ContractViolationError, match="unique"):
        PlotDefinition(
            result_id="variation-17-3",
            plot_type="distribution_matrix",
            variable_keys=("output:carry_m", "output:carry_m"),
        )


def test_plot_definition_dict_uses_json_array_for_variable_keys() -> None:
    definition = matrix_definition()

    document = definition.to_json_dict()

    assert document["variable_keys"] == ["input:swing.speed", "output:carry_m"]
    assert PlotDefinition.from_json_dict(document) == definition
