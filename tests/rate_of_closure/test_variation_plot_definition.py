"""Contract tests for reproducible variation plot definitions."""

from __future__ import annotations

import json
import math

import pytest

from rate_of_closure.variation.plot_definition import (
    PLOT_DEFINITION_SCHEMA_VERSION,
    PlotDefinition,
    read_plot_definition,
    write_plot_definition,
)
from shared.python.contracts import ContractViolationError

pytestmark = pytest.mark.unit


def _complete_geometric_definition(**overrides: object) -> PlotDefinition:
    values: dict[str, object] = {
        "result_id": "ensemble-contract",
        "plot_type": "swing_arc_overlay",
        "coordinate_frame": "app_frame:x_target,y_up,z_right",
        "point_id": "swing.clubhead.reference",
        "position_unit": "m",
        "alignment_basis": "common_simulation_time_s",
        "dispersion_metric": "rms-radius",
        "dispersion_unit": "m",
        "quiet_threshold": 0.005,
        "confidence_level": None,
        "min_quiet_duration_s": 0.0,
        "min_quiet_samples": 1,
        "selected_trial_index": 0,
        "camera_yaw_deg": -37.0,
        "camera_pitch_deg": 22.0,
        "camera_zoom": 1.2,
        "outcome_filter": "evaluated_hit",
        "phase_end_fraction": 0.75,
        "perturbation_source_key": "swing_sim.swing.yaw_deg",
        "perturbation_band": "Upper Third",
    }
    values.update(overrides)
    return PlotDefinition(**values)  # type: ignore[arg-type]


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
    )
    destination = tmp_path / "plot-definition.json"

    write_plot_definition(definition, destination)

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema_version"] == PLOT_DEFINITION_SCHEMA_VERSION == 2
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
    assert migrated.to_json_dict()["schema_version"] == 2


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_id", " ensemble-contract"),
        ("plot_type", "unknown"),
        ("coordinate_frame", " "),
        ("point_id", "swing.clubhead.reference "),
        ("position_unit", "mm"),
        ("alignment_basis", "sample-index"),
        ("selected_trial_index", True),
        ("selected_trial_index", 1.5),
        ("camera_yaw_deg", True),
        ("camera_yaw_deg", math.nan),
        ("camera_yaw_deg", math.inf),
        ("camera_pitch_deg", True),
        ("camera_pitch_deg", -90.0001),
        ("camera_pitch_deg", 90.0001),
        ("camera_zoom", True),
        ("camera_zoom", math.nan),
        ("camera_zoom", math.inf),
        ("phase_end_fraction", True),
        ("phase_end_fraction", math.nan),
        ("phase_end_fraction", math.inf),
        ("phase_end_fraction", 1.0001),
        ("outcome_filter", "hit"),
        ("perturbation_source_key", " swing_sim.swing.yaw_deg"),
        ("perturbation_band", "outer"),
    ],
)
def test_plot_definition_constructor_rejects_malformed_full_object_state(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ContractViolationError):
        _complete_geometric_definition(**{field: value})


def test_plot_definition_requires_a_source_for_a_perturbation_band() -> None:
    with pytest.raises(ContractViolationError, match="source"):
        _complete_geometric_definition(perturbation_source_key=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_id", " "),
        ("selected_trial_index", True),
        ("camera_yaw_deg", math.nan),
        ("camera_pitch_deg", math.nan),
        ("camera_zoom", math.nan),
        ("outcome_filter", "hit"),
    ],
)
def test_plot_definition_writer_revalidates_tampered_state(
    tmp_path,
    field: str,
    value: object,
) -> None:  # type: ignore[no-untyped-def]
    definition = _complete_geometric_definition()
    object.__setattr__(definition, field, value)
    destination = tmp_path / "must-not-exist.json"

    with pytest.raises(ContractViolationError):
        write_plot_definition(definition, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_type": "scalar_scatter"},
        {"plot_type": "geometric_variability", "point_id": "swing.wrist"},
        {"plot_type": "distribution_matrix"},
        {
            "plot_type": "swing_arc_overlay",
            "point_id": "swing.wrist",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
            "dispersion_metric": "rms-radius",
            "dispersion_unit": "m",
            "quiet_threshold": 0.0,
        },
        {
            "plot_type": "geometric_variability",
            "point_id": "swing.wrist",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
            "dispersion_metric": "rms-radius",
            "dispersion_unit": "m",
            "quiet_threshold": 0.005,
            "confidence_level": 0.95,
        },
        {
            "plot_type": "geometric_variability",
            "point_id": "swing.wrist",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
            "dispersion_metric": "confidence-ellipsoid-volume",
            "dispersion_unit": "m^3",
            "quiet_threshold": 1.0e-7,
            "confidence_level": None,
        },
    ],
)
def test_plot_definition_rejects_incomplete_or_invalid_state(kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ContractViolationError):
        PlotDefinition(result_id="ensemble-123", **kwargs)
