from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest

from rate_of_closure.variation.plot_definition import (
    PlotDefinition,
    write_plot_definition,
)
from shared.python.contracts import ContractViolationError
from tests.rate_of_closure._variation_plot_definition_support import (
    complete_geometric_definition,
    matrix_definition,
    scatter_definition,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_id", " ensemble-contract"),
        ("plot_type", "unknown"),
        ("coordinate_frame", " "),
        ("coordinate_frame", "app_frame:x_target,z_up,y_right"),
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
        complete_geometric_definition(**{field: value})


def test_plot_definition_requires_a_source_for_a_perturbation_band() -> None:
    with pytest.raises(ContractViolationError, match="source"):
        complete_geometric_definition(perturbation_source_key=None)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (scatter_definition, "coordinate_frame", "app_frame:x_target,y_up,z_right"),
        (scatter_definition, "point_id", "swing.clubhead.reference"),
        (scatter_definition, "camera_zoom", 1.0),
        (scatter_definition, "variable_keys", ("input:a", "output:b")),
        (matrix_definition, "coordinate_frame", "app_frame:x_target,y_up,z_right"),
        (matrix_definition, "x_variable_key", "input:swing.speed"),
        (matrix_definition, "selected_trial_index", 0),
        (complete_geometric_definition, "x_variable_key", "input:swing.speed"),
        (complete_geometric_definition, "variable_keys", ("input:a", "output:b")),
    ],
)
def test_plot_definition_rejects_inapplicable_fields(
    factory, field: str, value: object
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ContractViolationError, match="applicable"):
        factory(**{field: value})


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (complete_geometric_definition, "result_id", "ensemble\x00contract"),
        (complete_geometric_definition, "point_id", "swing\x1fclubhead"),
        (
            complete_geometric_definition,
            "perturbation_source_key",
            "swing_sim\x80yaw",
        ),
        (scatter_definition, "x_variable_key", "input\x7fspeed"),
        (matrix_definition, "variable_keys", ("input:a", "output:\x81b")),
    ],
)
def test_plot_definition_rejects_control_characters_in_stable_ids(
    factory, field: str, value: object
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ContractViolationError, match="control"):
        factory(**{field: value})


def test_python_constructor_normalizes_supported_real_scalars_before_json(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    definition = complete_geometric_definition(
        quiet_threshold=Fraction(1, 200),
        min_quiet_duration_s=np.float32(0.02),
        min_quiet_samples=np.int64(2),
        selected_trial_index=np.int64(3),
        camera_yaw_deg=Fraction(-37, 1),
        camera_pitch_deg=np.float32(22),
        camera_zoom=Fraction(6, 5),
        phase_end_fraction=Fraction(3, 4),
    )

    assert type(definition.quiet_threshold) is float
    assert type(definition.min_quiet_duration_s) is float
    assert type(definition.min_quiet_samples) is int
    assert type(definition.selected_trial_index) is int
    assert type(definition.camera_yaw_deg) is float
    assert type(definition.camera_pitch_deg) is float
    assert type(definition.camera_zoom) is float
    assert type(definition.phase_end_fraction) is float
    destination = tmp_path / "normalized.json"
    write_plot_definition(definition, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["camera_zoom"] == 1.2


@pytest.mark.parametrize("value", [Fraction(1, 2), np.float64(0.5)])
def test_plot_definition_parser_rejects_non_json_numeric_objects(value: object) -> None:
    document = complete_geometric_definition().to_json_dict()
    document["camera_zoom"] = value
    with pytest.raises(ContractViolationError):
        PlotDefinition.from_json_dict(document)


def test_python_numeric_conversion_overflow_fails_as_contract_violation() -> None:
    huge = 10**400
    with pytest.raises(ContractViolationError, match="finite JSON"):
        complete_geometric_definition(camera_zoom=huge)

    document = complete_geometric_definition().to_json_dict()
    document["camera_zoom"] = huge
    with pytest.raises(ContractViolationError, match="finite JSON"):
        PlotDefinition.from_json_dict(document)


def test_plot_definition_parser_rejects_control_and_inapplicable_state() -> None:
    document = complete_geometric_definition().to_json_dict()
    document["point_id"] = "swing\x00clubhead"
    with pytest.raises(ContractViolationError, match="control"):
        PlotDefinition.from_json_dict(document)

    document = complete_geometric_definition().to_json_dict()
    document["variable_keys"] = ["input:a", "output:b"]
    with pytest.raises(ContractViolationError, match="applicable"):
        PlotDefinition.from_json_dict(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_id", " "),
        ("result_id", "result\x00id"),
        ("selected_trial_index", True),
        ("camera_yaw_deg", math.nan),
        ("camera_pitch_deg", math.nan),
        ("camera_zoom", math.nan),
        ("outcome_filter", "hit"),
        ("variable_keys", ("input:a", "output:b")),
    ],
)
def test_plot_definition_writer_revalidates_tampered_state(
    tmp_path,
    field: str,
    value: object,
) -> None:  # type: ignore[no-untyped-def]
    definition = complete_geometric_definition()
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
