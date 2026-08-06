"""Contract tests for reproducible variation plot definitions."""

from __future__ import annotations

import json

import pytest

from rate_of_closure.variation.plot_definition import (
    PlotDefinition,
    write_plot_definition,
)
from shared.python.contracts import ContractViolationError

pytestmark = pytest.mark.unit


def test_plot_definition_round_trips_complete_geometric_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    definition = PlotDefinition(
        result_id="ensemble-123",
        plot_type="swing_arc_overlay",
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_id="swing.clubhead.reference",
        position_unit="m",
        alignment_basis="common_simulation_time_s",
        quiet_threshold_m=0.005,
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
    assert document["schema_version"] == 1
    assert document["result_id"] == "ensemble-123"
    assert document["quiet_threshold_m"] == pytest.approx(0.005)
    assert document["selected_trial_index"] == 2
    assert document["phase_end_fraction"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_type": "scalar_scatter"},
        {"plot_type": "geometric_variability", "point_id": "swing.wrist"},
        {
            "plot_type": "swing_arc_overlay",
            "point_id": "swing.wrist",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
            "quiet_threshold_m": 0.0,
        },
    ],
)
def test_plot_definition_rejects_incomplete_or_invalid_state(kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ContractViolationError):
        PlotDefinition(result_id="ensemble-123", **kwargs)
