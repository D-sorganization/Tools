"""Plot-ready, miss-safe facade for complete Rate simulation ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from rate_of_closure.variation.plot_data import (
    PlotBudget,
    ScalarVariableKind,
    build_ensemble_plot_dataset,
)
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    CONTACT_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    IMPACT_OUTPUT_NAMES,
    NUMERICAL_FAILURE,
    SHOT_OUTPUT_NAMES,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    EnsemblePositionTraces,
    LowVariabilityCriteria,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_POINT = "swing.clubhead.reference"


def _values(status: object, offset: float) -> MappingProxyType[str, float | None]:
    values: dict[str, float | None] = dict.fromkeys(ALL_OUTPUT_NAMES)
    if status is not NUMERICAL_FAILURE:
        for index, name in enumerate(CONTACT_OUTPUT_NAMES):
            values[name] = offset + index + 0.1
    if status is EVALUATED_HIT:
        for index, name in enumerate(IMPACT_OUTPUT_NAMES + SHOT_OUTPUT_NAMES):
            values[name] = offset + index + 10.0
    return MappingProxyType(values)


def _result() -> SimulationEnsembleResult:
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=1.0),),
        n_runs=3,
        seed=17,
    )
    outcomes = (
        SimulationTrialOutcome(0, EVALUATED_HIT, _values(EVALUATED_HIT, 0.0)),
        SimulationTrialOutcome(
            1,
            EVALUATED_NO_IMPACT,
            _values(EVALUATED_NO_IMPACT, 1.0),
        ),
        SimulationTrialOutcome(
            2,
            NUMERICAL_FAILURE,
            _values(NUMERICAL_FAILURE, 2.0),
            "FloatingPointError",
            "planted",
        ),
    )
    outputs = np.full((3, len(ALL_OUTPUT_NAMES)), np.nan)
    for outcome in outcomes:
        outputs[outcome.trial_index] = [
            np.nan if outcome.value(name) is None else outcome.value(name)
            for name in ALL_OUTPUT_NAMES
        ]
    variation = VariationDataset(
        plan=plan,
        input_names=(_FACE,),
        inputs=np.array([[-1.0], [0.5], [2.0]]),
        output_names=ALL_OUTPUT_NAMES,
        outputs=outputs,
        success=np.array([True, True, False]),
    )
    positions = np.full((3, 5, 1, 3), np.nan)
    positions[0, :, 0] = np.column_stack((np.arange(5), np.zeros(5), np.zeros(5)))
    positions[1, :, 0] = np.column_stack((np.arange(5), np.ones(5), np.zeros(5)))
    traces = EnsemblePositionTraces(
        variation=variation,
        sample_times_s=np.linspace(0.0, 0.04, 5),
        coordinate_frame="app_frame:x_target,y_up,z_right",
        point_ids=(_POINT,),
        positions_m=positions,
        sample_valid=np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, False, False, False, False],
            ]
        ),
        impact_sample_indices=np.array([4, -1, -1]),
    )
    return SimulationEnsembleResult(outcomes, variation, traces)


def test_facade_exposes_unit_bearing_inputs_impact_and_shot_axes() -> None:
    plot = build_ensemble_plot_dataset(_result(), result_id="seed-17")

    assert plot.result_id == "seed-17"
    assert plot.coordinate_frame == "app_frame:x_target,y_up,z_right"
    assert plot.variable(f"input:{_FACE}").unit == "deg"
    assert plot.variable("output:clubhead_speed_mps").kind is ScalarVariableKind.IMPACT
    assert plot.variable("output:clubhead_speed_mps").unit == "m/s"
    assert plot.variable("output:carry_m").kind is ScalarVariableKind.SHOT
    assert plot.variable("output:carry_m").unit == "m"


def test_scatter_pairs_only_available_values_and_reports_every_cohort() -> None:
    plot = build_ensemble_plot_dataset(_result())

    scatter = plot.scatter(f"input:{_FACE}", "output:closest_approach_m")

    np.testing.assert_array_equal(scatter.trial_indices, [0, 1])
    np.testing.assert_allclose(scatter.x, [-1.0, 0.5])
    assert scatter.cohorts == (EVALUATED_HIT, EVALUATED_NO_IMPACT)
    assert scatter.summary(EVALUATED_HIT).total == 1
    assert scatter.summary(EVALUATED_HIT).plotted == 1
    assert scatter.summary(EVALUATED_NO_IMPACT).plotted == 1
    assert scatter.summary(NUMERICAL_FAILURE).total == 1
    assert scatter.summary(NUMERICAL_FAILURE).plotted == 0
    assert scatter.summary(NUMERICAL_FAILURE).unavailable == 1


def test_shot_scatter_keeps_misses_and_failures_in_the_unavailable_ledger() -> None:
    plot = build_ensemble_plot_dataset(_result())

    scatter = plot.scatter("output:lateral_m", "output:carry_m")

    np.testing.assert_array_equal(scatter.trial_indices, [0])
    assert scatter.cohorts == (EVALUATED_HIT,)
    assert scatter.summary(EVALUATED_NO_IMPACT).unavailable == 1
    assert scatter.summary(NUMERICAL_FAILURE).unavailable == 1


def test_arc_overlay_retains_rows_and_limits_vertices() -> None:
    plot = build_ensemble_plot_dataset(_result())

    overlay = plot.arc_overlay(_POINT, PlotBudget(max_arc_vertices=9))

    assert overlay.cohorts == (
        EVALUATED_HIT,
        EVALUATED_NO_IMPACT,
        NUMERICAL_FAILURE,
    )
    np.testing.assert_array_equal(overlay.sample_indices, [0, 2, 4])
    assert overlay.positions_m.shape == (3, 3, 3)
    assert np.all(np.isnan(overlay.positions_m[2]))
    np.testing.assert_allclose(overlay.reference_positions_m[:, 1], 0.5)
    assert overlay.raw_vertex_count == 15
    assert overlay.rendered_vertex_count == 9
    assert overlay.coordinate_frame == plot.coordinate_frame
    assert overlay.position_unit == "m"


def test_filtered_arc_and_variability_share_exact_trial_and_time_view() -> None:
    plot = build_ensemble_plot_dataset(_result())
    selected = np.array([1], dtype=int)

    overlay = plot.arc_overlay(_POINT, trial_indices=selected, sample_count=3)
    variability = plot.geometric_variability(
        _POINT,
        LowVariabilityCriteria(max_rms_radius_m=1.0),
        trial_indices=selected,
        sample_count=3,
    )

    np.testing.assert_array_equal(overlay.trial_indices, selected)
    assert overlay.positions_m.shape == (1, 3, 3)
    assert overlay.cohorts == (EVALUATED_NO_IMPACT,)
    assert overlay.raw_vertex_count == 3
    np.testing.assert_array_equal(variability.valid_trial_count, [1, 1, 1])
    np.testing.assert_allclose(variability.sample_times_s, [0.0, 0.01, 0.02])
    assert variability.n_quiet_samples == 0


@pytest.mark.parametrize(
    ("trial_indices", "sample_count", "message"),
    [
        (np.array([], dtype=int), None, "at least one trial"),
        (np.array([0, 0], dtype=int), None, "must be unique"),
        (np.array([0], dtype=int), 0, "invalid sample_count"),
    ],
)
def test_filtered_arc_rejects_invalid_trial_and_time_views(
    trial_indices: np.ndarray,
    sample_count: int | None,
    message: str,
) -> None:
    plot = build_ensemble_plot_dataset(_result())

    with pytest.raises(ContractViolationError, match=message):
        plot.arc_overlay(
            _POINT,
            trial_indices=trial_indices,
            sample_count=sample_count,
        )


def test_geometric_variability_pins_covariance_envelope_and_quiet_zone() -> None:
    plot = build_ensemble_plot_dataset(_result())
    variability = plot.geometric_variability(
        _POINT,
        LowVariabilityCriteria(max_rms_radius_m=0.6, min_samples=5),
    )

    np.testing.assert_allclose(variability.rms_radius_m, 0.5)
    np.testing.assert_allclose(variability.principal_sigma_m[:, 0], np.sqrt(0.5))
    np.testing.assert_allclose(variability.principal_sigma_m[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_array_equal(variability.valid_trial_count, 2)
    assert variability.n_quiet_samples == 5
    assert len(variability.quiet_intervals) == 1
    assert variability.quiet_intervals[0].start_time_s == 0.0
    assert variability.quiet_intervals[0].end_time_s == pytest.approx(0.04)
    assert variability.alignment_basis == "common_simulation_time_s"


def test_geometric_variability_keeps_nonqualifying_samples_visible() -> None:
    plot = build_ensemble_plot_dataset(_result())
    variability = plot.geometric_variability(
        _POINT,
        LowVariabilityCriteria(max_rms_radius_m=0.4),
    )

    assert variability.n_quiet_samples == 0
    assert variability.quiet_intervals == ()
    np.testing.assert_allclose(variability.rms_radius_m, 0.5)


def test_plot_contract_rejects_unknown_axes_points_and_invalid_budgets() -> None:
    plot = build_ensemble_plot_dataset(_result())

    with pytest.raises(ContractViolationError, match="unknown scalar variable"):
        plot.scatter("input:missing", "output:carry_m")
    with pytest.raises(ContractViolationError, match="unknown point_id"):
        plot.arc_overlay("swing.missing")
    with pytest.raises(ContractViolationError, match="max_arc_vertices"):
        PlotBudget(max_arc_vertices=0)


def test_result_rejects_outcomes_that_are_not_in_trial_order() -> None:
    result = _result()

    with pytest.raises(ContractViolationError, match="trial order"):
        SimulationEnsembleResult(
            tuple(reversed(result.outcomes)),
            result.variation,
            result.traces,
        )


def test_web_visualization_contract_matches_python_identifiers() -> None:
    fixture_path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "variation_visualization_contract.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["coordinate_frame"] == "app_frame:x_target,y_up,z_right"
    assert fixture["point_ids"] == [
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    ]
    assert fixture["trial_statuses"] == [
        status.value
        for status in (
            EVALUATED_HIT,
            EVALUATED_NO_IMPACT,
            NUMERICAL_FAILURE,
        )
    ]
    assert fixture["output_names"] == list(ALL_OUTPUT_NAMES)
