"""Validated Morris global-sensitivity screening contracts (#4142 R13)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import (
    MorrisDesign,
    MorrisFactor,
    MorrisObservations,
    MorrisOutput,
    NoiseSpec,
    analyze_morris,
    generate_morris_design,
)

pytestmark = pytest.mark.physics


def _factors() -> tuple[MorrisFactor, ...]:
    return (
        MorrisFactor.from_noise_spec(
            NoiseSpec(
                variable_key="swing_sim.impact.delivery.face_angle_deg",
                spec_id="face-window",
                time_window_s=(0.01, 0.02),
                point_ids=("clubhead",),
            ),
            lower=0.0,
            upper=1.0,
        ),
        MorrisFactor(
            spec_id="speed-global",
            variable_key="swing_sim.impact.delivery.clubhead_speed_mps",
            lower=0.0,
            upper=1.0,
            unit="m/s",
        ),
    )


def _observations(
    response: Callable[[np.ndarray], float],
    *,
    trajectories: int = 12,
    outcomes: np.ndarray | None = None,
) -> MorrisObservations:
    design = generate_morris_design(_factors(), trajectories=trajectories, seed=73)
    values = np.empty((trajectories, 3, 1), dtype=float)
    for trajectory in range(trajectories):
        for sample in range(3):
            values[trajectory, sample, 0] = response(
                design.physical_points[trajectory, sample]
            )
    if outcomes is None:
        statuses = np.empty((trajectories, 3), dtype=object)
        statuses.fill(TrialEvaluationStatus.EVALUATED_HIT)
    else:
        statuses = np.asarray(outcomes, dtype=object)
    status_values = np.asarray(
        [str(getattr(item, "value", item)) for item in statuses.ravel()]
    ).reshape(statuses.shape)
    values[status_values != TrialEvaluationStatus.EVALUATED_HIT.value] = np.nan
    return MorrisObservations(
        design=design,
        outputs=(
            MorrisOutput(
                name="clubhead_x_m",
                unit="m",
                target_kind="state-point",
                target_time_s=0.03,
                target_point_id="clubhead",
                coordinate_frame="app_frame:x_target,y_up,z_right",
            ),
        ),
        values=values,
        outcomes=statuses,
    )


def test_additive_fixture_recovers_elementary_effects_and_source_locus() -> None:
    observations = _observations(lambda point: 2.0 * point[0] + 3.0 * point[1])

    report = analyze_morris(observations)
    face = report.estimate("face-window", "clubhead_x_m")
    speed = report.estimate("speed-global", "clubhead_x_m")

    assert face.mu_star == pytest.approx(2.0)
    assert speed.mu_star == pytest.approx(3.0)
    assert face.sigma == pytest.approx(0.0, abs=1e-12)
    assert speed.sigma == pytest.approx(0.0, abs=1e-12)
    assert face.mu_star_standard_error == pytest.approx(0.0, abs=1e-12)
    assert face.availability == "available"
    assert face.sample_adequacy == "adequate"
    assert face.source_time_window_s == (0.01, 0.02)
    assert face.source_point_ids == ("clubhead",)
    assert face.source_unit == "deg"
    assert face.source_lower == 0.0
    assert face.source_upper == 1.0
    assert face.output_unit == "m"
    assert face.coordinate_frame == "app_frame:x_target,y_up,z_right"
    assert face.target_time_s == pytest.approx(0.03)
    assert face.target_point_id == "clubhead"
    assert report.seed == 73
    assert report.total_design_samples == 36
    assert report.normalized_step == pytest.approx(2.0 / 3.0)


def test_interacting_fixture_reports_nonzero_sigma_without_claiming_causality() -> None:
    observations = _observations(lambda point: point[0] * point[1])

    report = analyze_morris(observations)

    assert report.estimate("face-window", "clubhead_x_m").sigma > 0.0
    assert report.estimate("speed-global", "clubhead_x_m").sigma > 0.0
    assert report.estimate("face-window", "clubhead_x_m").mu_star_standard_error > 0.0
    assert "nonlinearity" in report.interaction_caveat
    assert "interaction" in report.interaction_caveat


def test_constant_output_is_explicit_and_not_reported_as_unavailable() -> None:
    report = analyze_morris(_observations(lambda _point: 7.0))
    estimate = report.estimate("face-window", "clubhead_x_m")

    assert estimate.availability == "constant-output"
    assert estimate.sample_adequacy == "adequate"
    assert estimate.mu == 0.0
    assert estimate.mu_star == 0.0
    assert estimate.sigma == 0.0


def test_tiny_nonzero_effect_is_not_misclassified_as_constant() -> None:
    estimate = analyze_morris(_observations(lambda point: 1e-15 * point[0])).estimate(
        "face-window", "clubhead_x_m"
    )

    assert estimate.availability == "available"
    assert estimate.mu_star > 0.0


def test_insufficient_pairs_retain_no_impact_and_failure_denominators() -> None:
    statuses = np.empty((4, 3), dtype=object)
    statuses.fill(TrialEvaluationStatus.EVALUATED_HIT)
    statuses[0, :] = TrialEvaluationStatus.EVALUATED_NO_IMPACT
    statuses[1, :] = TrialEvaluationStatus.NUMERICAL_FAILURE
    report = analyze_morris(
        _observations(
            lambda point: point[0] + point[1], trajectories=4, outcomes=statuses
        ),
        minimum_effects=3,
    )

    for factor_id in ("face-window", "speed-global"):
        estimate = report.estimate(factor_id, "clubhead_x_m")
        assert estimate.availability == "insufficient-data"
        assert estimate.sample_adequacy == "insufficient"
        assert estimate.total_effect_pairs == 4
        assert estimate.valid_effect_pairs == 2
        assert estimate.no_impact_pairs == 1
        assert estimate.no_impact_unavailable_pairs == 1
        assert estimate.failed_pairs == 1
        assert estimate.nonfinite_pairs == 0
        assert np.isnan(estimate.mu_star)


def test_invalid_observations_reject_fabricated_no_impact_output() -> None:
    design = generate_morris_design(_factors(), trajectories=4, seed=11)
    values = np.zeros((4, 3, 1), dtype=float)
    outcomes = np.empty((4, 3), dtype=object)
    outcomes.fill(TrialEvaluationStatus.EVALUATED_NO_IMPACT)

    with pytest.raises(Exception, match="no-impact samples must not contain"):
        MorrisObservations(
            design=design,
            outputs=(MorrisOutput("carry_m", unit="m", target_kind="shot-outcome"),),
            values=values,
            outcomes=outcomes,
        )


def test_no_impact_state_metric_contributes_when_output_is_available() -> None:
    observations = _observations(lambda point: point[0] + point[1], trajectories=4)
    values = np.array(observations.values, copy=True)
    outcomes = np.array(observations.outcomes, dtype=object, copy=True)
    outcomes[0, :] = TrialEvaluationStatus.EVALUATED_NO_IMPACT
    design = observations.design
    for sample in range(3):
        values[0, sample, 0] = float(np.sum(design.physical_points[0, sample]))
    retained = MorrisObservations(
        design=design,
        outputs=observations.outputs,
        values=values,
        outcomes=outcomes,
    )

    estimate = analyze_morris(retained, minimum_effects=2).estimate(
        "face-window", "clubhead_x_m"
    )

    assert estimate.valid_effect_pairs == 4
    assert estimate.no_impact_pairs == 1
    assert estimate.no_impact_unavailable_pairs == 0


def test_design_is_deterministic_valid_and_changes_one_factor_per_step() -> None:
    first = generate_morris_design(_factors(), trajectories=9, levels=4, seed=91)
    second = generate_morris_design(_factors(), trajectories=9, levels=4, seed=91)

    assert np.array_equal(first.normalized_points, second.normalized_points)
    assert np.array_equal(first.changed_factor_indices, second.changed_factor_indices)
    assert np.all((first.normalized_points >= 0.0) & (first.normalized_points <= 1.0))
    changes = np.count_nonzero(np.diff(first.normalized_points, axis=1), axis=2)
    assert np.array_equal(changes, np.ones((9, 2), dtype=int))


@pytest.mark.parametrize("levels", [2, 3, 5])
def test_design_rejects_levels_that_violate_morris_grid_assumptions(
    levels: int,
) -> None:
    expected = "levels must be an integer >= 4" if levels < 4 else "levels must be even"
    with pytest.raises(Exception, match=expected):
        generate_morris_design(_factors(), trajectories=4, levels=levels, seed=0)


@pytest.mark.parametrize("invalid_count", [True, 4.0])
def test_design_rejects_non_integer_count_contracts(invalid_count: object) -> None:
    with pytest.raises(Exception, match="trajectories must be an integer"):
        generate_morris_design(_factors(), trajectories=invalid_count)  # type: ignore[arg-type]


def test_design_rejects_fractional_changed_factor_indices() -> None:
    design = generate_morris_design(_factors(), trajectories=2)
    with pytest.raises(Exception, match="changed_factor_indices must contain integers"):
        MorrisDesign(
            factors=design.factors,
            trajectories=design.trajectories,
            levels=design.levels,
            seed=design.seed,
            normalized_points=design.normalized_points,
            changed_factor_indices=design.changed_factor_indices.astype(float),
            signed_steps=design.signed_steps,
        )


def test_analysis_rejects_boolean_adequacy_threshold() -> None:
    with pytest.raises(Exception, match="minimum_effects must be an integer"):
        analyze_morris(_observations(lambda point: float(np.sum(point))), True)


def test_cross_runtime_golden_fixture_matches_serialized_report() -> None:
    report = analyze_morris(
        _observations(lambda point: 2.0 * point[0] + 3.0 * point[1])
    )
    fixture_path = (
        Path(__file__).parents[5]
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "morris_global_sensitivity_golden_v1.json"
    )

    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert report.to_json_dict() == expected
