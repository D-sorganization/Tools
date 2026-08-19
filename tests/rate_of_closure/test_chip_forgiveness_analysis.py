"""Decision-analysis contracts for chip-shot forgiveness studies."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rate_of_closure.variation.chip_forgiveness import (
    ChipStudyMetadata,
    ChipTrialCohort,
    ChipTrialRecord,
    summarize_chip_trials,
)
from rate_of_closure.variation.forgiveness_ranking import (
    ChipCandidateScore,
    pareto_frontier,
)


def _record(
    index: int,
    cohort: ChipTrialCohort,
    loss: float,
    *,
    constraint_violated: bool = False,
) -> ChipTrialRecord:
    return ChipTrialRecord(
        trial_index=index,
        cohort=cohort,
        loss=loss,
        constraint_violated=constraint_violated,
    )


def _metadata(candidate_id: str = "wedge-a") -> ChipStudyMetadata:
    return ChipStudyMetadata(
        candidate_id=candidate_id,
        plan_schema="swing-sim.variation-plan/v2",
        coordinate_frame="app_frame:x_target,y_up,z_right",
        seed=41,
        noise_model_id="delivery-correlated-v1",
        objective_id="30-yard-chip-balanced-v1",
        turf_profile_id="illustrative-firm-fairway",
        turf_calibration_status="illustrative",
        solver_id="rate-of-closure/canonical",
        sampling_design="iid-monte-carlo-joint",
        inference_method_id="wilson+mulberry32-iid-bootstrap-v1",
        limitations="Conditional engineering comparison only.",
    )


def test_summary_keeps_failures_in_all_trial_probabilities_and_loss() -> None:
    records = (
        _record(0, ChipTrialCohort.BALL_FIRST, 1.0),
        _record(1, ChipTrialCohort.BALL_FIRST, 2.0),
        _record(2, ChipTrialCohort.GROUND_FIRST, 4.0, constraint_violated=True),
        _record(3, ChipTrialCohort.NUMERICAL_FAILURE, 9.0, constraint_violated=True),
    )

    summary = summarize_chip_trials(
        _metadata(), records, cvar_tail_fraction=0.5, bootstrap_samples=512
    )

    assert summary.sample_count == 4
    assert summary.cohorts[ChipTrialCohort.BALL_FIRST].count == 2
    assert summary.cohorts[ChipTrialCohort.BALL_FIRST].probability == pytest.approx(0.5)
    failure_probability = summary.cohorts[ChipTrialCohort.NUMERICAL_FAILURE].probability
    assert failure_probability == pytest.approx(0.25)
    assert summary.expected_loss == pytest.approx(4.0)
    assert summary.cvar_loss == pytest.approx(6.5)
    assert summary.constraint_violation_rate == pytest.approx(0.5)
    assert summary.expected_loss_ci_low <= summary.expected_loss
    assert summary.expected_loss <= summary.expected_loss_ci_high


def test_wilson_intervals_are_bounded_and_non_degenerate() -> None:
    records = tuple(
        _record(
            index,
            ChipTrialCohort.BALL_FIRST if index < 7 else ChipTrialCohort.GROUND_FIRST,
            float(index),
        )
        for index in range(10)
    )

    summary = summarize_chip_trials(_metadata(), records, bootstrap_samples=128)
    interval = summary.cohorts[ChipTrialCohort.BALL_FIRST]

    assert 0.0 < interval.ci_low < interval.probability < interval.ci_high < 1.0
    assert interval.ci_low == pytest.approx(0.396778, abs=1e-6)
    assert interval.ci_high == pytest.approx(0.892209, abs=1e-6)


def test_seeded_bootstrap_and_convergence_are_reproducible() -> None:
    records = tuple(
        _record(index, ChipTrialCohort.BALL_ONLY, float(index % 5))
        for index in range(25)
    )

    first = summarize_chip_trials(_metadata(), records, bootstrap_samples=256)
    second = summarize_chip_trials(_metadata(), records, bootstrap_samples=256)

    assert first.expected_loss_ci_low == second.expected_loss_ci_low
    assert first.expected_loss_ci_high == second.expected_loss_ci_high
    assert (first.expected_loss_ci_low, first.expected_loss_ci_high) == pytest.approx(
        (1.4, 2.57)
    )
    assert first.convergence == second.convergence
    assert first.convergence[-1].sample_count == 25
    assert first.convergence[-1].mean_loss == pytest.approx(2.0)
    assert math.isfinite(first.convergence[-1].standard_error)


def test_bootstrap_matches_the_browser_golden_case() -> None:
    records = (
        _record(0, ChipTrialCohort.BALL_FIRST, 1.0),
        _record(1, ChipTrialCohort.BALL_FIRST, 2.0),
        _record(2, ChipTrialCohort.GROUND_FIRST, 4.0),
        _record(3, ChipTrialCohort.NUMERICAL_FAILURE, 9.0),
    )

    summary = summarize_chip_trials(
        replace(_metadata(), seed=7), records, bootstrap_samples=128
    )

    assert (summary.expected_loss_ci_low, summary.expected_loss_ci_high) == (
        pytest.approx(1.75),
        pytest.approx(7.6625),
    )


def test_trial_contract_rejects_nonfinite_or_sparse_indices() -> None:
    with pytest.raises(ValueError, match="loss must be finite"):
        _record(0, ChipTrialCohort.BALL_FIRST, math.nan)

    records = (
        _record(0, ChipTrialCohort.BALL_FIRST, 0.0),
        _record(2, ChipTrialCohort.BALL_FIRST, 0.0),
    )
    with pytest.raises(ValueError, match="canonical trial order"):
        summarize_chip_trials(_metadata(), records, bootstrap_samples=64)


def _summary(
    candidate_id: str,
    expected_loss: float,
    cvar_loss: float,
    clean_probability: float,
) -> ChipCandidateScore:
    return ChipCandidateScore(
        metadata=_metadata(candidate_id),
        expected_loss=expected_loss,
        cvar_loss=cvar_loss,
        clean_probability=clean_probability,
    )


def test_pareto_frontier_retains_tradeoffs_and_rejects_dominated_candidates() -> None:
    low_mean = _summary("low-mean", 1.0, 4.0, 0.70)
    low_tail = _summary("low-tail", 1.3, 2.0, 0.80)
    dominated = _summary("dominated", 2.0, 5.0, 0.60)

    ranked = pareto_frontier((dominated, low_tail, low_mean))

    assert tuple(item.metadata.candidate_id for item in ranked) == (
        "low-mean",
        "low-tail",
    )


def test_uncalibrated_turf_blocks_turf_ranking_claims() -> None:
    summary = summarize_chip_trials(
        _metadata(),
        (_record(0, ChipTrialCohort.BALL_FIRST, 0.0),),
        bootstrap_samples=64,
    )

    assert summary.supports_turf_rankings is False
    assert "illustrative" in summary.ranking_scope
