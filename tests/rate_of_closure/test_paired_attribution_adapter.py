"""Rate adapter tests for shared paired-attribution authority."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.variation.paired_attribution_adapter import (
    RATE_PAIRED_ATTRIBUTION_ADAPTER_ID,
    build_rate_paired_attribution_input,
)
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    CONTACT_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationTrialOutcome,
)
from shared.python.swing_sim.variation import sample_inputs
from shared.python.swing_sim.variation.paired_attribution import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NUMERICAL_FAILURE,
    AttributionTarget,
    compute_paired_attribution,
)
from shared.python.swing_sim.variation.tests.noise_response_test_support import (
    BALL_SPEED,
    LAUNCH_ANGLE,
    ResponseFixtureConfig,
    build_response_inputs,
    default_fixture_config,
)


def _values(status: object, offset: float) -> dict[str, float | None]:
    if status is NUMERICAL_FAILURE:
        return dict.fromkeys(ALL_OUTPUT_NAMES)
    values: dict[str, float | None] = {
        name: offset + index for index, name in enumerate(ALL_OUTPUT_NAMES)
    }
    if status is EVALUATED_NO_IMPACT:
        for name in set(ALL_OUTPUT_NAMES) - set(CONTACT_OUTPUT_NAMES):
            values[name] = None
    return values


def _outcome(index: int, status: object, offset: float) -> SimulationTrialOutcome:
    return SimulationTrialOutcome(
        index,
        status,
        _values(status, offset),
        failure_type="solver" if status is NUMERICAL_FAILURE else None,
        failure_message="bounded failure" if status is NUMERICAL_FAILURE else None,
    )


def _targets() -> tuple[AttributionTarget, ...]:
    return (
        AttributionTarget(
            target_id="clubhead-x-0.5",
            kind="state",
            unit="m",
            metric_id="position_x_m",
            coordinate_frame="swing.world",
            point_id="swing.clubhead.reference",
            coordinate_value=0.5,
            coordinate_unit="s",
        ),
        AttributionTarget(
            target_id="impact-speed",
            kind="impact",
            unit="m/s",
            metric_id="clubhead_speed_mps",
        ),
        AttributionTarget(
            target_id="carry",
            kind="shot",
            unit="m",
            metric_id="carry_m",
        ),
    )


def test_adapter_binds_resampled_state_impact_and_shot_pairs() -> None:
    response_input = build_response_inputs(default_fixture_config())[0]
    baseline = tuple(_outcome(index, EVALUATED_HIT, 10.0 + index) for index in range(4))
    perturbed = (
        _outcome(0, EVALUATED_HIT, 20.0),
        _outcome(1, EVALUATED_NO_IMPACT, 20.0),
        _outcome(2, NUMERICAL_FAILURE, 0.0),
        _outcome(3, EVALUATED_HIT, 23.0),
    )

    field_input = build_rate_paired_attribution_input(
        response_input, _targets(), baseline, perturbed
    )
    record = compute_paired_attribution(field_input)

    assert field_input.baseline_context.adapter_id == RATE_PAIRED_ATTRIBUTION_ADAPTER_ID
    assert field_input.source.variable_key.endswith("ball_speed_mph")
    assert field_input.source.unit == "mph"
    assert field_input.source.time_window_s is None
    assert record.availability[0].tolist() == [AVAILABILITY_AVAILABLE] * 3
    assert record.availability[1].tolist() == [
        AVAILABILITY_AVAILABLE,
        AVAILABILITY_NO_IMPACT,
        AVAILABILITY_NO_IMPACT,
    ]
    assert record.availability[2].tolist() == [AVAILABILITY_NUMERICAL_FAILURE] * 3
    assert record.availability[3].tolist() == [AVAILABILITY_AVAILABLE] * 3
    assert record.signed_response[0, 1] == 10.0
    assert np.isnan(record.signed_response[1, 1])


def test_adapter_preserves_trace_gap_as_missing_state_only() -> None:
    config = default_fixture_config()
    perturbed_valid = config.perturbed_valid.copy()
    perturbed_valid[0, 3, 1] = False
    altered = ResponseFixtureConfig(
        deltas=config.deltas,
        coefficients=config.coefficients,
        baseline_positions_m=config.baseline_positions_m,
        baseline_valid=config.baseline_valid,
        perturbed_valid=perturbed_valid,
    )
    response_input = build_response_inputs(altered)[0]
    outcomes = tuple(_outcome(index, EVALUATED_HIT, 10.0 + index) for index in range(4))

    record = compute_paired_attribution(
        build_rate_paired_attribution_input(
            response_input, _targets(), outcomes, outcomes
        )
    )

    assert record.availability[3].tolist() == [
        AVAILABILITY_MISSING,
        AVAILABILITY_AVAILABLE,
        AVAILABILITY_AVAILABLE,
    ]
    assert np.isnan(record.signed_response[3, 0])


@pytest.mark.parametrize(
    "target",
    [
        AttributionTarget(
            "bad-state",
            "state",
            "m",
            "not-position",
            "swing.world",
            "swing.wrist",
            0.5,
            "s",
        ),
        AttributionTarget("bad-impact", "impact", "m", "carry_m"),
        AttributionTarget("bad-shot", "shot", "m", "clubhead_speed_mps"),
    ],
)
def test_adapter_rejects_target_registry_or_kind_drift(
    target: AttributionTarget,
) -> None:
    response_input = build_response_inputs(default_fixture_config())[0]
    outcomes = tuple(_outcome(index, EVALUATED_HIT, 10.0 + index) for index in range(4))

    with pytest.raises(ValueError, match="target"):
        build_rate_paired_attribution_input(
            response_input, (target,), outcomes, outcomes
        )


def test_adapter_rejects_outcome_order_and_trace_contract_drift() -> None:
    response_input = build_response_inputs(default_fixture_config())[0]
    outcomes = tuple(_outcome(index, EVALUATED_HIT, 10.0 + index) for index in range(4))

    with pytest.raises(ValueError, match="outcome order"):
        build_rate_paired_attribution_input(
            response_input,
            _targets(),
            outcomes[::-1],
            outcomes,
        )


@pytest.mark.parametrize("design", ["discrete", "bounded", "correlated"])
def test_adapter_rejects_non_estimable_source_designs(design: str) -> None:
    from shared.python.swing_sim.variation.group_spec import PerturbationGroup
    from shared.python.swing_sim.variation.spec import NoiseSpec

    config = default_fixture_config()
    bounded = design == "bounded"
    specs = (
        NoiseSpec(
            BALL_SPEED,
            scale=2.0,
            lower=90.0 if bounded else None,
            upper=110.0 if bounded else None,
            spec_id="input.ball-speed",
        ),
        NoiseSpec(LAUNCH_ANGLE, scale=4.0, spec_id="input.launch-angle"),
    )
    groups = (
        (
            PerturbationGroup(
                group_id="joint-inputs",
                spec_ids=("input.ball-speed", "input.launch-angle"),
                matrix=((1.0, 0.5), (0.5, 1.0)),
            ),
        )
        if design == "correlated"
        else ()
    )
    altered = ResponseFixtureConfig(
        deltas=config.deltas,
        coefficients=config.coefficients,
        baseline_positions_m=config.baseline_positions_m,
        baseline_valid=config.baseline_valid,
        perturbed_valid=config.perturbed_valid,
        specs=specs,
        groups=groups,
        input_kinds=(
            "discrete" if design == "discrete" else "continuous",
            "continuous",
        ),
    )
    response_input = build_response_inputs(altered)[0]
    outcomes = tuple(_outcome(index, EVALUATED_HIT, 10.0 + index) for index in range(4))

    with pytest.raises(ValueError, match="independently estimable OAT"):
        build_rate_paired_attribution_input(
            response_input, _targets(), outcomes, outcomes
        )


def test_planted_intervention_pairs_recover_response_structure() -> None:
    config = default_fixture_config()
    planted_deltas = config.deltas[0]
    response_input = build_response_inputs(config)[0]

    speed_gain = 1.5
    carry_gain = 2.5
    baseline_speed = 45.0
    baseline_carry = 200.0

    baseline_outcomes = tuple(
        SimulationTrialOutcome(
            index,
            EVALUATED_HIT,
            {
                **_values(EVALUATED_HIT, 0.0),
                "clubhead_speed_mps": baseline_speed,
                "carry_m": baseline_carry,
            },
        )
        for index in range(4)
    )
    perturbed_outcomes = tuple(
        SimulationTrialOutcome(
            index,
            EVALUATED_HIT,
            {
                **_values(EVALUATED_HIT, 0.0),
                "clubhead_speed_mps": (
                    baseline_speed + speed_gain * planted_deltas[index]
                ),
                "carry_m": baseline_carry + carry_gain * planted_deltas[index],
            },
        )
        for index in range(4)
    )

    targets = _targets()
    field_input = build_rate_paired_attribution_input(
        response_input, targets, baseline_outcomes, perturbed_outcomes
    )
    record = compute_paired_attribution(field_input)

    # Recovered intervention delta matches planted deltas exactly
    recovered_deltas = record.perturbed_source_values - record.baseline_source_values
    assert np.array_equal(recovered_deltas, planted_deltas)

    # Recovered signed response matches planted scalar deltas
    expected_speed_delta = speed_gain * planted_deltas
    expected_carry_delta = carry_gain * planted_deltas
    assert np.allclose(record.signed_response[:, 1], expected_speed_delta)
    assert np.allclose(record.signed_response[:, 2], expected_carry_delta)

    # Recovered local response matches planted sensitivity / intervention structure
    assert np.allclose(record.local_response_per_source_unit[:, 1], speed_gain)
    assert np.allclose(record.local_response_per_source_unit[:, 2], carry_gain)

    # State target: coefficients[0][1][1][0] = 0.50, scale = 2.0 -> gain = 0.25
    expected_state_gain = 0.25
    expected_state_delta = expected_state_gain * planted_deltas
    assert np.allclose(record.signed_response[:, 0], expected_state_delta)
    assert np.allclose(record.local_response_per_source_unit[:, 0], expected_state_gain)

    # Detection: pseudorandom draws differ from planted intervention deltas
    plan = response_input.baseline.traces.variation.plan
    rng_draws = np.asarray(sample_inputs(plan), dtype=float)[:, 0]
    assert not np.array_equal(recovered_deltas, rng_draws)
    with pytest.raises(AssertionError):
        assert np.array_equal(rng_draws, planted_deltas)
