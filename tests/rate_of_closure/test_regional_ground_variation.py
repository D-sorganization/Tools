"""Seeded regional-ground material variation contracts for #4273."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from rate_of_closure.variation.regional_ground_variation import (
    GROUND_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY,
    INPUT_NORMAL_RESTITUTION_KEY,
    INPUT_ROLLING_RESISTANCE_KEY,
    GroundRegionalVariationRequest,
    GroundRegionalVariationTrial,
    register_ground_variation_variables,
    run_regional_ground_variation,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight import execute_regional_ground_from_flight
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _crossing_result,
    _launch,
    _plan,
    _settings,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation import registry as variation_registry
from tests.rate_of_closure.regional_ground_target_support import transfer_failure

BALL_SPEED_KEY = "swing_sim.flight.launch.ball_speed_mph"


@pytest.fixture(autouse=True)
def _isolated_ground_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register extension variables without leaking them across test modules."""
    isolated = dict(variation_registry.variable_registry())
    monkeypatch.setattr(variation_registry, "_REGISTRY", isolated)
    register_ground_variation_variables()


def _variation_plan(
    *,
    seed: int = 1729,
    n_runs: int = 4,
    include_restitution: bool = True,
    rolling_base: float = 0.04,
    rolling_lower: float | bool | None = 0.02,
    rolling_upper: float | None = 0.08,
) -> VariationPlan:
    noise = [
        NoiseSpec(
            GROUND_ROLLING_RESISTANCE_KEY,
            distribution="uniform",
            scale=0.02,
            lower=rolling_lower,
            upper=rolling_upper,
            spec_id="ground-rolling-resistance",
        )
    ]
    if include_restitution:
        noise.append(
            NoiseSpec(
                GROUND_NORMAL_RESTITUTION_KEY,
                distribution="uniform",
                scale=0.1,
                lower=0.2,
                upper=0.6,
                spec_id="ground-normal-restitution",
            )
        )
    return VariationPlan(
        mode="launch",
        base_variables={
            GROUND_NORMAL_RESTITUTION_KEY: 0.4,
            GROUND_ROLLING_RESISTANCE_KEY: rolling_base,
        },
        noise=tuple(noise),
        n_runs=n_runs,
        seed=seed,
    )


def _request(
    plan: VariationPlan | None = None,
    *,
    max_rows: int = 8,
) -> GroundRegionalVariationRequest:
    return GroundRegionalVariationRequest(
        plan=_variation_plan() if plan is None else plan,
        regional_plan=_plan(),
        result_id="seeded-ground-study",
        source_provenance="pytest/exact-parent-f13f0908",
        max_rows=max_rows,
        series_id="driver",
    )


def _json_bytes(dataset: object) -> str:
    return json.dumps(dataset.to_wire(), sort_keys=True, separators=(",", ":"))


def test_seed_is_deterministic_and_each_stream_is_subset_stable() -> None:
    failure = transfer_failure()
    request = _request()

    first = run_regional_ground_variation(request, lambda _trial: failure)
    second = run_regional_ground_variation(request, lambda _trial: failure)
    rolling_only = run_regional_ground_variation(
        _request(_variation_plan(include_restitution=False)), lambda _trial: failure
    )

    assert _json_bytes(first) == _json_bytes(second)
    assert [row.values[INPUT_ROLLING_RESISTANCE_KEY] for row in first.rows] == [
        row.values[INPUT_ROLLING_RESISTANCE_KEY] for row in rolling_only.rows
    ]
    changed_seed = run_regional_ground_variation(
        _request(_variation_plan(seed=1730)), lambda _trial: failure
    )
    assert [row.values[INPUT_ROLLING_RESISTANCE_KEY] for row in first.rows] != [
        row.values[INPUT_ROLLING_RESISTANCE_KEY] for row in changed_seed.rows
    ]


def test_trials_rebind_plan_identity_provenance_and_preserve_order() -> None:
    captured: list[GroundRegionalVariationTrial] = []
    failure = transfer_failure()

    def observe(trial: GroundRegionalVariationTrial):
        captured.append(trial)
        return failure

    dataset = run_regional_ground_variation(_request(), observe)

    assert [trial.trial_index for trial in captured] == list(range(4))
    assert [row.trial_index for row in dataset.rows] == list(range(4))
    assert [row.row_id for row in dataset.rows] == [
        f"series:driver/trial:{index}" for index in range(4)
    ]
    assert len({trial.input_sha256 for trial in captured}) == 4
    assert len({trial.regional_plan.request_id for trial in captured}) == 4
    for trial, row in zip(captured, dataset.rows, strict=True):
        digest = regional_plan_request_sha256(trial.regional_plan)
        assert trial.regional_plan.provenance.input_sha256 == trial.input_sha256
        assert row.values[INPUT_ROLLING_RESISTANCE_KEY] == pytest.approx(
            trial.sampled_values[GROUND_ROLLING_RESISTANCE_KEY]
        )
        assert row.values[INPUT_NORMAL_RESTITUTION_KEY] == pytest.approx(
            trial.sampled_values[GROUND_NORMAL_RESTITUTION_KEY]
        )
        assert row.attributes is not None
        assert row.attributes["variation_seed"] == "1729"
        assert row.attributes["variation_trial_index"] == str(trial.trial_index)
        assert row.attributes["variation_input_sha256"] == trial.input_sha256
        assert row.attributes["variation_regional_plan_sha256"] == digest


def test_real_pipeline_responds_to_sampled_rolling_resistance() -> None:
    request = _request(_variation_plan(include_restitution=False, n_runs=5))

    def execute(trial: GroundRegionalVariationTrial):
        settings = _settings()
        varied_surface = replace(
            settings.surface,
            rolling_resistance=trial.regional_plan.base_surface.rolling_resistance,
            normal_restitution=trial.regional_plan.base_surface.normal_restitution,
        )
        varied_settings = replace(settings, surface=varied_surface)
        return execute_regional_ground_from_flight(
            _crossing_result(),
            _launch(),
            varied_settings,
            trial.regional_plan,
            capture_speed_m_s=3.0,
        )

    dataset = run_regional_ground_variation(request, execute)
    rows = sorted(
        dataset.rows, key=lambda row: row.values[INPUT_ROLLING_RESISTANCE_KEY]
    )

    assert all(row.cohort == "complete" for row in rows)
    assert (
        rows[0].values["metric.total_distance"]
        > rows[-1].values["metric.total_distance"]
    )


def test_transfer_failures_keep_sampled_inputs_but_null_every_ground_metric() -> None:
    dataset = run_regional_ground_variation(
        _request(_variation_plan(include_restitution=False)),
        lambda _trial: transfer_failure(),
    )

    input_keys = {INPUT_ROLLING_RESISTANCE_KEY}
    for row in dataset.rows:
        assert row.cohort == "unavailable"
        assert row.values[INPUT_ROLLING_RESISTANCE_KEY] is not None
        assert all(
            value is None for key, value in row.values.items() if key not in input_keys
        )
        assert row.attributes is not None
        assert row.attributes["transfer_reason"] == "no_physical_contact"


@pytest.mark.parametrize(
    "plan_factory",
    [
        lambda: VariationPlan(
            mode="launch",
            base_variables={BALL_SPEED_KEY: 70.0},
            noise=(NoiseSpec(BALL_SPEED_KEY, lower=60.0, upper=80.0),),
            n_runs=2,
        ),
        lambda: _variation_plan(rolling_base=0.05),
        lambda: _variation_plan(rolling_lower=None),
        lambda: _variation_plan(rolling_lower=-0.01),
        lambda: _variation_plan(rolling_upper=1.01),
        lambda: _variation_plan(rolling_lower=False),
    ],
)
def test_invalid_keys_base_bounds_and_bool_fail_before_execution(
    plan_factory: Callable[[], VariationPlan],
) -> None:
    calls = 0
    plan = plan_factory()

    def forbidden(_trial: GroundRegionalVariationTrial):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run")

    with pytest.raises(ContractViolationError):
        run_regional_ground_variation(_request(plan), forbidden)
    assert calls == 0


def test_nonfinite_sample_and_row_cap_fail_before_execution() -> None:
    plan = _variation_plan(n_runs=3)
    object.__setattr__(plan.noise[0], "scale", float("nan"))
    calls = 0

    def forbidden(_trial: GroundRegionalVariationTrial):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run")

    with pytest.raises(ContractViolationError, match="finite"):
        run_regional_ground_variation(_request(plan), forbidden)
    with pytest.raises(ContractViolationError, match="max_rows"):
        run_regional_ground_variation(
            _request(_variation_plan(n_runs=3), max_rows=2), forbidden
        )
    assert calls == 0


def test_executor_must_return_exact_outcome_bound_to_the_sampled_plan() -> None:
    with pytest.raises(ContractViolationError, match="pipeline result or transfer"):
        run_regional_ground_variation(_request(), lambda _trial: object())

    baseline = execute_regional_ground_from_flight(
        _crossing_result(), _launch(), _settings(), _plan(), capture_speed_m_s=3.0
    )
    with pytest.raises(ContractViolationError, match="sampled regional plan"):
        run_regional_ground_variation(_request(), lambda _trial: baseline)
