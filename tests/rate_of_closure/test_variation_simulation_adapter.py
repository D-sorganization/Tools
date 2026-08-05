"""Rate-owned adapter from complete simulation runs to ensemble data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.variation.simulation_adapter import (
    APP_FRAME_ID,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationEnsembleRequest,
    run_simulation_ensemble,
    spatial_point_ids,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    NoiseSpec,
    VariationPlan,
    summary_stats,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_DRIVER = get_club("Driver 10.5°")


def _config(
    contact_mode: ContactMode,
    speed_mph: float = 100.0,
    source_kind: str = "double_pendulum",
) -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=speed_mph),
        club=_DRIVER,
        source_kind=source_kind,
        swing_duration_s=0.05,
        contact_mode=contact_mode,
    )


def _request(configs: tuple[SimulationConfig, ...]) -> SimulationEnsembleRequest:
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=1.0),),
        n_runs=len(configs),
        seed=11,
    )
    samples = np.linspace(-1.0, 1.0, len(configs)).reshape(-1, 1)
    return SimulationEnsembleRequest(plan=plan, sampled_inputs=samples, configs=configs)


def test_ensemble_distinguishes_hit_miss_and_numerical_failure() -> None:
    hit = _config(ContactMode.DELIVERY_INSPECTION)
    miss = _config(ContactMode.FIXED_BALL_CONTACT)
    failing = _config(ContactMode.DELIVERY_INSPECTION, speed_mph=99.0)

    def executor(config: SimulationConfig) -> SimulationRun:
        if config.scenario.clubhead_speed_mph == 99.0:
            raise RuntimeError("planted trial failure")
        return run_simulation(config)

    result = run_simulation_ensemble(_request((hit, miss, failing)), executor)

    assert tuple(item.status for item in result.outcomes) == (
        EVALUATED_HIT,
        EVALUATED_NO_IMPACT,
        NUMERICAL_FAILURE,
    )
    assert result.outcomes[2].failure_type == "RuntimeError"
    assert result.outcomes[2].failure_message == "planted trial failure"
    np.testing.assert_array_equal(result.variation.success, [True, True, False])
    assert result.variation.output_column("closest_approach_m").shape == (2,)
    assert result.variation.output_column("clubhead_speed_mps").shape == (1,)
    stats = {item.name: item for item in summary_stats(result.variation)}
    assert stats["closest_approach_m"].n == 2
    assert stats["clubhead_speed_mps"].n == 1


def test_miss_retains_geometry_and_contact_but_nulls_impact_and_shot_metrics() -> None:
    result = run_simulation_ensemble(
        _request(
            (
                _config(ContactMode.DELIVERY_INSPECTION),
                _config(ContactMode.FIXED_BALL_CONTACT),
            )
        )
    )
    miss = result.outcomes[1]

    assert miss.status == EVALUATED_NO_IMPACT
    assert miss.value("candidate_time_s") is not None
    assert miss.value("closest_approach_m") is not None
    assert miss.value("contact_margin_m") is not None
    for name in result.impact_output_names + result.shot_output_names:
        assert miss.value(name) is None

    traces = result.traces
    assert traces.coordinate_frame == APP_FRAME_ID
    assert traces.point_ids == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert np.all(traces.sample_valid[1])
    assert np.all(np.isfinite(traces.positions_m[1]))
    assert traces.impact_sample_indices[1] == -1


def test_numerical_failure_has_an_explicit_invalid_trace_row() -> None:
    good = _config(ContactMode.DELIVERY_INSPECTION)
    failing = replace(good, scenario=ImpactScenario(clubhead_speed_mph=99.0))

    def executor(config: SimulationConfig) -> SimulationRun:
        if config.scenario.clubhead_speed_mph == 99.0:
            raise FloatingPointError("non-finite state")
        return run_simulation(config)

    result = run_simulation_ensemble(_request((good, failing)), executor)

    assert not np.any(result.traces.sample_valid[1])
    assert np.all(np.isnan(result.traces.positions_m[1]))
    assert result.traces.impact_sample_indices[1] == -1
    assert all(value is None for value in result.outcomes[1].values.values())


def test_all_numerical_failures_return_an_all_invalid_common_trace() -> None:
    configs = (
        _config(ContactMode.DELIVERY_INSPECTION),
        _config(ContactMode.FIXED_BALL_CONTACT),
    )

    def executor(_config: SimulationConfig) -> SimulationRun:
        raise RuntimeError("all trials failed")

    result = run_simulation_ensemble(_request(configs), executor)

    assert all(item.status is NUMERICAL_FAILURE for item in result.outcomes)
    np.testing.assert_array_equal(result.variation.success, [False, False])
    assert result.traces.point_ids == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert result.traces.sample_times_s.shape == (51,)
    assert not np.any(result.traces.sample_valid)
    assert np.all(np.isnan(result.traces.positions_m))


def test_unexpected_programming_error_is_not_hidden_as_a_trial_failure() -> None:
    request = _request((_config(ContactMode.DELIVERY_INSPECTION),))

    def broken_executor(_config: SimulationConfig) -> SimulationRun:
        raise TypeError("programming defect")

    with pytest.raises(TypeError, match="programming defect"):
        run_simulation_ensemble(request, broken_executor)


def test_all_failure_grid_matches_the_configured_source_duration() -> None:
    fractional = replace(
        _config(ContactMode.DELIVERY_INSPECTION), swing_duration_s=0.0504
    )
    expected = run_simulation(fractional).swing_times

    def executor(_config: SimulationConfig) -> SimulationRun:
        raise RuntimeError("planted after-grid failure")

    result = run_simulation_ensemble(_request((fractional,)), executor)

    np.testing.assert_array_equal(result.traces.sample_times_s, expected)


def test_spatial_point_ids_are_explicit_and_not_torque_joint_ids() -> None:
    double = run_simulation(_config(ContactMode.DELIVERY_INSPECTION))
    triple = run_simulation(
        _config(
            ContactMode.DELIVERY_INSPECTION,
            source_kind="triple_pendulum",
        )
    )
    manual = run_simulation(
        _config(ContactMode.DELIVERY_INSPECTION, source_kind="manual")
    )

    assert spatial_point_ids(double) == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert spatial_point_ids(triple) == (
        "swing.pivot",
        "swing.elbow",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert spatial_point_ids(manual) == ("swing.clubhead.reference",)
    assert set(spatial_point_ids(double)).isdisjoint(double.swing_joint_ids)


def test_request_rejects_mixed_sources_before_execution() -> None:
    with pytest.raises(ContractViolationError, match="same source_kind"):
        _request(
            (
                _config(ContactMode.DELIVERY_INSPECTION),
                _config(ContactMode.DELIVERY_INSPECTION, source_kind="manual"),
            )
        )


def test_request_rejects_sample_shape_that_does_not_match_plan() -> None:
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=1.0),),
        n_runs=2,
    )
    with pytest.raises(ContractViolationError, match="sampled_inputs"):
        SimulationEnsembleRequest(
            plan=plan,
            sampled_inputs=np.zeros((2, 2)),
            configs=(
                _config(ContactMode.DELIVERY_INSPECTION),
                _config(ContactMode.FIXED_BALL_CONTACT),
            ),
        )
