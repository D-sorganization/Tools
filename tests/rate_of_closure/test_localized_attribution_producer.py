"""Production paired localized-attribution execution tests."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.variation.localized_attribution import Availability, TrialStatus
from rate_of_closure.variation.localized_attribution_producer import (
    LocalizedAttributionDesign,
    LocalizedAttributionProduction,
    produce_localized_attribution,
)
from rate_of_closure.variation.request_builder import (
    build_simulation_ensemble_request_from_samples,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import (
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    LocalizedTorqueOffset,
)
from shared.python.swing_sim.solver.solve import CancelledError, ProgressReport
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan

from .test_variation_simulation_request import (
    _SHOULDER_TORQUE_OFFSET,
    _WRIST_TORQUE_OFFSET,
    _base_config,
    _localized_spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _plan(*specs: NoiseSpec) -> VariationPlan:
    return VariationPlan(
        mode="swing",
        noise=specs,
        base_variables={spec.variable_key: 0.0 for spec in specs},
        n_runs=7,
        seed=41,
    )


def _state_target() -> object:
    from rate_of_closure.variation.localized_attribution import AttributionTarget

    return AttributionTarget(
        "state.clubhead.x.0_02",
        "state",
        "position_x_m",
        "m",
        "app-frame-cartesian-v1",
        0.02,
        "swing.clubhead.reference",
        "app_frame:x_target,y_up,z_right",
    )


def _scalar_target(kind: str, name: str, unit: str) -> object:
    from rate_of_closure.variation.localized_attribution import AttributionTarget

    convention = (
        "rate-of-closure-impact-v1" if kind == "impact" else "rate-of-closure-flight-v1"
    )
    return AttributionTarget(
        f"{kind}.{name}", kind, name, unit, convention, None, None, None
    )


def _design(
    config: SimulationConfig | None = None,
    *,
    include_wrist: bool = False,
) -> LocalizedAttributionDesign:
    specs = [_localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)]
    deltas = {specs[0].spec_id: 2.0}
    if include_wrist:
        wrist = _localized_spec(_WRIST_TORQUE_OFFSET, WRIST_JOINT_ID)
        specs.append(wrist)
        deltas[wrist.spec_id] = -3.0
    return LocalizedAttributionDesign(
        design_id="test.paired-localized.v1",
        source_plan=_plan(*specs),
        base_config=config or _base_config(),
        targets=(
            _state_target(),
            _scalar_target("impact", "clubhead_speed_mps", "m/s"),
            _scalar_target("shot", "carry_m", "m"),
        ),
        intervention_deltas_nm=deltas,
    )


def test_explicit_sample_builder_preserves_exact_pair_matrix() -> None:
    spec = _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)
    plan = replace(_plan(spec), n_runs=2)
    samples = np.array([[0.0], [2.5]], dtype=float)

    request = build_simulation_ensemble_request_from_samples(
        plan, _base_config(), samples
    )

    np.testing.assert_array_equal(request.sampled_inputs, samples)
    assert tuple(
        config.swing_run_config.commanded_torque_offsets[0].torque_nm
        for config in request.configs
    ) == (0.0, 2.5)


def test_producer_runs_deterministic_one_at_a_time_pairs_and_binds_identity() -> None:
    design = _design(include_wrist=True)
    seen: list[tuple[float, float]] = []

    def executor(config: SimulationConfig) -> SimulationRun:
        offsets = config.swing_run_config.commanded_torque_offsets
        seen.append(tuple(item.torque_nm for item in offsets))
        return run_simulation(config)

    first = produce_localized_attribution(design, executor=executor)
    second = produce_localized_attribution(design)

    assert seen == [(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, -3.0)]
    assert first.design_identity == second.design_identity
    assert first.request_identity == second.request_identity
    assert first.authority.authority_id == second.authority.authority_id
    assert tuple(
        (pair.baseline_trial_index, pair.perturbed_trial_index)
        for pair in first.authority.pairs
    ) == ((0, 1), (2, 3))
    assert len(first.authority.observations) == 6
    assert all(
        item.availability is Availability.AVAILABLE
        for item in first.authority.observations
    )
    with pytest.raises(ContractViolationError, match="bind the design identity"):
        LocalizedAttributionProduction(
            first.authority, "0" * 64, first.request_identity
        )


def test_no_impact_retains_state_and_types_impact_and_shot_unavailability() -> None:
    config = replace(_base_config(), contact_mode=ContactMode.FIXED_BALL_CONTACT)

    production = produce_localized_attribution(_design(config))

    by_kind = {
        target.kind: next(
            observation
            for observation in production.authority.observations
            if observation.target_id == target.target_id
        )
        for target in production.authority.targets
    }
    assert by_kind["state"].availability is Availability.AVAILABLE
    assert by_kind["impact"].availability is Availability.NO_IMPACT_UNAVAILABLE
    assert by_kind["shot"].availability is Availability.NO_IMPACT_UNAVAILABLE
    assert (
        production.authority.pairs[0].baseline_status is TrialStatus.EVALUATED_NO_IMPACT
    )


def test_numerical_failure_is_retained_for_only_the_failed_side() -> None:
    def executor(config: SimulationConfig) -> SimulationRun:
        torque = config.swing_run_config.commanded_torque_offsets[0].torque_nm
        if torque != 0.0:
            raise RuntimeError("planted perturbed failure")
        return run_simulation(config)

    authority = produce_localized_attribution(_design(), executor=executor).authority

    assert authority.pairs[0].baseline_status is TrialStatus.EVALUATED_HIT
    assert authority.pairs[0].perturbed_status is TrialStatus.NUMERICAL_FAILURE
    assert all(
        item.availability is Availability.NUMERICAL_FAILURE
        for item in authority.observations
    )


def test_progress_and_cancellation_cover_the_exact_pair_roster() -> None:
    cancel = threading.Event()
    reports: list[ProgressReport] = []

    def progress(report: ProgressReport) -> None:
        reports.append(report)
        if report.iteration == 1:
            cancel.set()

    with pytest.raises(CancelledError):
        produce_localized_attribution(
            _design(include_wrist=True), progress_cb=progress, cancel_event=cancel
        )

    assert [item.iteration for item in reports] == [1]

    completed: list[ProgressReport] = []
    produce_localized_attribution(
        _design(include_wrist=True), progress_cb=completed.append
    )
    assert [item.iteration for item in completed] == [1, 2, 3, 4]


def test_design_rejects_nonlocalized_sources_and_wrong_delta_roster() -> None:
    localized = _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)
    wrong_delta = {"unknown.spec": 1.0}
    with pytest.raises(ContractViolationError):
        LocalizedAttributionDesign(
            "bad.design",
            _plan(localized),
            _base_config(),
            (_state_target(),),
            wrong_delta,
        )

    global_spec = NoiseSpec("swing_sim.swing.yaw_deg", scale=1.0)
    with pytest.raises(ContractViolationError):
        LocalizedAttributionDesign(
            "bad.global",
            _plan(global_spec),
            _base_config(),
            (_state_target(),),
            {global_spec.spec_id: 1.0},
        )


def test_design_rejects_ambiguous_base_offsets_and_invalid_state_loci() -> None:
    spec = _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)
    offset_config = replace(
        _base_config(),
        swing_run_config=replace(
            _base_config().swing_run_config,
            commanded_torque_offsets=(
                LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 1.0),
            ),
        ),
    )
    with pytest.raises(ContractViolationError, match="pre-existing"):
        _design(offset_config)

    from rate_of_closure.variation.localized_attribution import AttributionTarget

    bad_target = AttributionTarget(
        "state.bad-time",
        "state",
        "position_x_m",
        "m",
        "app-frame-cartesian-v1",
        0.0205,
        "swing.clubhead.reference",
        "app_frame:x_target,y_up,z_right",
    )
    with pytest.raises(ContractViolationError, match="sample grid"):
        LocalizedAttributionDesign(
            "bad.state",
            _plan(spec),
            _base_config(),
            (bad_target,),
            {spec.spec_id: 1.0},
        )


@pytest.mark.parametrize(
    "samples",
    [np.array([[False], [True]]), np.array([[0.0], [np.inf]])],
)
def test_explicit_sample_builder_rejects_non_authoritative_values(
    samples: np.ndarray,
) -> None:
    spec = _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)
    with pytest.raises(ContractViolationError):
        build_simulation_ensemble_request_from_samples(
            replace(_plan(spec), n_runs=2), _base_config(), samples
        )
