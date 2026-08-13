"""Localized torque variation mapping and complete-ensemble behavior."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode, run_simulation
from rate_of_closure.variation.ensemble_chunks import CollectingEnsembleSink
from rate_of_closure.variation.request_builder import (
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    TRACE_CAPABILITIES,
)
from rate_of_closure.variation.simulation_adapter import (
    build_simulation_ensemble_request,
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import (
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    LocalizedTorqueOffset,
)
from shared.python.swing_sim.variation import (
    NoiseSpec,
    VariationPlan,
    run_variation,
    variable_registry,
)

from .test_variation_simulation_request import (
    _SHOULDER_TORQUE_OFFSET,
    _WRIST_TORQUE_OFFSET,
    _base_config,
    _localized_spec,
    _spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_builder_maps_deterministic_localized_joint_torque_offsets() -> None:
    plan = VariationPlan(
        mode="swing",
        base_variables={
            _SHOULDER_TORQUE_OFFSET: 1.0,
            _WRIST_TORQUE_OFFSET: -0.5,
        },
        noise=(
            _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID),
            _localized_spec(_WRIST_TORQUE_OFFSET, WRIST_JOINT_ID, scale=1.0),
        ),
        n_runs=3,
        seed=73,
    )

    first = build_simulation_ensemble_request(plan, _base_config())
    second = build_simulation_ensemble_request(plan, _base_config())

    assert first.sampled_inputs == pytest.approx(second.sampled_inputs)
    for row, config in zip(first.sampled_inputs, first.configs, strict=True):
        assert config.swing_run_config.commanded_torque_offsets == (
            LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), row[0]),
            LocalizedTorqueOffset(WRIST_JOINT_ID, (0.02, 0.04), row[1]),
        )


def test_locus_does_not_replace_spatial_trace_ids() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(_localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID),),
        n_runs=2,
        seed=11,
    )

    result = run_simulation_ensemble(
        build_simulation_ensemble_request(plan, _base_config())
    )

    assert result.traces.point_ids == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert result.traces.point_ids != (SHOULDER_JOINT_ID, WRIST_JOINT_ID)


def test_torque_history_pins_half_open_boundary_and_stable_ids() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(
            _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID),
            _localized_spec(_WRIST_TORQUE_OFFSET, WRIST_JOINT_ID),
        ),
        n_runs=2,
        seed=31,
    )
    request = build_simulation_ensemble_request(plan, _base_config())
    run = run_simulation(request.configs[0])
    start_index = int(round(0.02 / 0.001))
    end_index = int(round(0.04 / 0.001))

    assert run.swing_joint_ids == (SHOULDER_JOINT_ID, WRIST_JOINT_ID)
    np.testing.assert_allclose(run.swing_applied_torques_nm[:start_index], 0.0)
    np.testing.assert_allclose(
        run.swing_applied_torques_nm[start_index:end_index],
        np.broadcast_to(request.sampled_inputs[0], (end_index - start_index, 2)),
    )
    np.testing.assert_allclose(run.swing_applied_torques_nm[end_index:], 0.0)


def test_replay_is_chunk_size_independent_and_retains_typed_misses() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(
            _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID),
            _localized_spec(_WRIST_TORQUE_OFFSET, WRIST_JOINT_ID),
        ),
        n_runs=3,
        seed=41,
    )
    base = dataclasses.replace(
        _base_config(), contact_mode=ContactMode.FIXED_BALL_CONTACT
    )
    request = build_simulation_ensemble_request(plan, base)
    first = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=1
    )
    second = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=2
    )

    assert tuple(item.status for item in first.outcomes) == tuple(
        item.status for item in second.outcomes
    )
    assert all(item.status.value == "evaluated_no_impact" for item in first.outcomes)
    assert all(item.value("closest_approach_m") is not None for item in first.outcomes)
    np.testing.assert_array_equal(first.variation.inputs, second.variation.inputs)
    np.testing.assert_allclose(first.variation.outputs, second.variation.outputs)
    np.testing.assert_array_equal(first.traces.sample_valid, second.traces.sample_valid)
    np.testing.assert_allclose(first.traces.positions_m, second.traces.positions_m)


def test_capability_registry_is_explicit_and_topological() -> None:
    assert TRACE_CAPABILITIES["localized_torque_offsets"] == (
        (_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID),
        (_WRIST_TORQUE_OFFSET, WRIST_JOINT_ID),
    )
    assert (
        tuple(LOCALIZED_TORQUE_VARIABLE_JOINTS.items())
        == TRACE_CAPABILITIES["localized_torque_offsets"]
    )
    for key in LOCALIZED_TORQUE_VARIABLE_JOINTS:
        definition = variable_registry()[key]
        assert definition.unit == "N·m"
        assert definition.applicability == "localized_torque_only"


@pytest.mark.parametrize(
    "spec",
    [
        _spec(_SHOULDER_TORQUE_OFFSET, 1.0),
        NoiseSpec(
            _SHOULDER_TORQUE_OFFSET,
            scale=1.0,
            point_ids=(SHOULDER_JOINT_ID,),
        ),
        NoiseSpec(
            _SHOULDER_TORQUE_OFFSET,
            scale=1.0,
            time_window_s=(0.02, 0.04),
        ),
        _localized_spec(_SHOULDER_TORQUE_OFFSET, WRIST_JOINT_ID),
        NoiseSpec(
            _SHOULDER_TORQUE_OFFSET,
            scale=1.0,
            time_window_s=(0.02, 0.04),
            point_ids=(SHOULDER_JOINT_ID, WRIST_JOINT_ID),
        ),
        _localized_spec(
            _SHOULDER_TORQUE_OFFSET,
            SHOULDER_JOINT_ID,
            window=(0.19, 0.21),
        ),
    ],
)
def test_builder_rejects_incomplete_or_incompatible_loci(spec: NoiseSpec) -> None:
    plan = VariationPlan(mode="swing", noise=(spec,), n_runs=2)

    with pytest.raises(ContractViolationError, match="torque|locus|window|point"):
        build_simulation_ensemble_request(plan, _base_config())


def test_scalar_executor_rejects_commanded_torque_instead_of_ignoring_it() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(_spec(_SHOULDER_TORQUE_OFFSET, 1.0),),
        n_runs=2,
    )

    with pytest.raises(ContractViolationError, match="context-specific"):
        run_variation(plan)


def test_simulation_config_rejects_other_source_kind_and_overlong_window() -> None:
    valid = DoublePendulumRunConfig(
        commanded_torque_offsets=(
            LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.02, 0.04), 1.0),
        )
    )
    overlong = DoublePendulumRunConfig(
        commanded_torque_offsets=(
            LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.19, 0.21), 1.0),
        )
    )

    with pytest.raises(ContractViolationError, match="double-pendulum"):
        dataclasses.replace(
            _base_config(), source_kind="manual", swing_run_config=valid
        )
    with pytest.raises(ContractViolationError, match="run duration"):
        dataclasses.replace(_base_config(), swing_run_config=overlong)


def test_builder_rejects_locus_beyond_effective_rk4_grid_before_sampling() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(
            _localized_spec(
                _SHOULDER_TORQUE_OFFSET,
                SHOULDER_JOINT_ID,
                window=(0.2, 0.2002),
            ),
        ),
        n_runs=2,
    )
    base = dataclasses.replace(_base_config(), swing_duration_s=0.2004)

    with pytest.raises(ContractViolationError, match="effective RK4 duration"):
        build_simulation_ensemble_request(plan, base)
