"""Lossless per-trial records for governed Rate ensemble evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import ContactMode, SimulationConfig, run_simulation
from rate_of_closure.variation.complete_trial_record import (
    CompleteTrialRecordSource,
    build_complete_trial_record,
)
from rate_of_closure.variation.request_builder import build_simulation_ensemble_request
from rate_of_closure.variation.simulation_adapter import (
    build_ensemble_stream_header,
    run_simulation_ensemble_chunks,
)
from rate_of_closure.variation.simulation_types import (
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationEnsembleRequest,
)
from rate_of_closure.variation.trial_projection import (
    TrialCapture,
    project_simulation_outcome,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import SHOULDER_JOINT_ID
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    NoiseSpec,
    VariationPlan,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_DRIVER = get_club("Driver 10.5°")


def _config(
    contact_mode: ContactMode, source_kind: str = "double_pendulum"
) -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=100.0),
        club=_DRIVER,
        source_kind=source_kind,
        swing_duration_s=0.05,
        contact_mode=contact_mode,
    )


def _context(config: SimulationConfig):  # type: ignore[no-untyped-def]
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=1.0),),
        n_runs=1,
        seed=19,
    )
    inputs = np.array([[0.25]])
    request = SimulationEnsembleRequest(plan, inputs, (config,))
    return inputs[0], build_ensemble_stream_header(request)


def _record(config: SimulationConfig, capture: TrialCapture):  # type: ignore[no-untyped-def]
    sampled_inputs, header = _context(config)
    outcome = project_simulation_outcome(0, capture)
    source = CompleteTrialRecordSource(0, sampled_inputs, config)
    return build_complete_trial_record(source, capture, outcome, header)


def test_hit_retains_every_simulation_phase_and_identity() -> None:
    config = _config(ContactMode.DELIVERY_INSPECTION)
    run = run_simulation(config)

    record = _record(config, TrialCapture(run, None))

    assert record.status is EVALUATED_HIT
    assert record.source_kind == "double_pendulum"
    assert record.coordinate_frame == "app_frame:x_target,y_up,z_right"
    assert record.spatial_point_ids == (
        "swing.pivot",
        "swing.wrist",
        "swing.clubhead.reference",
    )
    assert record.torque_joint_ids == run.swing_joint_ids
    assert record.plan_sha256 != record.execution_sha256
    assert len(record.plan_sha256) == len(record.execution_sha256) == 64
    assert len(record.configuration_sha256) == 64
    assert record.source_revision_status in {"exact", "unavailable"}
    np.testing.assert_array_equal(record.swing_times_s, run.swing_times)
    np.testing.assert_array_equal(record.swing_positions_m, run.swing_positions)
    np.testing.assert_array_equal(record.swing_poses, run.swing_poses)
    np.testing.assert_array_equal(record.swing_twists, run.swing_twists)
    np.testing.assert_array_equal(record.swing_joint_positions_m, run.swing_joints)
    np.testing.assert_array_equal(
        record.swing_applied_torques_nm, run.swing_applied_torques_nm
    )
    np.testing.assert_array_equal(record.flight_times_s, run.flight_times)
    np.testing.assert_array_equal(record.flight_positions_m, run.flight_positions)
    np.testing.assert_array_equal(record.flight_velocities_mps, run.flight_velocities)
    assert record.impact_outcome is not None
    assert record.delivery_state is not None
    assert record.post_impact_state is not None
    assert record.launch_state is not None
    assert record.impact_time_s == run.impact_time_s
    assert 0 < record.pre_impact_sample_count <= len(run.swing_times)


def test_miss_retains_contact_and_swing_without_fabricating_downstream_phases() -> None:
    config = _config(ContactMode.FIXED_BALL_CONTACT)
    run = run_simulation(config)

    record = _record(config, TrialCapture(run, None))

    assert record.status is EVALUATED_NO_IMPACT
    assert record.candidate_time_s == run.impact_outcome.candidate_time_s
    assert record.impact_time_s is None
    assert record.impact_outcome is not None
    assert record.delivery_state is None
    assert record.post_impact_state is None
    assert record.launch_state is None
    assert record.flight_times_s.shape == (0,)
    assert record.flight_positions_m.shape == (0, 3)
    assert record.flight_velocities_mps.shape == (0, 3)
    assert len(record.swing_times_s) == len(run.swing_times)


def test_failure_retains_typed_diagnostics_without_fabricating_physics() -> None:
    config = _config(ContactMode.DELIVERY_INSPECTION)

    record = _record(config, TrialCapture(None, RuntimeError("planted failure")))

    assert record.status is NUMERICAL_FAILURE
    assert record.failure_type == "RuntimeError"
    assert record.failure_message == "planted failure"
    assert record.candidate_time_s is None
    assert record.impact_time_s is None
    assert record.impact_outcome is None
    assert record.delivery_state is None
    assert record.post_impact_state is None
    assert record.launch_state is None
    assert record.swing_times_s.shape == (0,)
    assert record.flight_times_s.shape == (0,)


@pytest.mark.parametrize(
    ("source_kind", "point_count", "torque_count"),
    (("manual", 1, 0), ("double_pendulum", 3, 2), ("triple_pendulum", 4, 0)),
)
def test_current_swing_sources_have_explicit_compatible_record_layouts(
    source_kind: str, point_count: int, torque_count: int
) -> None:
    config = _config(ContactMode.DELIVERY_INSPECTION, source_kind)
    run = run_simulation(config)

    record = _record(config, TrialCapture(run, None))

    assert len(record.spatial_point_ids) == point_count
    assert len(record.torque_joint_ids) == torque_count
    assert record.swing_joint_positions_m.shape[1] == (
        0 if source_kind == "manual" else point_count
    )
    assert record.swing_applied_torques_nm.shape[1] == torque_count


def test_record_owns_arrays_and_rejects_nonfinite_state() -> None:
    config = _config(ContactMode.DELIVERY_INSPECTION)
    run = run_simulation(config)
    record = _record(config, TrialCapture(run, None))

    assert not record.swing_positions_m.flags.writeable
    run.swing_positions[0, 0] += 1.0
    assert record.swing_positions_m[0, 0] != run.swing_positions[0, 0]

    invalid = replace(run, swing_positions=np.array(run.swing_positions, copy=True))
    invalid.swing_positions[0, 0] = np.nan
    with pytest.raises(
        ContractViolationError, match="swing_positions_m must be finite"
    ):
        _record(config, TrialCapture(invalid, None))
    with pytest.raises(ContractViolationError, match="trial units"):
        replace(record, units={})


def test_bounded_executor_delivers_aligned_complete_records_to_its_sink() -> None:
    hit = _config(ContactMode.DELIVERY_INSPECTION)
    miss = _config(ContactMode.FIXED_BALL_CONTACT)
    failure = replace(hit, scenario=ImpactScenario(clubhead_speed_mph=99.0))
    plan = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec(_FACE, scale=1.0),),
        n_runs=3,
        seed=23,
    )
    request = SimulationEnsembleRequest(
        plan, np.array([[-0.5], [0.0], [0.5]]), (hit, miss, failure)
    )

    def executor(config: SimulationConfig):  # type: ignore[no-untyped-def]
        if config.scenario.clubhead_speed_mph == 99.0:
            raise RuntimeError("bounded planted failure")
        return run_simulation(config)

    class Sink:
        def __init__(self) -> None:
            self.records = []

        def begin(self, _header) -> None:  # type: ignore[no-untyped-def]
            return None

        def accept(self, chunk) -> None:  # type: ignore[no-untyped-def]
            assert len(chunk.complete_records) == len(chunk.outcomes)
            self.records.extend(chunk.complete_records)

        def commit(self, _elapsed_s: float):  # type: ignore[no-untyped-def]
            return tuple(self.records)

        def abort(self) -> None:
            return None

    result = run_simulation_ensemble_chunks(
        request, Sink(), chunk_size=2, executor=executor
    )

    assert tuple(item.trial_index for item in result) == (0, 1, 2)
    assert tuple(item.status for item in result) == (
        EVALUATED_HIT,
        EVALUATED_NO_IMPACT,
        NUMERICAL_FAILURE,
    )
    assert result[2].failure_message == "bounded planted failure"


@pytest.mark.parametrize(
    ("spec", "adapter_id"),
    (
        (NoiseSpec("swing_sim.swing.yaw_deg", scale=0.1), "global_simulation_value/v1"),
        (
            NoiseSpec(
                "swing_sim.swing.shoulder_commanded_torque_offset_nm",
                scale=0.1,
                time_window_s=(0.01, 0.03),
                point_ids=(SHOULDER_JOINT_ID,),
            ),
            "localized_joint_torque_offset/v1",
        ),
    ),
)
def test_qualified_double_pendulum_adapters_emit_complete_golden_records(
    spec: NoiseSpec, adapter_id: str
) -> None:
    plan = VariationPlan(mode="swing", noise=(spec,), n_runs=1, seed=29)
    request = build_simulation_ensemble_request(
        plan, _config(ContactMode.DELIVERY_INSPECTION)
    )

    class Sink:
        def __init__(self) -> None:
            self.records = []

        def begin(self, _header) -> None:  # type: ignore[no-untyped-def]
            return None

        def accept(self, chunk) -> None:  # type: ignore[no-untyped-def]
            self.records.extend(chunk.complete_records)

        def commit(self, _elapsed_s: float):  # type: ignore[no-untyped-def]
            return tuple(self.records)

        def abort(self) -> None:
            return None

    records = run_simulation_ensemble_chunks(request, Sink(), chunk_size=1)

    assert len(records) == 1
    assert records[0].adapter_ids == (adapter_id,)
    assert records[0].source_kind == "double_pendulum"
    assert records[0].units["swing_applied_torques_nm"] == "N*m"
    assert records[0].swing_times_s.size > 0


@pytest.mark.parametrize("source_kind", ("manual", "triple_pendulum"))
def test_nonqualified_sources_fail_closed_before_adapter_execution(
    source_kind: str,
) -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec("swing_sim.swing.yaw_deg", scale=0.1),),
        n_runs=1,
    )

    with pytest.raises(ContractViolationError, match="double_pendulum source"):
        build_simulation_ensemble_request(
            plan, _config(ContactMode.DELIVERY_INSPECTION, source_kind)
        )


@pytest.mark.parametrize(
    "variable_key",
    (
        "swing_sim.flight.launch.ground_normal_restitution",
        "golf_club.turf.normal_stiffness_n_m",
    ),
)
def test_non_trace_adapters_fail_closed_before_complete_trial_builder(
    variable_key: str,
) -> None:
    with pytest.raises(ContractViolationError, match="registered|legal in swing"):
        plan = VariationPlan(
            mode="swing",
            noise=(NoiseSpec(variable_key, scale=0.1),),
            n_runs=1,
        )
        build_simulation_ensemble_request(
            plan, _config(ContactMode.DELIVERY_INSPECTION)
        )
