"""Source-neutral generalized-state authority for complete ensemble traces."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.simulation.sources import make_source
from rate_of_closure.variation.ensemble_chunks import (
    EnsembleStreamHeader,
    SimulationResultChunk,
    require_chunk_matches_header,
)
from rate_of_closure.variation.ensemble_trace_authority import (
    ChunkTraceAuthority,
    event_for_grid,
)
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from shared.python.contracts import ContractViolationError

from .test_variation_simulation_adapter import _config, _request

_SCENARIO = ImpactScenario(clubhead_speed_mph=100.0)


def test_manual_source_declares_pose_twist_as_its_complete_state() -> None:
    source = make_source("manual", _SCENARIO)

    assert source.generalized_state_ids == ()
    assert source.generalized_state_units == ()
    assert source.generalized_state_at(0.01).shape == (0,)


def test_double_source_exposes_canonical_generalized_state() -> None:
    source = make_source("double_pendulum", _SCENARIO, duration=0.01)
    expected = source.inner.state_at(0.005)

    assert source.generalized_state_ids == (
        "joint.shoulder.angle_rad",
        "joint.wrist.relative_angle_rad",
        "joint.shoulder.rate_rad_s",
        "joint.wrist.relative_rate_rad_s",
    )
    assert source.generalized_state_units == ("rad", "rad", "rad/s", "rad/s")
    np.testing.assert_allclose(
        source.generalized_state_at(0.005),
        (expected.theta1, expected.theta2, expected.omega1, expected.omega2),
    )


def test_triple_source_exposes_state_and_zero_commanded_torque() -> None:
    source = make_source("triple_pendulum", _SCENARIO, duration=0.01)

    assert source.generalized_state_ids == (
        "joint.shoulder.absolute_angle_rad",
        "joint.elbow.absolute_angle_rad",
        "joint.wrist.absolute_angle_rad",
        "joint.shoulder.absolute_rate_rad_s",
        "joint.elbow.absolute_rate_rad_s",
        "joint.wrist.absolute_rate_rad_s",
    )
    np.testing.assert_allclose(
        source.generalized_state_at(0.005), source.inner.state_at(0.005)
    )
    assert source.inner.joint_torques_at(0.005) == {
        "joint.shoulder": 0.0,
        "joint.elbow": 0.0,
        "joint.wrist": 0.0,
    }


def test_streamed_chunk_owns_full_state_torque_and_contact_events() -> None:
    class Sink:
        header: EnsembleStreamHeader | None = None
        chunks: list[SimulationResultChunk] = []

        def begin(self, header: EnsembleStreamHeader) -> None:
            self.header = header

        def accept(self, chunk: SimulationResultChunk) -> None:
            self.chunks.append(chunk)

        def commit(self, elapsed_s: float) -> tuple[SimulationResultChunk, ...]:
            assert elapsed_s >= 0.0
            return tuple(self.chunks)

        def abort(self) -> None:
            raise AssertionError("unexpected abort")

    evaluated: list[SimulationRun] = []

    def executor(config: SimulationConfig) -> SimulationRun:
        if config.scenario.clubhead_speed_mph == 99.0:
            raise RuntimeError("planted middle failure")
        run = run_simulation(config)
        evaluated.append(run)
        return run

    sink = Sink()
    chunks = run_simulation_ensemble_chunks(
        _request(
            (
                replace(_config(ContactMode.DELIVERY_INSPECTION), impact_time_s=0.03),
                _config(ContactMode.FIXED_BALL_CONTACT),
                _config(ContactMode.DELIVERY_INSPECTION, speed_mph=99.0),
            )
        ),
        sink,
        chunk_size=3,
        executor=executor,
    )

    authority = chunks[0].authority
    assert authority is not None
    assert authority.poses_app.flags.writeable is False
    assert authority.twists_app_si.shape[-1] == 6
    assert authority.generalized_states.shape[-1] == 4
    assert authority.applied_torques_nm.shape[-1] == 2
    assert authority.events[0] is not None
    assert authority.events[0].kind == "impact"
    assert authority.events[1] is not None
    assert authority.events[1].kind == "closest_approach"
    assert np.all(authority.preimpact_valid[1])
    assert authority.events[2] is None
    assert not np.any(chunks[0].sample_valid[2])
    assert not np.any(authority.preimpact_valid[2])
    assert np.all(np.isnan(authority.generalized_states[2]))
    hit_event = authority.events[0]
    assert hit_event is not None
    assert sink.header is not None
    expected_preimpact = (
        sink.header.sample_times_s <= hit_event.outcome.candidate_time_s
    )
    np.testing.assert_array_equal(authority.preimpact_valid[0], expected_preimpact)
    assert np.any(chunks[0].sample_valid[0] & ~authority.preimpact_valid[0])
    np.testing.assert_allclose(authority.poses_app[0], evaluated[0].swing_poses)
    np.testing.assert_allclose(authority.twists_app_si[0], evaluated[0].swing_twists)
    np.testing.assert_allclose(
        authority.generalized_states[0], evaluated[0].swing_generalized_states
    )
    np.testing.assert_allclose(
        authority.applied_torques_nm[0], evaluated[0].swing_applied_torques_nm
    )
    assert sink.header.authority_layout is not None


@pytest.mark.parametrize(
    ("source_kind", "state_count", "torque_count", "point_count"),
    (("manual", 0, 0, 1), ("triple_pendulum", 6, 3, 4)),
)
def test_streamed_authority_is_source_neutral(
    source_kind: str, state_count: int, torque_count: int, point_count: int
) -> None:
    class Sink:
        chunk: SimulationResultChunk | None = None

        def begin(self, header: EnsembleStreamHeader) -> None:
            assert len(header.point_ids) == point_count

        def accept(self, chunk: SimulationResultChunk) -> None:
            self.chunk = chunk

        def commit(self, elapsed_s: float) -> SimulationResultChunk:
            assert elapsed_s >= 0.0 and self.chunk is not None
            return self.chunk

        def abort(self) -> None:
            raise AssertionError("unexpected abort")

    sink = Sink()
    chunk = run_simulation_ensemble_chunks(
        _request((_config(ContactMode.DELIVERY_INSPECTION, source_kind=source_kind),)),
        sink,
        chunk_size=1,
    )

    assert chunk.authority is not None
    assert chunk.authority.generalized_states.shape[-1] == state_count
    assert chunk.authority.applied_torques_nm.shape[-1] == torque_count


def test_contact_event_cannot_diverge_from_typed_outcome() -> None:
    class Sink:
        header: EnsembleStreamHeader | None = None
        chunk: SimulationResultChunk | None = None

        def begin(self, header: EnsembleStreamHeader) -> None:
            self.header = header

        def accept(self, chunk: SimulationResultChunk) -> None:
            self.chunk = chunk

        def commit(self, elapsed_s: float) -> SimulationResultChunk:
            assert elapsed_s >= 0.0 and self.chunk is not None
            return self.chunk

        def abort(self) -> None:
            raise AssertionError("unexpected abort")

    sink = Sink()
    chunk = run_simulation_ensemble_chunks(
        _request(
            (replace(_config(ContactMode.DELIVERY_INSPECTION), impact_time_s=0.03),)
        ),
        sink,
        chunk_size=1,
    )
    assert sink.header is not None and chunk.authority is not None
    event = chunk.authority.events[0]
    assert event is not None
    forged_outcome = replace(
        event.outcome, candidate_time_s=event.outcome.candidate_time_s + 0.001
    )
    forged_event = event_for_grid(0, forged_outcome, sink.header.sample_times_s)
    forged_authority = ChunkTraceAuthority(
        chunk.authority.poses_app,
        chunk.authority.twists_app_si,
        chunk.authority.generalized_states,
        chunk.authority.applied_torques_nm,
        sink.header.sample_times_s[np.newaxis, :] <= forged_outcome.candidate_time_s,
        (forged_event,),
    )
    forged_chunk = replace(chunk, authority=forged_authority)

    with pytest.raises(ContractViolationError, match="candidate_time_s"):
        require_chunk_matches_header(sink.header, forged_chunk, 0)
