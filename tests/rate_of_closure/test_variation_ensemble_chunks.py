"""Deterministic lifecycle tests for bounded Rate result chunks."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode, SimulationConfig, SimulationRun
from rate_of_closure.variation.ensemble_chunks import (
    MAX_CHUNK_POSITION_CELLS,
    CollectingEnsembleSink,
    EnsembleStreamHeader,
    SimulationResultChunk,
)
from rate_of_closure.variation.ensemble_io import from_json_dict, to_json_dict
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble,
    run_simulation_ensemble_chunks,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.solver.solve import CancelledError

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _three_trial_request():
    return _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION),
            _config(ContactMode.FIXED_BALL_CONTACT),
            _config(ContactMode.DELIVERY_INSPECTION),
        )
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_collecting_sink_is_semantically_invariant_to_chunk_size(
    chunk_size: int,
) -> None:
    request = _three_trial_request()
    expected = run_simulation_ensemble(request)

    actual = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=chunk_size
    )

    assert actual.outcomes == expected.outcomes
    np.testing.assert_array_equal(actual.variation.inputs, expected.variation.inputs)
    np.testing.assert_array_equal(actual.variation.outputs, expected.variation.outputs)
    np.testing.assert_array_equal(actual.variation.success, expected.variation.success)
    np.testing.assert_array_equal(
        actual.traces.positions_m, expected.traces.positions_m
    )
    np.testing.assert_array_equal(
        actual.traces.sample_valid, expected.traces.sample_valid
    )
    np.testing.assert_array_equal(
        actual.traces.impact_sample_indices, expected.traces.impact_sample_indices
    )


class _RecordingSink:
    def __init__(
        self,
        fail_accept: bool = False,
        fail_begin: bool = False,
        fail_abort: bool = False,
    ) -> None:
        self.header: EnsembleStreamHeader | None = None
        self.starts: list[int] = []
        self.aborts = 0
        self.commits = 0
        self.fail_accept = fail_accept
        self.fail_begin = fail_begin
        self.fail_abort = fail_abort

    def begin(self, header: EnsembleStreamHeader) -> None:
        if self.fail_begin:
            raise RuntimeError("begin failure")
        self.header = header

    def accept(self, chunk: SimulationResultChunk) -> None:
        if self.fail_accept:
            raise RuntimeError("sink failure")
        self.starts.append(chunk.start_index)

    def commit(self, elapsed_s: float) -> tuple[int, ...]:
        assert elapsed_s >= 0.0
        self.commits += 1
        return tuple(self.starts)

    def abort(self) -> None:
        self.aborts += 1
        if self.fail_abort:
            raise RuntimeError("abort failure")


def test_stream_accepts_only_canonical_committed_prefixes() -> None:
    sink = _RecordingSink()
    progress: list[int] = []

    starts = run_simulation_ensemble_chunks(
        _three_trial_request(),
        sink,
        chunk_size=2,
        progress_cb=lambda report: progress.append(report.iteration),
    )

    assert starts == (0, 2)
    assert progress == [2, 3]
    assert sink.commits == 1
    assert sink.aborts == 0


def test_chunking_is_invariant_with_a_numerical_failure_in_a_middle_chunk() -> None:
    request = _request(
        (
            _config(ContactMode.DELIVERY_INSPECTION),
            _config(ContactMode.DELIVERY_INSPECTION, speed_mph=99.0),
            _config(ContactMode.FIXED_BALL_CONTACT),
        )
    )

    def executor(config: SimulationConfig) -> SimulationRun:
        from rate_of_closure.simulation import run_simulation

        if config.scenario.clubhead_speed_mph == 99.0:
            raise RuntimeError("planted middle failure")
        return run_simulation(config)

    expected = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=1, executor=executor
    )
    actual = run_simulation_ensemble_chunks(
        request, CollectingEnsembleSink(), chunk_size=2, executor=executor
    )

    assert actual.outcomes == expected.outcomes
    np.testing.assert_array_equal(actual.variation.inputs, expected.variation.inputs)
    np.testing.assert_array_equal(actual.variation.outputs, expected.variation.outputs)
    np.testing.assert_array_equal(
        actual.traces.positions_m, expected.traces.positions_m
    )
    np.testing.assert_array_equal(
        actual.traces.sample_valid, expected.traces.sample_valid
    )


def test_post_executor_cancellation_aborts_without_accepting_partial_chunk() -> None:
    request = _request((_config(ContactMode.DELIVERY_INSPECTION),))
    cancelled = threading.Event()
    sink = _RecordingSink()

    def executor(config: SimulationConfig) -> SimulationRun:
        from rate_of_closure.simulation import run_simulation

        result = run_simulation(config)
        cancelled.set()
        return result

    with pytest.raises(CancelledError):
        run_simulation_ensemble_chunks(
            request, sink, executor=executor, cancel_event=cancelled
        )

    assert sink.starts == []
    assert sink.commits == 0
    assert sink.aborts == 1


def test_cancellation_after_a_provisional_chunk_aborts_without_commit() -> None:
    cancelled = threading.Event()

    class _CancelAfterAcceptSink(_RecordingSink):
        def accept(self, chunk: SimulationResultChunk) -> None:
            super().accept(chunk)
            cancelled.set()

    sink = _CancelAfterAcceptSink()

    with pytest.raises(CancelledError):
        run_simulation_ensemble_chunks(
            _three_trial_request(), sink, chunk_size=1, cancel_event=cancelled
        )

    assert sink.starts == [0]
    assert sink.commits == 0
    assert sink.aborts == 1


def test_sink_failure_aborts_exactly_once_and_propagates() -> None:
    sink = _RecordingSink(fail_accept=True)

    with pytest.raises(RuntimeError, match="sink failure"):
        run_simulation_ensemble_chunks(_three_trial_request(), sink, chunk_size=1)

    assert sink.commits == 0
    assert sink.aborts == 1


def test_begin_failure_aborts_and_abort_failure_does_not_mask_primary() -> None:
    sink = _RecordingSink(fail_begin=True, fail_abort=True)

    with pytest.raises(RuntimeError, match="begin failure") as caught:
        run_simulation_ensemble_chunks(_three_trial_request(), sink, chunk_size=1)

    assert sink.aborts == 1
    assert "abort failure" in " ".join(caught.value.__notes__)


def test_commit_failure_aborts_collecting_sink_and_releases_provisional_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = CollectingEnsembleSink()

    def rejected_authority(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("final authority rejected")

    monkeypatch.setattr(
        "rate_of_closure.variation.ensemble_chunks.SimulationEnsembleResult",
        rejected_authority,
    )
    with pytest.raises(RuntimeError, match="final authority rejected"):
        run_simulation_ensemble_chunks(
            _request((_config(ContactMode.DELIVERY_INSPECTION),)),
            sink,
            chunk_size=1,
        )

    assert sink._finished is True
    assert sink._inputs is None
    assert sink._positions is None


@pytest.mark.parametrize("invalid", [True, 0.5])
def test_chunk_start_index_requires_an_exact_integer(invalid: object) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )

    with pytest.raises(ContractViolationError, match="non-negative integer"):
        SimulationResultChunk(
            invalid,  # type: ignore[arg-type]
            source.variation.inputs,
            source.outcomes,
            source.traces.positions_m,
            source.traces.sample_valid,
            source.traces.impact_sample_indices,
        )


@pytest.mark.parametrize(
    "invalid_frame",
    [" app_frame:x_target,y_up,z_right", "other_frame"],
)
def test_header_rejects_unsupported_coordinate_frame(invalid_frame: str) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )

    with pytest.raises(ContractViolationError, match="trimmed stable ID|unsupported"):
        EnsembleStreamHeader(
            source.variation.plan,
            source.variation.inputs,
            source.traces.sample_times_s,
            source.traces.point_ids,
            invalid_frame,
        )


@pytest.mark.parametrize(
    "invalid", [np.array([[True]], dtype=bool), np.array([["1.25"]], dtype=str)]
)
def test_header_rejects_coercive_sampled_input_domains(invalid: np.ndarray) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )

    with pytest.raises(ContractViolationError, match="real non-boolean"):
        EnsembleStreamHeader(
            source.variation.plan,
            invalid,
            source.traces.sample_times_s,
            source.traces.point_ids,
            source.traces.coordinate_frame,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([False, True], dtype=bool),
        np.array(["0.0", "1.0"], dtype=str),
        np.array([0.0 + 7.0j, 1.0 + 9.0j], dtype=complex),
        np.array([0.0, 1.0], dtype=object),
    ],
)
def test_header_rejects_coercive_time_domains(invalid: np.ndarray) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )

    with pytest.raises(ContractViolationError, match="real non-boolean"):
        EnsembleStreamHeader(
            source.variation.plan,
            source.variation.inputs,
            invalid,
            source.traces.point_ids,
            source.traces.coordinate_frame,
        )


def test_collected_authority_round_trips_through_the_strict_reader() -> None:
    result = run_simulation_ensemble_chunks(
        _three_trial_request(), CollectingEnsembleSink(), chunk_size=2
    )

    restored = from_json_dict(to_json_dict(result))

    assert restored.outcomes == result.outcomes
    assert restored.traces.coordinate_frame == result.traces.coordinate_frame
    np.testing.assert_array_equal(restored.variation.inputs, result.variation.inputs)
    np.testing.assert_array_equal(
        restored.traces.positions_m, result.traces.positions_m
    )


def test_compatibility_runner_preserves_per_trial_progress() -> None:
    reports: list[int] = []

    run_simulation_ensemble(
        _three_trial_request(),
        progress_cb=lambda report: reports.append(report.iteration),
    )

    assert reports == [1, 2, 3]


def test_oversized_position_tensor_is_rejected_before_owned_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )
    outcome = source.outcomes
    oversized = np.lib.stride_tricks.as_strided(
        np.zeros(1), shape=(1, MAX_CHUNK_POSITION_CELLS + 1, 1, 3), strides=(0, 0, 0, 0)
    )

    def unexpected_copy(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("oversized position tensor copied before its cap")

    monkeypatch.setattr(
        "rate_of_closure.variation.ensemble_chunks._owned_array", unexpected_copy
    )
    with pytest.raises(ContractViolationError, match="position cell limit"):
        SimulationResultChunk(
            0,
            source.variation.inputs,
            outcome,
            oversized,
            np.ones((1, oversized.shape[1]), dtype=bool),
            np.array([0]),
        )


def test_header_binding_rejects_wrong_impact_time_provenance() -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )
    sink = CollectingEnsembleSink()
    header = EnsembleStreamHeader(
        source.variation.plan,
        source.variation.inputs,
        source.traces.sample_times_s,
        source.traces.point_ids,
        source.traces.coordinate_frame,
    )
    sink.begin(header)
    wrong = source.traces.impact_sample_indices.copy()
    wrong[0] = max(0, wrong[0] - 1)
    chunk = SimulationResultChunk(
        0,
        source.variation.inputs,
        source.outcomes,
        source.traces.positions_m,
        source.traces.sample_valid,
        wrong,
    )

    with pytest.raises(ContractViolationError, match="impact-time provenance"):
        sink.accept(chunk)


def test_header_binding_rejects_changed_inputs_at_nonzero_chunk_offset() -> None:
    source = run_simulation_ensemble(_three_trial_request())
    header = EnsembleStreamHeader(
        source.variation.plan,
        source.variation.inputs,
        source.traces.sample_times_s,
        source.traces.point_ids,
        source.traces.coordinate_frame,
    )
    sink = CollectingEnsembleSink()
    sink.begin(header)
    first = SimulationResultChunk(
        0,
        source.variation.inputs[:1],
        source.outcomes[:1],
        source.traces.positions_m[:1],
        source.traces.sample_valid[:1],
        source.traces.impact_sample_indices[:1],
    )
    sink.accept(first)
    changed = source.variation.inputs[1:2].copy()
    changed[0, 0] += 123.0
    second = SimulationResultChunk(
        1,
        changed,
        source.outcomes[1:2],
        source.traces.positions_m[1:2],
        source.traces.sample_valid[1:2],
        source.traces.impact_sample_indices[1:2],
    )

    with pytest.raises(ContractViolationError, match="sampled inputs"):
        sink.accept(second)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[2]], dtype=int),
        np.array([[0.0]], dtype=float),
        np.array([["true"]], dtype=str),
    ],
)
def test_chunk_rejects_non_boolean_validity_domains(invalid: np.ndarray) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )
    repeated = np.broadcast_to(invalid, source.traces.sample_valid.shape)

    with pytest.raises(ContractViolationError, match="genuine boolean"):
        SimulationResultChunk(
            0,
            source.variation.inputs,
            source.outcomes,
            source.traces.positions_m,
            repeated,
            source.traces.impact_sample_indices,
        )


@pytest.mark.parametrize("invalid", [True, 0, -1, 10_000])
def test_chunk_size_must_fit_the_declared_trace_budget(invalid: object) -> None:
    with pytest.raises(ContractViolationError, match="chunk_size"):
        run_simulation_ensemble_chunks(
            _three_trial_request(),
            CollectingEnsembleSink(),
            chunk_size=invalid,  # type: ignore[arg-type]
        )
