"""Execution-control contracts for seeded regional-ground variation."""

from __future__ import annotations

import hashlib
import threading

import pytest

from rate_of_closure.variation import regional_ground_variation as variation_module
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
    GroundRegionalVariationTrial,
    register_ground_variation_variables,
    run_regional_ground_variation,
)
from shared.python.contracts import PreconditionError
from tests.rate_of_closure.regional_ground_target_support import transfer_failure
from tests.rate_of_closure.test_regional_ground_variation import (
    _json_bytes,
    _request,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_BASELINE_SUCCESS_SHA256 = (
    "671e5fd6c59aa1c068f2a3bd608ff7ef58c585b7ee4897ca49ef4ae73743f6a0"
)


@pytest.fixture(autouse=True)
def _registered_ground_variables() -> None:
    """Make the exact extension keys available to the request fixture."""
    register_ground_variation_variables()


def test_success_reports_typed_progress_without_changing_result_bytes() -> None:
    reports: list[GroundRegionalVariationProgress] = []
    hooks = GroundRegionalVariationHooks(progress_callback=reports.append)

    result = run_regional_ground_variation(
        _request(), lambda _trial: transfer_failure(), hooks=hooks
    )

    assert [(item.completed, item.total) for item in reports] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    payload = _json_bytes(result).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == _BASELINE_SUCCESS_SHA256


def test_precancel_stops_before_executor_and_publishes_no_partial_dataset() -> None:
    cancellation = threading.Event()
    cancellation.set()
    attempted: list[int] = []
    reports: list[GroundRegionalVariationProgress] = []

    def executor(trial: GroundRegionalVariationTrial):
        attempted.append(trial.trial_index)
        return transfer_failure()

    hooks = GroundRegionalVariationHooks(
        progress_callback=reports.append,
        cancellation_requested=cancellation.is_set,
    )
    with pytest.raises(GroundRegionalVariationCancelled) as raised:
        run_regional_ground_variation(_request(), executor, hooks=hooks)

    assert (raised.value.completed, raised.value.total) == (0, 4)
    assert attempted == []
    assert reports == []
    assert not hasattr(raised.value, "dataset")


def test_midrun_cancel_discards_inflight_trial_and_publishes_nothing() -> None:
    cancellation = threading.Event()
    attempted: list[int] = []
    reports: list[GroundRegionalVariationProgress] = []

    def executor(trial: GroundRegionalVariationTrial):
        attempted.append(trial.trial_index)
        if trial.trial_index == 1:
            cancellation.set()
        return transfer_failure()

    hooks = GroundRegionalVariationHooks(
        progress_callback=reports.append,
        cancellation_requested=cancellation.is_set,
    )
    with pytest.raises(GroundRegionalVariationCancelled) as raised:
        run_regional_ground_variation(_request(), executor, hooks=hooks)

    assert (raised.value.completed, raised.value.total) == (1, 4)
    assert attempted == [0, 1]
    assert [(item.completed, item.total) for item in reports] == [(1, 4)]
    assert not hasattr(raised.value, "dataset")


def test_progress_callback_failure_is_typed_and_cannot_publish_rows() -> None:
    attempted: list[int] = []

    def executor(trial: GroundRegionalVariationTrial):
        attempted.append(trial.trial_index)
        return transfer_failure()

    def broken_callback(progress: GroundRegionalVariationProgress) -> None:
        if progress.completed == 2:
            raise LookupError("progress sink unavailable")

    hooks = GroundRegionalVariationHooks(progress_callback=broken_callback)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_variation(_request(), executor, hooks=hooks)

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.PROGRESS_CALLBACK
    assert (failure.completed, failure.total) == (2, 4)
    assert failure.cause_type == "LookupError"
    assert failure.cause_message == "progress sink unavailable"
    assert isinstance(failure.__cause__, LookupError)
    assert attempted == [0, 1]
    assert not hasattr(failure, "dataset")


def test_executor_failure_is_typed_and_publishes_no_partial_dataset() -> None:
    attempted: list[int] = []
    reports: list[GroundRegionalVariationProgress] = []

    def executor(trial: GroundRegionalVariationTrial):
        attempted.append(trial.trial_index)
        if trial.trial_index == 1:
            raise ArithmeticError("planted executor failure")
        return transfer_failure()

    hooks = GroundRegionalVariationHooks(progress_callback=reports.append)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_variation(_request(), executor, hooks=hooks)

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.EXECUTOR
    assert (failure.completed, failure.total) == (1, 4)
    assert failure.cause_type == "ArithmeticError"
    assert failure.cause_message == "planted executor failure"
    assert attempted == [0, 1]
    assert [(item.completed, item.total) for item in reports] == [(1, 4)]
    assert not hasattr(failure, "dataset")


def test_validator_failure_is_typed_and_publishes_no_partial_dataset() -> None:
    attempted: list[int] = []
    reports: list[GroundRegionalVariationProgress] = []

    def executor(trial: GroundRegionalVariationTrial) -> object:
        attempted.append(trial.trial_index)
        if trial.trial_index == 1:
            return object()
        return transfer_failure()

    hooks = GroundRegionalVariationHooks(progress_callback=reports.append)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_variation(_request(), executor, hooks=hooks)  # type: ignore[arg-type]

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.VALIDATION
    assert (failure.completed, failure.total) == (1, 4)
    assert failure.cause_type == "PreconditionError"
    assert "executor must return an exact pipeline result" in failure.cause_message
    assert isinstance(failure.__cause__, PreconditionError)
    assert attempted == [0, 1]
    assert [(item.completed, item.total) for item in reports] == [(1, 4)]
    assert not hasattr(failure, "dataset")
    assert not hasattr(failure, "rows")


def test_publication_failure_is_typed_after_all_trials_without_partial_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports: list[GroundRegionalVariationProgress] = []

    def broken_publisher(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ensemble publisher unavailable")

    monkeypatch.setattr(
        variation_module, "build_regional_ground_study_ensemble", broken_publisher
    )
    hooks = GroundRegionalVariationHooks(progress_callback=reports.append)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_variation(
            _request(), lambda _trial: transfer_failure(), hooks=hooks
        )

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.PUBLICATION
    assert (failure.completed, failure.total) == (4, 4)
    assert failure.cause_type == "RuntimeError"
    assert [item.completed for item in reports] == [1, 2, 3, 4]
    assert not hasattr(failure, "dataset")


def test_cancellation_callback_failure_is_a_typed_terminal_failure() -> None:
    def broken_cancellation() -> bool:
        raise OSError("cancellation source unavailable")

    hooks = GroundRegionalVariationHooks(cancellation_requested=broken_cancellation)
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        run_regional_ground_variation(
            _request(), lambda _trial: transfer_failure(), hooks=hooks
        )

    failure = raised.value
    assert failure.stage is GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK
    assert (failure.completed, failure.total) == (0, 4)
    assert failure.cause_type == "OSError"
    assert not hasattr(failure, "dataset")
