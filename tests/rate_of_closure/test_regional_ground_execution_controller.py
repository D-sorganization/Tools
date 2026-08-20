"""Headless Qt contracts for regional-ground job submission."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from rate_of_closure.ui.pyqt6.regional_ground_execution_controller import (
    RegionalGroundExecutionController,
)
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)
from tests.rate_of_closure.test_regional_ground_execution_result import (
    _job,
    _result,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_controller_publishes_typed_progress_and_complete_bound_result(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    job = _job()
    expected = _result()
    worker_thread_ids: list[int] = []
    reports: list[GroundRegionalVariationProgress] = []
    failures: list[GroundRegionalVariationFailed] = []

    def submitter(job_argument, hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        worker_thread_ids.append(threading.get_ident())
        assert job_argument is job
        assert hooks.progress_callback is not None
        for completed in range(1, 5):
            hooks.progress_callback(GroundRegionalVariationProgress(completed, 4))
        return expected

    controller = RegionalGroundExecutionController(submitter)
    controller.progressed.connect(reports.append)
    controller.failed.connect(failures.append)

    with qtbot.waitSignal(controller.succeeded, timeout=5_000) as succeeded:
        controller.submit(job)
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    assert succeeded.args == [expected]
    assert [(item.completed, item.total) for item in reports] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert worker_thread_ids != [threading.get_ident()]
    assert failures == []
    assert controller.active_job_sha256 is None


def test_controller_cancel_emits_typed_terminal_without_result(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    started = threading.Event()
    successes: list[object] = []

    def submitter(_job_argument, hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        assert hooks.cancellation_requested is not None
        started.set()
        while not hooks.cancellation_requested():
            started.wait(0.005)
        raise GroundRegionalVariationCancelled(0, 4)

    controller = RegionalGroundExecutionController(submitter)
    controller.succeeded.connect(successes.append)
    with qtbot.waitSignal(controller.cancelled, timeout=5_000) as cancelled:
        controller.submit(_job())
        qtbot.waitUntil(started.is_set, timeout=2_000)
        assert controller.cancel()
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    terminal = cancelled.args[0]
    assert isinstance(terminal, GroundRegionalVariationCancelled)
    assert (terminal.completed, terminal.total) == (0, 4)
    assert successes == []


def test_controller_forwards_typed_authority_failure_without_result(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    cause = ArithmeticError("qualified authority unavailable")
    terminal = GroundRegionalVariationFailed(
        GroundRegionalVariationFailureStage.EXECUTOR, 0, 4, cause
    )
    successes: list[object] = []

    def submitter(_job_argument, _hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        raise terminal

    controller = RegionalGroundExecutionController(submitter)
    controller.succeeded.connect(successes.append)
    with qtbot.waitSignal(controller.failed, timeout=5_000) as failed:
        controller.submit(_job())
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    assert failed.args == [terminal]
    assert successes == []


def test_controller_wraps_untyped_submitter_exception_with_chained_cause(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    cause = LookupError("submission adapter disconnected")

    def submitter(_job_argument, _hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        raise cause

    controller = RegionalGroundExecutionController(submitter)
    with qtbot.waitSignal(controller.failed, timeout=5_000) as failed:
        controller.submit(_job())
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    terminal = failed.args[0]
    assert isinstance(terminal, GroundRegionalVariationFailed)
    assert terminal.stage is GroundRegionalVariationFailureStage.EXECUTOR
    assert terminal.__cause__ is cause


def test_controller_rejects_unbound_result_as_typed_validation_failure(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    unbound = replace(_result(), job_id="other-qualified-job")
    successes: list[object] = []

    def submitter(_job_argument, _hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        return unbound

    controller = RegionalGroundExecutionController(submitter)
    controller.succeeded.connect(successes.append)
    with qtbot.waitSignal(controller.failed, timeout=5_000) as failed:
        controller.submit(_job())
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    terminal = failed.args[0]
    assert isinstance(terminal, GroundRegionalVariationFailed)
    assert terminal.stage is GroundRegionalVariationFailureStage.VALIDATION
    assert terminal.cause_type == "ValueError"
    assert "job_id" in terminal.cause_message
    assert successes == []


def test_controller_rejects_concurrent_submission_and_idle_cancel(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    release = threading.Event()

    def submitter(_job_argument, _hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        release.wait(2.0)
        return _result()

    controller = RegionalGroundExecutionController(submitter)
    controller.submit(_job())
    assert controller.active_job_sha256 == _job().job_sha256
    with pytest.raises(RuntimeError, match="already running"):
        controller.submit(_job())
    release.set()
    qtbot.waitUntil(lambda: not controller.is_running, timeout=5_000)

    assert not controller.cancel()


def test_controller_shutdown_cancels_and_joins_worker(qtbot) -> None:  # type: ignore[no-untyped-def]
    started = threading.Event()
    successes: list[object] = []

    def submitter(_job_argument, hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        assert hooks.cancellation_requested is not None
        started.set()
        while not hooks.cancellation_requested():
            started.wait(0.005)
        raise GroundRegionalVariationCancelled(0, 4)

    controller = RegionalGroundExecutionController(submitter)
    controller.succeeded.connect(successes.append)
    controller.submit(_job())
    qtbot.waitUntil(started.is_set, timeout=2_000)

    controller.shutdown(timeout_ms=5_000)

    assert not controller.is_running
    assert controller.active_job_sha256 is None
    assert successes == []
