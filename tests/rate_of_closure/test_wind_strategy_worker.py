"""Focused QThread contract tests for wind-strategy execution."""

from __future__ import annotations

import math

from rate_of_closure.ui.pyqt6.wind_strategy_worker import WindStrategyWorker
from shared.python.swing_sim.flight import (
    LaunchConditions,
    ScalarDistribution,
    StrategyAnalysisRequest,
    TargetPoint,
    WindStrategy,
    WindUncertaintySpec,
)


def _request(trials: int = 2) -> StrategyAnalysisRequest:
    launch = LaunchConditions.from_imperial(145.0, 12.0, 2600.0)
    return StrategyAnalysisRequest(
        uncertainty=WindUncertaintySpec(
            trials=trials,
            seed=4199,
            true_speed_mps=ScalarDistribution("fixed", 4.0, minimum=0.0),
            true_from_bearing_deg=ScalarDistribution("fixed", 90.0),
            provenance="ui-test",
        ),
        strategies=(
            WindStrategy("current-launch", "Current Launch", launch, math.radians(0.1)),
        ),
        target=TargetPoint(220.0, 0.0),
    )


def test_worker_reports_exact_progress_and_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    worker = WindStrategyWorker(_request(2))
    reports: list[tuple[int, int]] = []
    worker.progressed.connect(lambda done, total: reports.append((done, total)))

    with qtbot.waitSignal(worker.succeeded, timeout=10_000) as completed:
        worker.start()
    worker.wait(10_000)

    assert reports == [(0, 2), (1, 2), (2, 2)]
    assert len(completed.args[0].outcomes) == 2


def test_worker_cancel_emits_cancelled_without_publishing_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    worker = WindStrategyWorker(_request())
    successes: list[object] = []
    worker.succeeded.connect(successes.append)

    with qtbot.waitSignal(worker.cancelled, timeout=2_000):
        worker.cancel()
        worker.start()
    worker.wait(2_000)

    assert worker.cancel_event.is_set()
    assert successes == []
