"""Dedicated PyQt worker lifecycle for explicit paired attribution."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.simulation import run_simulation  # noqa: E402
from rate_of_closure.ui.pyqt6.localized_attribution_worker import (  # noqa: E402
    LocalizedAttributionWorker,
)
from rate_of_closure.variation.localized_attribution_producer import (  # noqa: E402
    LocalizedAttributionProduction,
)

from .test_localized_attribution_producer import _design  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_worker_emits_exact_completed_trial_progress_and_production(qtbot) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def executor(config):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return run_simulation(config)

    worker = LocalizedAttributionWorker(_design(include_wrist=True), executor=executor)
    progress: list[int] = []
    productions: list[object] = []
    worker.progressed.connect(lambda report: progress.append(report.iteration))
    worker.succeeded.connect(productions.append)

    with qtbot.waitSignal(worker.finished, timeout=60_000):
        worker.start()

    assert worker.total_runs == 4
    assert progress == [1, 2, 3, 4]
    assert calls == 4
    assert len(productions) == 1
    assert isinstance(productions[0], LocalizedAttributionProduction)


def test_worker_cancel_before_start_emits_cancelled(qtbot) -> None:  # type: ignore[no-untyped-def]
    worker = LocalizedAttributionWorker(_design())
    worker.cancel()

    with qtbot.waitSignal(worker.cancelled, timeout=10_000):
        worker.start()
    worker.wait(10_000)


def test_worker_mid_run_cancel_preserves_cancel_semantics(qtbot) -> None:  # type: ignore[no-untyped-def]
    worker: LocalizedAttributionWorker

    def executor(config):  # type: ignore[no-untyped-def]
        worker.cancel()
        return run_simulation(config)

    worker = LocalizedAttributionWorker(_design(), executor=executor)
    with qtbot.waitSignal(worker.cancelled, timeout=30_000):
        worker.start()
    worker.wait(10_000)


def test_worker_surfaces_executor_failure(qtbot) -> None:  # type: ignore[no-untyped-def]
    def producer(_design, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected paired failure")

    worker = LocalizedAttributionWorker(_design(), producer=producer)
    failures: list[str] = []
    worker.failed.connect(failures.append)

    with qtbot.waitSignal(worker.finished, timeout=10_000):
        worker.start()

    assert failures == ["injected paired failure"]
