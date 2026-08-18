"""Responsive capability-optimization worker contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.capability_workflow import (  # noqa: E402
    CapabilityWorkflowInputs,
    build_capability_workflow,
)
from rate_of_closure.ui.pyqt6.capability_worker import (  # noqa: E402
    CapabilityOptimizationWorker,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_worker_publishes_result_dataset_and_exact_final_progress(qtbot) -> None:  # type: ignore[no-untyped-def]
    document = build_capability_workflow(
        CapabilityWorkflowInputs(
            candidate_budget=1, ensemble_size=2, alternatives_count=1
        )
    )
    worker = CapabilityOptimizationWorker(document)
    progress: list[tuple[int, int]] = []
    worker.progressed.connect(
        lambda completed, total: progress.append((completed, total))
    )

    with qtbot.waitSignal(worker.succeeded, timeout=20_000) as signal:
        worker.start()

    result, dataset = signal.args
    assert result.evaluations_attempted == 2
    assert len(dataset.rows) == 2
    assert progress[-1] == (2, 2)
    assert worker.wait(5_000)


def test_worker_honors_prestart_cancellation_without_partial_publication(qtbot) -> None:  # type: ignore[no-untyped-def]
    document = build_capability_workflow(
        CapabilityWorkflowInputs(
            candidate_budget=2, ensemble_size=2, alternatives_count=1
        )
    )
    worker = CapabilityOptimizationWorker(document)
    succeeded: list[object] = []
    worker.succeeded.connect(succeeded.append)
    worker.cancel()

    with qtbot.waitSignal(worker.cancelled, timeout=5_000) as signal:
        worker.start()

    assert signal.args == [0, 4]
    assert succeeded == []
    assert worker.wait(5_000)
