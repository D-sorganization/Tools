"""Tests for NeuralNetworkTrainingWorker.

Verifies that neural network training runs off the Qt main thread so the
event loop stays responsive while a long training job is in flight.
"""

from __future__ import annotations

import sys
import time
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is required for Qt worker tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for Qt worker tests")

from PyQt6.QtCore import QThread  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from data_processor.ui.async_workers import (  # noqa: E402
    NeuralNetworkTrainingWorker,
)


class _SlowTrainer:
    """Fake trainer whose ``train`` call blocks long enough to detect UI freeze."""

    def __init__(self, duration_s: float = 0.4) -> None:
        self.duration_s = duration_s
        self.train_thread: int | None = None

    def train(self, data: pd.DataFrame) -> dict:
        # Capture the OS thread id so the test can confirm the call is
        # off the main thread.
        self.train_thread = int(QThread.currentThreadId())  # type: ignore[arg-type]
        time.sleep(self.duration_s)
        return {"ok": True, "rows": len(data)}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"x": range(100), "y": range(100)})


def test_worker_runs_off_main_thread(qtbot: Any, sample_df: pd.DataFrame) -> None:
    """The trainer's ``train`` method must execute on a worker thread."""
    trainer = _SlowTrainer(duration_s=0.2)
    worker = NeuralNetworkTrainingWorker(sample_df, {"epochs": 1}, trainer=trainer)

    main_thread_id = int(QThread.currentThreadId())  # type: ignore[arg-type]

    results: list[object] = []
    worker.result_ready.connect(results.append)
    worker.start()

    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    worker.wait(2000)

    assert results == [{"ok": True, "rows": 100}]
    assert trainer.train_thread is not None
    assert trainer.train_thread != main_thread_id, (
        "train() ran on the Qt main thread — UI would freeze"
    )


def test_worker_ui_stays_responsive(qtbot: Any, sample_df: pd.DataFrame) -> None:
    """While the worker runs, the Qt event loop must still process events."""
    trainer = _SlowTrainer(duration_s=0.5)
    worker = NeuralNetworkTrainingWorker(sample_df, {"epochs": 1}, trainer=trainer)

    done = []
    worker.result_ready.connect(lambda r: done.append(r))
    worker.start()

    # If the UI thread were blocked, qtbot.waitUntil would itself stall.
    # The tight poll here confirms the event loop is alive during training.
    ticks = 0

    def still_ticking() -> bool:
        nonlocal ticks
        ticks += 1
        return bool(done)

    qtbot.waitUntil(still_ticking, timeout=5000)
    worker.wait(2000)
    assert ticks > 1, "Event loop did not tick during training — UI was blocked"


def test_worker_emits_error_on_failure(qtbot: Any, sample_df: pd.DataFrame) -> None:
    trainer = MagicMock()
    trainer.train.side_effect = RuntimeError("boom")
    worker = NeuralNetworkTrainingWorker(sample_df, {"epochs": 1}, trainer=trainer)

    errors: list[str] = []
    worker.error.connect(errors.append)
    worker.start()
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)
    worker.wait(2000)
    assert errors == ["boom"]


def test_worker_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError):
        NeuralNetworkTrainingWorker(None, {"epochs": 1})
    with pytest.raises(ValueError):
        NeuralNetworkTrainingWorker(pd.DataFrame({"a": [1]}), None)
