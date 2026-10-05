"""Race-free Qt waits shared by the rate_of_closure GUI worker tests (#5440).

A queued Qt signal reaches only the receivers connected when it is emitted.
A test that starts a worker and *then* attaches ``qtbot.waitSignal`` misses
the signal whenever the worker finishes first, and blocks for its full
timeout. That ties the 60 s pytest-timeout (thread method), which
``os._exit()``s the xdist worker. These helpers wait on widget STATE that
is already in place before the worker starts, and keep every wait strictly
below the pytest timeout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

# Strictly below the 60 s ``timeout`` in pyproject.toml.
PYTEST_TIMEOUT_MS = 60_000
QT_WAIT_MS = 30_000
JOIN_MS = 10_000


def wait_for_state(
    qtbot: Any,
    predicate: Callable[[], bool],
    describe: Callable[[], str],
    *,
    timeout_ms: int = QT_WAIT_MS,
) -> None:
    """Wait until ``predicate()`` holds, failing fast with ``describe()``.

    Precondition: ``0 < timeout_ms < PYTEST_TIMEOUT_MS``.
    Postcondition: ``predicate()`` was true, or the test failed with the
    caller's description of the stuck state (never a bare timeout).
    """
    assert 0 < timeout_ms < PYTEST_TIMEOUT_MS, (
        f"Qt wait {timeout_ms} ms must be below the {PYTEST_TIMEOUT_MS} ms "
        "pytest timeout"
    )
    try:
        qtbot.waitUntil(predicate, timeout=timeout_ms)
    except qtbot.TimeoutError:
        pytest.fail(f"still running after {timeout_ms} ms: {describe()}")


def join_worker(worker: Any, *, timeout_ms: int = JOIN_MS) -> None:
    """Assert the worker thread has joined.

    Precondition: ``0 < timeout_ms < PYTEST_TIMEOUT_MS``.
    """
    assert 0 < timeout_ms < PYTEST_TIMEOUT_MS
    assert worker.wait(timeout_ms), "worker thread did not join"
