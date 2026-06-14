"""P1AM desktop HTTP worker regressions for responsive GUI writes (#3352)."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("requests")

from PyQt6.QtCore import QThread, QTimer  # noqa: E402

from p1am_control_system.desktop import workers  # noqa: E402
from p1am_control_system.desktop.control_tab import ControlTab  # noqa: E402
from p1am_control_system.desktop.workers import HttpWorker  # noqa: E402


class _Response:
    text = "ok"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "time": [0.0, 1.0],
            "pid": {"pv": [0.0, 1.0], "cv": [0.1, 0.2]},
            "mpc": {"pv": [0.0, 0.8], "cv": [0.1, 0.15]},
        }


def _wait_until(qapp, predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert predicate()


def test_http_worker_converts_scalar_timeout_to_explicit_connect_read_tuple() -> None:
    worker = HttpWorker("POST", "http://backend/api", timeout=3.0)
    assert worker.timeout == (0.5, 3.0)

    worker = HttpWorker("POST", "http://backend/api", timeout=(0.25, 1.25))
    assert worker.timeout == (0.25, 1.25)


def test_requests_client_recovers_after_import_masking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workers, "requests", None)

    client = workers._requests_client()

    assert client is not None
    assert workers.requests is client


@pytest.mark.gui
def test_mpc_simulation_http_worker_keeps_qt_event_loop_responsive(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_thread_id = int(QThread.currentThreadId())  # type: ignore[arg-type]
    request_thread_ids: list[int] = []

    def _post(*_args, **_kwargs):
        request_thread_ids.append(int(QThread.currentThreadId()))  # type: ignore[arg-type]
        time.sleep(0.2)
        return _Response()

    monkeypatch.setattr(workers.requests, "post", _post)

    ticks = 0

    def _tick() -> None:
        nonlocal ticks
        ticks += 1

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(10)

    tab = ControlTab()
    tab._simulate_mpc()

    assert tab.btn_simulate_mpc.isEnabled() is False
    assert tab.btn_simulate_mpc.text() == "Simulating..."

    _wait_until(qapp, lambda: tab.btn_simulate_mpc.isEnabled())
    timer.stop()

    assert ticks > 0
    assert request_thread_ids and request_thread_ids[0] != main_thread_id
    assert tab.btn_simulate_mpc.text() == "Simulate PID vs MPC"
