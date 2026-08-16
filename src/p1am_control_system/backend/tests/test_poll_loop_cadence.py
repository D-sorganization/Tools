"""The control period must not be settable by a browser tab (issue #4008).

``PerformanceMode.LIGHTWEIGHT`` is a *rendering* concession the HMI makes when
its tab is hidden. It may decimate the WebSocket broadcast; it may never slow
the PLC scan, the alarm evaluation, the heater-relay decision or the E-stop
re-assert.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as backend_main  # noqa: E402
from performance import PerformanceMode, ScanScheduler  # noqa: E402


class _Clock:
    """Virtual monotonic clock driven by the loop's own (faked) sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def _restore_mode() -> Any:
    original = backend_main.perf_controller.mode
    yield
    backend_main.perf_controller.set_mode(original)


@pytest.mark.asyncio
async def test_lightweight_mode_does_not_slow_the_scan(
    monkeypatch: pytest.MonkeyPatch, _restore_mode: None
) -> None:
    backend_main.perf_controller.set_mode(PerformanceMode.LIGHTWEIGHT)
    clock = _Clock()
    sleeps: list[float] = []
    broadcasts: list[bool] = []

    async def fake_poll_once(**kwargs: Any) -> dict[str, Any]:
        broadcasts.append(bool(kwargs["broadcast"]))
        clock.advance(0.01)  # 10 ms of scan work
        return {"tags": []}

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.advance(delay)
        if len(sleeps) == 40:
            backend_main.shutdown_event.set()

    monkeypatch.setattr(backend_main, "_poll_once", fake_poll_once)
    monkeypatch.setattr(backend_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(backend_main.settings, "poll_interval_s", 0.1)
    monkeypatch.setattr(
        backend_main, "scan_scheduler", ScanScheduler(0.1, monotonic=clock)
    )

    backend_main.shutdown_event.clear()
    try:
        await backend_main.poll_plc_loop()
    finally:
        backend_main.shutdown_event.clear()

    # The scan cadence tracks settings.poll_interval_s, NOT the 2.0 s
    # lightweight interval a hidden browser tab asks for. The work time is
    # absorbed by the deadline, so each cycle is exactly one period.
    assert sleeps, "loop never slept"
    assert max(sleeps) == pytest.approx(0.09)
    assert backend_main.scan_scheduler.overrun_count == 0
    # Only the broadcast is decimated: 20 scans per frame at 0.1 s / 2.0 s.
    assert broadcasts.count(True) < len(broadcasts)
    assert broadcasts.count(True) >= 1
    assert backend_main.perf_controller.broadcast_every_n == 20


@pytest.mark.asyncio
async def test_performance_mode_broadcasts_every_scan(
    monkeypatch: pytest.MonkeyPatch, _restore_mode: None
) -> None:
    backend_main.perf_controller.set_mode(PerformanceMode.PERFORMANCE)
    broadcasts: list[bool] = []
    calls = {"n": 0}

    async def fake_poll_once(**kwargs: Any) -> dict[str, Any]:
        broadcasts.append(bool(kwargs["broadcast"]))
        return {"tags": []}

    async def fake_sleep(_delay: float) -> None:
        calls["n"] += 1
        if calls["n"] == 5:
            backend_main.shutdown_event.set()

    monkeypatch.setattr(backend_main, "_poll_once", fake_poll_once)
    monkeypatch.setattr(backend_main.asyncio, "sleep", fake_sleep)

    backend_main.shutdown_event.clear()
    try:
        await backend_main.poll_plc_loop()
    finally:
        backend_main.shutdown_event.clear()

    assert all(broadcasts)


@pytest.mark.asyncio
async def test_overruns_are_surfaced_on_the_performance_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    sched = ScanScheduler(0.1, monotonic=clock)
    clock.advance(0.5)  # a scan blew straight through its deadline
    sched.next_sleep_s()
    monkeypatch.setattr(backend_main, "scan_scheduler", sched)

    cfg = await backend_main.get_performance()

    assert cfg.scan_overruns == 1
    assert cfg.scan_interval_s == pytest.approx(backend_main.settings.poll_interval_s)


@pytest.mark.asyncio
async def test_historian_write_failures_are_surfaced() -> None:
    cfg = await backend_main.get_performance()
    assert cfg.historian_write_failures >= 0
