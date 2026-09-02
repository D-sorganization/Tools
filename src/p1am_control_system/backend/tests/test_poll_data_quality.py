"""Data-quality contract for one PLC scan (issue #4004).

The poll loop must never present held or simulated numbers to the control laws,
the alarm engine or the historian as if they were measurements. On real
hardware a dropped link is a FAULT: a gap in the trend is correct, fabricated
continuity is not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DataSource  # noqa: E402
from poll_runtime import DataQualityTracker, poll_once  # noqa: E402


class _Status:
    def model_dump(self) -> dict[str, str]:
        return {"state": "ok"}


class _PLC:
    def __init__(self, *, connected: bool, tags: dict[str, float] | None) -> None:
        self.connected = connected
        self._tags = tags
        self.estop_flag: bool | None = None
        self.trigger_count = 0
        self.heartbeats = 0

    async def read_tags(self) -> dict[str, float] | None:
        return self._tags

    async def trigger_estop(self) -> None:
        self.trigger_count += 1

    def set_estop_active(self, active: bool) -> None:
        self.estop_flag = active

    async def write_heartbeat(self) -> bool:
        self.heartbeats += 1
        return True


class _PLCWithoutHeartbeat:
    """A client that predates the #4044 heartbeat seam (must still poll)."""

    def __init__(self, tags: dict[str, float]) -> None:
        self.connected = True
        self._tags = tags

    async def read_tags(self) -> dict[str, float]:
        return self._tags

    async def trigger_estop(self) -> None:  # pragma: no cover - not exercised
        return None


class _Simulator:
    def __init__(self, tags: dict[str, float]) -> None:
        self._tags = tags
        self.read_count = 0

    async def read_tags(self) -> dict[str, float]:
        self.read_count += 1
        return self._tags


class _Service:
    def __init__(self) -> None:
        self.seen_tags: list[dict[str, float] | None] = []
        self.estop_flag: bool | None = None
        self.engage_calls = 0
        self.controller = type(
            "Controller",
            (),
            {"config": type("Config", (), {"command_tag": "TAG_10"})()},
        )()

    async def poll(self, tags: dict[str, float] | None) -> _Status:
        self.seen_tags.append(tags)
        return _Status()

    def engage_estop(self) -> None:
        self.engage_calls += 1

    def set_estop_active(self, active: bool) -> None:
        self.estop_flag = active


class _Alicats:
    def get_devices_data(self) -> list[dict[str, str]]:
        return [{"device_id": "A"}]


class _Ws:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class _AlarmEngine:
    """Reports Normal for every tag — the flap-clears-alarm reproducer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def update_tag(self, name: str, value: float) -> list[dict[str, str]]:
        self.calls.append((name, value))
        return [{"current_state": "State.Normal"}]


class _Historian:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def submit(self, record: Any) -> bool:
        self.records.append(record)
        return True


async def _scan(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "plc": _PLC(connected=True, tags={"TAG_0": 1.0}),
        "backup": None,
        "simulated": False,
        "latest_tag_values": {},
        "ws": _Ws(),
        "alicats": _Alicats(),
        "power_supply": _Service(),
        "temperature": _Service(),
        "alarm_engine": _AlarmEngine(),
        "active_alarm_map": {},
        "estop_active": False,
        "historian": _Historian(),
    }
    kwargs.update(overrides)
    payload: dict[str, Any] = await poll_once(**kwargs)
    return payload


@pytest.mark.asyncio
async def test_real_driver_disconnect_is_a_fault_not_simulated_data() -> None:
    """#4004: a dropped link must never be back-filled by the simulator."""
    simulator = _Simulator({"TAG_0": 999.0})
    power = _Service()
    temp = _Service()
    alarms = _AlarmEngine()
    hist = _Historian()
    active = {
        "TAG_0": {
            "tag_id": "TAG_0",
            "tag_name": "TAG_0",
            "state": "HiHi",
            "severity": 2,
            "acknowledged": False,
            "timestamp": "now",
        }
    }

    payload = await _scan(
        plc=_PLC(connected=False, tags=None),
        backup=simulator,
        simulated=False,
        latest_tag_values={"TAG_0": 90.0},
        power_supply=power,
        temperature=temp,
        alarm_engine=alarms,
        active_alarm_map=active,
        historian=hist,
    )

    assert simulator.read_count == 0, "simulator must not drive a real-driver scan"
    assert payload["data_source"] == DataSource.FAULT
    assert payload["plc_connected"] is False
    assert payload["simulated"] is False
    # The control laws get no measurement at all.
    assert power.seen_tags == [None]
    assert temp.seen_tags == [None]
    # The alarm engine is not run, so an active HiHi cannot silently clear.
    assert alarms.calls == []
    assert active["TAG_0"]["state"] == "HiHi"
    # The historian records a gap, not fabricated continuity.
    assert all(rec.tags is None for rec in hist.records)


@pytest.mark.asyncio
async def test_real_driver_fault_arms_the_write_seam_interlocks() -> None:
    """Without a measurement no energizing command may be issued (#4004)."""
    power = _Service()
    temp = _Service()
    plc = _PLC(connected=False, tags=None)

    await _scan(
        plc=plc,
        backup=None,
        simulated=False,
        power_supply=power,
        temperature=temp,
        latest_tag_values={},
    )

    assert power.estop_flag is True
    assert temp.estop_flag is True
    assert plc.estop_flag is True
    # The one-way controller latch is NOT engaged — a data gap is not an E-stop.
    assert power.engage_calls == 0


@pytest.mark.asyncio
async def test_real_driver_read_hiccup_holds_for_display_only() -> None:
    """Held values may animate the HMI but must not reach the control path."""
    power = _Service()
    alarms = _AlarmEngine()
    hist = _Historian()

    payload = await _scan(
        plc=_PLC(connected=True, tags=None),  # connected, read returned None
        backup=None,
        simulated=False,
        latest_tag_values={"TAG_0": 56.5},
        power_supply=power,
        alarm_engine=alarms,
        historian=hist,
    )

    assert payload["data_source"] == DataSource.HELD
    assert payload["tags"][0] == 56.5  # displayed, clearly marked as held
    assert power.seen_tags == [None]
    assert alarms.calls == []
    assert all(rec.tags is None for rec in hist.records)


@pytest.mark.asyncio
async def test_simulator_driver_still_drives_the_bench_and_marks_data() -> None:
    """Bench mode keeps working, but every row is stamped ``simulated``."""
    simulator = _Simulator({"TAG_0": 2.5})
    power = _Service()
    hist = _Historian()

    payload = await _scan(
        plc=_PLC(connected=False, tags=None),
        backup=simulator,
        simulated=True,
        latest_tag_values={},
        power_supply=power,
        historian=hist,
    )

    assert simulator.read_count == 1
    assert payload["data_source"] == DataSource.SIMULATED
    assert payload["simulated"] is True
    assert power.seen_tags == [{"TAG_0": 2.5}]
    assert [rec.quality for rec in hist.records] == [DataSource.SIMULATED]
    assert hist.records[0].tags == {"TAG_0": 2.5}


@pytest.mark.asyncio
async def test_live_scan_marks_data_live_and_beats_the_watchdog() -> None:
    """#4044: a good scan is the host's proof-of-life to the firmware."""
    plc = _PLC(connected=True, tags={"TAG_0": 4.0})
    hist = _Historian()

    payload = await _scan(plc=plc, latest_tag_values={}, historian=hist)

    assert payload["data_source"] == DataSource.LIVE
    assert payload["plc_connected"] is True
    assert plc.heartbeats == 1
    assert [rec.quality for rec in hist.records] == [DataSource.LIVE]


@pytest.mark.asyncio
async def test_missing_heartbeat_seam_does_not_break_the_scan() -> None:
    payload = await _scan(plc=_PLCWithoutHeartbeat({"TAG_0": 4.0}))

    assert payload["data_source"] == DataSource.LIVE


@pytest.mark.asyncio
async def test_diagnostics_are_published_in_the_frame() -> None:
    payload = await _scan(diagnostics={"scan_overruns": 7})

    assert payload["diagnostics"]["scan_overruns"] == 7


@pytest.mark.asyncio
async def test_broadcast_can_be_decimated_without_skipping_the_scan() -> None:
    """#4008: throttling the socket must not throttle the control scan."""
    ws = _Ws()
    power = _Service()

    payload = await _scan(ws=ws, power_supply=power, broadcast=False)

    assert ws.messages == []
    assert power.seen_tags == [{"TAG_0": 1.0}]  # the scan still ran
    assert payload["data_source"] == DataSource.LIVE


class TestDataQualityTracker:
    def test_emits_an_event_only_on_transition(self) -> None:
        tracker = DataQualityTracker()

        first = tracker.observe(DataSource.LIVE)
        again = tracker.observe(DataSource.LIVE)
        degraded = tracker.observe(DataSource.FAULT)

        assert first is not None
        assert again is None
        assert degraded is not None
        assert "fault" in degraded.description.lower()
        assert degraded.severity == 2

    def test_rejects_an_unknown_source(self) -> None:
        tracker = DataQualityTracker()
        with pytest.raises(ValueError):
            tracker.observe("made-up")

    def test_rejects_a_non_string_source(self) -> None:
        tracker = DataQualityTracker()
        with pytest.raises(TypeError):
            tracker.observe(object())
