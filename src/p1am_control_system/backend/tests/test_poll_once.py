"""Single-scan poll-loop tests for the P1AM backend.

The infinite poll loop delegates one scan to ``_poll_once`` so safety,
broadcast, historian, and alarm commit behavior can be tested without sleeping
or starting background tasks.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import _connect_once, _poll_once  # noqa: E402
from models import EventLog, RoutingConfig  # noqa: E402
from signal_quality import SignalFrameFactory  # noqa: E402


class _Status:
    def model_dump(self) -> dict[str, str]:
        return {"state": "ok"}


class _FakePLC:
    def __init__(self, *, connected: bool, tags: dict[str, float] | None) -> None:
        self.connected = connected
        self._tags = tags
        self.trigger_count = 0

    async def read_tags(self) -> dict[str, float] | None:
        return self._tags

    async def trigger_estop(self) -> None:
        self.trigger_count += 1


class _FakeConnectPLC:
    def __init__(self, routing: RoutingConfig | None) -> None:
        self.connected = False
        self.connect_count = 0
        self.trigger_count = 0
        self.routing = routing

    async def connect(self) -> bool:
        self.connect_count += 1
        self.connected = True
        return True

    async def trigger_estop(self) -> None:
        self.trigger_count += 1

    async def read_routing(self) -> RoutingConfig | None:
        return self.routing


class _FakeSimulator:
    def __init__(self, tags: dict[str, float] | None) -> None:
        self._tags = tags
        self.read_count = 0

    async def read_tags(self) -> dict[str, float] | None:
        self.read_count += 1
        return self._tags


class _FakePowerSupply:
    def __init__(self) -> None:
        self.seen_tags: list[dict[str, float] | None] = []
        self.controller = type(
            "Controller",
            (),
            {"config": type("Config", (), {"command_tag": "TAG_10"})()},
        )()

    async def poll(self, tags: dict[str, float] | None) -> _Status:
        self.seen_tags.append(tags)
        return _Status()


class _FakeAlicats:
    def get_devices_data(self) -> list[dict[str, str]]:
        return [{"device_id": "A"}]


class _FakeWsManager:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class _FakeAlarmEngine:
    def update_tag(self, name: str, value: float) -> list[dict[str, str]]:
        if name == "TAG_1" and value > 9.0:
            return [{"current_state": "State.High"}]
        return []


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[EventLog] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def add(self, row: EventLog) -> None:
        self.added.append(row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _session_factory(session: _FakeSession) -> Iterator[_FakeSession]:
    yield session


@pytest.mark.asyncio
async def test_poll_once_offline_falls_back_to_simulator_and_broadcasts_payload() -> (
    None
):
    # When the PLC is NOT connected (offline / dev), the simulator drives tags.
    session = _FakeSession()
    latest = {"TAG_0": 0.0, "TAG_1": 0.0}
    plc = _FakePLC(connected=False, tags=None)
    simulator = _FakeSimulator({"TAG_0": 2.5, "TAG_1": 10.0})
    power = _FakePowerSupply()
    ws = _FakeWsManager()
    logged_scans: list[dict[str, float]] = []

    def fake_log_scan(
        _session: _FakeSession,
        tags: dict[str, float],
        **_: object,
    ) -> int:
        logged_scans.append(dict(tags))
        return len(tags)

    payload = await _poll_once(
        plc=plc,
        backup=simulator,
        latest_tag_values=latest,
        ws=ws,
        alicats=_FakeAlicats(),
        power_supply=power,
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        session_factory=lambda: _session_factory(session),
        estop_active=False,
        log_scan=fake_log_scan,
    )

    assert simulator.read_count == 1
    assert latest == {"TAG_0": 2.5, "TAG_1": 10.0}
    assert power.seen_tags == [{"TAG_0": 2.5, "TAG_1": 10.0}]
    assert payload["tags"][:2] == [2.5, 10.0]
    assert payload["tag_samples"]["TAG_0"]["quality"] == "simulated"
    assert payload["comms_health"]["quality"] == "simulated"
    assert ws.messages == [payload]
    assert logged_scans == [{"TAG_0": 2.5, "TAG_1": 10.0}]
    assert len(session.added) == 1
    assert session.commits == 1
    assert session.rollbacks == 0
    assert session.closed is True


@pytest.mark.asyncio
async def test_poll_once_connected_read_hiccup_holds_last_good() -> None:
    # A connected PLC whose read momentarily fails (returns None) must HOLD the
    # last good values, NOT substitute the offline simulator's fake readings —
    # otherwise a comms hiccup shows as a spurious drop to ~0 and feeds the
    # control law a false "cold".
    session = _FakeSession()
    last_good = {"TAG_0": 56.5, "TAG_1": 2.5}
    latest = dict(last_good)
    plc = _FakePLC(connected=True, tags=None)  # connected but read fails
    simulator = _FakeSimulator({"TAG_0": 0.0, "TAG_1": 0.0})  # must NOT be used
    power = _FakePowerSupply()
    ws = _FakeWsManager()
    alarm_calls: list[dict[str, float]] = []

    def process_events(
        _engine: object,
        tags: dict[str, float],
        _active: dict[str, dict[str, Any]],
    ) -> list[EventLog]:
        alarm_calls.append(tags)
        return []

    payload = await _poll_once(
        plc=plc,
        backup=simulator,
        latest_tag_values=latest,
        ws=ws,
        alicats=_FakeAlicats(),
        power_supply=power,
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        session_factory=lambda: _session_factory(session),
        estop_active=False,
        log_scan=lambda _s, _t, **_kw: 0,
        process_events=process_events,
    )

    assert simulator.read_count == 0  # simulator never consulted while connected
    assert latest == last_good  # held, not zeroed
    assert payload["tags"][:2] == [56.5, 2.5]
    assert payload["tag_samples"]["TAG_0"]["quality"] == "stale"
    assert payload["tag_samples"]["TAG_0"]["diagnostic_reason"] == "read_timeout"
    assert payload["comms_health"]["quality"] == "stale"
    assert power.seen_tags == [last_good]
    assert alarm_calls == []


@pytest.mark.asyncio
async def test_poll_once_reasserts_estop_every_connected_scan() -> None:
    session = _FakeSession()
    plc = _FakePLC(connected=True, tags={"TAG_0": 1.0})

    await _poll_once(
        plc=plc,
        backup=_FakeSimulator({"TAG_0": 9.0}),
        latest_tag_values={"TAG_0": 0.0},
        ws=_FakeWsManager(),
        alicats=_FakeAlicats(),
        power_supply=_FakePowerSupply(),
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        session_factory=lambda: _session_factory(session),
        estop_active=True,
    )

    assert plc.trigger_count == 1


@pytest.mark.asyncio
async def test_poll_once_rolls_back_historian_and_alarm_transaction() -> None:
    session = _FakeSession()

    def failing_log_scan(
        _session: _FakeSession,
        _tags: dict[str, float],
        **_: object,
    ) -> int:
        raise RuntimeError("disk unavailable")

    await _poll_once(
        plc=_FakePLC(connected=False, tags=None),
        backup=_FakeSimulator({"TAG_1": 10.0}),
        latest_tag_values={"TAG_1": 0.0},
        ws=_FakeWsManager(),
        alicats=_FakeAlicats(),
        power_supply=_FakePowerSupply(),
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        session_factory=lambda: _session_factory(session),
        estop_active=False,
        log_scan=failing_log_scan,
    )

    assert session.commits == 0
    assert session.rollbacks == 1
    assert session.closed is True


@pytest.mark.asyncio
async def test_poll_frames_increment_one_shared_scan_sequence() -> None:
    factory = SignalFrameFactory()
    sequences: list[int] = []
    latest = {"TAG_0": 0.0}
    for value in (1.0, 2.0):
        payload = await _poll_once(
            plc=_FakePLC(connected=True, tags={"TAG_0": value}),
            backup=_FakeSimulator(None),
            latest_tag_values=latest,
            ws=_FakeWsManager(),
            alicats=_FakeAlicats(),
            power_supply=_FakePowerSupply(),
            alarm_engine=_FakeAlarmEngine(),
            active_alarm_map={},
            session_factory=lambda: _session_factory(_FakeSession()),
            estop_active=False,
            signal_frames=factory,
            log_scan=lambda _s, _t, **_kw: 0,
        )
        sequences.append(payload["comms_health"]["sequence"])

    assert sequences == [1, 2]


@pytest.mark.asyncio
async def test_connect_once_syncs_routing_and_reasserts_estop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routing = RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=["TAG_1"],
        pids=[],
        interlocks={},
    )
    plc = _FakeConnectPLC(routing)
    applied: list[RoutingConfig] = []
    passthrough_calls: list[RoutingConfig] = []

    async def fake_passthrough(
        _plc: _FakeConnectPLC,
        config: RoutingConfig,
        *,
        command_tag: str,
        logger: Any,
    ) -> RoutingConfig:
        assert command_tag
        assert logger is not None
        passthrough_calls.append(config)
        return config

    synced = await _connect_once(
        plc=plc,
        power_supply=_FakePowerSupply(),
        apply_config=applied.append,
        estop_active=True,
        ensure_passthrough=fake_passthrough,
    )

    assert synced is routing
    assert plc.connect_count == 1
    assert plc.trigger_count == 1
    assert passthrough_calls == [routing]
    assert applied == [routing]
