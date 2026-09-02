"""Single-scan poll-loop tests for the P1AM backend.

The infinite poll loop delegates one scan to ``poll_once`` so safety,
broadcast, historian, and alarm commit behavior can be tested without sleeping
or starting background tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import connect_once, poll_once  # noqa: E402
from models import RoutingConfig  # noqa: E402


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


class _FakeHistorian:
    """Historian sink double — the scan queues records, it never writes SQLite."""

    def __init__(self) -> None:
        self.records: list[Any] = []

    def submit(self, record: Any) -> bool:
        self.records.append(record)
        return True


@pytest.mark.asyncio
async def test_poll_once_offline_falls_back_to_simulator_and_broadcasts_payload() -> (
    None
):
    # On a SIMULATOR driver the backup simulator drives the bench plant, and
    # every row it produces is stamped as simulated (issue #4004).
    hist = _FakeHistorian()
    latest = {"TAG_0": 0.0, "TAG_1": 0.0}
    plc = _FakePLC(connected=False, tags=None)
    simulator = _FakeSimulator({"TAG_0": 2.5, "TAG_1": 10.0})
    power = _FakePowerSupply()
    ws = _FakeWsManager()

    payload = await poll_once(
        plc=plc,
        backup=simulator,
        simulated=True,
        latest_tag_values=latest,
        ws=ws,
        alicats=_FakeAlicats(),
        power_supply=power,
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        estop_active=False,
        historian=hist,
    )

    assert simulator.read_count == 1
    assert latest == {"TAG_0": 2.5, "TAG_1": 10.0}
    assert power.seen_tags == [{"TAG_0": 2.5, "TAG_1": 10.0}]
    assert payload["tags"][:2] == [2.5, 10.0]
    assert payload["data_source"] == "simulated"
    assert ws.messages == [payload]
    assert len(hist.records) == 1
    assert hist.records[0].tags == {"TAG_0": 2.5, "TAG_1": 10.0}
    assert hist.records[0].quality == "simulated"
    assert len(hist.records[0].events) == 1  # the TAG_1 High transition


@pytest.mark.asyncio
async def test_poll_once_connected_read_hiccup_holds_last_good() -> None:
    # A connected PLC whose read momentarily fails (returns None) HOLDS the last
    # good values for the HMI so the trace does not flicker to ~0 — but the held
    # numbers are marked and must not reach the control law (issue #4004).
    hist = _FakeHistorian()
    last_good = {"TAG_0": 56.5, "TAG_1": 2.5}
    latest = dict(last_good)
    plc = _FakePLC(connected=True, tags=None)  # connected but read fails
    simulator = _FakeSimulator({"TAG_0": 0.0, "TAG_1": 0.0})  # must NOT be used
    power = _FakePowerSupply()
    ws = _FakeWsManager()

    payload = await poll_once(
        plc=plc,
        backup=simulator,
        latest_tag_values=latest,
        ws=ws,
        alicats=_FakeAlicats(),
        power_supply=power,
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        estop_active=False,
        historian=hist,
    )

    assert simulator.read_count == 0  # simulator never consulted while connected
    assert latest == last_good  # held, not zeroed
    assert payload["tags"][:2] == [56.5, 2.5]
    assert payload["data_source"] == "held"
    assert power.seen_tags == [None]  # the control law is told there is no data
    assert all(rec.tags is None for rec in hist.records)


@pytest.mark.asyncio
async def test_poll_once_reasserts_estop_every_connected_scan() -> None:
    plc = _FakePLC(connected=True, tags={"TAG_0": 1.0})

    await poll_once(
        plc=plc,
        backup=_FakeSimulator({"TAG_0": 9.0}),
        latest_tag_values={"TAG_0": 0.0},
        ws=_FakeWsManager(),
        alicats=_FakeAlicats(),
        power_supply=_FakePowerSupply(),
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        estop_active=True,
    )

    assert plc.trigger_count == 1


@pytest.mark.asyncio
async def test_poll_once_never_touches_a_database_session() -> None:
    """#4023: persistence is queued for the writer task, not done inline."""
    hist = _FakeHistorian()

    await poll_once(
        plc=_FakePLC(connected=False, tags=None),
        backup=_FakeSimulator({"TAG_1": 10.0}),
        simulated=True,
        latest_tag_values={"TAG_1": 0.0},
        ws=_FakeWsManager(),
        alicats=_FakeAlicats(),
        power_supply=_FakePowerSupply(),
        alarm_engine=_FakeAlarmEngine(),
        active_alarm_map={},
        estop_active=False,
        historian=hist,
    )

    assert len(hist.records) == 1
    assert isinstance(hist.records[0].events, tuple)


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

    synced = await connect_once(
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
