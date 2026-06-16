"""Unit tests for alarm-event processing.

Uses a tiny fake alarm engine so the transition logic is tested without the
real Rust engine or any limits config.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alarm_processing import process_alarm_events, severity_for_state  # noqa: E402


class FakeEngine:
    """Returns canned transition events per tag: {tag: [state, ...]}."""

    def __init__(self, scripted: dict[str, list[str]]) -> None:
        self._scripted = scripted

    def update_tag(self, name: str, _value: float) -> list[dict[str, Any]]:
        return [{"current_state": f"State.{s}"} for s in self._scripted.get(name, [])]


class TestSeverity:
    @pytest.mark.parametrize(
        "state,expected",
        [("Normal", 0), ("Low", 1), ("High", 1), ("LoLo", 2), ("HiHi", 2), ("?", 0)],
    )
    def test_severity_for_state(self, state: str, expected: int) -> None:
        assert severity_for_state(state) == expected


class TestProcessAlarmEvents:
    def test_new_alarm_added_with_severity(self) -> None:
        active: dict[str, Any] = {}
        engine = FakeEngine({"TAG_5": ["LoLo"]})
        events = process_alarm_events(engine, {"TAG_5": 0.0}, active)
        assert active["TAG_5"]["state"] == "LoLo"
        assert active["TAG_5"]["severity"] == 2
        assert active["TAG_5"]["acknowledged"] is False
        assert len(events) == 1
        assert events[0].severity == 2

    def test_return_to_normal_drops_acknowledged_alarm(self) -> None:
        active: dict[str, Any] = {
            "TAG_5": {
                "tag_id": "TAG_5",
                "tag_name": "TAG_5",
                "state": "LoLo",
                "severity": 2,
                "acknowledged": True,
                "timestamp": "t",
            }
        }
        engine = FakeEngine({"TAG_5": ["Normal"]})
        process_alarm_events(engine, {"TAG_5": 50.0}, active)
        assert "TAG_5" not in active  # acknowledged + normal -> cleared

    def test_return_to_normal_keeps_unacked_as_normal(self) -> None:
        active: dict[str, Any] = {
            "TAG_5": {
                "tag_id": "TAG_5",
                "tag_name": "TAG_5",
                "state": "LoLo",
                "severity": 2,
                "acknowledged": False,
                "timestamp": "t",
            }
        }
        engine = FakeEngine({"TAG_5": ["Normal"]})
        process_alarm_events(engine, {"TAG_5": 50.0}, active)
        assert active["TAG_5"]["state"] == "Normal"  # still visible until acked

    def test_no_events_when_nothing_transitions(self) -> None:
        active: dict[str, Any] = {}
        events = process_alarm_events(FakeEngine({}), {"TAG_0": 1.0}, active)
        assert events == []
        assert active == {}

    def test_rejects_non_dict_tags(self) -> None:
        with pytest.raises(TypeError):
            process_alarm_events(FakeEngine({}), [], {})  # type: ignore[arg-type]

    def test_rejects_non_dict_active(self) -> None:
        with pytest.raises(TypeError):
            process_alarm_events(FakeEngine({}), {}, [])  # type: ignore[arg-type]

    def test_rejects_engine_without_update_tag(self) -> None:
        with pytest.raises(TypeError):
            process_alarm_events(object(), {}, {})
