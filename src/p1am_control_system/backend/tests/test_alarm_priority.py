"""Per-tag alarm priority override (SCADA F03, issue #5526).

Contract under test (see ``alarm_priority.py``):

* No override => the historical state->severity map, byte for byte.
* An override changes the severity of THAT tag only.
* Invalid override => ``ValueError`` at the config boundary.
* ``BadQuality`` is a safety floor: it is trip severity whatever the override.
* A stored override that is somehow unreadable at resolve time fails CLOSED to
  trip severity (2), never to a quieter tier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

import scada_fallback  # noqa: E402
from alarm_priority import (  # noqa: E402
    FAIL_CLOSED_SEVERITY,
    AlarmPriority,
    parse_alarm_priority,
    resolve_severity,
)
from alarm_processing import (  # noqa: E402
    build_alarm_entry,
    process_alarm_events,
    severity_for_state,
)
from models import InterlockConfig, RoutingConfig  # noqa: E402

_GOLDEN = {
    "Normal": 0,
    "Low": 1,
    "High": 1,
    "LoLo": 2,
    "HiHi": 2,
    "BadQuality": 2,
    "Unknown": 0,
}


class _Engine:
    def __init__(self, scripted: dict[str, list[str]]) -> None:
        self._scripted = scripted

    def update_tag(self, name: str, _value: float) -> list[dict[str, Any]]:
        return [{"current_state": f"State.{s}"} for s in self._scripted.get(name, [])]


@pytest.mark.parametrize(("state", "expected"), _GOLDEN.items())
def test_default_classification_is_unchanged(state: str, expected: int) -> None:
    assert severity_for_state(state) == expected
    assert resolve_severity(state, None) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (1, AlarmPriority.WARNING),
        (2, AlarmPriority.TRIP),
        ("warning", AlarmPriority.WARNING),
        ("TRIP", AlarmPriority.TRIP),
        (AlarmPriority.TRIP, AlarmPriority.TRIP),
    ],
)
def test_parse_accepts_valid_forms(raw: Any, expected: AlarmPriority | None) -> None:
    assert parse_alarm_priority(raw) is expected


@pytest.mark.parametrize("raw", [0, 3, -1, 1.5, True, "", "critical", [], {}])
def test_parse_rejects_invalid_with_clear_error(raw: Any) -> None:
    with pytest.raises(ValueError, match="alarm_priority"):
        parse_alarm_priority(raw)


def test_override_changes_severity_of_alarm_states() -> None:
    assert resolve_severity("HiHi", AlarmPriority.WARNING) == 1
    assert resolve_severity("Low", AlarmPriority.TRIP) == 2


def test_override_never_creates_severity_for_normal() -> None:
    assert resolve_severity("Normal", AlarmPriority.TRIP) == 0


def test_bad_quality_is_a_safety_floor() -> None:
    assert resolve_severity("BadQuality", AlarmPriority.WARNING) == 2


def test_unreadable_override_fails_closed_to_trip() -> None:
    assert FAIL_CLOSED_SEVERITY == 2
    assert resolve_severity("Low", "garbage") == 2  # type: ignore[arg-type]
    assert resolve_severity("Normal", "garbage") == 0  # type: ignore[arg-type]


def test_override_affects_only_that_tag() -> None:
    active: dict[str, dict[str, Any]] = {}
    engine = _Engine({"A": ["HiHi"], "B": ["HiHi"]})
    rows = process_alarm_events(
        engine,
        {"A": 1.0, "B": 1.0},
        active,
        priorities={"A": AlarmPriority.WARNING},
    )
    assert active["A"]["severity"] == 1
    assert active["B"]["severity"] == 2
    assert [r.severity for r in rows] == [1, 2]


def test_bad_quality_event_stays_trip_under_override() -> None:
    active: dict[str, dict[str, Any]] = {}
    rows = process_alarm_events(
        _Engine({"A": ["BadQuality"]}),
        {"A": float("nan")},
        active,
        priorities={"A": AlarmPriority.WARNING},
    )
    assert active["A"]["severity"] == 2
    assert rows[0].severity == 2


def test_build_alarm_entry_honours_priority() -> None:
    entry = build_alarm_entry(
        "A", "HiHi", timestamp="t", priority=AlarmPriority.WARNING
    )
    assert entry["severity"] == 1
    assert build_alarm_entry("A", "HiHi", timestamp="t")["severity"] == 2


def test_interlock_config_validates_priority_at_boundary() -> None:
    assert InterlockConfig().alarm_priority is None
    assert InterlockConfig(alarm_priority="warning").alarm_priority is (
        AlarmPriority.WARNING
    )
    assert InterlockConfig(alarm_priority=2).alarm_priority is AlarmPriority.TRIP
    with pytest.raises(ValueError, match="alarm_priority"):
        InterlockConfig(alarm_priority=0)
    with pytest.raises(ValueError, match="alarm_priority"):
        InterlockConfig(alarm_priority="nonsense")


def test_priority_round_trips_through_the_config_store() -> None:
    from config_store import load_model, save_model
    from sqlmodel import Session, SQLModel, create_engine

    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    config = RoutingConfig(
        input_routing=[],
        output_routing=[],
        pids=[],
        interlocks={
            "A": InterlockConfig(alarm_priority="warning"),
            "B": InterlockConfig(),
        },
    )
    with Session(engine) as session:
        save_model(session, "routing", config)
        loaded = load_model(session, "routing", RoutingConfig)
    assert loaded is not None
    assert loaded.interlocks["A"].alarm_priority is AlarmPriority.WARNING
    assert loaded.interlocks["B"].alarm_priority is None


def test_system_state_exposes_priorities_from_active_config() -> None:
    from state import SystemState

    config = RoutingConfig(
        input_routing=[],
        output_routing=[],
        pids=[],
        interlocks={
            "A": InterlockConfig(alarm_priority="warning"),
            "B": InterlockConfig(),
        },
    )
    state = SystemState(
        alarm_engine_factory=lambda _c: scada_fallback.AlarmEngine({}),
        config=config,
    )
    assert state.alarm_priorities() == {"A": AlarmPriority.WARNING}


def _engines() -> list[Any]:
    engines: list[Any] = [pytest.param(scada_fallback.AlarmEngine, id="python")]
    try:
        from tools_core import scada as rust_scada
    except ImportError:  # pragma: no cover - wheel absent on this interpreter
        return engines
    engines.append(pytest.param(rust_scada.AlarmEngine, id="rust"))
    return engines


@pytest.mark.parametrize("engine_cls", _engines())
def test_engine_base_severity_matches_python_default_map(engine_cls: Any) -> None:
    """Parity: the engine owns the BASE tier; the override is applied above it.

    Each engine's reported severity must equal the Python default map for every
    state it can emit, so an un-overridden tag is identical on both paths.
    """
    limits = {"T": {"lolo": 10.0, "low": 20.0, "high": 80.0, "hihi": 90.0}}
    for value, state in [
        (50.0, "Normal"),
        (15.0, "Low"),
        (5.0, "LoLo"),
        (85.0, "High"),
        (95.0, "HiHi"),
        (float("nan"), "BadQuality"),
    ]:
        engine = engine_cls(limits)
        engine.update_tag("T", value)
        active = engine.get_active_alarms()
        if state == "Normal":
            assert active == []
            continue
        assert active[0]["severity"] == severity_for_state(state)
