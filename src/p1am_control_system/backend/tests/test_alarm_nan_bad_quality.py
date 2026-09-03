"""NaN/Inf must read as BadQuality -- never Normal -- in both alarm engines.

Issue #3973: ``update_tag(tag, nan)`` computed ``Normal`` (every band
comparison is False for NaN), emitted a state-change event, and
``process_alarm_events`` then resolved a live HiHi. Both the pure-Python
fallback and the Rust ``tools_core.scada`` engine had the same hole, so a
parity test could not see it. This module drives the same scenario through
every engine that is importable and asserts they agree, event for event.

The Rust engine is built from ``rust_core/tools-core/src/scada.rs`` in CI.
If an *installed* ``tools_core`` wheel predates the fix the parity case FAILS
rather than skips: a stale wheel silently diverging from the source is the
exact condition the test exists to catch.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import scada_fallback
from alarm_processing import (
    BAD_QUALITY_STATE,
    process_alarm_events,
    severity_for_state,
    state_name,
)

_LIMITS = {"T1": {"lolo": 10.0, "low": 20.0, "high": 80.0, "hihi": 90.0}}


def _engines() -> list[Any]:
    engines: list[Any] = [pytest.param(scada_fallback.AlarmEngine, id="python")]
    try:
        from tools_core import scada as rust_scada
    except ImportError:  # pragma: no cover - wheel absent on this interpreter
        return engines
    engines.append(pytest.param(rust_scada.AlarmEngine, id="rust"))
    return engines


@pytest.fixture(params=_engines())
def engine_cls(request: pytest.FixtureRequest) -> Any:
    return request.param


def _states(engine: Any) -> str:
    return state_name(engine.get_alarm_state("T1")["state"])


def test_nan_is_bad_quality_never_normal(engine_cls: Any) -> None:
    engine = engine_cls(_LIMITS)
    events = engine.update_tag("T1", float("nan"))
    assert len(events) == 1
    assert state_name(events[0]["current_state"]) == BAD_QUALITY_STATE, (
        f"{engine_cls.__module__}.{engine_cls.__name__} classified NaN as "
        f"{state_name(events[0]['current_state'])}; if this is the Rust engine "
        "the installed tools_core wheel is stale relative to scada.rs -- rebuild it"
    )
    assert _states(engine) == BAD_QUALITY_STATE
    active = engine.get_active_alarms()
    assert len(active) == 1
    assert state_name(active[0]["state"]) == BAD_QUALITY_STATE
    assert active[0]["severity"] == severity_for_state(BAD_QUALITY_STATE) == 2


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_infinity_is_bad_quality(engine_cls: Any, bad: float) -> None:
    engine = engine_cls(_LIMITS)
    engine.update_tag("T1", bad)
    assert _states(engine) == BAD_QUALITY_STATE


def test_nan_does_not_resolve_an_active_hihi(engine_cls: Any) -> None:
    """The #3973 scenario end to end through process_alarm_events."""
    engine = engine_cls(_LIMITS)
    active: dict[str, dict[str, Any]] = {}

    process_alarm_events(engine, {"T1": 95.0}, active)
    assert active["T1"]["state"] == "HiHi"

    rows = process_alarm_events(engine, {"T1": float("nan")}, active)
    assert "T1" in active, "a NaN read must not clear the alarm"
    assert active["T1"]["state"] == BAD_QUALITY_STATE
    assert active["T1"]["severity"] == 2
    assert active["T1"]["acknowledged"] is False
    assert len(rows) == 1
    assert "not a number" in rows[0].description
    assert rows[0].severity == 2

    # A finite reading re-classifies normally; the alarm only resolves on a
    # real in-band measurement.
    process_alarm_events(engine, {"T1": 50.0}, active)
    assert active["T1"]["state"] == "Normal"


def test_bad_quality_is_acknowledgeable(engine_cls: Any) -> None:
    engine = engine_cls(_LIMITS)
    engine.update_tag("T1", float("nan"))
    assert engine.acknowledge_alarm("T1", "operator") is True
    assert engine.get_alarm_state("T1")["acknowledged"] is True


def test_disabled_sides_never_fire(engine_cls: Any) -> None:
    """The backend feeds None as -inf/+inf; the engine must never select it."""
    engine = engine_cls(
        {"T1": {"lolo": -math.inf, "low": -math.inf, "high": 95.0, "hihi": 100.0}}
    )
    assert engine.update_tag("T1", 0.0) == []
    assert _states(engine) == "Normal"
    engine.update_tag("T1", 96.0)
    assert _states(engine) == "High"


def test_python_and_rust_emit_identical_event_sequences() -> None:
    """Parity: the two engines must agree on every transition, NaN included."""
    rust_scada = pytest.importorskip("tools_core.scada")
    readings = [50.0, 95.0, float("nan"), float("nan"), 85.0, float("inf"), 15.0, 50.0]

    def run(cls: Any) -> list[tuple[str, str]]:
        engine = cls(_LIMITS)
        out: list[tuple[str, str]] = []
        for value in readings:
            for ev in engine.update_tag("T1", value):
                out.append(
                    (state_name(ev["previous_state"]), state_name(ev["current_state"]))
                )
        return out

    python_seq = run(scada_fallback.AlarmEngine)
    rust_seq = run(rust_scada.AlarmEngine)
    assert python_seq == rust_seq
    assert python_seq == [
        ("Normal", "HiHi"),
        ("HiHi", BAD_QUALITY_STATE),
        (BAD_QUALITY_STATE, "High"),
        ("High", BAD_QUALITY_STATE),
        (BAD_QUALITY_STATE, "Low"),
        ("Low", "Normal"),
    ]
