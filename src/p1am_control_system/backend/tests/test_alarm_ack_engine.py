"""Regression tests for issue #4034 — alarm ack must reach the alarm engine.

``POST /api/alarms/{tag}/acknowledge`` used to flip a flag in
``SystemState.active_alarms`` only. ``AlarmEngine.acknowledge_alarm`` had no
production caller, so its bookkeeping (and the ``acknowledged_by`` audit field)
read ``False``/``None`` forever, and because ``apply_config`` clears
``active_alarms`` and rebuilds the engine on every routing deploy and every
reconnect-time ``_publish_active_config``, the acknowledgement silently
vanished — the operator saw a dealt-with alarm come back unacknowledged.
"""

from __future__ import annotations

from typing import Any

import pytest
import scada_fallback
from alarm_processing import process_alarm_events
from defaults import default_routing_config
from models import InterlockConfig, RoutingConfig
from state import SystemState

_TAG = "TAG_1"


def _build_alarm_engine(config: RoutingConfig) -> scada_fallback.AlarmEngine:
    """Mirror of ``main.build_alarm_engine`` over the pure-Python fallback."""
    return scada_fallback.AlarmEngine(
        {
            tag_name: {
                "lolo": interlock.lolo_limit,
                "low": interlock.low_limit,
                "high": interlock.high_limit,
                "hihi": interlock.hihi_limit,
            }
            for tag_name, interlock in config.interlocks.items()
        }
    )


def _tripped_state() -> SystemState:
    """A SystemState with ``_TAG`` driven into HiHi through the real engine."""
    state = SystemState(alarm_engine_factory=_build_alarm_engine)
    process_alarm_events(state.alarm_engine, {_TAG: 150.0}, state.active_alarms)
    assert state.active_alarms[_TAG]["state"] == "HiHi"
    return state


def test_acknowledgement_reaches_the_alarm_engine() -> None:
    state = _tripped_state()

    assert state.acknowledge_alarm(_TAG, user="operator-jane") is True

    engine_view = state.alarm_engine.get_alarm_state(_TAG)
    assert engine_view["acknowledged"] is True
    assert engine_view["acknowledged_by"] == "operator-jane"


def test_acknowledgement_populates_the_audit_field_on_the_live_map() -> None:
    state = _tripped_state()

    state.acknowledge_alarm(_TAG, user="operator-jane")

    assert state.active_alarms[_TAG]["acknowledged"] is True
    assert state.active_alarms[_TAG]["acknowledged_by"] == "operator-jane"


def test_unacknowledged_alarms_expose_a_null_audit_field() -> None:
    state = _tripped_state()

    assert state.active_alarms[_TAG]["acknowledged"] is False
    assert state.active_alarms[_TAG]["acknowledged_by"] is None


def test_acknowledgement_survives_a_routing_redeploy() -> None:
    """apply_config runs on every deploy AND every reconnect (#4034)."""
    state = _tripped_state()
    state.acknowledge_alarm(_TAG, user="operator-jane")

    state.apply_config(default_routing_config())

    assert _TAG in state.active_alarms, "the alarm must survive the rebuild"
    assert state.active_alarms[_TAG]["state"] == "HiHi"
    assert state.active_alarms[_TAG]["acknowledged"] is True
    assert state.active_alarms[_TAG]["acknowledged_by"] == "operator-jane"
    assert state.alarm_engine.get_alarm_state(_TAG)["acknowledged"] is True


def test_unacknowledged_alarm_also_survives_a_redeploy() -> None:
    state = _tripped_state()

    state.apply_config(default_routing_config())

    assert state.active_alarms[_TAG]["state"] == "HiHi"
    assert state.active_alarms[_TAG]["acknowledged"] is False


def test_redeploy_drops_alarms_for_tags_no_longer_configured() -> None:
    state = _tripped_state()
    state.acknowledge_alarm(_TAG, user="operator-jane")

    trimmed = default_routing_config()
    trimmed.interlocks = {
        "TAG_2": InterlockConfig(
            lolo_limit=0.0, low_limit=5.0, high_limit=95.0, hihi_limit=100.0
        )
    }
    state.apply_config(trimmed)

    assert state.active_alarms == {}


def test_acknowledged_alarm_that_returned_to_normal_is_cleared() -> None:
    state = _tripped_state()
    process_alarm_events(state.alarm_engine, {_TAG: 50.0}, state.active_alarms)
    assert state.active_alarms[_TAG]["state"] == "Normal"

    assert state.acknowledge_alarm(_TAG, user="operator-jane") is True
    assert _TAG not in state.active_alarms


def test_acknowledge_alarm_validates_its_inputs() -> None:
    state = _tripped_state()

    not_a_tag: Any = 1
    not_a_user: Any = object()
    with pytest.raises(TypeError):
        state.acknowledge_alarm(not_a_tag)
    with pytest.raises(TypeError):
        state.acknowledge_alarm(_TAG, user=not_a_user)
    with pytest.raises(ValueError):
        state.acknowledge_alarm(_TAG, user="   ")

    # None is the documented "unattributed operator" default, not an error.
    from state import DEFAULT_ACK_USER

    assert state.acknowledge_alarm(_TAG, user=None) is True
    assert state.active_alarms[_TAG]["acknowledged_by"] == DEFAULT_ACK_USER


def test_engine_without_the_ack_api_is_tolerated() -> None:
    """The engine is duck-typed; a minimal stub must not break the endpoint."""

    def _dict_engine(config: RoutingConfig) -> dict[str, Any]:
        return {"interlocks": dict(config.interlocks)}

    state = SystemState(alarm_engine_factory=_dict_engine)
    state.active_alarms[_TAG] = {
        "tag_id": _TAG,
        "state": "High",
        "acknowledged": False,
        "acknowledged_by": None,
    }

    assert state.acknowledge_alarm(_TAG, user="operator-jane") is True
    assert state.active_alarms[_TAG]["acknowledged_by"] == "operator-jane"

    state.apply_config(default_routing_config())
    assert state.active_alarms == {}


class _StubSession:
    """Minimal stand-in for the SQLModel session the endpoint logs through."""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commits = 0

    def add(self, row: Any) -> None:
        self.added.append(row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:  # pragma: no cover - defensive
        raise AssertionError("acknowledgement must not roll back")


def test_acknowledge_endpoint_forwards_the_operator_to_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#4034: main.py's endpoint never reached AlarmEngine.acknowledge_alarm."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    try:
        import main
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"P1AM backend not importable in this environment: {exc}")

    import asyncio

    state = _tripped_state()
    monkeypatch.setattr(main, "control_context", state)
    session = _StubSession()

    result = asyncio.run(
        main.acknowledge_alarm(
            _TAG, payload=main.AlarmAckPayload(user="operator-jane"), db=session
        )
    )

    assert result["status"] == "success"
    assert session.commits == 1
    assert state.alarm_engine.get_alarm_state(_TAG)["acknowledged_by"] == (
        "operator-jane"
    )
    assert state.active_alarms[_TAG]["acknowledged_by"] == "operator-jane"


def test_acknowledge_endpoint_defaults_the_operator_when_no_body_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    try:
        import main
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"P1AM backend not importable in this environment: {exc}")

    import asyncio

    from state import DEFAULT_ACK_USER

    state = _tripped_state()
    monkeypatch.setattr(main, "control_context", state)

    result = asyncio.run(main.acknowledge_alarm(_TAG, payload=None, db=_StubSession()))

    assert result["status"] == "success"
    assert state.alarm_engine.get_alarm_state(_TAG)["acknowledged"] is True
    assert state.active_alarms[_TAG]["acknowledged_by"] == DEFAULT_ACK_USER


def test_fallback_and_rust_alarm_engines_share_the_ack_signature() -> None:
    """Both engines must accept ``acknowledge_alarm(tag_id, user)`` (#4034)."""
    engine = _build_alarm_engine(default_routing_config())
    engine.update_tag(_TAG, 150.0)

    assert engine.acknowledge_alarm(_TAG, "operator-jane") is True
    assert engine.get_active_alarms()[0]["acknowledged_by"] == "operator-jane"
