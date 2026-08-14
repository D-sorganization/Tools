"""Pure-logic regressions for the desktop HMI alarm/telemetry layer.

Covers:

* **#4012** — acknowledging a still-active alarm must not hide it, and an alarm
  whose condition cleared must stop annunciating.
* **#4019a** — high-high / low-low trip points come from the configured
  ``hihi_limit``/``lolo_limit``, never from a synthesised ``high_limit + 5``.
* **#4019b** — the PLC connection label is derived from the telemetry frame
  rather than hardcoded to ``"Simulating"``.
* **#4022** — alarm transitions are debounced/deduplicated before they are
  written to SQLite, writes are batched off the caller's thread, and the event
  table has a retention policy.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from p1am_control_system.desktop.alarm_state import (
    AlarmEventDebouncer,
    AlarmStateMachine,
    InterlockLimitError,
    classify_value,
    interlock_for_index,
    validate_interlocks,
)
from p1am_control_system.desktop.connection_state import (
    CONNECTED,
    OFFLINE,
    SIMULATING,
    derive_connection_status,
)
from p1am_control_system.desktop.event_logger import EventLogger


class _Interlock:
    """Minimal stand-in for ``backend.models.InterlockConfig``."""

    def __init__(
        self, lolo: float, low: float, high: float, hihi: float
    ) -> None:  # noqa: D107
        self.lolo_limit = lolo
        self.low_limit = low
        self.high_limit = high
        self.hihi_limit = hihi


NARROW = _Interlock(lolo=3.0, low=5.0, high=95.0, hihi=97.0)


# ---------------------------------------------------------------------------
# #4019a — configured HH/LL trip points
# ---------------------------------------------------------------------------


def test_hh_uses_configured_hihi_limit_not_high_plus_five() -> None:
    """H=95, HIHI=97: a value of 98 is HH, matching the firmware trip."""
    # The removed code synthesised hh_thresh = high + 5.0 = 100.0 and would have
    # classified 98.0 as a mere "H" — a severity below the plant's.
    assert classify_value(98.0, NARROW) == "HH"
    assert classify_value(97.0, NARROW) == "HH"
    assert classify_value(96.0, NARROW) == "H"
    assert classify_value(95.0, NARROW) == "H"


def test_ll_uses_configured_lolo_limit_not_low_minus_five() -> None:
    """LOW=5, LOLO=3: a value of 3 is LL, not merely L."""
    assert classify_value(3.0, NARROW) == "LL"
    assert classify_value(4.0, NARROW) == "L"
    assert classify_value(5.0, NARROW) == "L"


def test_classify_value_returns_none_inside_the_band() -> None:
    assert classify_value(50.0, NARROW) is None


def test_classify_value_rejects_non_numeric_value() -> None:
    with pytest.raises(TypeError):
        classify_value("hot", NARROW)


def test_validate_interlocks_rejects_inverted_thresholds() -> None:
    """lolo <= low <= high <= hihi is a precondition; violations fail loudly."""
    bad = {
        "TAG_0": _Interlock(lolo=0.0, low=5.0, high=95.0, hihi=100.0),
        "TAG_1": _Interlock(lolo=0.0, low=5.0, high=95.0, hihi=90.0),
    }
    with pytest.raises(InterlockLimitError) as exc:
        validate_interlocks(bad)
    assert "TAG_1" in str(exc.value)
    assert "TAG_0" not in str(exc.value)


def test_interlock_limit_error_is_a_value_error() -> None:
    assert issubclass(InterlockLimitError, ValueError)


def test_validate_interlocks_rejects_non_numeric_limits() -> None:
    with pytest.raises(TypeError):
        validate_interlocks({"TAG_0": _Interlock(0.0, 5.0, "95", 100.0)})


def test_validate_interlocks_accepts_ordered_limits() -> None:
    validate_interlocks({"TAG_0": _Interlock(0.0, 5.0, 95.0, 100.0)})


def test_interlock_for_index_resolves_tag_named_mapping_keys() -> None:
    """Backend serves ``dict[str, InterlockConfig]`` keyed ``TAG_<n>``."""
    interlocks = {"TAG_0": NARROW, "TAG_1": _Interlock(0.0, 1.0, 2.0, 3.0)}
    assert interlock_for_index(interlocks, 0) is NARROW
    assert interlock_for_index(interlocks, 1) is not NARROW
    assert interlock_for_index(interlocks, 7) is None


def test_interlock_for_index_supports_sequence_configs() -> None:
    assert interlock_for_index([NARROW], 0) is NARROW
    assert interlock_for_index([NARROW], 3) is None


# ---------------------------------------------------------------------------
# #4012 — acknowledge / clear transitions
# ---------------------------------------------------------------------------


def test_acknowledged_but_still_active_alarm_keeps_annunciating() -> None:
    """Acking a HH silences the flash but must NOT hide the condition (#4012a)."""
    machine = AlarmStateMachine()
    machine.evaluate(0, 99.0, NARROW)
    assert (0, "HH") in machine.active_alarms
    assert (0, "HH") in machine.unacknowledged_alarms

    machine.acknowledge([(0, "HH")])

    state = machine.annunciator_state()
    assert state.has_hhll is True, "colour must follow the active condition"
    assert state.unacked_hhll is False, "flash must follow acknowledgement"
    assert (0, "HH") in machine.active_alarms

    # A further scan with the tag still above HIHI keeps it annunciated and
    # must not re-flag it as unacknowledged.
    machine.evaluate(0, 99.0, NARROW)
    assert machine.annunciator_state().has_hhll is True
    assert machine.annunciator_state().unacked_hhll is False


def test_cleared_alarm_stops_flashing(tmp_path: Path) -> None:
    """Return to normal drops the key from BOTH sets (#4012b)."""
    machine = AlarmStateMachine()
    machine.evaluate(0, 99.0, NARROW)
    assert machine.unacknowledged_alarms

    transitions = machine.evaluate(0, 50.0, NARROW)

    assert machine.active_alarms == set()
    assert machine.unacknowledged_alarms == set()
    assert [t.kind for t in transitions] == ["cleared"]
    assert machine.annunciator_state() == (False, False, False, False)


def test_alarm_re_raised_after_clearing_is_unacknowledged_again() -> None:
    machine = AlarmStateMachine()
    machine.evaluate(0, 99.0, NARROW)
    machine.acknowledge([(0, "HH")])
    machine.evaluate(0, 50.0, NARROW)
    machine.evaluate(0, 99.0, NARROW)
    assert machine.annunciator_state().unacked_hhll is True


def test_acknowledge_only_clears_the_supplied_keys() -> None:
    """An alarm that arrived between render and click stays unacknowledged."""
    machine = AlarmStateMachine()
    machine.evaluate(0, 99.0, NARROW)
    displayed = frozenset(machine.unacknowledged_alarms)

    # A second alarm races in after the header was rendered.
    machine.evaluate(1, 99.0, NARROW)

    acked = machine.acknowledge(displayed)

    assert acked == [(0, "HH")]
    assert machine.unacknowledged_alarms == {(1, "HH")}
    assert machine.annunciator_state().unacked_hhll is True


def test_acknowledge_rejects_non_iterable_keys() -> None:
    machine = AlarmStateMachine()
    with pytest.raises(TypeError):
        machine.acknowledge(None)


def test_annunciator_state_separates_severity_from_acknowledgement() -> None:
    machine = AlarmStateMachine()
    machine.evaluate(0, 98.0, NARROW)  # HH
    machine.evaluate(1, 4.0, NARROW)  # L
    machine.acknowledge([(0, "HH")])

    state = machine.annunciator_state()
    assert (state.has_hhll, state.has_hl) == (True, True)
    assert (state.unacked_hhll, state.unacked_hl) == (False, True)


def test_evaluate_promotes_h_to_hh_without_leaving_a_stale_h() -> None:
    machine = AlarmStateMachine()
    machine.evaluate(0, 98.0, NARROW)
    assert machine.active_alarms == {(0, "HH")}
    machine.evaluate(0, 95.5, NARROW)
    assert machine.active_alarms == {(0, "H")}


def test_evaluate_rejects_bad_tag_id() -> None:
    machine = AlarmStateMachine()
    with pytest.raises(TypeError):
        machine.evaluate("0", 50.0, NARROW)


def test_evaluate_transition_messages_quote_the_configured_limit() -> None:
    machine = AlarmStateMachine()
    (transition,) = machine.evaluate(2, 99.0, NARROW)
    assert transition.kind == "raised"
    assert transition.tag_id == 2
    assert transition.alarm_type == "HH"
    assert "97.00" in transition.message


# ---------------------------------------------------------------------------
# #4019b — connection status derived from the telemetry frame
# ---------------------------------------------------------------------------


def test_derive_connection_status_defaults_to_connected() -> None:
    """A live plant must never be mislabelled as a bench simulation."""
    assert derive_connection_status({"tags": [1.0, 2.0]}) == CONNECTED


def test_derive_connection_status_honours_explicit_plc_connected_flag() -> None:
    assert derive_connection_status({"plc_connected": True}) == CONNECTED
    assert derive_connection_status({"plc_connected": False}) == SIMULATING


def test_derive_connection_status_honours_positive_simulated_flag() -> None:
    assert derive_connection_status({"simulated": True}) == SIMULATING
    assert derive_connection_status({"simulated": False}) == CONNECTED


def test_derive_connection_status_reports_offline_on_degraded_polling() -> None:
    frame = {"polling_status": {"status": "degraded", "consecutive_failures": 9}}
    assert derive_connection_status(frame) == OFFLINE


def test_derive_connection_status_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        derive_connection_status(["tags"])


# ---------------------------------------------------------------------------
# #4022 — debounce, batching, retention
# ---------------------------------------------------------------------------


def test_debouncer_coalesces_a_dithering_tag_into_one_counted_event() -> None:
    """A thermocouple chattering on its limit is ONE event with a count."""
    debouncer = AlarmEventDebouncer(window_s=5.0)
    key = (0, "HH", "ALARM")

    first = debouncer.submit(key, "ALARM", "Tag 0 High-High", now=100.0)
    assert [msg for _lvl, msg in first] == ["Tag 0 High-High"]

    for tick in range(1, 10):
        coalesced = debouncer.submit(
            key, "ALARM", "Tag 0 High-High", now=100.0 + tick * 0.1
        )
        assert coalesced == [], "repeats inside the window must not be written"

    released = debouncer.flush(now=106.0)
    assert len(released) == 1
    level, message = released[0]
    assert level == "ALARM"
    assert "9" in message and "repeat" in message.lower()

    # Nothing left to release.
    assert debouncer.flush(now=200.0) == []


def test_debouncer_emits_distinct_keys_immediately() -> None:
    debouncer = AlarmEventDebouncer(window_s=5.0)
    assert debouncer.submit((0, "HH", "ALARM"), "ALARM", "a", now=1.0)
    assert debouncer.submit((1, "HH", "ALARM"), "ALARM", "b", now=1.0)


def test_debouncer_emits_again_once_the_window_expires() -> None:
    debouncer = AlarmEventDebouncer(window_s=5.0)
    key = (0, "HH", "ALARM")
    assert debouncer.submit(key, "ALARM", "a", now=1.0)
    assert debouncer.submit(key, "ALARM", "a", now=2.0) == []
    assert debouncer.submit(key, "ALARM", "a", now=99.0)


def test_debouncer_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        AlarmEventDebouncer(window_s=0.0)


def test_event_logger_async_writes_happen_off_the_calling_thread(
    tmp_path: Path,
) -> None:
    """Alarm logging must not fsync-commit on the Qt GUI thread (#4022)."""
    db_file = tmp_path / "events.db"
    event_logger = EventLogger(str(db_file))
    try:
        for index in range(25):
            event_logger.log_event_async(
                event_type="alarm_trip",
                severity="CRITICAL",
                operator="Operator",
                description=f"event {index}",
            )
        writer = event_logger.async_writer
        assert writer.thread is not threading.current_thread()
        assert event_logger.flush_async(timeout=10.0) is True
        assert len(event_logger.fetch_logs()) == 25
    finally:
        event_logger.close()


def test_event_logger_async_validates_before_queueing(tmp_path: Path) -> None:
    event_logger = EventLogger(str(tmp_path / "events.db"))
    try:
        with pytest.raises(ValueError):
            event_logger.log_event_async(
                event_type="", severity="INFO", operator=None, description="x"
            )
    finally:
        event_logger.close()


def test_event_logger_purges_rows_older_than_the_retention_window(
    tmp_path: Path,
) -> None:
    event_logger = EventLogger(str(tmp_path / "events.db"))
    now = datetime.now()
    event_logger.log_event(
        "alarm_trip", "CRITICAL", None, "ancient", timestamp=now - timedelta(days=400)
    )
    event_logger.log_event("alarm_trip", "CRITICAL", None, "recent", timestamp=now)

    removed = event_logger.purge_older_than(90)

    assert removed == 1
    remaining = event_logger.fetch_logs()
    assert [row[5] for row in remaining] == ["recent"]


def test_event_logger_purge_rejects_negative_retention(tmp_path: Path) -> None:
    event_logger = EventLogger(str(tmp_path / "events.db"))
    with pytest.raises(ValueError):
        event_logger.purge_older_than(-1)
    with pytest.raises(TypeError):
        event_logger.purge_older_than("90")
