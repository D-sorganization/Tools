"""Deterministic professional alarm lifecycle and performance contracts."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alarm_lifecycle import (  # noqa: E402
    AlarmDefinition,
    AlarmLifecycle,
    AlarmManager,
    AlarmPriority,
)
from identity import Principal, Role  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

NOW = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
OPERATOR = Principal("operator.1", "Operator One", Role.OPERATOR)


def _definition(tag: str = "TAG_0") -> AlarmDefinition:
    return AlarmDefinition(
        tag=tag,
        low_limit=10.0,
        high_limit=90.0,
        priority=AlarmPriority.HIGH,
        deadband=2.0,
        on_delay=timedelta(seconds=2),
        off_delay=timedelta(seconds=1),
        help_text="Check the synthetic source and upstream permissive.",
        suppression_rules=frozenset({"synthetic.maintenance"}),
    )


def test_alarm_activation_and_return_honor_delay_and_deadband() -> None:
    manager = AlarmManager([_definition()])

    assert manager.evaluate("TAG_0", 95.0, NOW).lifecycle is AlarmLifecycle.INACTIVE
    active = manager.evaluate("TAG_0", 95.0, NOW + timedelta(seconds=2))
    assert active.lifecycle is AlarmLifecycle.UNACKNOWLEDGED
    assert active.condition == "high"
    assert active.first_out_sequence == 1

    # Inside the high-side deadband: remain active and do not start return delay.
    assert (
        manager.evaluate("TAG_0", 89.0, NOW + timedelta(seconds=3)).condition == "high"
    )
    manager.evaluate("TAG_0", 87.0, NOW + timedelta(seconds=4))
    returned = manager.evaluate("TAG_0", 87.0, NOW + timedelta(seconds=5))
    assert returned.lifecycle is AlarmLifecycle.RETURNED_UNACKNOWLEDGED


def test_acknowledged_alarm_clears_only_after_return() -> None:
    manager = AlarmManager([_definition()])
    manager.evaluate("TAG_0", 95.0, NOW)
    manager.evaluate("TAG_0", 95.0, NOW + timedelta(seconds=2))

    acknowledged = manager.acknowledge("TAG_0", OPERATOR, NOW + timedelta(seconds=3))
    assert acknowledged.lifecycle is AlarmLifecycle.ACKNOWLEDGED
    manager.evaluate("TAG_0", 50.0, NOW + timedelta(seconds=4))
    assert (
        manager.evaluate("TAG_0", 50.0, NOW + timedelta(seconds=5)).lifecycle
        is AlarmLifecycle.INACTIVE
    )


def test_authorized_shelving_requires_reason_and_expires() -> None:
    manager = AlarmManager([_definition()])
    manager.evaluate("TAG_0", 95.0, NOW)
    manager.evaluate("TAG_0", 95.0, NOW + timedelta(seconds=2))

    shelved = manager.shelve(
        "TAG_0",
        OPERATOR,
        reason="Synthetic maintenance",
        until=NOW + timedelta(minutes=10),
        now=NOW + timedelta(seconds=3),
    )
    assert shelved.lifecycle is AlarmLifecycle.SHELVED
    assert (
        manager.snapshot("TAG_0", NOW + timedelta(minutes=11)).lifecycle
        is AlarmLifecycle.UNACKNOWLEDGED
    )
    with pytest.raises(ValueError, match="reason"):
        manager.shelve("TAG_0", OPERATOR, "", NOW + timedelta(minutes=1), NOW)


def test_only_designed_suppression_rules_can_hide_alarm() -> None:
    manager = AlarmManager([_definition()])
    with pytest.raises(ValueError, match="not designed"):
        manager.set_suppression("TAG_0", "ad-hoc", True, NOW)

    suppressed = manager.set_suppression("TAG_0", "synthetic.maintenance", True, NOW)
    assert suppressed.lifecycle is AlarmLifecycle.SUPPRESSED
    assert suppressed.suppression_rule == "synthetic.maintenance"


def test_first_out_order_help_and_performance_report_are_deterministic() -> None:
    manager = AlarmManager([_definition("TAG_0"), _definition("TAG_1")])
    for tag, offset in (("TAG_1", 0), ("TAG_0", 1)):
        manager.evaluate(tag, 95.0, NOW + timedelta(seconds=offset))
        manager.evaluate(tag, 95.0, NOW + timedelta(seconds=offset + 2))
    manager.acknowledge("TAG_1", OPERATOR, NOW + timedelta(seconds=5))

    snapshots = manager.active_snapshots(NOW + timedelta(seconds=5))
    assert [item.tag for item in snapshots] == ["TAG_1", "TAG_0"]
    assert snapshots[0].help_text.startswith("Check the synthetic")
    report = manager.performance_report()
    assert report.activations == 2
    assert report.acknowledged_activations == 1
    assert report.mean_acknowledgement_seconds == pytest.approx(3.0)
