"""Application-service contracts for generic routing-to-alarm adaptation."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alarm_service import AlarmService, manager_from_routing  # noqa: E402
from models import InterlockConfig, RoutingConfig  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _routing() -> RoutingConfig:
    return RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=[],
        pids=[],
        interlocks={
            "TAG_0": InterlockConfig(
                lolo_limit=0,
                low_limit=10,
                high_limit=90,
                hihi_limit=100,
            )
        },
    )


def test_routing_adapter_builds_generic_nonconfidential_alarm_definition() -> None:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    service = AlarmService(manager_from_routing(_routing()), clock=lambda: now)

    service.observe({"TAG_0": 95}, now)
    assert service.active() == []  # representative one-second on-delay
    service.observe({"TAG_0": 95}, now + timedelta(seconds=1))

    active = service.active()[0]
    assert active.tag == "TAG_0"
    assert active.help_text == "Review signal quality and the generic process context."


def test_routing_adapter_rejects_invalid_limit_order() -> None:
    invalid = _routing()
    invalid.interlocks["TAG_0"].high_limit = 5

    with pytest.raises(ValueError, match="ordered"):
        manager_from_routing(invalid)
