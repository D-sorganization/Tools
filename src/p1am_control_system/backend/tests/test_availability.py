"""F15 command authority, ordered buffering, and safe fault behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from availability import AvailabilityPolicy, AvailabilityService, BufferedSample


def _service() -> AvailabilityService:
    return AvailabilityService(
        AvailabilityPolicy(
            recovery_time_objective=timedelta(minutes=5),
            recovery_point_objective=timedelta(seconds=30),
            max_clock_skew=timedelta(seconds=2),
            buffer_capacity=10,
        )
    )


def test_exactly_one_command_authority_is_enforced() -> None:
    service = _service()

    lease = service.acquire_authority("SYNTHETIC.CONTROLLER.PRIMARY")

    with pytest.raises(PermissionError, match="already held"):
        service.acquire_authority("SYNTHETIC.CONTROLLER.SECONDARY")
    assert service.authority == lease


def test_offline_buffer_reconciles_ordered_unique_samples() -> None:
    service = _service()
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    service.set_transport_available(False)
    service.ingest(BufferedSample(sequence=1, timestamp=start, value=10))
    service.ingest(
        BufferedSample(sequence=2, timestamp=start + timedelta(seconds=1), value=11)
    )

    with pytest.raises(ValueError, match="strictly increase"):
        service.ingest(BufferedSample(sequence=3, timestamp=start, value=12))

    service.set_transport_available(True)
    reconciled = service.reconcile()

    assert [sample.sequence for sample in reconciled] == [1, 2]
    assert service.reconcile() == []


def test_hmi_loss_rejects_energizing_but_allows_deenergizing_command() -> None:
    service = _service()
    service.acquire_authority("SYNTHETIC.CONTROLLER.PRIMARY")
    service.inject_fault("hmi_unavailable")

    energize = service.command("SYNTHETIC.HEATER.ENABLE", energizing=True)
    deenergize = service.command("SYNTHETIC.HEATER.ENABLE", energizing=False)

    assert energize.accepted is False
    assert energize.fail_closed is True
    assert deenergize.accepted is True


def test_health_report_exposes_recovery_and_clock_contracts() -> None:
    service = _service()
    service.report_clock_skew(timedelta(seconds=3))

    health = service.health()

    assert health.recovery_time_objective_seconds == 300
    assert health.recovery_point_objective_seconds == 30
    assert health.clock_ordering_reliable is False
    assert health.command_authority is None
