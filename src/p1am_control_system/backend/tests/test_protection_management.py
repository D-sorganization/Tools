"""F07 contracts for first-out capture and managed synthetic bypasses."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from identity import Principal, Role
from protection_management import (
    BypassRequest,
    ProtectionCategory,
    ProtectionDefinition,
    ProtectionService,
)

UTC = UTC


def principal(role: Role) -> Principal:
    return Principal(subject=f"user.{role}", display_name=str(role).title(), role=role)


def service(now: datetime) -> ProtectionService:
    return ProtectionService(
        definitions=(
            ProtectionDefinition(
                protection_id="SYNTHETIC.REACTOR.HIGH_PRESSURE",
                category=ProtectionCategory.INTERLOCK,
                consequences=("SYNTHETIC.FEED stops", "SYNTHETIC.VENT opens"),
                bypassable=True,
            ),
            ProtectionDefinition(
                protection_id="SYNTHETIC.REACTOR.INDEPENDENT_TRIP",
                category=ProtectionCategory.INDEPENDENT_PROTECTION,
                consequences=("Synthetic heater power removed",),
                bypassable=False,
            ),
        ),
        now=lambda: now,
    )


def test_trip_group_preserves_first_out_and_consequences() -> None:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    protections = service(now)

    first = protections.trip("SYNTHETIC.REACTOR.HIGH_PRESSURE", group_id="trip-1")
    second = protections.trip("SYNTHETIC.REACTOR.INDEPENDENT_TRIP", group_id="trip-1")

    assert first.first_out is True
    assert second.first_out is False
    assert first.consequences == ("SYNTHETIC.FEED stops", "SYNTHETIC.VENT opens")
    assert second.category is ProtectionCategory.INDEPENDENT_PROTECTION


def test_managed_bypass_requires_role_reason_expiry_and_banner() -> None:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    protections = service(now)
    request = BypassRequest(
        protection_id="SYNTHETIC.REACTOR.HIGH_PRESSURE",
        reason="Synthetic FAT verification",
        expires_at=now + timedelta(hours=2),
    )

    with pytest.raises(PermissionError):
        protections.request_bypass(request, principal(Role.OPERATOR))

    bypass = protections.request_bypass(request, principal(Role.ENGINEER))

    assert bypass.active is True
    assert bypass.banner_required is True
    assert bypass.actor == "user.engineer"
    assert bypass.reason == "Synthetic FAT verification"
    assert protections.active_bypasses() == [bypass]


def test_non_bypassable_policy_and_expiry_are_fail_closed() -> None:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    protections = service(now)

    with pytest.raises(ValueError, match="non-bypassable"):
        protections.request_bypass(
            BypassRequest(
                protection_id="SYNTHETIC.REACTOR.INDEPENDENT_TRIP",
                reason="Not permitted",
                expires_at=now + timedelta(minutes=5),
            ),
            principal(Role.ADMIN),
        )

    with pytest.raises(ValueError, match="future"):
        protections.request_bypass(
            BypassRequest(
                protection_id="SYNTHETIC.REACTOR.HIGH_PRESSURE",
                reason="Expired request",
                expires_at=now,
            ),
            principal(Role.ENGINEER),
        )


def test_expired_bypass_is_not_reported_active() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    protections = ProtectionService(
        definitions=(
            ProtectionDefinition(
                protection_id="SYNTHETIC.PUMP.LOW_FLOW",
                category=ProtectionCategory.INTERLOCK,
                consequences=("Synthetic pump stops",),
                bypassable=True,
            ),
        ),
        now=lambda: clock[0],
    )
    protections.request_bypass(
        BypassRequest(
            protection_id="SYNTHETIC.PUMP.LOW_FLOW",
            reason="Timed synthetic test",
            expires_at=clock[0] + timedelta(minutes=1),
        ),
        principal(Role.ENGINEER),
    )

    clock[0] += timedelta(minutes=2)
    assert protections.active_bypasses() == []
