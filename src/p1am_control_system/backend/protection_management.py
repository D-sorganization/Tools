"""Synthetic first-out, consequence, and managed-bypass domain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from identity import Principal, Role
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _required_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


class ProtectionCategory(StrEnum):
    CONTROL = "control"
    INTERLOCK = "interlock"
    INDEPENDENT_PROTECTION = "independent_protection"


class ProtectionDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    protection_id: str
    category: ProtectionCategory
    consequences: tuple[str, ...] = Field(min_length=1)
    bypassable: bool

    @field_validator("protection_id")
    @classmethod
    def _synthetic_only(cls, value: str) -> str:
        normalized = _required_text(value, "protection_id")
        if not normalized.startswith("SYNTHETIC."):
            raise ValueError("protection_id must begin with SYNTHETIC.")
        return normalized


class BypassRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    protection_id: str
    reason: str = Field(min_length=8, max_length=500)
    expires_at: datetime

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str) -> str:
        return _required_text(value, "reason")


class TripRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    protection_id: str
    group_id: str
    category: ProtectionCategory
    consequences: tuple[str, ...]
    occurred_at: datetime
    first_out: bool


class ManagedBypass(BaseModel):
    model_config = ConfigDict(frozen=True)

    protection_id: str
    actor: str
    reason: str
    requested_at: datetime
    expires_at: datetime
    banner_required: Literal[True] = True
    active: Literal[True] = True


class ProtectionService:
    """Thread-local domain service; API audit middleware supplies durable audit."""

    def __init__(
        self,
        definitions: Sequence[ProtectionDefinition],
        now: Callable[[], datetime],
    ) -> None:
        indexed = {definition.protection_id: definition for definition in definitions}
        if len(indexed) != len(definitions):
            raise ValueError("protection identifiers must be unique")
        self._definitions = indexed
        self._now = now
        self._trips: list[TripRecord] = []
        self._bypasses: list[ManagedBypass] = []

    def _definition(self, protection_id: str) -> ProtectionDefinition:
        try:
            return self._definitions[protection_id]
        except KeyError as exc:
            raise KeyError(f"unknown protection: {protection_id}") from exc

    def trip(self, protection_id: str, *, group_id: str) -> TripRecord:
        definition = self._definition(protection_id)
        normalized_group = _required_text(group_id, "group_id")
        first_out = not any(
            record.group_id == normalized_group for record in self._trips
        )
        record = TripRecord(
            protection_id=definition.protection_id,
            group_id=normalized_group,
            category=definition.category,
            consequences=definition.consequences,
            occurred_at=self._now(),
            first_out=first_out,
        )
        self._trips.append(record)
        return record

    def request_bypass(
        self, request: BypassRequest, principal: Principal
    ) -> ManagedBypass:
        if principal.role not in {Role.ENGINEER, Role.ADMIN}:
            raise PermissionError("engineer or admin role required")
        definition = self._definition(request.protection_id)
        if not definition.bypassable:
            raise ValueError("protection policy is non-bypassable")
        requested_at = self._now()
        if request.expires_at <= requested_at:
            raise ValueError("bypass expiry must be in the future")
        if request.expires_at - requested_at > timedelta(hours=24):
            raise ValueError("bypass duration cannot exceed 24 hours")
        bypass = ManagedBypass(
            protection_id=request.protection_id,
            actor=principal.subject,
            reason=request.reason,
            requested_at=requested_at,
            expires_at=request.expires_at,
        )
        self._bypasses.append(bypass)
        return bypass

    def active_bypasses(self) -> list[ManagedBypass]:
        now = self._now()
        return [bypass for bypass in self._bypasses if bypass.expires_at > now]

    def definitions(self) -> list[ProtectionDefinition]:
        return list(self._definitions.values())

    def trips(self) -> list[TripRecord]:
        return list(self._trips)


def representative_protections() -> tuple[ProtectionDefinition, ...]:
    """Non-confidential protection examples for the synthetic process."""
    return (
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
    )
