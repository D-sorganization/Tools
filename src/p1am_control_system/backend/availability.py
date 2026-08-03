"""Single command authority, ordered buffering, recovery, and HMI-loss policy."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _synthetic(value: str) -> str:
    if not value.startswith("SYNTHETIC."):
        raise ValueError("identifiers must begin with SYNTHETIC.")
    return value


class AvailabilityPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    recovery_time_objective: timedelta
    recovery_point_objective: timedelta
    max_clock_skew: timedelta
    buffer_capacity: int = Field(gt=0)

    @model_validator(mode="after")
    def _positive_contracts(self) -> AvailabilityPolicy:
        if any(
            value <= timedelta(0)
            for value in (
                self.recovery_time_objective,
                self.recovery_point_objective,
                self.max_clock_skew,
            )
        ):
            raise ValueError("recovery and clock contracts must be positive")
        return self


class AuthorityLease(BaseModel):
    model_config = ConfigDict(frozen=True)

    lease_id: str
    holder: str

    _holder_is_synthetic = field_validator("holder")(_synthetic)


class BufferedSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int = Field(gt=0)
    timestamp: datetime
    value: float

    @field_validator("timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sample timestamp must include a UTC offset")
        return value


class AvailabilityCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: str
    energizing: bool
    accepted: bool
    fail_closed: bool
    reason: str


class AvailabilityHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    recovery_time_objective_seconds: float
    recovery_point_objective_seconds: float
    clock_ordering_reliable: bool
    command_authority: str | None
    transport_available: bool
    hmi_available: bool
    buffered_samples: int
    data_classification: Literal["synthetic"] = "synthetic"


class AvailabilityService:
    def __init__(self, policy: AvailabilityPolicy) -> None:
        self._policy = policy
        self._authority: AuthorityLease | None = None
        self._transport_available = True
        self._hmi_available = True
        self._clock_skew = timedelta(0)
        self._buffer: list[BufferedSample] = []
        self._last_sequence = 0
        self._last_timestamp: datetime | None = None

    @property
    def authority(self) -> AuthorityLease | None:
        return self._authority

    def acquire_authority(self, holder: str) -> AuthorityLease:
        if self._authority is not None:
            raise PermissionError(
                f"command authority is already held by {self._authority.holder}"
            )
        lease = AuthorityLease(lease_id=uuid.uuid4().hex, holder=holder)
        self._authority = lease
        return lease

    def release_authority(self, lease_id: str) -> None:
        if self._authority is None or self._authority.lease_id != lease_id:
            raise PermissionError("only the active lease may release authority")
        self._authority = None

    def set_transport_available(self, available: bool) -> None:
        self._transport_available = available

    def ingest(self, sample: BufferedSample) -> None:
        if sample.sequence <= self._last_sequence:
            raise ValueError("sample sequences must strictly increase")
        if (
            self._last_timestamp is not None
            and sample.timestamp <= self._last_timestamp
        ):
            raise ValueError("sample timestamps must strictly increase")
        if (
            not self._transport_available
            and len(self._buffer) >= self._policy.buffer_capacity
        ):
            raise OverflowError("offline buffer capacity exceeded")
        self._last_sequence = sample.sequence
        self._last_timestamp = sample.timestamp
        if not self._transport_available:
            self._buffer.append(sample)

    def reconcile(self) -> list[BufferedSample]:
        if not self._transport_available:
            raise RuntimeError("transport must recover before reconciliation")
        reconciled = list(self._buffer)
        self._buffer.clear()
        return reconciled

    def inject_fault(self, fault: Literal["hmi_unavailable", "authority_loss"]) -> None:
        if fault == "hmi_unavailable":
            self._hmi_available = False
        elif fault == "authority_loss":
            self._authority = None

    def report_clock_skew(self, skew: timedelta) -> None:
        self._clock_skew = abs(skew)

    def command(self, target: str, *, energizing: bool) -> AvailabilityCommandResult:
        target = _synthetic(target)
        if self._authority is None:
            return AvailabilityCommandResult(
                target=target,
                energizing=energizing,
                accepted=False,
                fail_closed=True,
                reason="No command authority",
            )
        if energizing and not self._hmi_available:
            return AvailabilityCommandResult(
                target=target,
                energizing=True,
                accepted=False,
                fail_closed=True,
                reason="Energizing commands are blocked while the HMI is unavailable",
            )
        return AvailabilityCommandResult(
            target=target,
            energizing=energizing,
            accepted=True,
            fail_closed=False,
            reason="Accepted by the single synthetic command authority",
        )

    def health(self) -> AvailabilityHealth:
        return AvailabilityHealth(
            recovery_time_objective_seconds=self._policy.recovery_time_objective.total_seconds(),
            recovery_point_objective_seconds=self._policy.recovery_point_objective.total_seconds(),
            clock_ordering_reliable=self._clock_skew <= self._policy.max_clock_skew,
            command_authority=self._authority.holder if self._authority else None,
            transport_available=self._transport_available,
            hmi_available=self._hmi_available,
            buffered_samples=len(self._buffer),
        )
