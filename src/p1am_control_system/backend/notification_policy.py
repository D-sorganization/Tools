"""Deterministic alarm notification, escalation, rate-limit, and audit policy."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SECRET = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|credential)\s*[:=]\s*\S+"
)


def _redact(message: str) -> str:
    return _SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", message)


class AlarmNotice(BaseModel):
    model_config = ConfigDict(frozen=True)

    alarm_id: str
    priority: Literal["high", "critical"]
    occurred_at: datetime
    message: str = Field(min_length=1, max_length=1000)

    @field_validator("alarm_id")
    @classmethod
    def _synthetic_alarm(cls, value: str) -> str:
        if not value.startswith("SYNTHETIC."):
            raise ValueError("alarm_id must begin with SYNTHETIC.")
        return value


class NotificationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_delay: timedelta
    escalation_delay: timedelta
    primary_recipient: str = Field(min_length=1)
    escalation_recipient: str = Field(min_length=1)
    suppressed_alarm_ids: frozenset[str] = frozenset()
    max_deliveries: int = Field(default=20, gt=0)
    rate_limit_window: timedelta = timedelta(minutes=5)

    @model_validator(mode="after")
    def _valid_delays(self) -> NotificationPolicy:
        if self.initial_delay < timedelta(0):
            raise ValueError("initial_delay must be nonnegative")
        if self.escalation_delay < self.initial_delay:
            raise ValueError("escalation_delay cannot precede initial_delay")
        if self.rate_limit_window <= timedelta(0):
            raise ValueError("rate_limit_window must be positive")
        return self


class NotificationChannel(Protocol):
    def send(self, recipient: str, message: str) -> None: ...


class NotificationAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    alarm_id: str
    recipient: str | None
    stage: Literal["primary", "escalation", "policy", "acknowledgment"]
    outcome: Literal["delivered", "suppressed", "cancelled", "rate_limited"]
    occurred_at: datetime
    message: str
    actor: str | None = None


class NotificationService:
    def __init__(
        self,
        policy: NotificationPolicy,
        channel: NotificationChannel,
        now: Callable[[], datetime],
    ) -> None:
        self._policy = policy
        self._channel = channel
        self._now = now
        self._active: dict[str, AlarmNotice] = {}
        self._completed_stages: set[tuple[str, str]] = set()
        self._delivery_times: list[datetime] = []
        self._audit: list[NotificationAudit] = []

    @property
    def policy(self) -> NotificationPolicy:
        return self._policy

    @staticmethod
    def _message(notice: AlarmNotice) -> str:
        return _redact(f"{notice.alarm_id}: {notice.message}")

    def raise_alarm(self, notice: AlarmNotice) -> None:
        if notice.alarm_id in self._active:
            raise ValueError("alarm is already active")
        if notice.alarm_id in self._policy.suppressed_alarm_ids:
            self._audit.append(
                NotificationAudit(
                    alarm_id=notice.alarm_id,
                    recipient=None,
                    stage="policy",
                    outcome="suppressed",
                    occurred_at=self._now(),
                    message=self._message(notice),
                )
            )
            return
        self._active[notice.alarm_id] = notice

    def acknowledge(self, alarm_id: str, actor: str) -> None:
        try:
            notice = self._active.pop(alarm_id)
        except KeyError as exc:
            raise KeyError(f"unknown active alarm: {alarm_id}") from exc
        self._audit.append(
            NotificationAudit(
                alarm_id=alarm_id,
                recipient=None,
                stage="acknowledgment",
                outcome="cancelled",
                occurred_at=self._now(),
                message=self._message(notice),
                actor=actor,
            )
        )

    def _rate_limited(self, now: datetime) -> bool:
        cutoff = now - self._policy.rate_limit_window
        self._delivery_times = [
            value for value in self._delivery_times if value > cutoff
        ]
        return len(self._delivery_times) >= self._policy.max_deliveries

    def tick(self) -> list[NotificationAudit]:
        now = self._now()
        delivered: list[NotificationAudit] = []
        stages: tuple[tuple[Literal["primary", "escalation"], timedelta, str], ...] = (
            ("primary", self._policy.initial_delay, self._policy.primary_recipient),
            (
                "escalation",
                self._policy.escalation_delay,
                self._policy.escalation_recipient,
            ),
        )
        for notice in self._active.values():
            for stage, delay, recipient in stages:
                key = (notice.alarm_id, stage)
                if key in self._completed_stages or now - notice.occurred_at < delay:
                    continue
                outcome: Literal["delivered", "rate_limited"]
                message = self._message(notice)
                if self._rate_limited(now):
                    outcome = "rate_limited"
                else:
                    self._channel.send(recipient, message)
                    self._delivery_times.append(now)
                    outcome = "delivered"
                audit = NotificationAudit(
                    alarm_id=notice.alarm_id,
                    recipient=recipient,
                    stage=stage,
                    outcome=outcome,
                    occurred_at=now,
                    message=message,
                )
                self._audit.append(audit)
                self._completed_stages.add(key)
                if outcome == "delivered":
                    delivered.append(audit)
        return delivered

    def audit(self) -> list[NotificationAudit]:
        return list(self._audit)
