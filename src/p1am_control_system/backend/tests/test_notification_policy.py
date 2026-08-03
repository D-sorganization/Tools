"""F14 deterministic notification delay, suppression, and escalation contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from notification_policy import AlarmNotice, NotificationPolicy, NotificationService


class RecordingChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, recipient: str, message: str) -> None:
        self.sent.append((recipient, message))


def _notice(
    alarm_id: str, now: datetime, message: str = "Synthetic alarm"
) -> AlarmNotice:
    return AlarmNotice(
        alarm_id=alarm_id,
        priority="high",
        occurred_at=now,
        message=message,
    )


def test_delay_then_escalation_and_delivery_audit_are_deterministic() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    channel = RecordingChannel()
    service = NotificationService(
        NotificationPolicy(
            initial_delay=timedelta(minutes=1),
            escalation_delay=timedelta(minutes=3),
            primary_recipient="synthetic.on-call.primary",
            escalation_recipient="synthetic.on-call.escalation",
        ),
        channel,
        now=lambda: clock[0],
    )
    service.raise_alarm(_notice("SYNTHETIC.ALARM.HIGH", clock[0]))

    assert service.tick() == []
    clock[0] += timedelta(minutes=1)
    primary = service.tick()
    clock[0] += timedelta(minutes=2)
    escalated = service.tick()

    assert primary[0].recipient == "synthetic.on-call.primary"
    assert escalated[0].recipient == "synthetic.on-call.escalation"
    assert channel.sent == [
        ("synthetic.on-call.primary", "SYNTHETIC.ALARM.HIGH: Synthetic alarm"),
        ("synthetic.on-call.escalation", "SYNTHETIC.ALARM.HIGH: Synthetic alarm"),
    ]
    assert [audit.outcome for audit in service.audit()] == ["delivered", "delivered"]


def test_suppression_acknowledgment_cancellation_and_redaction() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    channel = RecordingChannel()
    service = NotificationService(
        NotificationPolicy(
            initial_delay=timedelta(seconds=10),
            escalation_delay=timedelta(minutes=1),
            primary_recipient="synthetic.primary",
            escalation_recipient="synthetic.escalation",
            suppressed_alarm_ids=frozenset({"SYNTHETIC.ALARM.SUPPRESSED"}),
        ),
        channel,
        now=lambda: clock[0],
    )
    service.raise_alarm(_notice("SYNTHETIC.ALARM.SUPPRESSED", clock[0]))
    service.raise_alarm(
        _notice(
            "SYNTHETIC.ALARM.ACKED",
            clock[0],
            "Synthetic alarm password=do-not-expose",
        )
    )
    service.acknowledge("SYNTHETIC.ALARM.ACKED", "operator.one")
    clock[0] += timedelta(minutes=2)

    assert service.tick() == []
    assert channel.sent == []
    assert {audit.outcome for audit in service.audit()} == {"suppressed", "cancelled"}
    assert all("do-not-expose" not in audit.message for audit in service.audit())


def test_rate_limit_blocks_burst_and_records_attempt() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    channel = RecordingChannel()
    service = NotificationService(
        NotificationPolicy(
            initial_delay=timedelta(0),
            escalation_delay=timedelta(hours=1),
            primary_recipient="synthetic.primary",
            escalation_recipient="synthetic.escalation",
            max_deliveries=1,
            rate_limit_window=timedelta(minutes=5),
        ),
        channel,
        now=lambda: clock[0],
    )
    service.raise_alarm(_notice("SYNTHETIC.ALARM.ONE", clock[0]))
    service.raise_alarm(_notice("SYNTHETIC.ALARM.TWO", clock[0]))

    service.tick()

    assert len(channel.sent) == 1
    assert [audit.outcome for audit in service.audit()] == ["delivered", "rate_limited"]
