import logging
import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import hardware
from alarm_processing import build_alarm_entry, state_name
from defaults import default_routing_config
from models import RoutingConfig

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

logger = logging.getLogger("dcs_backend.state")

# Credited operator when an acknowledgement arrives without an identity (the
# API key tiers are anonymous). Recorded verbatim in the audit field so a
# reader can tell "unattributed" from a named operator.
DEFAULT_ACK_USER = "operator"


def default_tag_values() -> dict[str, float]:
    return {f"TAG_{i}": 0.0 for i in range(32)}


def _finite_or_none(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not a reading.

    Absent, non-numeric and non-finite values all mean "no measurement". They
    are deliberately NOT coerced to a number: a NaN compares False against every
    threshold and a substituted 0.0 sits below every high limit, so either one
    silently resolves an active alarm to Normal instead of raising it.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


class SystemState:
    """Owns mutable backend control state and keeps PLC clients in sync."""

    def __init__(
        self,
        *,
        alarm_engine_factory: Callable[[RoutingConfig], Any],
        config: RoutingConfig | None = None,
    ) -> None:
        self.latest_tags = default_tag_values()
        self.active_config = config or default_routing_config()
        self.alarm_engine_factory = alarm_engine_factory
        self.alarm_engine = alarm_engine_factory(self.active_config)
        self.active_alarms: dict[str, dict[str, Any]] = {}
        self.e_stop_active = False
        self.tuning_sessions: dict[int, dict[str, Any]] = {}

    def attach_clients(self, *clients: Any) -> None:
        for client in clients:
            client.tuning_sessions = self.tuning_sessions
            client.active_config = self.active_config

    def apply_config(self, config: RoutingConfig, *clients: Any) -> None:
        """Adopt a new routing config, carrying alarm acknowledgements across.

        This runs on every operator deploy *and* on every reconnect-time
        ``_publish_active_config``. It rebuilds the alarm engine from the new
        interlocks, so the alarm state — including acknowledgement — has to be
        migrated through the engine's public API or the operator's ack silently
        disappears (issue #4034). Alarms for tags dropped from the new config
        are correctly forgotten.
        """
        carried = self._snapshot_engine_alarms()
        self.active_config = config
        self.alarm_engine = self.alarm_engine_factory(config)
        self.active_alarms.clear()
        self._restore_engine_alarms(carried)
        for client in clients:
            client.active_config = config

    def engage_estop(self) -> None:
        self.e_stop_active = True

    def clear_estop(self) -> None:
        self.e_stop_active = False

    def reset_tag_values(self) -> None:
        self.latest_tags.clear()
        self.latest_tags.update(default_tag_values())

    def write_tag(self, tag_name: str, value: float) -> None:
        """Record an operator force in the live tag map.

        Raises:
            TypeError: If ``value`` is not a number.
            hardware.NonFiniteValueError: If ``value`` is NaN/Inf. A NaN in
                ``latest_tags`` would be fed to the alarm engine on the next
                snapshot and read as a sensor fault (#3973/#3974).
        """
        self.latest_tags[tag_name] = hardware.require_finite_value(value, "value")

    def acknowledge_alarm(self, tag_id: str, user: str | None = None) -> bool:
        """Acknowledge ``tag_id`` on both the live map and the alarm engine.

        The engine is the durable record: ``apply_config`` rebuilds
        ``active_alarms`` from it, so an acknowledgement that never reached the
        engine would not survive a routing deploy or a PLC reconnect.

        Args:
            tag_id: Tag whose alarm is being acknowledged.
            user: Operator credited in the ``acknowledged_by`` audit field.
                ``None`` records :data:`DEFAULT_ACK_USER`.

        Returns:
            True if an alarm was acknowledged, False if ``tag_id`` has no live
            alarm.

        Raises:
            TypeError: If ``tag_id`` is not a str, or ``user`` is neither a str
                nor ``None``.
            ValueError: If ``user`` is blank.
        """
        if not isinstance(tag_id, str):
            raise TypeError(f"tag_id must be a str, got {type(tag_id).__name__}")
        if user is None:
            user = DEFAULT_ACK_USER
        if not isinstance(user, str):
            raise TypeError(f"user must be a str or None, got {type(user).__name__}")
        if not user.strip():
            raise ValueError("user must be a non-empty identifier")

        alarm = self.active_alarms.get(tag_id)
        if alarm is None:
            return False

        self._engine_acknowledge(tag_id, user)
        alarm["acknowledged"] = True
        alarm["acknowledged_by"] = user
        if alarm.get("state") == "Normal":
            del self.active_alarms[tag_id]
        return True

    # -- alarm-engine plumbing ----------------------------------------------
    def _engine_acknowledge(self, tag_id: str, user: str) -> bool:
        """Forward the ack to the engine; tolerate engines without the API.

        Both the Rust ``tools_core.scada.AlarmEngine`` and the pure-Python
        ``scada_fallback.AlarmEngine`` expose ``acknowledge_alarm(tag_id,
        user)``, so this is the one call that works against either. A tag that
        is already Normal (or unknown to the engine) is not an error — the live
        map still owns clearing it.
        """
        ack = getattr(self.alarm_engine, "acknowledge_alarm", None)
        if not callable(ack):
            logger.debug("Alarm engine exposes no acknowledge_alarm; local ack only.")
            return False
        try:
            return bool(ack(tag_id, user))
        except KeyError:
            logger.warning(
                "Acknowledged alarm for tag %r that the alarm engine does not "
                "track; the acknowledgement will not survive a redeploy.",
                tag_id,
            )
            return False

    def _snapshot_engine_alarms(self) -> list[dict[str, Any]]:
        """Read the engine's active alarms before it is replaced.

        Each record is enriched with the live map's ``timestamp`` so an alarm
        keeps its original raise time across a redeploy.
        """
        get_active = getattr(self.alarm_engine, "get_active_alarms", None)
        if not callable(get_active):
            return []
        try:
            records = list(get_active())
        except Exception as exc:  # noqa: BLE001 - a deploy must never fail here
            logger.warning("Could not snapshot alarm engine state: %s", exc)
            return []

        for record in records:
            local = self.active_alarms.get(str(record.get("tag_id", "")))
            if local is not None and local.get("timestamp"):
                record["timestamp"] = local["timestamp"]
        return records

    def _restore_engine_alarms(self, snapshot: list[dict[str, Any]]) -> None:
        """Replay a snapshot into the freshly built engine and live map.

        Uses only the engine's public API (``update_tag``,
        ``acknowledge_alarm``, ``get_alarm_state``) so it behaves identically
        against the Rust ``tools_core.scada`` engine and the pure-Python
        fallback. The rebuilt engine — not the snapshot — is the authority on
        the resulting state, because the new config may carry different limits.
        Tags the new config no longer defines raise ``KeyError`` from
        ``update_tag`` and are dropped.

        A snapshot record carrying no usable ``value`` is replayed *without*
        touching the engine, and keeps the state it was snapshotted in. Feeding
        the engine a substituted ``0.0`` would resolve almost any alarm to
        Normal, and this method runs on every routing deploy AND every
        reconnect-time ``_publish_active_config`` — after ``active_alarms`` has
        already been cleared. A live HiHi would therefore be erased from both
        the engine and the live map by a fabricated reading, which is the exact
        failure the poll loop's "only a real measurement may move the alarm
        state machine" rule exists to prevent. ``get_active_alarms()`` is only
        contracted to agree between the Rust and Python engines on the keys this
        method reads, not on carrying ``value`` at all, so the absence of a
        value is an expected shape, not an error.
        """
        if not snapshot:
            return
        update = getattr(self.alarm_engine, "update_tag", None)
        read_state = getattr(self.alarm_engine, "get_alarm_state", None)
        if not callable(update) or not callable(read_state):
            return

        stamp = datetime.now(UTC).isoformat()
        for entry in snapshot:
            tag_id = str(entry.get("tag_id", ""))
            if not tag_id:
                continue

            snapshot_state = state_name(entry.get("state", "Normal"))
            replay_value = _finite_or_none(entry.get("value"))
            try:
                if replay_value is None:
                    # No measurement to replay: retain what we snapshotted
                    # rather than inventing a reading that clears the alarm.
                    if snapshot_state == "Normal":
                        continue
                    logger.warning(
                        "Alarm snapshot for %r carries no usable value; "
                        "retaining state %s across the rebuild without "
                        "re-evaluating the engine.",
                        tag_id,
                        snapshot_state,
                    )
                    self.active_alarms[tag_id] = build_alarm_entry(
                        tag_id,
                        snapshot_state,
                        timestamp=str(entry.get("timestamp") or stamp),
                        acknowledged=bool(entry.get("acknowledged", False)),
                        acknowledged_by=entry.get("acknowledged_by"),
                    )
                    continue

                update(tag_id, replay_value)
                if entry.get("acknowledged"):
                    self._engine_acknowledge(
                        tag_id, str(entry.get("acknowledged_by") or DEFAULT_ACK_USER)
                    )
                current = read_state(tag_id)
            except (KeyError, TypeError, ValueError):
                # Tag removed from the new config, or an unusable record.
                continue

            state = state_name(current.get("state", "Normal"))
            if state == "Normal":
                continue
            self.active_alarms[tag_id] = build_alarm_entry(
                tag_id,
                state,
                timestamp=str(entry.get("timestamp") or stamp),
                acknowledged=bool(current.get("acknowledged", False)),
                acknowledged_by=current.get("acknowledged_by"),
            )
