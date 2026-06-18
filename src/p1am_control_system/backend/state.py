from collections.abc import Callable
from typing import Any

from defaults import default_routing_config
from models import RoutingConfig


def default_tag_values() -> dict[str, float]:
    return {f"TAG_{i}": 0.0 for i in range(32)}


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
        self.active_config = config
        self.alarm_engine = self.alarm_engine_factory(config)
        self.active_alarms.clear()
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
        self.latest_tags[tag_name] = value

    def acknowledge_alarm(self, tag_id: str) -> bool:
        alarm = self.active_alarms.get(tag_id)
        if alarm is None:
            return False
        alarm["acknowledged"] = True
        if alarm.get("state") == "Normal":
            del self.active_alarms[tag_id]
        return True
