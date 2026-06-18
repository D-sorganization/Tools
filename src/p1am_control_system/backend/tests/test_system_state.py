from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from defaults import default_routing_config
from mpc import simulate_pid_vs_mpc
from state import SystemState


def _alarm_engine(config: Any) -> dict[str, Any]:
    return {"interlocks": dict(config.interlocks)}


def test_system_state_applies_config_and_syncs_clients() -> None:
    state = SystemState(alarm_engine_factory=_alarm_engine)
    client = SimpleNamespace(active_config=None, tuning_sessions=None)
    backup = SimpleNamespace(active_config=None, tuning_sessions=None)
    state.active_alarms["TAG_1"] = {"state": "High"}

    state.attach_clients(client, backup)
    config = default_routing_config()
    config.pids[0].setpoint = 42.0
    state.apply_config(config, client, backup)

    assert state.active_config is config
    assert client.active_config is config
    assert backup.active_config is config
    assert state.active_alarms == {}
    assert client.tuning_sessions is state.tuning_sessions
    assert backup.tuning_sessions is state.tuning_sessions


def test_system_state_estop_and_tag_writes_mutate_owned_state() -> None:
    state = SystemState(alarm_engine_factory=_alarm_engine)

    state.write_tag("TAG_7", 12.5)
    state.engage_estop()
    state.reset_tag_values()

    assert state.e_stop_active is True
    assert state.latest_tags["TAG_7"] == 0.0
    assert len(state.latest_tags) == 32

    state.clear_estop()

    assert state.e_stop_active is False


def test_system_state_acknowledges_and_clears_normal_alarm() -> None:
    state = SystemState(alarm_engine_factory=_alarm_engine)
    state.active_alarms["TAG_2"] = {"state": "Normal", "acknowledged": False}

    assert state.acknowledge_alarm("TAG_2") is True
    assert "TAG_2" not in state.active_alarms
    assert state.acknowledge_alarm("missing") is False


def test_mpc_simulation_helper_preserves_response_shape() -> None:
    payload = SimpleNamespace(
        prediction_horizon=10,
        control_horizon=3,
        setpoint=50.0,
        rho=0.1,
        process_gain=1.2,
        process_tau=5.0,
        process_delay=1.0,
    )

    result = simulate_pid_vs_mpc(payload)

    assert result["status"] == "success"
    assert len(result["time"]) == 50
    assert set(result["pid"]) == {"pv", "cv"}
    assert set(result["mpc"]) == {"pv", "cv"}
    assert len(result["pid"]["pv"]) == len(result["mpc"]["pv"]) == 50
