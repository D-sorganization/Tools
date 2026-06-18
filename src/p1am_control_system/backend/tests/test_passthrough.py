"""Tests for the power-supply PID pass-through helpers + tag_map declaration.

Covers the pure config-shaping logic behind the connect-time auto-repair
(#3550) and the now-declared ``tag_map`` attribute on every PLC client (#3540).
These import only ``defaults``/``simulator_client``/``modbus_client`` — no
``tools_core``/``fastapi`` — so they gate in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pymodbus")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from defaults import (  # noqa: E402
    default_routing_config,
    ensure_pid_passthrough,
    is_pid_passthrough,
)
from modbus_client import AsyncModbusManager  # noqa: E402
from models import PIDConfig  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402

COMMAND_TAG = "TAG_10"


def _passthrough_config():
    config = default_routing_config()
    config.pids[0] = PIDConfig(
        pv_tag="TAG_1",
        cv_tag=COMMAND_TAG,
        setpoint=12.0,
        kp=1.0,
        ki=0.0,
        kd=0.0,
    )
    return config


def _broken_config():
    config = default_routing_config()
    # Simulate the post-NVRAM-reset unmapped state (cv=TAG_255, kp=0).
    config.pids[0] = PIDConfig(
        pv_tag="TAG_1",
        cv_tag="TAG_255",
        setpoint=7.5,
        kp=0.0,
        ki=0.0,
        kd=0.0,
    )
    return config


class TestIsPidPassthrough:
    def test_true_for_passthrough(self) -> None:
        assert is_pid_passthrough(_passthrough_config(), 0, COMMAND_TAG) is True

    def test_false_for_unmapped_loop(self) -> None:
        assert is_pid_passthrough(_broken_config(), 0, COMMAND_TAG) is False

    def test_false_for_wrong_command_tag(self) -> None:
        assert is_pid_passthrough(_passthrough_config(), 0, "TAG_11") is False

    def test_false_for_out_of_range_index(self) -> None:
        assert is_pid_passthrough(_passthrough_config(), 99, COMMAND_TAG) is False


class TestEnsurePidPassthrough:
    def test_noop_when_already_passthrough(self) -> None:
        config = _passthrough_config()
        result, repaired = ensure_pid_passthrough(config, 0, COMMAND_TAG)
        assert repaired is False
        assert result is config  # unchanged, same object

    def test_repairs_unmapped_loop_preserving_setpoint(self) -> None:
        config = _broken_config()
        original_setpoint = config.pids[0].setpoint
        result, repaired = ensure_pid_passthrough(config, 0, COMMAND_TAG)
        assert repaired is True
        assert is_pid_passthrough(result, 0, COMMAND_TAG) is True
        assert result.pids[0].cv_tag == COMMAND_TAG
        assert result.pids[0].kp == 1.0
        assert result.pids[0].setpoint == pytest.approx(original_setpoint)
        # Original config is not mutated (deep copy).
        assert config.pids[0].cv_tag == "TAG_255"

    def test_raises_on_out_of_range_index(self) -> None:
        with pytest.raises(ValueError):
            ensure_pid_passthrough(_passthrough_config(), 99, COMMAND_TAG)


class TestTagMapDeclared:
    def test_simulator_has_tag_map_attribute(self) -> None:
        sim = SimulatedPLCClient()
        # Declared on the base class — a real, empty dict, not a missing attr.
        assert sim.tag_map == {}

    def test_modbus_client_has_tag_map_attribute(self) -> None:
        client = AsyncModbusManager(host="127.0.0.1")
        assert client.tag_map == {}
