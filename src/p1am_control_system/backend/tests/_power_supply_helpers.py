"""Shared helpers for the split power_supply test files.

Kept under an underscore so pytest doesn't try to collect this as a test
module. Each helper returns a fresh PowerSupplyController in a known state
so individual tests can start from idle / armed / running without repeating
the boilerplate.
"""

from __future__ import annotations

import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from typing import Any

from power_supply import (  # noqa: E402  (path setup above must run first)
    PowerSupplyConfig,
    PowerSupplyController,
)


def test_config(**overrides: Any) -> PowerSupplyConfig:
    """A stable config for controller-logic tests.

    Pinned to the historical 100 A / 50 V / 50 A / 1000 W values so these tests
    exercise the *control law* (scaling, trips, clamps) independent of the
    production defaults in PowerSupplyConfig — which are tuned to a specific
    bench supply and may change. Tests that assert the production defaults
    construct PowerSupplyConfig() directly instead.
    """
    base: dict[str, Any] = {
        "current_full_scale_a": 100.0,
        "voltage_full_scale_v": 50.0,
        "current_setpoint_max_a": 50.0,
        "power_alarm_max_w": 1000.0,
    }
    base.update(overrides)
    return PowerSupplyConfig(**base)


def fresh_idle_controller() -> PowerSupplyController:
    """Return a controller in IDLE state (stable test config)."""
    return PowerSupplyController(test_config())


def fresh_armed_controller() -> PowerSupplyController:
    """Return a controller already in ARMED state (permissive on)."""
    c = fresh_idle_controller()
    c.set_permissive(True)
    return c


def fresh_running_controller(sp_a: float = 10.0) -> PowerSupplyController:
    """Return a controller in RUNNING state with a non-zero current setpoint."""
    c = fresh_armed_controller()
    c.set_current_setpoint(sp_a)
    return c


def fresh_armed_controller_unclamped() -> PowerSupplyController:
    """Return an ARMED controller with the output clamp opened to 100 %.

    The default config applies a 20 % safety clamp on the commanded output.
    Tests that exercise the proportional-scaling / slew / full-scale laws in
    isolation use this so the clamp doesn't cap the value under test.
    """
    c = PowerSupplyController(test_config(output_clamp_percent=100.0))
    c.set_permissive(True)
    return c
