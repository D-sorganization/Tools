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

from power_supply import (  # noqa: E402  (path setup above must run first)
    PowerSupplyConfig,
    PowerSupplyController,
)


def fresh_idle_controller() -> PowerSupplyController:
    """Return a controller in IDLE state (default config)."""
    return PowerSupplyController(PowerSupplyConfig())


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
    c = PowerSupplyController(PowerSupplyConfig(output_clamp_percent=100.0))
    c.set_permissive(True)
    return c
