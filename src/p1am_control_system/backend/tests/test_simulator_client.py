"""Smoke tests for ``SimulatedPLCClient`` step + command seams (issue #3537).

The simulator is the data source for demos and most backend test runs, so its
FOPDT/PID step and the public command seams (``write_tag``,
``write_pid_setpoint``) need at least smoke coverage.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator_client import SimulatedPLCClient  # noqa: E402


def test_read_tags_returns_full_tag_table() -> None:
    sim = SimulatedPLCClient()
    tags = asyncio.run(sim.read_tags())
    assert tags is not None
    assert len(tags) >= 32
    # Safety/status tags the step always sets.
    assert tags["TAG_0"] == pytest.approx(1.0)  # normal safety state
    assert all(isinstance(v, float) for v in tags.values())


def test_read_tags_none_when_disconnected() -> None:
    sim = SimulatedPLCClient()
    asyncio.run(sim.disconnect())
    assert asyncio.run(sim.read_tags()) is None


def test_step_drives_pv_toward_setpoint() -> None:
    sim = SimulatedPLCClient()
    pid = sim.active_config.pids[0]
    pv_tag = pid.pv_tag
    # Run many 100ms steps; the closed-loop sim should move PV off zero toward
    # the (positive) setpoint.
    for _ in range(200):
        asyncio.run(sim.read_tags())
    assert sim.simulated_tags[pv_tag] > 0.0


def test_write_pid_setpoint_updates_active_config() -> None:
    sim = SimulatedPLCClient()
    ok = asyncio.run(sim.write_pid_setpoint(0, 73.0))
    assert ok is True
    assert sim.active_config.pids[0].setpoint == pytest.approx(73.0)


def test_write_pid_setpoint_rejects_bad_index() -> None:
    sim = SimulatedPLCClient()
    assert asyncio.run(sim.write_pid_setpoint(99, 1.0)) is False


def test_trigger_and_clear_estop_toggle_latch() -> None:
    sim = SimulatedPLCClient()
    assert asyncio.run(sim.trigger_estop()) is True
    assert sim.e_stop_active is True
    assert all(v == 0.0 for v in sim.simulated_tags.values())
    assert asyncio.run(sim.clear_estop()) is True
    assert sim.e_stop_active is False
