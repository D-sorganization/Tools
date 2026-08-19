"""Guard the Python-owned localized-torque golden consumed by the web kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.python.swing_sim import reference
from shared.python.swing_sim.localized_torque import (
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    LocalizedTorqueOffset,
    add_localized_offsets,
)
from shared.python.swing_sim.types import PendulumParameters, PendulumState

_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "localized_torque_python_golden_v1.json"
)


def _series(
    parameters: PendulumParameters,
    offsets: tuple[LocalizedTorqueOffset, ...],
    base_torques_nm: tuple[float, float],
) -> dict[str, Any]:
    torque_at = lambda time_s: add_localized_offsets(  # noqa: E731
        base_torques_nm, offsets, time_s
    )
    states = reference.simulate_forced(
        parameters,
        PendulumState(0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0),
        0.001,
        5,
        torque_at,
    )
    return {
        "base_torques_nm": list(base_torques_nm),
        "states": [[float(value) for value in row] for row in states],
        "sampled_torques_nm": [list(torque_at(index * 0.001)) for index in range(6)],
    }


def _authority_document() -> dict[str, Any]:
    parameters = PendulumParameters.golf_default()
    offsets = (
        LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.001, 0.003), 3.0),
        LocalizedTorqueOffset(WRIST_JOINT_ID, (0.002, 0.004), -2.0),
    )
    return {
        "schema_version": 1,
        "authority": "shared.python.swing_sim.reference",
        "dt_s": 0.001,
        "n_steps": 5,
        "gravity_m_s2": [0, 0],
        "initial_state": [0, 0, 0, 0],
        "parameters": {
            name: getattr(parameters, name)
            for name in ("m1", "l1", "lc1", "i1", "m2", "l2", "lc2", "i2", "d1", "d2")
        },
        "commands": [
            {
                "joint_id": offset.joint_id,
                "time_window_s": list(offset.time_window_s),
                "torque_nm": offset.torque_nm,
            }
            for offset in offsets
        ],
        "passive": _series(parameters, offsets, (0.0, 0.0)),
        "prescribed": _series(parameters, offsets, (20.0, -5.0)),
    }


def test_localized_torque_web_golden_is_exact_python_output() -> None:
    assert json.loads(_FIXTURE.read_text(encoding="utf-8")) == _authority_document()
