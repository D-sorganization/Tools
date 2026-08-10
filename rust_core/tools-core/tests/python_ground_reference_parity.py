"""Seeded Python/Rust execution parity over qualified planar ground cases."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import tools_core

from shared.python.swing_sim.ground import GroundSimulationRequest, run_ground_reference

FIXTURE = (
    Path(__file__).parents[3]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)
CASE_COUNT = 20
SEED = 4275


def dot(left: list[float], right: list[float]) -> float:
    """Return a deterministic three-vector dot product."""
    return sum(left[index] * right[index] for index in range(3))


def add(left: list[float], right: list[float]) -> list[float]:
    """Return a deterministic three-vector sum."""
    return [left[index] + right[index] for index in range(3)]


def scale(value: list[float], factor: float) -> list[float]:
    """Scale a three-vector."""
    return [component * factor for component in value]


def tangent(value: list[float], normal: list[float]) -> list[float]:
    """Project a vector into the declared plane."""
    return add(value, scale(normal, -dot(value, normal)))


def case_request(
    template: dict[str, Any], index: int, rng: random.Random
) -> dict[str, Any]:
    """Build one valid deterministic request in the shared resolver-free scope."""
    request = deepcopy(template)
    # Python's canonical resolver-free domain fixes its axis to world x, so the
    # cross-runtime corpus must use the horizontal plane. Tilted planes remain
    # valid in the compiled runtime and are covered by native invariant tests.
    normal = [0.0, 1.0, 0.0]
    height = 0.0
    contact = add([0.0, height, 0.0], scale(normal, request["ball_radius_m"]))
    surface_velocity = tangent(
        [rng.uniform(-0.08, 0.08), 0.0, rng.uniform(-0.08, 0.08)], normal
    )
    launch_tangent = tangent(
        [rng.uniform(0.5, 1.4), 0.0, rng.uniform(-0.2, 0.2)], normal
    )
    incoming_velocity = add(add(surface_velocity, launch_tangent), scale(normal, -0.1))
    spin = [rng.uniform(-4.0, 4.0) for _ in range(3)]
    request["request_id"] = f"compiled-parity-{index:02d}"
    request["surface"]["height_m"] = height
    request["surface"]["normal_unit"] = normal
    request["surface"]["surface_velocity_m_s"] = surface_velocity
    request["surface"]["normal_restitution"] = rng.uniform(0.05, 0.45)
    request["surface"]["static_friction"] = 0.45
    request["surface"]["kinetic_friction"] = rng.uniform(0.1, 0.35)
    request["surface"]["rolling_resistance"] = rng.uniform(0.03, 0.12)
    request["last_separated_state"]["position_m"] = add(contact, scale(normal, 0.001))
    request["first_penetrating_state"]["position_m"] = add(
        contact, scale(normal, -0.001)
    )
    for key in ("last_separated_state", "first_penetrating_state"):
        request[key]["velocity_m_s"] = incoming_velocity
        request[key]["angular_velocity_rad_s"] = spin
    request["max_time_s"] = 0.6
    request["output_interval_s"] = 0.05
    request["max_events"] = 64
    return request


def immediate_capture_request(template: dict[str, Any]) -> dict[str, Any]:
    """Build the exact zero-slip first-impact edge case."""
    request = deepcopy(template)
    speed = 1.0
    spin = -speed / float(request["ball_radius_m"])
    request["request_id"] = "compiled-parity-immediate-capture"
    for key in ("last_separated_state", "first_penetrating_state"):
        request[key]["velocity_m_s"] = [speed, -0.04, 0.0]
        request[key]["angular_velocity_rad_s"] = [0.0, 0.0, spin]
    return request


def assert_parity(payload: dict[str, Any], execution_json: str, label: str) -> None:
    """Assert exact canonical Python/compiled parity for one payload."""
    python_request = GroundSimulationRequest.from_dict(payload)
    canonical_request = python_request.to_json()
    expected = run_ground_reference(python_request).to_json()
    actual = tools_core.run_flight_to_ground_reference_v1(
        canonical_request, execution_json
    )
    assert actual == expected, f"compiled parity mismatch for {label}"


def main() -> None:
    """Assert exact canonical execution parity for the seeded corpus."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    execution = dict(fixture["execution"])
    execution["schema_version"] = fixture["execution_schema_version"]
    execution_json = json.dumps(execution, separators=(",", ":"))
    rng = random.Random(SEED)
    for index in range(CASE_COUNT):
        payload = case_request(fixture["request"], index, rng)
        assert_parity(payload, execution_json, f"seeded case {index}")
    assert_parity(
        immediate_capture_request(fixture["request"]),
        execution_json,
        "immediate capture",
    )


if __name__ == "__main__":
    main()
