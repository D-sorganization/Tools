"""Seeded Python/Rust parity over qualified tilted ground properties."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import tools_core

from shared.python.swing_sim.ground import GroundSimulationRequest, run_ground_reference
from shared.python.swing_sim.ground.tests.conformance_support import (
    SEEDED_PROPERTY_CASE_COUNT,
    SEEDED_PROPERTY_SEED,
    build_seeded_property_requests,
)

FIXTURE = (
    Path(__file__).parents[3]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)


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
    """Assert exact canonical parity for the seeded tilted-property sweep."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    execution = dict(fixture["execution"])
    execution["schema_version"] = fixture["execution_schema_version"]
    execution_json = json.dumps(execution, separators=(",", ":"))
    requests = build_seeded_property_requests(fixture["request"])
    assert len(requests) == SEEDED_PROPERTY_CASE_COUNT
    assert SEEDED_PROPERTY_SEED == 4275
    for index, payload in enumerate(requests):
        assert_parity(payload, execution_json, f"seeded case {index}")
    assert_parity(
        immediate_capture_request(fixture["request"]),
        execution_json,
        "immediate capture",
    )


if __name__ == "__main__":
    main()
