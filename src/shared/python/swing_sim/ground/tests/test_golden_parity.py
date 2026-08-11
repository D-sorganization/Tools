"""Shared golden-fixture parity tests for future TypeScript/Rust adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground import (
    GroundSimulationRequest,
    GroundSimulationResult,
)

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "flight_to_ground_golden_v1.json"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_python_round_trips_the_shared_ground_contract_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = GroundSimulationRequest.from_dict(fixture["request"])
    result = GroundSimulationResult.from_dict(fixture["result"])

    assert request.to_dict() == fixture["request"]
    assert result.to_dict() == fixture["result"]
    assert _digest(request.to_json()) == fixture["request_sha256"]
    assert _digest(result.to_json()) == fixture["result_sha256"]


def test_shared_fixture_pins_cross_runtime_numeric_tokens() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    policy = fixture["numeric_policy_cases"]

    assert canonical_numeric_json(policy["values"]) == policy["expected_json"]
    for value in policy["rejected_values"].values():
        with pytest.raises(ValueError):
            canonical_numeric_json(value)
