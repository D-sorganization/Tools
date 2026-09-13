"""Bidirectional flight-model policy pins across Python, web, and Morris (#4457).

This module verifies that:
1. Python's `FlightModelType` enum values match the authority models list in the
   Morris UI parity fixture.
2. The browser variation flight model (`waterloo_penner`) is a registered member
   of `FlightModelType`.
3. Python `VariationPlan` accepts all 7 authority models and serializes/round-trips
   them faithfully.
4. Python `VariationPlan` rejects unknown or unregistered flight models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    NoiseSpec,
    VariationPlan,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[6]
_FIXTURE_PATH = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "morris_ui_parity_v1.json"
)

# The single model supported by the in-browser trajectory solver
BROWSER_VARIATION_FLIGHT_MODEL = "waterloo_penner"


def _sample_noise_spec() -> NoiseSpec:
    return NoiseSpec(
        variable_key=f"{CATEGORY_LAUNCH}.ball_speed_mph",
        distribution="uniform",
        scale=2.0,
        spec_id="test_launch_speed",
    )


def test_python_flight_models_pin_morris_authority_fixture() -> None:
    """Ensure Python FlightModelType matches the authority fixture used by Morris UI."""
    assert _FIXTURE_PATH.is_file(), f"Morris parity fixture missing at {_FIXTURE_PATH}"
    fixture_data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    authority_models = fixture_data["authority_flight_models"]

    py_models = [m.value for m in FlightModelType]
    assert py_models == authority_models, (
        f"Python FlightModelType {py_models} does not match "
        f"Morris authority fixture {authority_models}"
    )


def test_browser_variation_model_is_member_of_python_flight_models() -> None:
    """Ensure the browser's supported flight model is recognized by Python."""
    registered = {m.value for m in FlightModelType}
    assert BROWSER_VARIATION_FLIGHT_MODEL in registered


@pytest.mark.parametrize("model_name", [m.value for m in FlightModelType])
def test_variation_plan_accepts_all_authority_flight_models(model_name: str) -> None:
    """Python VariationPlan accepts all 7 authority flight models."""
    plan = VariationPlan(
        mode="launch",
        noise=(_sample_noise_spec(),),
        n_runs=10,
        seed=42,
        flight_model=model_name,
    )
    assert plan.flight_model == model_name

    # Check to_json_dict / from_json_dict round-trip
    data = plan.to_json_dict()
    assert data["flight_model"] == model_name
    reconstituted = VariationPlan.from_json_dict(data)
    assert reconstituted.flight_model == model_name

    # Check dumps / loads round-trip
    serialized = plan.dumps()
    reconstituted_json = VariationPlan.loads(serialized)
    assert reconstituted_json.flight_model == model_name


@pytest.mark.parametrize(
    "invalid_model",
    [
        "unknown_model",
        "morris_custom",
        "waterloo-penner",
        "",
        "WATERLOO_PENNER",
    ],
)
def test_variation_plan_rejects_unregistered_flight_models(invalid_model: str) -> None:
    """Python VariationPlan rejects any model not in FlightModelType."""
    with pytest.raises(
        ContractViolationError, match="registered FlightModelType value"
    ):
        VariationPlan(
            mode="launch",
            noise=(_sample_noise_spec(),),
            n_runs=10,
            seed=42,
            flight_model=invalid_model,
        )
