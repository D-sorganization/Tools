"""Contract test pinning the public API surface of swing_sim.flight.

Downstream consumers (impact stage #4106, UpstreamDrift, web backend)
import from the subpackage facade only; this test fails loudly when the
surface changes so removals are always deliberate.
"""

from __future__ import annotations

import dataclasses

import pytest

import shared.python.swing_sim.flight as flight

EXPECTED_PUBLIC_API = {
    "DEFAULT_BACKSPIN_AXIS",
    "BallFlightModel",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "FlightModelRegistry",
    "FlightModelType",
    "FlightResult",
    "FlightSimulatorProtocol",
    "LaunchConditions",
    "MacDonaldHanzelyModel",
    "TrajectoryPoint",
    "WIND_SCHEMA_VERSION",
    "WaterlooPennerModel",
    "WindGust",
    "WindScenario",
    "compare_models",
    "compute_flight_metrics",
    "derive_launch_conditions",
    "from_flight_frame",
    "is_rust_available",
    "simulate",
    "simulate_trajectory_rust",
    "to_flight_frame",
}


@pytest.mark.contract
def test_public_api_surface_is_pinned() -> None:
    assert set(flight.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.contract
def test_all_exports_resolve() -> None:
    for name in flight.__all__:
        assert getattr(flight, name) is not None, f"{name} did not resolve"


@pytest.mark.contract
def test_swing_sim_top_level_facade_unchanged() -> None:
    """The flight port must not widen the parent swing_sim facade (#4104)."""
    import shared.python.swing_sim as swing_sim

    assert "flight" not in swing_sim.__all__


@pytest.mark.contract
def test_value_types_are_frozen_dataclasses() -> None:
    for cls in (
        flight.LaunchConditions,
        flight.TrajectoryPoint,
        flight.FlightResult,
        flight.ConstantCoefficientSpec,
        flight.WindGust,
        flight.WindScenario,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} not a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


@pytest.mark.contract
def test_model_type_values_are_pinned() -> None:
    assert {m.value for m in flight.FlightModelType} == {
        "waterloo_penner",
        "macdonald_hanzely",
        "nathan",
        "ballantyne",
        "jcole",
        "rospie_dl",
        "charry_l3",
    }
