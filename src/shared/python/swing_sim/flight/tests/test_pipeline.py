"""Pipeline-seam tests: DI protocol conformance and convenience simulate."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from shared.python.swing_sim.flight import (
    FlightModelRegistry,
    FlightResult,
    FlightSimulatorProtocol,
    LaunchConditions,
    TrajectoryPoint,
    WaterlooPennerModel,
    simulate,
)

LAUNCH = LaunchConditions(
    ball_speed=74.0, launch_angle=math.radians(12.0), spin_rate=2600.0
)


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Prevent cross-test pollution of the class-level model dict."""
    FlightModelRegistry.reset()
    yield
    FlightModelRegistry.reset()


@pytest.mark.contract
def test_every_registry_model_satisfies_protocol() -> None:
    for model in FlightModelRegistry.get_all_models():
        assert isinstance(model, FlightSimulatorProtocol)


@pytest.mark.contract
def test_mock_simulator_satisfies_protocol() -> None:
    class _Mock:
        def simulate(
            self,
            launch: LaunchConditions,
            max_time: float = 10.0,
            dt: float = 0.01,
        ) -> FlightResult:
            point = TrajectoryPoint(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            return FlightResult((point,), "mock")

    assert isinstance(_Mock(), FlightSimulatorProtocol)


@pytest.mark.unit
def test_simulate_default_model_is_waterloo_penner() -> None:
    result = simulate(LAUNCH)
    reference = WaterlooPennerModel().simulate(LAUNCH)
    assert result.model_name == "Waterloo/Penner"
    assert result.carry_distance == pytest.approx(reference.carry_distance)


@pytest.mark.unit
def test_simulate_selects_model_by_name() -> None:
    result = simulate(LAUNCH, model_name="macdonald_hanzely")
    assert result.model_name == "MacDonald-Hanzely"
    assert result.carry_distance > 0.0


@pytest.mark.unit
def test_simulate_rejects_unknown_model_and_missing_launch() -> None:
    with pytest.raises(ValueError, match="unknown flight model"):
        simulate(LAUNCH, model_name="not_a_model")
    with pytest.raises(ValueError, match="launch"):
        simulate(None)  # type: ignore[arg-type]
