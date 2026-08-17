"""Cooperative-cancellation contracts for surface-aware flight integration."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from shared.python.swing_sim.flight import (
    BallFlightModel,
    FlightCancellationCallbackError,
    FlightSimulationCancelled,
    LaunchConditions,
    SurfaceFlightSimulationSettings,
    WaterlooPennerModel,
)
from shared.python.swing_sim.ground import GroundFrame, GroundSurfaceProfile


def _launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=70.0,
        launch_angle=math.radians(12.0),
        spin_rate=2500.0,
    )


def _settings() -> SurfaceFlightSimulationSettings:
    return SurfaceFlightSimulationSettings(
        launch_relative_surface=GroundSurfaceProfile(
            surface_id="cancellation-test-plane",
            provider_id="pytest",
            provider_version="1.0.0",
            frame=GroundFrame.TARGET,
            height_m=-0.02135,
            normal_unit=(0.0, 1.0, 0.0),
            surface_velocity_m_s=(0.0, 0.0, 0.0),
            normal_restitution=0.4,
            static_friction=0.35,
            kinetic_friction=0.25,
            rolling_resistance=0.04,
            firmness_pa=1_000_000.0,
            hardness_fraction=0.7,
            grass_height_m=0.01,
            compressibility_fraction=0.2,
            compression_damping_fraction=0.2,
            turf_density_kg_m3=180.0,
            moisture_fraction=0.3,
        ),
        max_time_s=10.0,
        output_interval_s=0.01,
    )


def test_pre_solve_cancellation_never_enters_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cancelled flight must not enter solve_ivp")

    monkeypatch.setattr(
        "shared.python.swing_sim.flight.models.solve_ivp",
        forbidden,
    )

    with pytest.raises(FlightSimulationCancelled):
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=lambda: True,
        )


def test_mid_solve_cancellation_stops_at_derivative_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation_requested = False
    first_derivative_completed = False

    def should_cancel() -> bool:
        return cancellation_requested

    def controlled_solve(derivatives, _bounds, initial, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal cancellation_requested, first_derivative_completed
        derivatives(0.0, initial)
        first_derivative_completed = True
        cancellation_requested = True
        derivatives(0.01, initial)
        raise AssertionError("second derivative must observe cancellation")

    monkeypatch.setattr(
        "shared.python.swing_sim.flight.models.solve_ivp",
        controlled_solve,
    )
    with pytest.raises(FlightSimulationCancelled):
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=should_cancel,
        )

    assert first_derivative_completed is True


@pytest.mark.parametrize("returned", [0, 1, None, "false"])
def test_non_boolean_callback_result_is_a_typed_defect(returned: object) -> None:
    with pytest.raises(FlightCancellationCallbackError) as raised:
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=lambda: cast(Any, returned),
        )

    assert isinstance(raised.value.__cause__, TypeError)


def test_raising_callback_is_typed_and_preserves_original_cause() -> None:
    cause = RuntimeError("private cancellation authority detail")

    def broken() -> bool:
        raise cause

    with pytest.raises(FlightCancellationCallbackError) as raised:
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=broken,
        )

    assert raised.value.__cause__ is cause


def test_non_callable_callback_is_rejected_at_public_boundary() -> None:
    with pytest.raises(TypeError, match="cancellation_requested"):
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=cast(Any, 1),
        )


def test_cancellation_during_dense_output_publishes_no_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = False

    def dense_output(_time_s: float) -> np.ndarray:
        nonlocal cancel
        cancel = True
        return np.array([0.0, 0.0, 0.1, 1.0, 0.0, -1.0])

    fake_solution = SimpleNamespace(
        t=np.array([0.0, 0.02]),
        y=np.array(
            [
                [0.0, 0.02],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, -1.0],
            ]
        ),
        sol=dense_output,
    )
    monkeypatch.setattr(
        "shared.python.swing_sim.flight.models.solve_ivp",
        lambda *_args, **_kwargs: fake_solution,
    )

    with pytest.raises(FlightSimulationCancelled):
        WaterlooPennerModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=lambda: cancel,
        )


def test_cancellation_after_metrics_prevents_result_publication() -> None:
    cancellation_requested = False

    class CancelAfterMetricsModel(WaterlooPennerModel):
        def _compute_metrics(self, trajectory):  # type: ignore[no-untyped-def]
            nonlocal cancellation_requested
            result = super()._compute_metrics(trajectory)
            cancellation_requested = True
            return result

    with pytest.raises(FlightSimulationCancelled):
        CancelAfterMetricsModel().simulate_to_surface(
            _launch(),
            _settings(),
            cancellation_requested=lambda: cancellation_requested,
        )


def test_always_false_callback_preserves_exact_solver_result() -> None:
    model = WaterlooPennerModel()
    baseline = model.simulate_to_surface(_launch(), _settings())
    controlled = model.simulate_to_surface(
        _launch(),
        _settings(),
        cancellation_requested=lambda: False,
    )

    assert controlled.model_name == baseline.model_name
    assert controlled.carry_distance == baseline.carry_distance
    assert controlled.flight_time == baseline.flight_time
    assert len(controlled.trajectory) == len(baseline.trajectory)
    for expected, actual in zip(
        baseline.trajectory,
        controlled.trajectory,
        strict=True,
    ):
        assert actual.time == expected.time
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.velocity, expected.velocity)
        np.testing.assert_array_equal(
            actual.angular_velocity_rad_s,
            expected.angular_velocity_rad_s,
        )


def test_surface_cancellation_is_additive_and_keyword_only() -> None:
    simulate = inspect.signature(BallFlightModel.simulate)
    surface = inspect.signature(BallFlightModel.simulate_to_surface)

    assert tuple(simulate.parameters) == ("self", "launch", "max_time", "dt")
    assert tuple(surface.parameters) == (
        "self",
        "launch",
        "settings",
        "cancellation_requested",
    )
    callback = surface.parameters["cancellation_requested"]
    assert callback.kind is inspect.Parameter.KEYWORD_ONLY
    assert callback.default is None
