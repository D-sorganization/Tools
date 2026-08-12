"""Flight-pipeline seam: DI protocol + one-call convenience simulate.

Mirrors the DI design of UpstreamDrift
``src/shared/python/physics/swing_ball_flight_pipeline.py``
(``FlightSimulatorProtocol``): the impact stage (#4106) derives
:class:`LaunchConditions` via
:func:`shared.python.swing_sim.flight.launch.derive_launch_conditions`
and plugs them straight into any :class:`FlightSimulatorProtocol`
implementation — every :class:`BallFlightModel` already satisfies it,
and tests inject mocks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .registry import FlightModelRegistry, FlightModelType
from .types import FlightResult, LaunchConditions


@runtime_checkable
class FlightSimulatorProtocol(Protocol):
    """Minimal protocol for any ball-flight simulator.

    Satisfied by every :class:`BallFlightModel` in this package; the Rust
    facade's :func:`simulate_trajectory_rust` can be adapted trivially.
    Tests inject mocks.
    """

    def simulate(
        self,
        launch: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Return a :class:`FlightResult` for the given launch conditions."""
        ...


def simulate(
    launch: LaunchConditions,
    model_name: str = "waterloo_penner",
    max_time: float = 10.0,
    dt: float = 0.01,
) -> FlightResult:
    """Simulate a ball flight with a registry model selected by name.

    Args:
        launch: Flight-frame launch conditions (radians / RPM / SI).
        model_name: A :class:`FlightModelType` value string, e.g.
            ``"waterloo_penner"`` or ``"macdonald_hanzely"``.
        max_time: Maximum simulated time [s], > 0.
        dt: Trajectory sampling interval [s], > 0.

    Returns:
        The :class:`FlightResult` from the selected model.

    Raises:
        ValueError: If ``launch`` is missing or ``model_name`` is not a
            known :class:`FlightModelType` value.
    """
    if launch is None:
        raise ValueError("launch must be provided")
    try:
        model_type = FlightModelType(model_name)
    except ValueError as exc:
        valid = ", ".join(m.value for m in FlightModelType)
        raise ValueError(
            f"unknown flight model {model_name!r}; valid names: {valid}"
        ) from exc
    model = FlightModelRegistry.get_model(model_type)
    return model.simulate(launch, max_time=max_time, dt=dt)


__all__ = ["FlightSimulatorProtocol", "simulate"]
