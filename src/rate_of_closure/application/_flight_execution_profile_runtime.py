"""Bounded Waterloo execution-profile parsing and physical recomputation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from rate_of_closure.application._workspace_validation import exact_mapping
from shared.python.swing_sim.flight import (
    CancellationCheck,
    FlightGroundTransferSettings,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    LaunchConditions,
    SurfaceFlightSimulationSettings,
    TrajectoryPoint,
    compute_flight_metrics,
    launch_relative_surface,
    raise_if_flight_cancelled,
)

WATERLOO_SETTING_IDS = ("max_time_s", "sample_every", "step_s")
_SETTING_FIELDS = frozenset(WATERLOO_SETTING_IDS)
_MAX_TIME_S = 120.0
_MIN_STEP_S = 0.0001
_MAX_STEP_S = 0.1
_MAX_SAMPLE_EVERY = 10_000
_MAX_RETAINED_INTERVAL_S = 1.0
_CANCELLATION_POLL_INTERVAL = 256


@dataclass(frozen=True, slots=True)
class WaterlooSettings:
    """Validated settings for the exact Waterloo execution profile."""

    max_time_s: float
    step_s: float
    sample_every: int

    @property
    def retained_interval_s(self) -> float:
        return self.step_s * self.sample_every


def _positive_bounded(value: object, name: str, maximum: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if not 0.0 < number <= maximum:
        raise ValueError(f"{name} lies outside its profile bound")
    return number


def parse_waterloo_settings(values: Mapping[str, float]) -> WaterlooSettings:
    """Return exact bounded settings or reject the profile fail closed."""
    data = exact_mapping(values, _SETTING_FIELDS, "flight profile settings")
    max_time_s = _positive_bounded(data["max_time_s"], "max_time_s", _MAX_TIME_S)
    step_s = _positive_bounded(data["step_s"], "step_s", _MAX_STEP_S)
    if step_s < _MIN_STEP_S:
        raise ValueError("step_s lies outside its profile bound")
    sample_value = data["sample_every"]
    if (
        type(sample_value) not in (int, float)
        or not math.isfinite(float(sample_value))
        or not float(sample_value).is_integer()
    ):
        raise ValueError("sample_every must be a finite whole number")
    sample_every = int(sample_value)
    if not 1 <= sample_every <= _MAX_SAMPLE_EVERY:
        raise ValueError("sample_every lies outside its profile bound")
    settings = WaterlooSettings(max_time_s, step_s, sample_every)
    if settings.retained_interval_s > _MAX_RETAINED_INTERVAL_S:
        raise ValueError("retained sample interval lies outside its profile bound")
    return settings


def _retained_trajectory(
    result: FlightResult,
    sample_every: int,
    cancellation_requested: CancellationCheck | None,
) -> list[TrajectoryPoint]:
    retained: list[TrajectoryPoint] = []
    for retained_ordinal, raw_ordinal in enumerate(
        range(0, len(result.trajectory), sample_every)
    ):
        if retained_ordinal % _CANCELLATION_POLL_INTERVAL == 0:
            raise_if_flight_cancelled(cancellation_requested)
        retained.append(result.trajectory[raw_ordinal])
    if result.trajectory and (
        not retained or retained[-1] is not result.trajectory[-1]
    ):
        retained.append(result.trajectory[-1])
    raise_if_flight_cancelled(cancellation_requested)
    return retained


def recompute_waterloo(
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    settings: WaterlooSettings,
    cancellation_requested: CancellationCheck | None = None,
) -> FlightResult:
    """Recompute one profile with cancellation through post-processing."""
    surface = launch_relative_surface(
        transfer.surface,
        launch.ball_radius,
        launch.ball_setup,
    )
    simulation = SurfaceFlightSimulationSettings(
        surface,
        settings.max_time_s,
        settings.step_s,
    )
    model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
    if cancellation_requested is None:
        raw = model.simulate_to_surface(launch, simulation)
    else:
        raw = model.simulate_to_surface(
            launch,
            simulation,
            cancellation_requested=cancellation_requested,
        )
    retained = _retained_trajectory(
        raw,
        settings.sample_every,
        cancellation_requested,
    )
    metrics = compute_flight_metrics(retained, raw.model_name)
    raise_if_flight_cancelled(cancellation_requested)
    return metrics


__all__ = [
    "WATERLOO_SETTING_IDS",
    "WaterlooSettings",
    "parse_waterloo_settings",
    "recompute_waterloo",
]
