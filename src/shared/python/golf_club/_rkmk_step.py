"""Shared private RK4 step in local material SE(3) charts.

The caller owns state construction, constitutive domains and response/work
semantics. This kernel supplies neither contact events nor a stability claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_se3 import right_jacobian

_State = TypeVar("_State")
_Response = TypeVar("_Response")
_RK4_WEIGHTS = (1 / 6, 1 / 3, 1 / 3, 1 / 6)


@dataclass(frozen=True)
class RkmkStepModel(Generic[_State, _Response]):
    """Explicit typed ports; shift must own/validate H0 Exp(q) and body twists.

    Evaluate must recompute a response at the supplied state/time. Rates must
    return its material twist derivatives. No callback may mutate input state.
    The enclosing trajectory validates physical laws and evaluation budgets.
    """

    shift: Callable[[_State, np.ndarray, np.ndarray], _State]
    evaluate: Callable[[_State, float], _Response]
    twists: Callable[[_State], object]
    rates: Callable[[_Response], object]


@dataclass(frozen=True)
class _Stage(Generic[_Response]):
    chart_rate: np.ndarray
    acceleration: np.ndarray
    response: _Response


def _velocities(model: RkmkStepModel[_State, _Response], state: _State) -> np.ndarray:
    value = model.twists(state)
    shape = np.asarray(value).shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != 6:
        raise ValueError("material velocities need one six-axis row per pose")
    return finite_array(value, shape, "material velocities")


def material_chart_rates(coordinates: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Use the canonical body differential for finite, matching six-axis rows."""
    shape = velocities.shape
    if len(shape) != 2 or shape[1] != 6 or shape[0] < 1:
        raise ValueError("material chart needs one six-axis row per pose")
    coordinates = finite_array(coordinates, shape, "chart coordinates")
    velocities = finite_array(velocities, shape, "material velocities")
    return np.array(
        [
            np.linalg.solve(right_jacobian(coordinate), velocity)
            for coordinate, velocity in zip(coordinates, velocities, strict=True)
        ]
    )


def _stage(
    model: RkmkStepModel[_State, _Response],
    initial: _State,
    increments: tuple[np.ndarray, np.ndarray],
    time_s: float,
) -> _Stage[_Response]:
    coordinates, velocity_change = increments
    velocities = _velocities(model, initial) + velocity_change
    state = model.shift(initial, coordinates, velocities)
    response = model.evaluate(state, time_s)
    chart_rate = material_chart_rates(coordinates, velocities)
    acceleration = finite_array(model.rates(response), velocities.shape, "acceleration")
    return _Stage(chart_rate, acceleration, response)


def _weighted(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = tuple(values)
    return np.asarray(
        sum(
            (
                weight * value
                for weight, value in zip(_RK4_WEIGHTS, arrays, strict=True)
            ),
            start=np.zeros_like(arrays[0]),
        )
    )


def rkmk_step(
    model: RkmkStepModel[_State, _Response],
    initial: _State,
    response: _Response,
    cell: tuple[float, float, float],
) -> tuple[_State, _Response, tuple[tuple[_Response, float], ...]]:
    """Advance one validated cell with a cached consistent start response.

    Require a finite forward cell with its exact arithmetic midpoint, owned
    finite material states and valid constitutive callbacks. Return the final
    state/response and separate RK-weighted stage responses, or raise. Four
    new responses are evaluated. Smooth fourth order does not imply that a
    discontinuous contact switch is resolved or that energy is conserved.
    """
    start, middle, end = finite_array(cell, (3,), "time cell")
    step_s = float(end - start)
    if not (0 <= start < middle < end and middle == start + step_s / 2):
        raise ValueError("time cell requires forward endpoints and their midpoint")
    velocities = _velocities(model, initial)
    acceleration = finite_array(model.rates(response), velocities.shape, "acceleration")
    stages = [_Stage(velocities, acceleration, response)]
    for fraction, time_s in zip((0.5, 0.5, 1.0), (middle, middle, end), strict=True):
        previous = stages[-1]
        increments = (
            step_s * fraction * previous.chart_rate,
            step_s * fraction * previous.acceleration,
        )
        stages.append(_stage(model, initial, increments, float(time_s)))
    final = model.shift(
        initial,
        step_s * _weighted(stage.chart_rate for stage in stages),
        velocities + step_s * _weighted(stage.acceleration for stage in stages),
    )
    work = tuple(
        (stage.response, weight)
        for stage, weight in zip(stages, _RK4_WEIGHTS, strict=True)
    )
    return final, model.evaluate(final, float(end)), work


__all__ = ()
