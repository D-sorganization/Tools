"""Amplitude convergence of inertial nonlinear motion to the rotating tangent."""

import json
from collections.abc import Callable

import numpy as np
import pytest

from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_moving_contracts import MovingChainState
from shared.python.golf_club._shaft_se3 import log_pose, right_jacobian

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_rotating_disturbance import _DURATION_S, _disturbed_problem
from .test_shaft_trajectory_geometry import _reference
from .test_shaft_trajectory_rotation import (
    _SPIN_RAD_S,
    _body_twist,
    _problem,
    _rotation,
)


def _rotating_chart(
    equilibrium: MovingChainState, state: MovingChainState, time_s: float
) -> np.ndarray:
    observer = np.linalg.inv(_rotation(time_s))
    coordinates, rates = [], []
    for reference, pose, velocity in zip(
        equilibrium.poses, state.poses, state.twists, strict=True
    ):
        relative = np.linalg.solve(np.asarray(reference), observer @ np.asarray(pose))
        coordinate = log_pose(relative)
        relative_velocity = np.asarray(velocity) - _body_twist(np.asarray(pose))
        coordinates.append(coordinate)
        rates.append(np.linalg.solve(right_jacobian(coordinate), relative_velocity))
    scales = _scales()
    dimensions = np.array([scales.length_m] * 3 + [1.0] * 3)
    return np.r_[
        (np.asarray(coordinates) / dimensions).ravel(),
        (scales.time_s * np.asarray(rates) / dimensions).ravel(),
    ]


@pytest.mark.parametrize("coarse_amplitude_m", [0.001, 0.0005])
def test_rotating_tangent_error_is_quadratic_in_resolved_bending_amplitude(
    coarse_amplitude_m: float,
    record_property: Callable[[str, object], None],
) -> None:
    count = 2
    _, equilibrium = _problem(count)
    chain, _ = _model(count, damping=0, spin=_SPIN_RAD_S)
    model = constant_gripped_model(chain, equilibrium.poses, _controls(), _scales())
    rows = []
    for amplitude in (coarse_amplitude_m, coarse_amplitude_m / 2):
        problem, initial = _disturbed_problem(count, amplitude)
        coarse = _reference(problem, initial, 1e-10)
        tight = _reference(problem, initial, 1e-12)
        exact = _rotating_chart(equilibrium, tight, _DURATION_S)
        repeated = _rotating_chart(equilibrium, coarse, _DURATION_S)
        initial_chart = _rotating_chart(equilibrium, initial, 0)
        predicted = np.asarray(model.scaled_state_at(initial_chart, _DURATION_S))
        error = float(np.linalg.norm(exact - predicted))
        reference_error = float(np.linalg.norm(exact - repeated))
        assert reference_error < 0.01 * error
        assert error < 0.01 * np.linalg.norm(predicted)
        rows.append(
            {
                "amplitude_m": amplitude,
                "scaled_linearization_error": error,
                "scaled_reference_difference": reference_error,
            }
        )
    ratio = (
        rows[0]["scaled_linearization_error"] / rows[1]["scaled_linearization_error"]
    )
    assert 3 < ratio < 5
    assert model.stability_status == "unqualified"
    record_property(
        "study", json.dumps({"elements": count, "time_s": _DURATION_S, "rows": rows})
    )
