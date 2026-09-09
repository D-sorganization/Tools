"""Finite-history corotation and independent finite-root rotating continuum."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_moving_contracts import (
    InertialMovingChain,
    MovingChainControls,
    MovingChainState,
    MovingGripAttachment,
)
from shared.python.golf_club._shaft_moving_trajectory import (
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
    integrate_moving_chain,
)

from .shaft_trajectory_oracles import AxialRodReference
from .test_shaft_equilibrium import _controls as _equilibrium_controls
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model

_SPIN_RAD_S = 3.0
_DURATION_S = 0.002


def _rotation(time_s: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([_SPIN_RAD_S * time_s, 0, 0]).as_matrix()
    return transform


def _body_twist(pose: np.ndarray) -> np.ndarray:
    omega = np.array([_SPIN_RAD_S, 0, 0])
    return np.r_[pose[:3, :3].T @ np.cross(omega, pose[:3, 3]), pose[:3, :3].T @ omega]


def _problem(count: int) -> tuple[MovingTrajectoryProblem, MovingChainState]:
    base, seed = _model(count, damping=0, spin=_SPIN_RAD_S)
    equilibrium = replace(
        _equilibrium_controls(), force_tolerance_n=1e-11, moment_tolerance_nm=1e-12
    )
    loaded = solve_gripped_chain(base, seed, equilibrium)
    shaft = base.shaft
    assert not shaft.elastic.loads, "fixed spatial loads must permit corotation"
    ports = tuple(
        MovingGripAttachment(
            port.node,
            port.grip,
            replace(
                port.anchor,
                twist=_body_twist(np.asarray(port.anchor.pose)),
                twist_rate=np.zeros(6),
            ),
        )
        for port in base.grips
    )
    inertial = replace(
        shaft, frame=replace(shaft.frame, angular_velocity_rad_s=(0, 0, 0))
    )
    chain = InertialMovingChain(inertial, ports)
    state = MovingChainState(
        loaded.poses, np.array([_body_twist(p) for p in loaded.poses]), "observer"
    )

    def history(time_s: float) -> tuple:
        return tuple(
            replace(port.anchor, pose=_rotation(time_s) @ np.asarray(port.anchor.pose))
            for port in ports
        )

    controls = MovingChainControls(equilibrium.strain_limits, _scales())
    return MovingTrajectoryProblem(chain, controls, history), state


def _continuum_positions(count: int) -> np.ndarray:
    rod = AxialRodReference()
    wave = _SPIN_RAD_S * np.sqrt(rod.density_kg_m / rod.rigidity_n)
    phase = wave * rod.length_m
    axial, tip = rod.rigidity_n, rod.tip_mass_kg * _SPIN_RAD_S**2
    matrix = np.array(
        [
            [-rod.root_stiffness_n_m, axial * wave],
            [
                -axial * wave * np.sin(phase) - tip * np.cos(phase),
                axial * wave * np.cos(phase) - tip * np.sin(phase),
            ],
        ]
    )
    coefficients = np.linalg.solve(matrix, [axial, axial])
    np.testing.assert_allclose(matrix @ coefficients, [axial, axial], atol=1e-11)
    position = rod.positions(count)
    return coefficients[0] * np.cos(wave * position) + coefficients[1] * np.sin(
        wave * position
    )


@pytest.mark.parametrize("count", [2, 4, 8])
def test_loaded_corotation_preserves_exact_motion_and_all_work_ports(
    count: int,
) -> None:
    problem, initial = _problem(count)
    # Refining the mesh also resolves its faster retained modes in time.
    steps = 4 * count
    request = MovingTrajectoryControls((0, _DURATION_S), steps, 2 * steps + 1)
    result = integrate_moving_chain(problem, initial, request)
    for sample in result.samples:
        expected = _rotation(sample.time_s) @ np.asarray(initial.poses)
        np.testing.assert_allclose(sample.state.poses, expected, atol=2e-10, rtol=0)
        np.testing.assert_allclose(
            sample.state.twists, initial.twists, atol=2e-8, rtol=0
        )
        assert sample.applied_work_j == 0
        assert abs(sample.anchor_work_j) < 1e-12
        assert sample.dissipated_energy_j == 0
        assert abs(sample.energy_balance_error_j) < 1e-10
    assert result.stability_status == "unqualified"


def test_finite_root_rotating_shape_converges_to_independent_continuum(
    record_property: Callable[[str, object], None],
) -> None:
    errors = []
    for count in (2, 4, 8):
        _, state = _problem(count)
        expected = _continuum_positions(count)
        actual = np.asarray(state.poses)[:, 2, 3]
        errors.append(float(np.max(abs(actual - expected))))
        # Positive extension is this radial benchmark's result, not a modal claim.
        assert actual[-1] - actual[0] > 1
        assert actual[0] > 0
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] < 1e-6
    record_property(
        "study",
        json.dumps(
            {
                "spin_rad_s": _SPIN_RAD_S,
                "elements": [2, 4, 8],
                "position_errors_m": errors,
            }
        ),
    )


def test_relative_grip_inertance_is_not_added_as_rotating_ground_mass() -> None:
    # In exact corotation the relative grip coordinates are constant, so their
    # inertial storage is zero even though the root has absolute acceleration.
    problem, initial = _problem(2)
    request = MovingTrajectoryControls((0, _DURATION_S), 8, 17)
    result = integrate_moving_chain(problem, initial, request)
    for sample in result.samples:
        response = sample.response.grip_responses[0]
        assert abs(response.storage.inertial_energy_j) < 1e-20
        assert np.linalg.norm(np.asarray(sample.state.twists)[0, :3]) > 1e-3
