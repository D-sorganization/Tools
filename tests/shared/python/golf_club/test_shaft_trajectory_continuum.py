"""Separate full nonlinear trajectory time error from continuum mesh error."""

import json
from collections.abc import Callable

import numpy as np
import pytest

from shared.python.golf_club._shaft_moving_contracts import (
    InertialMovingChain,
    MovingChainControls,
    MovingChainState,
    MovingGripAttachment,
)
from shared.python.golf_club._shaft_moving_trajectory import (
    MovingTrajectory,
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
    integrate_moving_chain,
)

from .shaft_trajectory_oracles import AxialRodReference
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model

_DURATION_S = 0.01


def _simulate(count: int, steps: int) -> MovingTrajectory:
    oracle = AxialRodReference()
    base, poses = _model(count, damping=0)
    port = base.grips[0]
    chain = InertialMovingChain(
        base.shaft, (MovingGripAttachment(port.node, port.grip, port.anchor),)
    )
    poses[:, 2, 3] += oracle.continuum_state(count, 0)[: count + 1]
    initial = MovingChainState(poses, np.zeros((count + 1, 6)), "observer")
    controls = MovingChainControls((0.2, 0.2, 0.2, 2, 2, 2), _scales())
    problem = MovingTrajectoryProblem(chain, controls, lambda _: (port.anchor,))
    request = MovingTrajectoryControls((0, _DURATION_S), steps, 2 * steps + 1)
    return integrate_moving_chain(problem, initial, request)


def _axial_state(result: MovingTrajectory, count: int) -> np.ndarray:
    state = result.samples[-1].state
    return np.r_[
        np.asarray(state.poses)[:, 2, 3] - AxialRodReference().positions(count),
        np.asarray(state.twists)[:, 2],
    ]


def _check_energy_and_invariant_subspace(result: MovingTrajectory, count: int) -> None:
    oracle = AxialRodReference()
    initial_energy = oracle.discrete_energy(count, oracle.continuum_state(count, 0))
    for sample in result.samples:
        twists, poses = np.asarray(sample.state.twists), np.asarray(sample.state.poses)
        np.testing.assert_allclose(twists[:, [0, 1, 3, 4, 5]], 0, atol=1e-10)
        np.testing.assert_allclose(
            poses[:, :3, :3], np.tile(np.eye(3), (count + 1, 1, 1)), atol=1e-12
        )
        assert sample.applied_work_j == 0
        assert sample.anchor_work_j == 0
        assert sample.dissipated_energy_j == 0
    final = result.samples[-1]
    energy = oracle.discrete_energy(count, _axial_state(result, count))
    assert final.response.total_energy_j == pytest.approx(energy, abs=1e-13)
    assert final.energy_balance_error_j == pytest.approx(
        energy - initial_energy, abs=1e-13
    )
    assert abs(final.energy_balance_error_j) / initial_energy < 1e-5


def test_time_refinement_uses_independent_semidiscrete_motion(
    record_property: Callable[[str, object], None],
) -> None:
    oracle, count = AxialRodReference(), 2
    reference = oracle.discrete_state(count, _DURATION_S)
    errors = []
    for steps in (16, 32, 64):
        result = _simulate(count, steps)
        errors.append(oracle.error(count, _axial_state(result, count), reference))
        _check_energy_and_invariant_subspace(result, count)
    assert 3 < errors[0] / errors[1] < 5
    assert 3 < errors[1] / errors[2] < 5
    assert errors[-1] < 1e-5
    record_property(
        "study",
        json.dumps(
            {
                "elements": count,
                "steps": [16, 32, 64],
                "time_s": _DURATION_S,
                "scaled_time_errors": errors,
            }
        ),
    )


@pytest.mark.parametrize("coarse_count", [2, 4])
def test_joint_mesh_time_refinement_separates_error_sources(
    coarse_count: int, record_property: Callable[[str, object], None]
) -> None:
    oracle = AxialRodReference()
    mesh_errors, joint_errors, rows = [], [], []
    for count in (coarse_count, 2 * coarse_count):
        exact = oracle.continuum_state(count, _DURATION_S)
        discrete = oracle.discrete_state(count, _DURATION_S)
        result = _simulate(count, 16 * count)
        actual = _axial_state(result, count)
        spatial = oracle.error(count, discrete, exact)
        temporal = oracle.error(count, actual, discrete)
        total = oracle.error(count, actual, exact)
        mesh_errors.append(spatial)
        joint_errors.append(total)
        rows.append(
            {
                "elements": count,
                "steps": 16 * count,
                "time_s": _DURATION_S,
                "spatial_error": spatial,
                "time_error": temporal,
                "joint_error": total,
                "energy_balance_error_j": result.samples[-1].energy_balance_error_j,
            }
        )
        # Time error remains below 10% of the independent spatial error.
        assert temporal < 0.1 * spatial
        assert total <= spatial + temporal + 1e-12
        assert total < 0.002 * (2 / count) ** 2
        _check_energy_and_invariant_subspace(result, count)
    assert 3 < mesh_errors[0] / mesh_errors[1] < 5
    assert 3 < joint_errors[0] / joint_errors[1] < 5
    record_property("study", json.dumps(rows))


def test_reference_continuum_energy_is_conserved_and_discrete_energy_converges(
    record_property: Callable[[str, object], None],
) -> None:
    oracle = AxialRodReference()
    reference = oracle.continuum_energy(0)
    for time_s in (0.01, 0.03, 0.05):
        assert oracle.continuum_energy(time_s) == pytest.approx(reference, rel=1e-12)
    errors = []
    for count in (2, 4, 8):
        initial = oracle.continuum_state(count, 0)
        errors.append(abs(oracle.discrete_energy(count, initial) - reference))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1
    assert errors[-1] / reference < 0.001
    record_property(
        "study",
        json.dumps(
            {
                "omega_rad_s": oracle.omega_rad_s,
                "energy_j": reference,
                "elements": [2, 4, 8],
                "spatial_energy_errors_j": errors,
            }
        ),
    )
