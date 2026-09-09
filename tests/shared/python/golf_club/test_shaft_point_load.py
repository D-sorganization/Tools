"""Independent spatial work and energy checks for finite-rotation point loads."""

import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

from shared.python.golf_club._shaft_point_load import SpatialPointLoad
from shared.python.golf_club._shaft_se3 import exp_twist


def _pose() -> np.ndarray:
    return np.asarray(exp_twist([0.2, -0.1, 0.7, 0.4, -0.3, 0.2]))


def _generator(q: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = np.cross(q[3:], np.eye(3)).T
    matrix[:3, 3] = q[:3]
    return matrix


def _spatial_work_oracle(
    load: SpatialPointLoad, pose: np.ndarray, q: np.ndarray
) -> np.ndarray:
    # Differentiate 4x4 poses directly; do not reuse the production 6x6 map.
    current = pose @ expm(_generator(q))
    point = np.r_[load.offset_m, 1.0]
    work = []
    for axis in np.eye(6):
        derivative = pose @ expm_frechet(
            _generator(q), _generator(axis), compute_expm=False
        )
        spin = derivative[:3, :3] @ current[:3, :3].T
        omega = np.array([spin[2, 1], spin[0, 2], spin[1, 0]])
        work.append(load.force_n @ (derivative @ point)[:3] + load.couple_nm @ omega)
    return np.asarray(work)


@pytest.mark.parametrize("couple", [[0, 0, 0], [0.7, -0.4, 0.2]])
def test_offset_load_and_complete_tangent_match_spatial_virtual_work(
    couple: list[float],
) -> None:
    load = SpatialPointLoad([3, -7, 11], couple, [0.03, -0.04, 0.08])
    pose = _pose()
    result = load.linearize(pose)
    np.testing.assert_allclose(
        result.wrench, _spatial_work_oracle(load, pose, np.zeros(6)), atol=1e-13
    )
    step = 1e-5
    tangent = np.column_stack(
        [
            (
                _spatial_work_oracle(load, pose, step * axis)
                - _spatial_work_oracle(load, pose, -step * axis)
            )
            / (2 * step)
            for axis in np.eye(6)
        ]
    )
    np.testing.assert_allclose(result.tangent, tangent, rtol=2e-9, atol=2e-9)


def test_dead_force_tangent_is_negative_potential_hessian() -> None:
    load = SpatialPointLoad([2, -3, 7], [0, 0, 0], [0.1, 0.03, -0.02])
    pose = _pose()
    step = 1e-4
    hessian = np.array(
        [
            [
                sum(
                    si
                    * sj
                    * load.force_potential(pose @ exp_twist(step * (si * a + sj * b)))
                    for si in (-1, 1)
                    for sj in (-1, 1)
                )
                / (4 * step**2)
                for b in np.eye(6)
            ]
            for a in np.eye(6)
        ]
    )
    result = load.linearize(pose)
    np.testing.assert_allclose(result.tangent, -hessian, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(result.tangent, result.tangent.T, atol=2e-15)


def test_constant_spatial_couple_keeps_nonconservative_rotational_tangent() -> None:
    load = SpatialPointLoad([0, 0, 0], [2, -3, 5], [0, 0, 0])
    result = load.linearize(np.eye(4))
    expected = np.cross(load.couple_nm, np.eye(3)).T / 2
    np.testing.assert_allclose(result.tangent[3:, 3:], expected, atol=1e-15)
    assert np.linalg.norm(result.tangent - result.tangent.T) > 1
    assert load.force_potential(_pose()) == 0  # Explicitly force-only, not total.


def test_power_and_tangent_are_observer_invariant() -> None:
    pose, observer = _pose(), exp_twist([-0.4, 0.6, 0.2, -0.7, 0.1, 0.4])
    load = SpatialPointLoad([3, 2, -5], [0.2, -0.7, 0.3], [0.03, 0.04, 0.05])
    rotated = SpatialPointLoad(
        observer[:3, :3] @ load.force_n,
        observer[:3, :3] @ load.couple_nm,
        load.offset_m,
    )
    body_velocity = np.array([0.2, -0.4, 0.6, 1.2, -0.7, 0.3])
    result, transformed = load.linearize(pose), rotated.linearize(observer @ pose)
    np.testing.assert_allclose(transformed.wrench, result.wrench, atol=3e-15)
    np.testing.assert_allclose(transformed.tangent, result.tangent, atol=3e-15)
    assert load.power(pose, body_velocity) == pytest.approx(
        rotated.power(observer @ pose, body_velocity), abs=2e-15
    )
    assert load.power(pose, body_velocity) == pytest.approx(
        result.wrench @ body_velocity
    )


@pytest.mark.parametrize("field", ["force_n", "couple_nm", "offset_m"])
@pytest.mark.parametrize("bad", [[True, 0, 0], ["1", "2", "3"], [0, np.nan, 0], [1, 2]])
def test_load_contract_rejects_nonphysical_numeric_inputs(
    field: str, bad: object
) -> None:
    args: dict[str, object] = {
        "force_n": [0, 0, 1],
        "couple_nm": [0, 0, 0],
        "offset_m": [0, 0, 0],
    }
    args[field] = bad
    with pytest.raises((TypeError, ValueError)):
        SpatialPointLoad(**args)


def test_load_inputs_and_results_do_not_alias() -> None:
    force = np.array([1.0, 2.0, 3.0])
    load = SpatialPointLoad(force, [0, 0, 0], [0, 0, 0])
    force[:] = 0
    result = load.linearize(np.eye(4))
    result.wrench[:] = 0
    result.tangent[:] = 0
    assert load.force_n == (1, 2, 3)
    np.testing.assert_array_equal(load.linearize(np.eye(4)).wrench[:3], [1, 2, 3])


@pytest.mark.parametrize("method", ["linearize", "force_potential", "power"])
def test_load_evaluation_rejects_nonrigid_pose(method: str) -> None:
    load = SpatialPointLoad([1, 0, 0], [0, 0, 0], [0, 0, 0])
    pose = np.eye(4)
    pose[0, 0] = 2
    args = (pose, np.zeros(6)) if method == "power" else (pose,)
    with pytest.raises(ValueError):
        getattr(load, method)(*args)


@pytest.mark.parametrize("bad", [[0] * 5, [np.inf] * 6, [True] * 6])
def test_power_rejects_invalid_material_velocity(bad: object) -> None:
    load = SpatialPointLoad([1, 0, 0], [0, 0, 0], [0, 0, 0])
    with pytest.raises((TypeError, ValueError)):
        load.power(np.eye(4), bad)
