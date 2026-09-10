"""Physical and continuum controls for deformed loaded-shaft operators."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh

from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_chain import SectionChain
from shared.python.golf_club._shaft_equilibrium import solve_clamped_chain
from shared.python.golf_club._shaft_inertia import SectionInertia
from shared.python.golf_club._shaft_loaded_dynamics import linearized_chain_dynamics
from shared.python.golf_club._shaft_rotating_chain import RotatingSectionChain
from shared.python.golf_club._shaft_se3 import exp_twist

from .test_shaft_chain import _loads, _poses, _sections
from .test_shaft_equilibrium import _controls
from .test_shaft_frame_inertia import _frame, _velocity_oracle
from .test_shaft_inertia import _moved, _point, _samples, _vee
from .test_shaft_rotating_chain import _radial_rod


def _chain() -> RotatingSectionChain:
    return RotatingSectionChain(
        SectionChain(_sections(), _loads(True)),
        (SectionInertia(_samples()),) * 2,
        _frame(),
    )


def test_deformed_mass_and_gyroscopic_force_match_physical_point_motion() -> None:
    chain, poses = _chain(), _poses()
    result = linearized_chain_dynamics(chain, poses)
    velocity = np.linspace(-0.4, 0.7, 18)
    energy, gyro, step = 0.0, np.zeros(18), 1e-6
    for index in range(2):
        pair, local = poses[index : index + 2], velocity[6 * index : 6 * index + 12]
        gyro[6 * index : 6 * index + 12] += _velocity_oracle(pair, local)
        for sample in _samples():
            point = _point(pair, sample.fraction)
            plus = _point(_moved(pair, local, step), sample.fraction)
            minus = _point(_moved(pair, local, -step), sample.fraction)
            rate, body = (plus - minus) / (2 * step), sample.body
            com_velocity = rate[:3, 3] + rate[:3, :3] @ body.center_of_mass_m
            spin = _vee(point[:3, :3].T @ rate[:3, :3])
            energy += (
                body.mass_kg * (com_velocity @ com_velocity)
                + spin @ body.inertia_at_com_kg_m2 @ spin
            ) / 2
    assert velocity @ result.mass @ velocity / 2 == pytest.approx(energy, rel=2e-8)
    np.testing.assert_allclose(result.gyroscopic @ velocity, gyro, atol=5e-8)
    np.testing.assert_allclose(result.mass, result.mass.T, atol=1e-15)
    np.testing.assert_allclose(result.gyroscopic + result.gyroscopic.T, 0, atol=3e-14)
    assert abs(velocity @ result.gyroscopic @ velocity) < 3e-14


def test_stiffness_differentiates_material_balance_even_away_from_a_root() -> None:
    chain, poses, step = _chain(), _poses(), 1e-5
    result = linearized_chain_dynamics(chain, poses)
    direction = np.linspace(-0.2, 0.3, 18)
    plus = chain.linearize(_moved(poses, direction, step))
    minus = chain.linearize(_moved(poses, direction, -step))
    np.testing.assert_allclose(
        result.stiffness @ direction,
        (plus.residual - minus.residual) / (2 * step),
        atol=2e-7,
    )
    assert np.linalg.norm(result.stiffness - result.stiffness.T) > 0.1
    np.testing.assert_array_equal(result.residual, chain.linearize(poses).residual)


def test_loaded_axial_operators_match_consistent_rod_mass_and_spin_softening() -> None:
    frame = RotatingFrameState("observer", (20, 0, 0), (0, 0, 0), (0, 0, 0))
    chain, seed = _radial_rod(2, frame)
    root = solve_clamped_chain(chain, seed, _controls())
    result = linearized_chain_dynamics(chain, root.poses)
    axial = np.array([2, 8, 14])
    mass = 0.2 * 0.5 / 6 * np.array([[2, 1, 0], [1, 4, 1], [0, 1, 2]])
    elastic = 1000 / 0.5 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    np.testing.assert_allclose(result.mass[np.ix_(axial, axial)], mass, atol=1e-14)
    np.testing.assert_allclose(
        result.stiffness[np.ix_(axial, axial)], elastic - 20**2 * mass, atol=1e-9
    )
    np.testing.assert_allclose(result.gyroscopic[np.ix_(axial, axial)], 0, atol=1e-14)
    assert root.poses[-1, 2, 3] > 1


def test_guided_axial_frequency_converges_to_continuum_rotating_rod_limit() -> None:
    # All nonaxial DOFs are constrained: this is NOT a full free-shaft mode.
    # mu*u_tt = EA*u_XX + mu*Omega²*u; u(0)=0, u_X(1)=0.
    frame = RotatingFrameState("observer", (20, 0, 0), (0, 0, 0), (0, 0, 0))
    exact_squared = 1000 / 0.2 * (np.pi / 2) ** 2 - 20**2
    errors = []
    for count in (2, 4, 8):
        chain, seed = _radial_rod(count, frame)
        root = solve_clamped_chain(chain, seed, _controls())
        result = linearized_chain_dynamics(chain, root.poses)
        axial = np.arange(8, 6 * (count + 1), 6)
        eigenvalues = eigh(
            result.stiffness[np.ix_(axial, axial)],
            result.mass[np.ix_(axial, axial)],
            eigvals_only=True,
        )
        errors.append(abs(eigenvalues[0] - exact_squared))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] / exact_squared < 0.004


def test_operator_arrays_are_fresh_and_inputs_and_prior_snapshots_are_isolated() -> (
    None
):
    chain, poses = _chain(), _poses()
    before = poses.copy()
    result = linearized_chain_dynamics(chain, poses)
    saved = result.mass.copy()
    result.mass[:] = 0
    result.gyroscopic[:] = 0
    result.stiffness[:] = 0
    result.residual[:] = 0
    again = linearized_chain_dynamics(chain, poses)
    np.testing.assert_array_equal(again.mass, saved)
    np.testing.assert_array_equal(poses, before)
    assert np.linalg.norm(again.gyroscopic) > 0
    assert np.linalg.norm(again.stiffness) > 0
    assert np.linalg.norm(again.residual) > 0


def test_zero_spin_and_common_observer_rotation_preserve_material_operators() -> None:
    chain, poses = _chain(), _poses()
    # No applied observer-fixed wrenches in this objectivity control.
    chain = replace(chain, elastic=SectionChain(_sections(), ()))
    rotation = exp_twist([0, 0, 0, 0.4, -0.3, 0.2])
    frame = _frame()
    rotated = RotatingFrameState(
        "rotated-observer",
        rotation[:3, :3] @ frame.angular_velocity_rad_s,
        rotation[:3, :3] @ frame.angular_acceleration_rad_s2,
        rotation[:3, :3] @ frame.origin_acceleration_m_s2,
    )
    first = linearized_chain_dynamics(chain, poses)
    second = linearized_chain_dynamics(replace(chain, frame=rotated), rotation @ poses)
    for name in ("mass", "gyroscopic", "stiffness", "residual"):
        np.testing.assert_allclose(
            getattr(first, name), getattr(second, name), atol=2e-10
        )
    stationary = replace(frame, angular_velocity_rad_s=(0, 0, 0))
    zero = linearized_chain_dynamics(replace(chain, frame=stationary), poses)
    np.testing.assert_array_equal(zero.gyroscopic, 0)
    np.testing.assert_array_equal(zero.mass, first.mass)


def test_invalid_chain_or_pose_is_refused_and_singular_quadrature_is_not_repaired() -> (
    None
):
    chain, poses = _chain(), _poses()
    with pytest.raises(TypeError, match="RotatingSectionChain"):
        linearized_chain_dynamics(None, poses)
    with pytest.raises(ValueError, match="poses"):
        linearized_chain_dynamics(chain, poses[:2])
    bad = poses.copy()
    bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        linearized_chain_dynamics(chain, bad)
    lumped = SectionInertia((replace(_samples()[0], fraction=0),))
    result = linearized_chain_dynamics(replace(chain, inertias=(lumped,) * 2), poses)
    np.testing.assert_allclose(result.mass[-6:, -6:], 0, atol=1e-25)
