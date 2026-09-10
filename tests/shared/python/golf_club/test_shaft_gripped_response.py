"""Independent elastic-rod and power controls for finite-support response."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_gripped_response as response_module
from shared.python.golf_club._shaft_gripped_chain import GrippedSectionChain
from shared.python.golf_club._shaft_gripped_dynamics import linearized_gripped_dynamics
from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_gripped_response import gripped_tip_compliance

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_chain import _attachment
from .test_shaft_harmonic_response import _fixture, _request


def _model(count: int = 1, damping: float = 2.0, spin: float = 0) -> tuple:
    shaft, poses = _fixture(count, spin)
    attachment = _attachment(400)
    attachment = replace(
        attachment,
        grip=replace(
            attachment.grip,
            inertance_factor=np.diag([0, 0, np.sqrt(0.03), 0, 0, 0]),
            damping_factor=np.diag([0, 0, np.sqrt(damping), 0, 0, 0]),
        ),
    )
    return GrippedSectionChain(shaft, (attachment,)), poses


@pytest.mark.parametrize("frequency", [0, 10, 25, 50])
@pytest.mark.parametrize("damping", [0.0, 2.0])
def test_axial_transfer_matches_independent_two_node_boundary_pencil(
    frequency: float, damping: float
) -> None:
    chain, poses = _model(damping=damping)
    response = gripped_tip_compliance(chain, poses, _controls(), _request(frequency))
    rod = 1000 * np.array([[1, -1], [-1, 1]])
    mass = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    boundary = 400 + 1j * frequency * damping
    pencil = rod - frequency**2 * mass + np.diag([boundary, 0])
    expected = np.linalg.solve(pencil, [0, 1])
    np.testing.assert_allclose(response.nodal_displacement[[2, 8], 2], expected)
    assert response.displacement_compliance[2, 2] == pytest.approx(expected[1])
    support = -(boundary - frequency**2 * 0.03) * expected[0]
    assert response.grip_wrench_transfers[0][2, 2] == pytest.approx(support)
    np.testing.assert_allclose(
        response.velocity_mobility, 1j * frequency * response.displacement_compliance
    )
    np.testing.assert_array_equal(response.grip_wrenches[0], np.zeros(6))
    assert response.stability_status == "unqualified"
    assert response.frame == chain.shaft.frame


@pytest.mark.parametrize("frequency", [10, 25, 50])
def test_mesh_converges_to_rod_with_dynamic_root_and_tip_boundaries(
    frequency: float,
) -> None:
    # EA*u'' + mu*w^2*u=0; EA*u'(0)=dg*u(0);
    # EA*u'(L)-mt*w^2*u(L)=F. These are synthetic SI parameters.
    wave = frequency * np.sqrt(0.2 / 1000)
    support = 400 + 2j * frequency - 0.03 * frequency**2
    shape = np.cos(wave) + support * np.sin(wave) / (1000 * wave)
    denominator = (
        -1000 * wave * np.sin(wave)
        + support * np.cos(wave)
        - 0.1 * frequency**2 * shape
    )
    exact = shape / denominator
    errors = []
    for count in (2, 4, 8):
        chain, poses = _model(count)
        result = gripped_tip_compliance(chain, poses, _controls(), _request(frequency))
        errors.append(abs(result.displacement_compliance[2, 2] - exact))
    assert 3.7 < errors[0] / errors[1] < 4.3
    assert 3.7 < errors[1] / errors[2] < 4.3
    assert errors[-1] / abs(exact) < 0.002


def test_offset_wrench_mobility_and_cycle_power_are_conjugate() -> None:
    chain, poses = _model(2)
    offset = np.array([0.02, -0.04, 0.03])
    origin = gripped_tip_compliance(chain, poses, _controls(), _request())
    result = gripped_tip_compliance(
        chain, poses, _controls(), replace(_request(), point_offset_m=offset)
    )
    force = np.array([0.7, -0.2, 0.8])
    torque = np.array([0.02, 0.04, -0.01])
    load = np.r_[force, torque]
    nodal_load = np.r_[force, torque + np.cross(offset, force)]
    motion = origin.displacement_compliance @ nodal_load
    point = np.r_[motion[:3] + np.cross(motion[3:], offset), motion[3:]]
    np.testing.assert_allclose(result.displacement_compliance @ load, point, atol=1e-12)
    np.testing.assert_allclose(
        result.grip_wrench_transfers[0] @ load,
        origin.grip_wrench_transfers[0] @ nodal_load,
        atol=1e-11,
    )
    assert np.vdot(load, point) == pytest.approx(np.vdot(nodal_load, motion))
    operators = linearized_gripped_dynamics(chain, poses)
    nodal = result.nodal_displacement @ load
    loss = 0.5 * 25**2 * np.vdot(nodal, operators.damping @ nodal).real
    supplied = 0.5 * np.vdot(load, result.velocity_mobility @ load).real
    assert loss > 0
    assert supplied == pytest.approx(loss, rel=1e-10)
    np.testing.assert_allclose(
        result.displacement_compliance, result.displacement_compliance.T, atol=1e-12
    )


def test_repeated_interior_supports_retain_separate_physical_reactions() -> None:
    shaft, poses = _fixture(2)
    first = _attachment(100, 1)
    first = replace(first, anchor=replace(first.anchor, pose=poses[1]))
    second = replace(first, grip=_attachment(200).grip)
    chain = GrippedSectionChain(shaft, (first, second))
    response = gripped_tip_compliance(chain, poses, _controls(), _request(0))
    assert response.displacement_compliance[2, 2] == pytest.approx(1 / 300 + 0.5 / 1000)
    assert response.nodal_displacement[2, 2] == pytest.approx(1 / 300)
    assert response.grip_wrench_transfers[0][2, 2] == pytest.approx(-1 / 3)
    assert response.grip_wrench_transfers[1][2, 2] == pytest.approx(-2 / 3)


def test_coordinate_scale_preserves_si_response_and_fresh_arrays() -> None:
    chain, poses = _model(2)
    first = gripped_tip_compliance(chain, poses, _controls(), _request())
    second = gripped_tip_compliance(
        chain, poses, _controls(), replace(_request(), length_m=0.01)
    )
    np.testing.assert_allclose(
        first.displacement_compliance, second.displacement_compliance, atol=1e-12
    )
    assert first.scaled_rcond != second.scaled_rcond
    assert not np.shares_memory(first.nodal_displacement, second.nodal_displacement)


def test_unbalanced_root_and_strain_domain_are_rechecked() -> None:
    chain, poses = _model()
    poses[:, 2, 3] += 0.01  # Unstrained shaft, unbalanced finite root support.
    with pytest.raises(ValueError, match="balance"):
        gripped_tip_compliance(chain, poses, _controls(), _request())
    poses[-1, 2, 3] += 0.2
    with pytest.raises(ValueError, match="strain"):
        gripped_tip_compliance(chain, poses, _controls(), _request())


def test_singular_static_boundary_and_overflow_are_refused() -> None:
    chain, poses = _model()
    attachment = replace(
        chain.grips[0],
        grip=replace(chain.grips[0].grip, stiffness_factor=np.zeros((6, 6))),
    )
    with pytest.raises(ValueError, match="singular|condition"):
        gripped_tip_compliance(
            replace(chain, grips=(attachment,)), poses, _controls(), _request(0)
        )
    with pytest.raises(ValueError, match="numerical"):
        gripped_tip_compliance(chain, poses, _controls(), _request(1e308))


def test_wrong_controls_are_refused_at_the_boundary() -> None:
    chain, poses = _model()
    with pytest.raises(TypeError, match="TipHarmonicControls"):
        gripped_tip_compliance(chain, poses, _controls(), None)
    with pytest.raises(TypeError, match="EquilibriumControls"):
        gripped_tip_compliance(chain, poses, None, _request())


def test_rotating_loaded_pencil_preserves_transport_and_preload_reactions() -> None:
    chain, seed = _model(2, spin=10)
    root = solve_gripped_chain(chain, seed, _controls())
    operators = linearized_gripped_dynamics(chain, root.poses)
    result = gripped_tip_compliance(chain, root.poses, _controls(), _request())
    assert np.linalg.norm(operators.gyroscopic) > 0
    # Check the physical SI force balance after the solver's length scaling.
    # This integration check supplements the independent rod/power oracles.
    dynamic = (
        operators.stiffness
        - 25**2 * operators.mass
        + 25j * (operators.gyroscopic + operators.damping)
    )
    force = np.zeros((18, 6))
    force[-6:] = np.eye(6)
    np.testing.assert_allclose(dynamic @ result.nodal_displacement, force, atol=1e-11)
    np.testing.assert_allclose(
        result.grip_wrenches[0], root.grip_responses[0].root_wrench, atol=1e-11
    )
    assert result.frame.angular_velocity_rad_s == (10, 0, 0)
    assert result.stability_status == "unqualified"


def test_well_conditioned_cancellation_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, poses = _model()
    actual = linearized_gripped_dynamics(chain, poses)
    manufactured = replace(
        actual,
        stiffness=np.eye(12),
        mass=(1 - 1e-14) * np.eye(12),
        gyroscopic=np.zeros((12, 12)),
        damping=np.zeros((12, 12)),
    )
    monkeypatch.setattr(
        response_module, "balanced_gripped_dynamics", lambda *_: manufactured
    )
    with pytest.raises(ValueError, match="coefficient"):
        gripped_tip_compliance(chain, poses, _controls(), _request(1))
