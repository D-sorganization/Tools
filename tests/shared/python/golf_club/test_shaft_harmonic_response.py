"""Continuum, force/torque port and refusal controls for frozen tip response."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_harmonic_response as harmonic_module
from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_body_assembly import attach_nodal_body
from shared.python.golf_club._shaft_equilibrium import solve_clamped_chain
from shared.python.golf_club._shaft_harmonic_response import (
    TipHarmonicControls,
    clamped_tip_compliance,
)
from shared.python.golf_club._shaft_loaded_dynamics import LoadedChainDynamics

from .test_shaft_body_assembly import _body, _principal_head
from .test_shaft_equilibrium import _controls
from .test_shaft_rotating_chain import _radial_rod


def _request(frequency: float = 25) -> TipHarmonicControls:
    return TipHarmonicControls(frequency, (0, 0, 0), 1.0, 1e-12, 1e-10)


def _fixture(count: int = 1, spin: float = 0) -> tuple:
    frame = RotatingFrameState("observer", (spin, 0, 0), (0, 0, 0), (0, 0, 0))
    bare, poses = _radial_rod(count, frame)
    return attach_nodal_body(bare, count, _principal_head()), poses


@pytest.mark.parametrize("frequency", [0, 25, 50])
def test_one_element_axial_compliance_and_support_follow_rod_balance(
    frequency: float,
) -> None:
    chain, poses = _fixture()
    response = clamped_tip_compliance(chain, poses, _controls(), _request(frequency))
    # One linear rod: M_tip = mu*L/3 + mt; M_root,tip = mu*L/6.
    expected = 1 / (1000 - frequency**2 * (0.2 / 3 + 0.1))
    support = -(1000 + frequency**2 * 0.2 / 6) * expected
    assert response.displacement_compliance[2, 2] == pytest.approx(expected)
    assert response.support_wrench_transfer[2, 2] == pytest.approx(support)
    np.testing.assert_allclose(
        response.velocity_mobility,
        1j * frequency * response.displacement_compliance,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        response.displacement_compliance, response.displacement_compliance.T, atol=1e-12
    )
    np.testing.assert_array_equal(response.support_wrench, np.zeros(6))
    assert np.max(response.relative_residuals) < 1e-10
    assert response.stability_status == "unqualified"


@pytest.mark.parametrize("frequency", [25, 50])
def test_tip_compliance_converges_to_continuum_dynamic_boundary(
    frequency: float,
) -> None:
    # EA phi'' + mu*w^2 phi=0; EA phi'(L)-mt*w^2 phi(L)=P; phi(0)=0.
    wave = frequency * np.sqrt(0.2 / 1000)
    exact = np.sin(wave) / (
        1000 * wave * np.cos(wave) - 0.1 * frequency**2 * np.sin(wave)
    )
    errors = []
    for count in (2, 4, 8):
        chain, poses = _fixture(count)
        result = clamped_tip_compliance(chain, poses, _controls(), _request(frequency))
        errors.append(abs(result.displacement_compliance[2, 2] - exact))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] / abs(exact) < 0.002


def test_offset_port_transfers_force_moment_motion_and_power() -> None:
    frame = RotatingFrameState("observer", (10, 0, 0), (0, 0, 0), (0, 0, 0))
    bare, seed = _radial_rod(2, frame)
    chain = attach_nodal_body(bare, 2, _body())
    root = solve_clamped_chain(chain, seed, _controls())
    origin = clamped_tip_compliance(chain, root.poses, _controls(), _request())
    offset = np.array([0.02, -0.04, 0.03])
    point = clamped_tip_compliance(
        chain, root.poses, _controls(), replace(_request(), point_offset_m=offset)
    )
    force, torque = np.array([0.7, -0.2, 0.8]), np.array([0.02, 0.04, -0.01])
    nodal_load = np.r_[force, torque + np.cross(offset, force)]
    motion = origin.displacement_compliance @ nodal_load
    expected = np.r_[motion[:3] + np.cross(motion[3:], offset), motion[3:]]
    load = np.r_[force, torque]
    np.testing.assert_allclose(
        point.displacement_compliance @ load, expected, atol=1e-11
    )
    np.testing.assert_allclose(
        point.support_wrench_transfer @ load,
        origin.support_wrench_transfer @ nodal_load,
        atol=1e-10,
    )
    assert np.vdot(load, expected) == pytest.approx(np.vdot(nodal_load, motion))
    assert abs(np.real(np.vdot(load, point.velocity_mobility @ load))) < 1e-9
    np.testing.assert_allclose(point.support_wrench, root.support_wrench, atol=1e-10)


def test_coordinate_scale_changes_conditioning_without_changing_response() -> None:
    chain, poses = _fixture(2)
    first = clamped_tip_compliance(chain, poses, _controls(), _request())
    second = clamped_tip_compliance(
        chain, poses, _controls(), replace(_request(), length_m=0.01)
    )
    np.testing.assert_allclose(
        first.displacement_compliance, second.displacement_compliance, atol=1e-10
    )
    assert first.scaled_rcond != second.scaled_rcond


def test_unresolved_resonance_is_refused_without_added_damping() -> None:
    chain, poses = _fixture()
    resonant = np.sqrt(1000 / (0.2 / 3 + 0.1))
    with pytest.raises(ValueError, match="condition|singular"):
        clamped_tip_compliance(chain, poses, _controls(), _request(resonant))


def test_well_conditioned_dynamic_matrix_can_hide_unresolved_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D=(1-(1-epsilon))*I has condition number one but loses its coefficient
    # accuracy near coincident resonances. A small linear-solve residual is
    # insufficient to resolve this pencil relative to the supplied M and K.
    chain, poses = _fixture()
    mass, stiffness = np.zeros((12, 12)), np.zeros((12, 12))
    mass[6:, 6:] = (1 - 1e-14) * np.eye(6)
    stiffness[6:, 6:] = np.eye(6)
    operators = LoadedChainDynamics(np.zeros(12), mass, np.zeros((12, 12)), stiffness)
    monkeypatch.setattr(
        harmonic_module, "balanced_clamped_dynamics", lambda *_: operators
    )
    with pytest.raises(ValueError, match="coefficient"):
        clamped_tip_compliance(chain, poses, _controls(), _request(1))


def test_frequency_square_overflow_is_refused() -> None:
    chain, poses = _fixture()
    with pytest.raises(ValueError, match="numerical"):
        clamped_tip_compliance(chain, poses, _controls(), _request(1e308))


def test_unbalanced_and_out_of_domain_shapes_are_refused() -> None:
    chain, poses = _fixture(spin=10)
    with pytest.raises(ValueError, match="balance"):
        clamped_tip_compliance(chain, poses, _controls(), _request())
    poses[-1, 2, 3] += 0.2
    with pytest.raises(ValueError, match="strain"):
        clamped_tip_compliance(chain, poses, _controls(), _request())


@pytest.mark.parametrize(
    "changes",
    [
        {"angular_frequency_rad_s": -1},
        {"angular_frequency_rad_s": True},
        {"angular_frequency_rad_s": 1j},
        {"point_offset_m": [False, 0, 0]},
        {"point_offset_m": [0, 0]},
        {"length_m": 0},
        {"dynamic_rcond_floor": 1},
        {"residual_tolerance": 0},
    ],
)
def test_malformed_harmonic_controls_are_refused(changes: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_request(), **changes)
