"""Rigid-head mechanics and continuum tip-mass controls for loaded shafts."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq

from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_body_assembly import attach_nodal_body
from shared.python.golf_club._shaft_equilibrium import solve_clamped_chain
from shared.python.golf_club._shaft_loaded_dynamics import linearized_chain_dynamics
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole

from .test_shaft_chain import _poses
from .test_shaft_equilibrium import _controls
from .test_shaft_frame_inertia import _frame
from .test_shaft_inertia import _moved
from .test_shaft_rotating_chain import _radial_rod


def _body() -> ComponentMassProperties:
    return ComponentMassProperties(
        "synthetic-head",
        ComponentRole.HEAD,
        "material",
        0.1,
        (0.04, -0.02, 0.03),
        ((0.004, 0.0003, -0.0002), (0.0003, 0.005, 0.0004), (-0.0002, 0.0004, 0.006)),
    )


def _wrench(
    body: ComponentMassProperties, pose: np.ndarray, frame: RotatingFrameState
) -> np.ndarray:
    rotation = pose[:3, :3]
    offset = rotation @ body.center_of_mass_m
    center = pose[:3, 3] + offset
    inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
    omega, alpha = frame.angular_velocity_rad_s, frame.angular_acceleration_rad_s2
    force = body.mass_kg * (
        frame.origin_acceleration_m_s2
        + np.cross(alpha, center)
        + np.cross(omega, np.cross(omega, center))
    )
    torque = (
        np.cross(offset, force) + inertia @ alpha + np.cross(omega, inertia @ omega)
    )
    return np.r_[rotation.T @ force, rotation.T @ torque]


@pytest.mark.parametrize("node", [0, 1, 2, np.int64(2)])
def test_attachment_is_local_and_preserves_full_rigid_body_inertia(node: int) -> None:
    bare, _ = _radial_rod(2, _frame())
    body, poses = _body(), _poses()
    attached = attach_nodal_body(bare, node, body)
    before = linearized_chain_dynamics(bare, poses)
    after = linearized_chain_dynamics(attached, poses)
    expected = np.zeros(18)
    expected[6 * node : 6 * node + 6] = _wrench(body, poses[node], _frame())
    np.testing.assert_allclose(after.residual - before.residual, expected, atol=1e-12)
    motion = np.linspace(-0.3, 0.8, 18)
    local = motion[6 * node : 6 * node + 6]
    center_velocity = local[:3] + np.cross(local[3:], body.center_of_mass_m)
    energy = (
        body.mass_kg * center_velocity @ center_velocity
        + local[3:] @ body.inertia_at_com_kg_m2 @ local[3:]
    ) / 2
    assert motion @ (after.mass - before.mass) @ motion / 2 == pytest.approx(
        energy, abs=2e-15
    )
    assert attached.elastic is bare.elastic and attached.frame is bare.frame
    assert sum(len(item.samples) for item in attached.inertias) == 5
    assert sum(len(item.samples) for item in bare.inertias) == 4


def test_offset_head_tangent_and_coriolis_wrench_follow_physical_motion() -> None:
    bare, _ = _radial_rod(2, _frame())
    body, poses = _body(), _poses()
    chain = attach_nodal_body(bare, 2, body)
    before, after = (
        linearized_chain_dynamics(bare, poses),
        linearized_chain_dynamics(chain, poses),
    )
    direction = np.linspace(-0.2, 0.3, 18)
    plus, minus = _moved(poses, direction, 1e-5), _moved(poses, direction, -1e-5)
    expected = np.zeros(18)
    expected[-6:] = (
        _wrench(body, plus[-1], _frame()) - _wrench(body, minus[-1], _frame())
    ) / 2e-5
    np.testing.assert_allclose(
        (after.stiffness - before.stiffness) @ direction, expected, atol=2e-10
    )
    rotation = poses[-1, :3, :3]
    offset = rotation @ body.center_of_mass_m
    inertia = rotation @ body.inertia_at_com_kg_m2 @ rotation.T
    omega = np.asarray(_frame().angular_velocity_rad_s)
    angular = rotation @ direction[-3:]
    velocity = rotation @ direction[-6:-3] + np.cross(angular, offset)
    force = 2 * body.mass_kg * np.cross(omega, velocity)
    torque = (
        np.cross(offset, force)
        + inertia @ np.cross(omega, angular)
        + np.cross(angular, inertia @ omega)
        + np.cross(omega, inertia @ angular)
    )
    expected[-6:] = np.r_[rotation.T @ force, rotation.T @ torque]
    np.testing.assert_allclose(
        (after.gyroscopic - before.gyroscopic) @ direction, expected, atol=2e-14
    )
    assert abs(direction @ (after.gyroscopic - before.gyroscopic) @ direction) < 1e-14


def _principal_head() -> ComponentMassProperties:
    # The one-dimensional reference requires spin about a principal head axis:
    # products of inertia would apply Omega x (I Omega) and bend/twist the rod.
    head = replace(
        _body(),
        center_of_mass_m=(0, 0, 0),
        inertia_at_com_kg_m2=tuple(map(tuple, np.diag([0.004, 0.005, 0.006]))),
    )
    omega = np.array([10, 0, 0])
    np.testing.assert_array_equal(
        np.cross(omega, head.inertia_at_com_kg_m2 @ omega), np.zeros(3)
    )
    return head


def test_tip_mass_changes_loaded_extension_support_and_guided_frequency() -> None:
    # L=1, EA=1000, mu=.2, tip mass=.1, Omega=10: synthetic axial control.
    frame = RotatingFrameState("observer", (10, 0, 0), (0, 0, 0), (0, 0, 0))
    head = _principal_head()
    wave = 10 * np.sqrt(0.2 / 1000)
    amplitude = 1 / (wave * np.cos(wave) - head.mass_kg * 10**2 / 1000 * np.sin(wave))
    exact_tip = amplitude * np.sin(wave)
    exact_support = -1000 * (amplitude * wave - 1)
    root_number = brentq(
        lambda x: x * np.tan(x) - 0.2 / head.mass_kg, 1e-6, np.pi / 2 - 1e-6
    )
    exact_squared = 1000 / 0.2 * root_number**2 - 10**2
    position_errors, frequency_errors = [], []
    for count in (2, 4, 8):
        bare, seed = _radial_rod(count, frame)
        chain = attach_nodal_body(bare, count, head)
        root = solve_clamped_chain(chain, seed, _controls())
        operators = linearized_chain_dynamics(chain, root.poses)
        axial = np.arange(8, 6 * (count + 1), 6)
        frequency_squared = eigh(
            operators.stiffness[np.ix_(axial, axial)],
            operators.mass[np.ix_(axial, axial)],
            eigvals_only=True,
        )[0]
        position_errors.append(abs(root.poses[-1, 2, 3] - exact_tip))
        frequency_errors.append(abs(frequency_squared - exact_squared))
    for errors in (position_errors, frequency_errors):
        assert 3.8 < errors[0] / errors[1] < 4.2
        assert 3.8 < errors[1] / errors[2] < 4.2
    assert position_errors[-1] < 1e-6
    assert frequency_errors[-1] / exact_squared < 0.002
    assert root.support_wrench[2] == pytest.approx(exact_support, abs=0.002)


@pytest.mark.parametrize(
    "node,error",
    [
        (-1, ValueError),
        (3, ValueError),
        (True, TypeError),
        (np.bool_(False), TypeError),
        (1.0, TypeError),
    ],
)
def test_invalid_node_is_refused(node: object, error: type[Exception]) -> None:
    chain, _ = _radial_rod(2, _frame())
    with pytest.raises(error):
        attach_nodal_body(chain, node, _body())


def test_malformed_attachment_and_material_frame_mismatch_are_refused() -> None:
    chain, _ = _radial_rod(2, _frame())
    with pytest.raises(TypeError):
        attach_nodal_body(object(), 1, _body())
    with pytest.raises(TypeError):
        attach_nodal_body(chain, 1, object())
    with pytest.raises(ValueError, match="frame"):
        attach_nodal_body(chain, 1, replace(_body(), frame_id="head-local"))
