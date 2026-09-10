"""Energy and virtual-work oracles for the finite-rotation elastic element."""

import numpy as np
import pytest

from shared.python.golf_club._shaft_se3 import exp_twist
from shared.python.golf_club._shaft_section import SectionElement


def _element() -> SectionElement:
    # Synthetic anisotropic, coupled stiffness; not an identified golf shaft.
    factor = np.diag([4.0, 5.0, 20.0, 2.0, 3.0, 1.5])
    factor[0, 4], factor[2, 5] = 0.3, -0.8
    return SectionElement(0.8, [0, 0, 0.8, 0, 0, 0], factor.T @ factor)


def _loaded_poses() -> tuple[np.ndarray, np.ndarray]:
    left = exp_twist([0.1, -0.2, 0.3, 0.3, -0.2, 0.1])
    return left, left @ exp_twist([0.02, -0.01, 0.82, 0.2, -0.3, 0.15])


def _energy_at(
    element: SectionElement, poses: tuple[np.ndarray, np.ndarray], q: np.ndarray
) -> float:
    return float(
        element.energy(poses[0] @ exp_twist(q[:6]), poses[1] @ exp_twist(q[6:]))
    )


@pytest.mark.parametrize("component", [0, 1, 2, 3, 4, 5])
def test_constant_strain_energy_has_correct_length_and_si_scaling(
    component: int,
) -> None:
    length, strain = 0.7, 0.012
    stiffness = np.diag([90, 100, 500, 2, 3, 4])
    reference = np.array([0, 0, length, 0, 0, 0])
    loaded = reference.copy()
    loaded[component] += length * strain
    element = SectionElement(length, reference, stiffness)
    assert element.energy(np.eye(4), exp_twist(reference)) == pytest.approx(
        0, abs=1e-25
    )
    assert element.energy(np.eye(4), exp_twist(loaded)) == pytest.approx(
        0.5 * length * stiffness[component, component] * strain**2, rel=2e-12
    )


def test_loaded_energy_gradient_matches_independent_virtual_work() -> None:
    element, poses = _element(), _loaded_poses()
    response = element.linearize(*poses)
    step = 1e-6
    numerical = np.array(
        [
            (
                _energy_at(element, poses, step * axis)
                - _energy_at(element, poses, -step * axis)
            )
            / (2 * step)
            for axis in np.eye(12)
        ]
    )
    np.testing.assert_allclose(response.gradient, numerical, rtol=2e-7, atol=2e-9)
    assert response.energy_j == pytest.approx(element.energy(*poses))


def test_complete_loaded_tangent_matches_fixed_chart_energy_curvature() -> None:
    element, poses = _element(), _loaded_poses()
    response = element.linearize(*poses)
    # Scalar-energy second differences do not reuse the analytic force mapping.
    step = 2e-4
    axes = np.eye(12)
    numerical = np.array(
        [
            [
                sum(
                    sign_i
                    * sign_j
                    * _energy_at(
                        element, poses, step * (sign_i * row + sign_j * column)
                    )
                    for sign_i in (-1, 1)
                    for sign_j in (-1, 1)
                )
                / (4 * step**2)
                for column in axes
            ]
            for row in axes
        ]
    )
    np.testing.assert_allclose(response.tangent, numerical, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(response.tangent, response.tangent.T, rtol=0, atol=2e-12)
    # Prestress terms are material to this fixture, not symmetrized away.
    assert np.linalg.norm(response.tangent - response.material_tangent) > 0.5


def test_observer_change_preserves_body_chart_energy_gradient_and_tangent() -> None:
    element, poses = _element(), _loaded_poses()
    observer = exp_twist([-0.4, 1.2, 0.5, -0.5, 0.3, 0.8])
    before = element.linearize(*poses)
    after = element.linearize(observer @ poses[0], observer @ poses[1])
    assert after.energy_j == pytest.approx(before.energy_j, rel=1e-13)
    np.testing.assert_allclose(after.gradient, before.gradient, atol=1e-12)
    np.testing.assert_allclose(after.tangent, before.tangent, atol=1e-11)


def test_unloaded_element_has_six_rigid_modes_and_positive_strain_modes() -> None:
    element = _element()
    response = element.linearize(np.eye(4), exp_twist([0, 0, 0.8, 0, 0, 0]))
    eigenvalues = np.linalg.eigvalsh(response.tangent)
    np.testing.assert_allclose(eigenvalues[:6], 0, atol=1e-12)
    assert np.all(eigenvalues[6:] > 1)
    np.testing.assert_allclose(response.gradient, 0, atol=1e-12)


@pytest.mark.parametrize("axis", np.eye(6))
def test_loaded_rigid_motion_performs_no_work_or_energy_curvature(
    axis: np.ndarray,
) -> None:
    poses = _loaded_poses()
    response = _element().linearize(*poses)
    # Conjugation H^-1 Exp(t*s) H gives the exact local straight chart path.
    direction = np.concatenate(
        [
            np.r_[
                pose[:3, :3].T @ (axis[:3] + np.cross(axis[3:], pose[:3, 3])),
                pose[:3, :3].T @ axis[3:],
            ]
            for pose in poses
        ]
    )
    assert response.gradient @ direction == pytest.approx(0, abs=2e-13)
    assert direction @ response.tangent @ direction == pytest.approx(0, abs=2e-12)


@pytest.mark.parametrize("curvature", [1e-12, 0.2, 2.0])
def test_pure_bending_keeps_shear_and_axial_energy_zero(curvature: float) -> None:
    length, bending_stiffness = 0.8, 3.0
    # Make spurious shear/axial strain expensive without imposing infinite GA.
    stiffness = np.diag([1e7, 1e7, 1e8, 2, bending_stiffness, 4])
    element = SectionElement(length, [0, 0, length, 0, 0, 0], stiffness)
    pose = exp_twist([0, 0, length, 0, length * curvature, 0])
    assert element.energy(np.eye(4), pose) == pytest.approx(
        0.5 * length * bending_stiffness * curvature**2, rel=3e-13, abs=1e-24
    )


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "0.8"])
def test_length_contract(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SectionElement(value, [0, 0, 0.8, 0, 0, 0], np.eye(6))


@pytest.mark.parametrize(
    "kind", ["asymmetric", "singular", "negative", "nan", "boolean"]
)
def test_stiffness_contract(kind: str) -> None:
    stiffness = np.eye(6)
    if kind == "asymmetric":
        stiffness[0, 1] = 0.01
    elif kind == "singular":
        stiffness[0, 0] = 0
    elif kind == "negative":
        stiffness[0, 0] = -1
    elif kind == "nan":
        stiffness[0, 0] = np.nan
    else:
        stiffness = stiffness.astype(bool)
    with pytest.raises((TypeError, ValueError)):
        SectionElement(0.8, [0, 0, 0.8, 0, 0, 0], stiffness)


def test_section_copies_inputs_and_returns_fresh_derivatives() -> None:
    reference, stiffness = np.array([0, 0, 0.8, 0, 0, 0]), np.eye(6)
    element = SectionElement(0.8, reference, stiffness)
    reference[:] = 100
    stiffness[:] = 0
    poses = _loaded_poses()
    response = element.linearize(*poses)
    original = response.gradient.copy()
    response.gradient[:] = 0
    np.testing.assert_array_equal(element.linearize(*poses).gradient, original)
