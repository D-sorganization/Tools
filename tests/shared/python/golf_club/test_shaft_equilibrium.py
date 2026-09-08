"""Independent static benchmarks for the private, unqualified root solve."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_chain import IndexedPointLoad, SectionChain
from shared.python.golf_club._shaft_equilibrium import (
    EquilibriumControls,
    moving_residual_jacobian,
    solve_clamped_chain,
)
from shared.python.golf_club._shaft_point_load import SpatialPointLoad
from shared.python.golf_club._shaft_se3 import exp_twist
from shared.python.golf_club._shaft_section import SectionElement


def _controls() -> EquilibriumControls:
    return EquilibriumControls(
        strain_limits=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        force_tolerance_n=1e-8,
        moment_tolerance_nm=1e-9,
        translation_step_m=0.05,
        rotation_step_rad=0.1,
        max_iterations=30,
        max_backtracks=15,
    )


def _fixture(force: object, moment: object) -> tuple[SectionChain, np.ndarray]:
    sections = tuple(
        SectionElement(length, [0, 0, length, 0, 0, 0], np.diag([80, 90, 300, 3, 4, 2]))
        for length in (0.4, 0.6)
    )
    load = IndexedPointLoad(2, SpatialPointLoad(force, moment, [0, 0, 0]))
    poses = np.array([exp_twist([0, 0, z, 0, 0, 0]) for z in (0, 0.4, 1)])
    return SectionChain(sections, (load,)), poses


def test_axial_equilibrium_displacement_reaction_and_energy() -> None:
    chain, seed = _fixture([0, 0, 3], [0, 0, 0])
    original = seed.copy()
    result = solve_clamped_chain(chain, seed, _controls())
    np.testing.assert_allclose(result.poses[:, 2, 3], [0, 0.404, 1.01], atol=1e-10)
    np.testing.assert_allclose(result.support_wrench, [0, 0, -3, 0, 0, 0], atol=1e-8)
    assert result.elastic_energy_j == pytest.approx(3**2 / (2 * 300))
    assert result.force_residual_n <= 1e-8
    assert result.moment_residual_nm <= 1e-9
    assert result.stability_status == "unqualified"
    np.testing.assert_array_equal(seed, original)
    np.testing.assert_array_equal(result.poses[0], seed[0])


def test_pure_end_moment_recovers_constant_curvature_arc() -> None:
    chain, seed = _fixture([0, 0, 0], [0, 0.4, 0])
    result = solve_clamped_chain(chain, seed, _controls())
    curvature = 0.4 / 4
    for pose, arclength in zip(result.poses, (0, 0.4, 1), strict=True):
        # Independent circular centerline and rotation, not section interpolation.
        angle = curvature * arclength
        expected = [(1 - np.cos(angle)) / curvature, 0, np.sin(angle) / curvature]
        np.testing.assert_allclose(pose[:3, 3], expected, atol=2e-9)
        np.testing.assert_allclose(
            pose[:3, 2], [np.sin(angle), 0, np.cos(angle)], atol=2e-9
        )
    np.testing.assert_allclose(result.support_wrench, [0, 0, 0, 0, -0.4, 0], atol=1e-8)
    assert result.elastic_energy_j == pytest.approx(0.4**2 / (2 * 4), rel=1e-8)


def test_moving_residual_jacobian_matches_all_material_force_derivatives() -> None:
    chain, poses = _fixture([0.2, -0.1, 0.3], [0.1, 0.2, -0.1])
    poses[1] = poses[1] @ exp_twist([0.003, -0.001, 0.002, 0.01, 0.02, -0.01])
    response = chain.linearize(poses)
    derivative = moving_residual_jacobian(response)
    numerical = np.empty((18, 18))
    step = 1e-5
    for column, axis in enumerate(np.eye(18)):
        direction = axis.reshape(-1, 6)
        plus = np.array(
            [h @ exp_twist(step * a) for h, a in zip(poses, direction, strict=True)]
        )
        minus = np.array(
            [h @ exp_twist(-step * a) for h, a in zip(poses, direction, strict=True)]
        )
        numerical[:, column] = (
            chain.linearize(plus).residual - chain.linearize(minus).residual
        ) / (2 * step)
    np.testing.assert_allclose(derivative, numerical, rtol=2e-7, atol=1e-7)
    assert np.linalg.norm(response.tangent - numerical) > 0.1


def test_equilibrium_is_invariant_under_common_observer_motion() -> None:
    chain, seed = _fixture([0.05, 0.02, 1], [0.02, 0.04, 0.01])
    original = solve_clamped_chain(chain, seed, _controls())
    observer = exp_twist([0.3, -0.2, 0.8, 0.3, -0.2, 0.1])
    rotation = observer[:3, :3]
    load = chain.loads[0].load
    transformed = SectionChain(
        chain.sections,
        (
            IndexedPointLoad(
                2,
                SpatialPointLoad(
                    rotation @ load.force_n, rotation @ load.couple_nm, load.offset_m
                ),
            ),
        ),
    )
    result = solve_clamped_chain(transformed, observer @ seed, _controls())
    np.testing.assert_allclose(result.poses, observer @ original.poses, atol=3e-9)
    np.testing.assert_allclose(
        result.support_wrench, original.support_wrench, atol=2e-8
    )


def test_zero_load_returns_fresh_converged_seed_without_iteration() -> None:
    chain, seed = _fixture([0, 0, 0], [0, 0, 0])
    result = solve_clamped_chain(chain, seed, _controls())
    assert result.iterations == 0
    np.testing.assert_array_equal(result.poses, seed)
    assert not np.shares_memory(result.poses, seed)


def test_combined_axial_torsion_recovers_independent_section_strains() -> None:
    chain, seed = _fixture([0, 0, 3], [0, 0, 0.2])
    result = solve_clamped_chain(chain, seed, _controls())
    for pose, length in zip(result.poses, (0, 0.4, 1), strict=True):
        np.testing.assert_allclose(pose[:3, 3], [0, 0, 1.01 * length], atol=1e-10)
        angle = 0.1 * length
        np.testing.assert_allclose(
            pose[:3, 0], [np.cos(angle), np.sin(angle), 0], atol=1e-10
        )
    np.testing.assert_allclose(result.support_wrench, [0, 0, -3, 0, 0, -0.2], atol=1e-8)
    assert result.elastic_energy_j == pytest.approx(3**2 / 600 + 0.2**2 / 4)


def test_root_load_changes_reaction_without_changing_free_solution() -> None:
    chain, seed = _fixture([0, 0, 0], [0, 0, 0])
    chain = SectionChain(
        chain.sections,
        (IndexedPointLoad(0, SpatialPointLoad([1, 2, 3], [4, 5, 6], [0, 0, 0])),),
    )
    result = solve_clamped_chain(chain, seed, _controls())
    assert result.iterations == 0
    np.testing.assert_array_equal(result.poses, seed)
    np.testing.assert_array_equal(result.support_wrench, [-1, -2, -3, -4, -5, -6])


def test_control_limits_are_copied_and_malformed_seed_is_rejected() -> None:
    limits = [0.05] * 6
    controls = replace(_controls(), strain_limits=limits)
    limits[0] = 42
    assert controls.strain_limits[0] == 0.05
    chain, seed = _fixture([0, 0, 0], [0, 0, 0])
    seed[1, 0, 0] = 2
    with pytest.raises(ValueError, match="rotation"):
        solve_clamped_chain(chain, seed, controls)
    with pytest.raises(TypeError, match="SectionChain"):
        solve_clamped_chain(None, seed, controls)


def test_out_of_domain_seed_and_unreachable_load_fail_closed() -> None:
    chain, seed = _fixture([0, 0, 3], [0, 0, 0])
    controls = replace(_controls(), strain_limits=(0.001,) * 6)
    with pytest.raises(RuntimeError, match="converg|admissible"):
        solve_clamped_chain(chain, seed, controls)
    seed[2, 2, 3] += 0.1
    with pytest.raises(ValueError, match="strain"):
        solve_clamped_chain(chain, seed, controls)


def test_iteration_budget_never_returns_an_unfinished_state() -> None:
    chain, seed = _fixture([0.2, 0, 1], [0, 0.4, 0])
    with pytest.raises(RuntimeError, match="converg"):
        solve_clamped_chain(chain, seed, replace(_controls(), max_iterations=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("force_tolerance_n", 0),
        ("moment_tolerance_nm", float("nan")),
        ("translation_step_m", -1),
        ("rotation_step_rad", np.pi),
        ("max_iterations", True),
        ("max_backtracks", 0),
        ("strain_limits", [1] * 5),
        ("strain_limits", [True] * 6),
        ("strain_limits", [0] * 6),
    ],
)
def test_invalid_controls_are_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), **{field: value})


def test_section_strain_is_fresh_and_uses_material_reference() -> None:
    chain, seed = _fixture([0, 0, 0], [0, 0, 0])
    section = chain.sections[0]
    right = exp_twist([0, 0, 0.404, 0, 0, 0])
    strain = section.strain(seed[0], right)
    np.testing.assert_allclose(strain, [0, 0, 0.01, 0, 0, 0], atol=1e-14)
    strain[:] = 42
    assert section.strain(seed[0], right)[2] == pytest.approx(0.01)
