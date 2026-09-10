"""Independent work and equilibrium identities for assembled section chains."""

import numpy as np
import pytest

from shared.python.golf_club._shaft_chain import IndexedPointLoad, SectionChain
from shared.python.golf_club._shaft_point_load import SpatialPointLoad
from shared.python.golf_club._shaft_se3 import exp_twist
from shared.python.golf_club._shaft_section import SectionElement


def _sections() -> tuple[SectionElement, ...]:
    stiffness = np.diag([50.0, 60.0, 300.0, 3.0, 4.0, 2.0])
    return tuple(
        SectionElement(length, [0, 0, length, 0, 0, 0], stiffness)
        for length in (0.4, 0.6)
    )


def _poses() -> np.ndarray:
    first = exp_twist([0.1, -0.2, 0.3, 0.2, -0.1, 0.3])
    second = first @ exp_twist([0.01, -0.02, 0.405, 0.06, -0.04, 0.02])
    third = second @ exp_twist([-0.02, 0.01, 0.607, -0.04, 0.08, -0.03])
    return np.array([first, second, third])


def _loads(couple: bool = False) -> tuple[IndexedPointLoad, ...]:
    return (
        IndexedPointLoad(1, SpatialPointLoad([0.2, -0.1, 0.3], [0, 0, 0], [0, 0, 0])),
        IndexedPointLoad(
            2, SpatialPointLoad([0.1, 0.3, -0.2], [0, 0, 0], [0.02, 0, 0.01])
        ),
        IndexedPointLoad(
            2,
            SpatialPointLoad(
                [-0.3, 0.1, 0.2],
                [0.1, -0.2, 0.3] if couple else [0, 0, 0],
                [0, 0.03, 0],
            ),
        ),
    )


def _potential(
    chain: SectionChain, poses: np.ndarray, coordinates: np.ndarray
) -> float:
    moved = np.array(
        [
            pose @ exp_twist(q)
            for pose, q in zip(poses, coordinates.reshape(-1, 6), strict=True)
        ]
    )
    elastic = sum(
        section.energy(moved[i], moved[i + 1])
        for i, section in enumerate(chain.sections)
    )
    # Independent spatial dead-force potential, not the production load method.
    external = sum(
        -np.asarray(item.load.force_n)
        @ (moved[item.node, :3, 3] + moved[item.node, :3, :3] @ item.load.offset_m)
        for item in chain.loads
    )
    return float(elastic + external)


def test_assembled_dead_force_residual_and_tangent_match_scalar_energy() -> None:
    chain, poses = SectionChain(_sections(), _loads()), _poses()
    result = chain.linearize(poses)
    step, zero = 2e-5, np.zeros(18)

    def energy(q: np.ndarray) -> float:
        return _potential(chain, poses, q)

    gradient = np.array(
        [(energy(step * a) - energy(-step * a)) / (2 * step) for a in np.eye(18)]
    )
    np.testing.assert_allclose(result.residual, gradient, rtol=1e-7, atol=2e-8)
    assert result.elastic_energy_j + result.force_potential_j == pytest.approx(
        energy(zero)
    )
    # Three mixed directions exercise every row, inter-node block and repeated load.
    for direction in np.random.default_rng(917).normal(size=(3, 18)):
        expected = (
            energy(step * direction) - 2 * energy(zero) + energy(-step * direction)
        ) / step**2
        assert direction @ result.tangent @ direction == pytest.approx(
            expected, rel=3e-6, abs=2e-5
        )
    np.testing.assert_allclose(result.tangent, result.tangent.T, atol=2e-12)


def test_every_chain_tangent_entry_matches_potential_curvature() -> None:
    chain, poses, step = SectionChain(_sections(), _loads()), _poses(), 2e-4
    tangent = chain.linearize(poses).tangent
    numerical = np.empty_like(tangent)
    for row, row_axis in enumerate(np.eye(18)):
        for column, column_axis in enumerate(np.eye(18)):
            numerical[row, column] = sum(
                sign_r
                * sign_c
                * _potential(
                    chain, poses, step * (sign_r * row_axis + sign_c * column_axis)
                )
                for sign_r, sign_c in ((-1, -1), (-1, 1), (1, -1), (1, 1))
            ) / (4 * step**2)
    np.testing.assert_allclose(tangent, numerical, rtol=4e-5, atol=5e-6)


def test_unloaded_free_chain_retains_six_rigid_modes() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = [0, 0.4, 1.0]
    response = SectionChain(_sections(), ()).linearize(poses)
    eigenvalues = np.linalg.eigvalsh(response.tangent)
    np.testing.assert_allclose(eigenvalues[:6], 0, atol=1e-12)
    assert np.all(eigenvalues[6:] > 1)
    np.testing.assert_allclose(response.residual, 0, atol=1e-12)


def test_repeated_node_couples_retain_their_nonsymmetric_load_tangent() -> None:
    poses = _poses()
    unloaded = SectionChain(_sections(), ()).linearize(poses)
    chain = SectionChain(_sections(), _loads(True))
    response = chain.linearize(poses)
    expected_force, expected_tangent = unloaded.residual.copy(), unloaded.tangent.copy()
    for item in chain.loads:
        load = item.load.linearize(poses[item.node])
        rows = slice(6 * item.node, 6 * (item.node + 1))
        expected_force[rows] -= load.wrench
        expected_tangent[rows, rows] -= load.tangent
    np.testing.assert_allclose(response.residual, expected_force, atol=1e-13)
    np.testing.assert_allclose(response.tangent, expected_tangent, atol=1e-13)
    assert np.linalg.norm(response.tangent - response.tangent.T) > 0.1


def test_two_axial_elements_recover_common_force_and_support_reaction() -> None:
    sections, tension = _sections(), 3.0
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, 2, 3] = 0.4 * (1 + tension / 300)
    poses[2, 2, 3] = 1.0 * (1 + tension / 300)
    load = IndexedPointLoad(2, SpatialPointLoad([0, 0, tension], [0, 0, 0], [0, 0, 0]))
    response = SectionChain(sections, (load,)).linearize(poses)
    np.testing.assert_allclose(response.residual[6:], 0, atol=3e-13)
    np.testing.assert_allclose(
        response.residual[:6], [0, 0, -tension, 0, 0, 0], atol=3e-13
    )
    assert response.elastic_energy_j == pytest.approx(tension**2 / (2 * 300))


def test_common_observer_change_preserves_residual_tangent_and_force_work() -> None:
    poses, loads = _poses(), _loads(True)
    observer = exp_twist([0.4, -0.3, 0.8, -0.2, 0.6, 0.1])
    rotated = tuple(
        IndexedPointLoad(
            item.node,
            SpatialPointLoad(
                observer[:3, :3] @ item.load.force_n,
                observer[:3, :3] @ item.load.couple_nm,
                item.load.offset_m,
            ),
        )
        for item in loads
    )
    before = SectionChain(_sections(), loads).linearize(poses)
    after = SectionChain(_sections(), rotated).linearize(observer @ poses)
    np.testing.assert_allclose(after.residual, before.residual, atol=2e-12)
    np.testing.assert_allclose(after.tangent, before.tangent, atol=2e-11)
    assert after.elastic_energy_j == pytest.approx(before.elastic_energy_j, rel=1e-12)
    shift = sum(
        np.asarray(item.load.force_n) @ observer[:3, :3].T @ observer[:3, 3]
        for item in loads
    )
    assert after.force_potential_j == pytest.approx(before.force_potential_j - shift)


@pytest.mark.parametrize("node", [-1, True, 1.0, "1"])
def test_load_node_refuses_invalid_indices(node: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        IndexedPointLoad(node, _loads()[0].load)


def test_chain_requires_sections_and_in_range_typed_loads() -> None:
    with pytest.raises(ValueError, match="section"):
        SectionChain((), ())
    with pytest.raises(TypeError, match="section"):
        SectionChain((None,), ())
    with pytest.raises(TypeError, match="load"):
        IndexedPointLoad(0, None)
    with pytest.raises(TypeError, match="load"):
        SectionChain(_sections(), (None,))
    with pytest.raises(ValueError, match="node"):
        SectionChain(_sections(), (IndexedPointLoad(3, _loads()[0].load),))


def test_chain_copies_inputs_and_returns_fresh_arrays() -> None:
    sections, loads, poses = list(_sections()), list(_loads()), _poses()
    chain = SectionChain(sections, loads)
    before = poses.copy()
    first = chain.linearize(poses)
    sections.clear()
    loads.clear()
    first.residual[:] = 0
    first.tangent[:] = 0
    second = chain.linearize(poses)
    assert np.linalg.norm(second.residual) > 0.1
    assert np.linalg.norm(second.tangent) > 1
    np.testing.assert_array_equal(poses, before)
    assert chain.node_count == 3


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 4, 4)),
        np.zeros((3, 4, 4)),
        np.full((3, 4, 4), np.nan),
        np.full((3, 4, 4), True),
    ],
)
def test_chain_refuses_invalid_pose_collections(bad: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        SectionChain(_sections(), ()).linearize(bad)
