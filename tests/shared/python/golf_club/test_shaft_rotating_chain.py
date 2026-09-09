"""Continuum and force-balance oracles for rotating loaded-chain roots."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_chain import SectionChain
from shared.python.golf_club._shaft_equilibrium import (
    moving_residual_jacobian,
    solve_clamped_chain,
)
from shared.python.golf_club._shaft_inertia import InertiaSample, SectionInertia
from shared.python.golf_club._shaft_rotating_chain import RotatingSectionChain
from shared.python.golf_club._shaft_se3 import exp_twist
from shared.python.golf_club._shaft_section import SectionElement
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole

from .test_shaft_chain import _loads, _poses, _sections
from .test_shaft_equilibrium import _controls, _fixture
from .test_shaft_frame_inertia import _frame
from .test_shaft_inertia import _moved, _samples


def _radial_rod(
    count: int, frame: RotatingFrameState
) -> tuple[RotatingSectionChain, np.ndarray]:
    """Synthetic L=1 m, EA=1000 N, reference mass/length=0.2 kg/m rod."""
    length = 1 / count
    section = SectionElement(
        length, [0, 0, length, 0, 0, 0], np.diag([500, 500, 1000, 10, 10, 5])
    )
    points, weights = np.polynomial.legendre.leggauss(2)
    samples = tuple(
        InertiaSample(
            float((point + 1) / 2),
            ComponentMassProperties(
                "synthetic-section",
                ComponentRole.SHAFT,
                "material",
                0.2 * length * weight / 2,
                (0, 0, 0),
                tuple(map(tuple, np.diag([1e-4, 1e-4, 2e-4]) * length * weight / 2)),
            ),
        )
        for point, weight in zip(points, weights, strict=True)
    )
    chain = SectionChain((section,) * count, ())
    poses = np.array(
        [exp_twist([0, 0, z, 0, 0, 0]) for z in np.linspace(0, 1, count + 1)]
    )
    return RotatingSectionChain(chain, (SectionInertia(samples),) * count, frame), poses


def test_zero_frame_preserves_static_loads_tangents_and_existing_root_solver() -> None:
    elastic, seed = _fixture([0, 0, 3], [0, 0, 0.02])
    frame = RotatingFrameState("observer", (0, 0, 0), (0, 0, 0), (0, 0, 0))
    rotating = RotatingSectionChain(elastic, (SectionInertia(_samples()),) * 2, frame)
    before, after = elastic.linearize(seed), rotating.linearize(seed)
    np.testing.assert_array_equal(before.residual, after.residual)
    np.testing.assert_array_equal(before.tangent, after.tangent)
    static_root = solve_clamped_chain(elastic, seed, _controls())
    loaded_root = solve_clamped_chain(rotating, seed, _controls())
    np.testing.assert_allclose(loaded_root.poses, static_root.poses, atol=2e-12)
    np.testing.assert_allclose(
        loaded_root.support_wrench, static_root.support_wrench, atol=2e-10
    )
    assert loaded_root.stability_status == "unqualified"


def test_rotating_chain_tangent_uses_the_same_chart_as_elastic_and_applied_work() -> (
    None
):
    chain = RotatingSectionChain(
        SectionChain(_sections(), _loads(True)),
        (SectionInertia(_samples()),) * 2,
        _frame(),
    )
    poses, step = _poses(), 1e-5
    response = chain.linearize(poses)
    moving = moving_residual_jacobian(response)
    columns = []
    for axis in np.eye(18):
        plus = chain.linearize(_moved(poses, axis, step))
        minus = chain.linearize(_moved(poses, axis, -step))
        columns.append((plus.residual - minus.residual) / (2 * step))
    np.testing.assert_allclose(moving, np.column_stack(columns), atol=2e-7, rtol=2e-7)
    assert np.linalg.norm(response.tangent - response.tangent.T) > 0.1


def test_uniform_origin_acceleration_matches_exact_axial_displacement_and_support() -> (
    None
):
    frame = RotatingFrameState("observer", (0, 0, 0), (0, 0, 0), (0, 0, 5))
    chain, seed = _radial_rod(3, frame)
    result = solve_clamped_chain(chain, seed, _controls())
    z = seed[:, 2, 3]
    expected = z - (0.2 * 5 / 1000) * (z - z**2 / 2)
    np.testing.assert_allclose(result.poses[:, 2, 3], expected, atol=2e-10)
    np.testing.assert_allclose(result.support_wrench, [0, 0, 1, 0, 0, 0], atol=2e-8)
    np.testing.assert_array_equal(result.poses[0], seed[0])


def test_radial_spin_roots_converge_to_independent_continuum_solution() -> None:
    # EA*r'' + mu*omega²*r = 0, r(0)=0 and free-tip r'(L)=1.
    frame = RotatingFrameState("observer", (20, 0, 0), (0, 0, 0), (0, 0, 0))
    wave_number = 20 * np.sqrt(0.2 / 1000)
    tip = np.tan(wave_number) / wave_number
    support = -1000 * (1 / np.cos(wave_number) - 1)
    errors = []
    for count in (2, 4, 8):
        chain, seed = _radial_rod(count, frame)
        result = solve_clamped_chain(chain, seed, _controls())
        errors.append(abs(result.poses[-1, 2, 3] - tip))
        assert result.force_residual_n < 1e-8
        assert result.moment_residual_nm < 1e-9
        assert result.stability_status == "unqualified"
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2
    assert errors[-1] < 1e-5
    assert result.support_wrench[2] == pytest.approx(support, abs=0.02)


def test_rotating_root_refuses_declared_material_domain_overrun() -> None:
    frame = RotatingFrameState("observer", (20, 0, 0), (0, 0, 0), (0, 0, 0))
    chain, seed = _radial_rod(2, frame)
    controls = replace(
        _controls(), strain_limits=(0.001,) * 6, max_iterations=6, max_backtracks=4
    )
    with pytest.raises(RuntimeError, match="no converged state"):
        solve_clamped_chain(chain, seed, controls)


def test_rotating_chain_requires_one_explicit_inertia_quadrature_per_section() -> None:
    elastic, _ = _fixture([0, 0, 0], [0, 0, 0])
    inertia = SectionInertia(_samples())
    with pytest.raises(ValueError, match="per section"):
        RotatingSectionChain(elastic, (inertia,), _frame())
    with pytest.raises(TypeError, match="SectionInertia"):
        RotatingSectionChain(elastic, (inertia, None), _frame())
    with pytest.raises(TypeError, match="SectionChain"):
        RotatingSectionChain(None, (inertia, inertia), _frame())
    with pytest.raises(TypeError, match="RotatingFrameState"):
        RotatingSectionChain(elastic, (inertia, inertia), None)
