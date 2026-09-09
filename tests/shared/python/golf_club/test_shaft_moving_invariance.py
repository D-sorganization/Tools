"""Independent momentum, observer/scaling and refusal checks for moving shafts."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from shared.python.golf_club import _shaft_moving_chain as moving
from shared.python.golf_club._shaft_chain import IndexedPointLoad
from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_inertia import SectionInertia
from shared.python.golf_club._shaft_point_load import SpatialPointLoad

from .test_shaft_equilibrium import _controls as _equilibrium_controls
from .test_shaft_gripped_response import _model
from .test_shaft_inertia import _hat, _moved, _point, _samples, _vee
from .test_shaft_moving_chain import _advance, _axial_case, _fixture, _nonplanar_case


def test_common_inertial_observer_rotation_and_translation_preserve_response() -> None:
    chain, state, controls = _nonplanar_case()
    before = moving.moving_chain_response(chain, state, controls)
    observer = expm(_hat(np.array([0.5, -0.3, 0.2, 0.4, -0.3, 0.5])))
    rotation = observer[:3, :3]
    elastic = chain.shaft.elastic
    loads = tuple(
        replace(
            item,
            load=replace(
                item.load,
                force_n=rotation @ item.load.force_n,
                couple_nm=rotation @ item.load.couple_nm,
            ),
        )
        for item in elastic.loads
    )
    grips = tuple(
        replace(port, anchor=replace(port.anchor, pose=observer @ port.anchor.pose))
        for port in chain.grips
    )
    changed = replace(
        chain,
        shaft=replace(chain.shaft, elastic=replace(elastic, loads=loads)),
        grips=grips,
    )
    changed_state = replace(state, poses=observer @ np.asarray(state.poses))
    after = moving.moving_chain_response(changed, changed_state, controls)
    np.testing.assert_allclose(
        after.twist_rates, before.twist_rates, rtol=2e-9, atol=2e-9
    )
    for name in (
        "total_energy_j",
        "energy_rate_w",
        "applied_power_w",
        "anchor_power_w",
        "dissipated_power_w",
    ):
        assert getattr(after, name) == pytest.approx(
            getattr(before, name), rel=2e-9, abs=2e-9
        )


@pytest.mark.parametrize("length", [0.05, 2.0])
def test_length_scaling_changes_conditioning_not_physical_acceleration(
    length: float,
) -> None:
    chain, state, controls = _nonplanar_case()
    before = moving.moving_chain_response(chain, state, controls)
    changed = replace(controls, scales=replace(controls.scales, length_m=length))
    after = moving.moving_chain_response(chain, state, changed)
    np.testing.assert_allclose(
        after.twist_rates, before.twist_rates, rtol=1e-10, atol=1e-10
    )
    assert after.energy_rate_w == pytest.approx(before.energy_rate_w, rel=1e-10)


def _free_chain() -> tuple:
    base, poses = _model(2)
    _, _, controls = _fixture()
    inertias = (SectionInertia(_samples(2)),) * 2
    load = IndexedPointLoad(
        2, SpatialPointLoad([0.2, -0.3, 0.4], [0.01, 0.02, 0.03], [0.05, 0.03, -0.02])
    )
    elastic = replace(base.shaft.elastic, loads=(load,))
    shaft = replace(base.shaft, elastic=elastic, inertias=inertias)
    offsets = np.array(
        [
            [0.01, -0.01, 0, 0.02, -0.03, 0.01],
            [0, 0.01, 0.01, -0.01, 0.02, 0.01],
            [0.01, 0, 0.02, 0.03, 0, -0.02],
        ]
    )
    poses = np.array([h @ expm(_hat(q)) for h, q in zip(poses, offsets, strict=True)])
    state = moving.MovingChainState(
        poses, np.arange(18).reshape(3, 6) * 0.01, "observer"
    )
    return moving.InertialMovingChain(shaft, ()), state, controls


def _five_point_derivative(
    function: Callable[[float], np.ndarray], step: float
) -> np.ndarray:
    return (
        8 * (function(step) - function(-step))
        - (function(2 * step) - function(-2 * step))
    ) / (12 * step)


def _field_momentum(
    chain: moving.InertialMovingChain,
    state: moving.MovingChainState,
    inner_step: float = 1e-3,
) -> np.ndarray:
    poses, twists = np.asarray(state.poses), np.asarray(state.twists)
    momentum = np.zeros(6)
    for index, inertia in enumerate(chain.shaft.inertias):
        pair, velocity = poses[index : index + 2], twists[index : index + 2].ravel()
        for sample in inertia.samples:
            center = _point(pair, sample.fraction)
            derivative = _five_point_derivative(
                lambda step, pair=pair, velocity=velocity, fraction=sample.fraction: (
                    _point(_moved(pair, velocity, step), fraction)
                ),
                inner_step,
            )
            body, rotation = sample.body, center[:3, :3]
            com = center[:3, 3] + rotation @ body.center_of_mass_m
            com_rate = derivative[:3, 3] + derivative[:3, :3] @ body.center_of_mass_m
            spin = _vee(derivative[:3, :3] @ rotation.T)
            linear = body.mass_kg * com_rate
            angular = (
                np.cross(com, linear)
                + rotation @ body.inertia_at_com_kg_m2 @ rotation.T @ spin
            )
            momentum += np.r_[linear, angular]
    return momentum


def test_multisection_com_and_spin_momentum_rates_equal_external_wrench() -> None:
    chain, state, controls = _free_chain()
    response = moving.moving_chain_response(chain, state, controls)
    rates = np.asarray(response.twist_rates)
    elastic = chain.shaft.elastic
    load = elastic.loads[0].load
    tip = np.asarray(state.poses)[-1]
    point = tip[:3, 3] + tip[:3, :3] @ load.offset_m
    expected = np.r_[
        load.force_n, np.asarray(load.couple_nm) + np.cross(point, load.force_n)
    ]

    # Nested two-point differences at 1e-6 and 2e-5 amplified logm roundoff
    # to 3e-5 on Python 3.11. Resolve the independent momentum derivative
    # with fourth-order stencils and check both differentiation scales.
    def momentum(step: float, inner_step: float = 1e-3) -> np.ndarray:
        return _field_momentum(chain, _advance(state, rates, step), inner_step)

    estimates = [
        _five_point_derivative(momentum, step) for step in (1e-3, 5e-4, 2.5e-4)
    ]
    errors = [float(np.max(np.abs(value - expected))) for value in estimates]
    assert errors[1] < errors[0] / 8
    assert errors[2] < errors[1] / 4
    np.testing.assert_allclose(estimates[-1], expected, rtol=2e-5, atol=2e-5)
    refined_inner = _five_point_derivative(lambda step: momentum(step, 5e-4), 2.5e-4)
    np.testing.assert_allclose(refined_inner, estimates[-1], rtol=0, atol=1e-7)
    np.testing.assert_allclose(refined_inner, expected, rtol=0, atol=1e-7)
    corrupted = _five_point_derivative(
        lambda step: _field_momentum(chain, _advance(state, 0.9 * rates, step)),
        2.5e-4,
    )
    assert np.max(np.abs(corrupted - expected)) > 1e-2
    assert response.grip_responses == ()
    assert response.anchor_power_w == response.dissipated_power_w == 0
    assert response.energy_rate_w == pytest.approx(response.applied_power_w, abs=2e-10)


@pytest.mark.parametrize("tiny", [False, True])
def test_inaccurate_mass_solve_is_refused(
    monkeypatch: pytest.MonkeyPatch, tiny: bool
) -> None:
    chain, state, controls = _axial_case()
    if tiny:
        chain, state, controls = _fixture()
        load = IndexedPointLoad(
            1, SpatialPointLoad([0, 0, 1e-180], [0, 0, 0], [0, 0, 0])
        )
        elastic = replace(chain.shaft.elastic, loads=(load,))
        chain = replace(chain, shaft=replace(chain.shaft, elastic=elastic))
    original = np.linalg.solve

    def corrupt(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        answer = original(matrix, rhs)
        return (
            0.9 * answer if matrix.shape == (12, 12) and rhs.shape == (12,) else answer
        )

    monkeypatch.setattr(np.linalg, "solve", corrupt)
    with pytest.raises(ValueError, match="residual"):
        moving.moving_chain_response(chain, state, controls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("twists", [[True] * 6] * 2),
        ("twists", np.zeros((2, 6), complex)),
        ("twists", np.zeros((2, 5))),
        ("poses", np.zeros((2, 4, 4))),
        ("observer_id", ""),
    ],
)
def test_owned_state_contracts_are_strict(field: str, value: object) -> None:
    _, state, _ = _fixture()
    with pytest.raises((TypeError, ValueError)):
        replace(state, **{field: value})


@pytest.mark.parametrize("limits", [[0] * 6, [-1] * 6, [1] * 5, [True] * 6])
def test_strain_limits_require_positive_real_six_vectors(limits: object) -> None:
    _, _, controls = _fixture()
    with pytest.raises((TypeError, ValueError)):
        replace(controls, strain_limits=limits)


@pytest.mark.parametrize("spin", [0.0, 3.0])
def test_loaded_rotating_balance_matches_inertial_corotating_motion(
    spin: float,
) -> None:
    base, poses = _model(spin=spin)
    equilibrium = replace(
        _equilibrium_controls(), force_tolerance_n=1e-11, moment_tolerance_nm=1e-12
    )
    loaded = solve_gripped_chain(base, poses, equilibrium)
    frame = base.shaft.frame
    omega = np.asarray(frame.angular_velocity_rad_s)
    twists = np.array(
        [
            np.r_[h[:3, :3].T @ np.cross(omega, h[:3, 3]), h[:3, :3].T @ omega]
            for h in loaded.poses
        ]
    )
    ports = []
    for port in base.grips:
        h = np.asarray(port.anchor.pose)
        anchor_twist = np.r_[
            h[:3, :3].T @ np.cross(omega, h[:3, 3]), h[:3, :3].T @ omega
        ]
        anchor = replace(port.anchor, twist=anchor_twist, twist_rate=np.zeros(6))
        ports.append(moving.MovingGripAttachment(port.node, port.grip, anchor))
    inertial_frame = replace(frame, angular_velocity_rad_s=(0, 0, 0))
    shaft = replace(base.shaft, frame=inertial_frame)
    chain = moving.InertialMovingChain(shaft, tuple(ports))
    state = moving.MovingChainState(loaded.poses, twists, frame.frame_id)
    controls = moving.MovingChainControls(
        equilibrium.strain_limits, _fixture()[2].scales
    )
    response = moving.moving_chain_response(chain, state, controls)
    np.testing.assert_allclose(response.twist_rates, 0, atol=2e-7)
    assert abs(response.power_residual_w) < 1e-9
    if spin:
        assert np.linalg.norm(loaded.poses[-1, :3, 3] - loaded.poses[0, :3, 3]) > 1
