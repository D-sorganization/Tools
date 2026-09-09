"""Independent moving-boundary force, mass and energy controls for Tools #5072."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from shared.python.golf_club import _shaft_moving_chain as moving
from shared.python.golf_club._shaft_chain import IndexedPointLoad
from shared.python.golf_club._shaft_point_load import SpatialPointLoad

from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_inertia import _hat


def _fixture() -> tuple:
    base, poses = _model()
    attachment = base.grips[0]
    grip = moving.MovingGripAttachment(
        attachment.node, attachment.grip, attachment.anchor
    )
    chain = moving.InertialMovingChain(base.shaft, (grip,))
    state = moving.MovingChainState(poses, np.zeros((2, 6)), "observer")
    controls = moving.MovingChainControls((0.2, 0.2, 0.2, 2, 2, 2), _scales())
    return chain, state, controls


def _axial_case() -> tuple:
    chain, state, controls = _fixture()
    positions, velocities = np.asarray(state.poses).copy(), np.zeros((2, 6))
    positions[:, 2, 3] += [0.01, 0.03]
    velocities[:, 2] = [0.2, -0.1]
    port = chain.grips[0]
    anchor_pose = np.eye(4)
    anchor_pose[2, 3] = -0.01
    anchor = replace(
        port.anchor,
        pose=anchor_pose,
        twist=[0, 0, 0.3, 0, 0, 0],
        twist_rate=[0, 0, 0.4, 0, 0, 0],
    )
    load = IndexedPointLoad(1, SpatialPointLoad([0, 0, 0.7], [0, 0, 0], [0, 0, 0]))
    elastic = replace(chain.shaft.elastic, loads=(load,))
    chain = replace(
        chain,
        shaft=replace(chain.shaft, elastic=elastic),
        grips=(replace(port, anchor=anchor),),
    )
    return chain, replace(state, poses=positions, twists=velocities), controls


def test_moving_anchor_matches_independent_two_mass_equations_and_power() -> None:
    chain, state, controls = _axial_case()
    response = moving.moving_chain_response(chain, state, controls)
    q, v = np.array([0.01, 0.03]), np.array([0.2, -0.1])
    shaft_mass = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0, 0.1])
    total_mass = shaft_mass + np.diag([0.03, 0])
    force = [400 * (q[0] + 0.01) + 2 * (v[0] - 0.3) - 0.03 * 0.4, 0]
    elastic = 1000 * np.array([[1, -1], [-1, 1]]) @ q
    exact = np.linalg.solve(total_mass, np.array([0, 0.7]) - elastic - force)
    np.testing.assert_allclose(
        np.asarray(response.twist_rates)[:, 2], exact, rtol=2e-12
    )
    np.testing.assert_allclose(
        np.asarray(response.twist_rates)[:, [0, 1, 3, 4, 5]], 0, atol=2e-11
    )
    effort = 0.03 * (exact[0] - 0.4) + 2 * (v[0] - 0.3) + 400 * (q[0] + 0.01)
    expected_energy = (
        v @ shaft_mass @ v / 2
        + 1000 * (q[1] - q[0]) ** 2 / 2
        + 0.03 * (v[0] - 0.3) ** 2 / 2
        + 400 * (q[0] + 0.01) ** 2 / 2
    )
    assert response.total_energy_j == pytest.approx(expected_energy, rel=2e-12)
    assert response.applied_power_w == pytest.approx(0.7 * v[1])
    assert response.anchor_power_w == pytest.approx(effort * 0.3)
    assert response.dissipated_power_w == pytest.approx(2 * (v[0] - 0.3) ** 2)
    assert response.energy_rate_w == pytest.approx(
        0.7 * v[1] - effort * 0.3 - 2 * (v[0] - 0.3) ** 2, abs=2e-11
    )
    assert abs(response.power_residual_w) < 2e-11
    assert response.balance_residual <= controls.scales.residual_tolerance
    assert response.stability_status == "unqualified"


def _advance(
    state: moving.MovingChainState, rates: np.ndarray, step: float
) -> moving.MovingChainState:
    poses = np.array(
        [
            pose @ expm(_hat(step * v + step**2 * a / 2))
            for pose, v, a in zip(
                np.asarray(state.poses), np.asarray(state.twists), rates, strict=True
            )
        ]
    )
    return replace(state, poses=poses, twists=np.asarray(state.twists) + step * rates)


def _advance_anchor(
    port: moving.MovingGripAttachment, step: float
) -> moving.MovingGripAttachment:
    anchor = port.anchor
    velocity, rate = np.asarray(anchor.twist), np.asarray(anchor.twist_rate)
    pose = np.asarray(anchor.pose) @ expm(_hat(step * velocity + step**2 * rate / 2))
    return replace(
        port, anchor=replace(anchor, pose=pose, twist=velocity + step * rate)
    )


def _nonplanar_case() -> tuple:
    chain, state, controls = _fixture()
    offsets = [
        [0.01, -0.02, 0.005, 0.1, -0.06, 0.04],
        [0.02, 0.01, 0.025, -0.02, 0.08, -0.04],
    ]
    poses = np.array(
        [
            h @ expm(_hat(np.array(delta)))
            for h, delta in zip(state.poses, offsets, strict=True)
        ]
    )
    twists = np.array(
        [[0.2, -0.1, 0.3, 0.4, -0.3, 0.2], [-0.1, 0.4, -0.2, -0.2, 0.1, 0.3]]
    )
    port = chain.grips[0]
    factor = np.diag([0.12, 0.14, 0.17, 0.02, 0.03, 0.04])
    factor[0, 4] = 0.025
    law = replace(port.grip, inertance_factor=factor, damping_factor=2 * factor)
    anchor = replace(
        port.anchor,
        pose=expm(_hat(np.array([0.02, 0.03, -0.01, 0.2, 0.1, -0.1]))),
        twist=[0.1, 0.2, -0.2, 0.2, -0.1, 0.3],
        twist_rate=[-0.2, 0.3, 0.1, 0.1, 0.2, -0.1],
    )
    load = IndexedPointLoad(
        1, SpatialPointLoad([0.2, -0.3, 0.4], [0.03, -0.02, 0.01], [0.02, -0.01, 0.03])
    )
    elastic = replace(chain.shaft.elastic, loads=(load,))
    chain = replace(
        chain,
        shaft=replace(chain.shaft, elastic=elastic),
        grips=(replace(port, grip=law, anchor=anchor),),
    )
    return chain, replace(state, poses=poses, twists=twists), controls


def test_nonplanar_total_energy_derivative_includes_both_moving_ports() -> None:
    chain, state, controls = _nonplanar_case()
    response = moving.moving_chain_response(chain, state, controls)
    rates = np.asarray(response.twist_rates)
    step = 2e-7
    energies = []
    for sign in (-1, 1):
        advanced = replace(
            chain, grips=tuple(_advance_anchor(p, sign * step) for p in chain.grips)
        )
        shifted = _advance(state, rates, sign * step)
        energies.append(
            moving.moving_chain_response(advanced, shifted, controls).total_energy_j
        )
    derivative = (energies[1] - energies[0]) / (2 * step)
    assert derivative == pytest.approx(response.energy_rate_w, rel=2e-6, abs=2e-6)
    assert derivative == pytest.approx(
        response.applied_power_w
        - response.anchor_power_w
        - response.dissipated_power_w,
        rel=2e-6,
        abs=2e-6,
    )
    assert abs(response.anchor_power_w) > 0.01
    assert response.dissipated_power_w > 0


def test_repeated_grips_add_inertance_and_prescribed_anchor_drive() -> None:
    chain, state, controls = _axial_case()
    port = chain.grips[0]
    doubled = replace(chain, grips=(port, port))
    equivalent = replace(
        port.grip,
        **{
            name: np.sqrt(2) * np.asarray(getattr(port.grip, name))
            for name in ("inertance_factor", "damping_factor", "stiffness_factor")
        },
    )
    single = replace(chain, grips=(replace(port, grip=equivalent),))
    first = moving.moving_chain_response(doubled, state, controls)
    second = moving.moving_chain_response(single, state, controls)
    np.testing.assert_allclose(
        first.twist_rates, second.twist_rates, rtol=2e-12, atol=2e-12
    )
    assert len(first.grip_responses) == 2
    assert first.total_energy_j == pytest.approx(second.total_energy_j)
    assert first.anchor_power_w == pytest.approx(second.anchor_power_w)


@pytest.mark.parametrize(
    "field",
    [
        "angular_velocity_rad_s",
        "angular_acceleration_rad_s2",
        "origin_acceleration_m_s2",
    ],
)
def test_accelerating_or_rotating_observer_is_not_treated_as_inertial(
    field: str,
) -> None:
    chain, _, _ = _fixture()
    frame = replace(chain.shaft.frame, **{field: [np.nextafter(0.0, 1.0), 0, 0]})
    with pytest.raises(ValueError, match="inertial"):
        replace(chain, shaft=replace(chain.shaft, frame=frame))


def test_strain_and_singular_mass_are_refused_without_regularization() -> None:
    chain, state, controls = _fixture()
    poses = np.asarray(state.poses).copy()
    poses[1, 2, 3] += 0.3
    with pytest.raises(ValueError, match="strain"):
        moving.moving_chain_response(chain, replace(state, poses=poses), controls)
    inertia = chain.shaft.inertias[0]
    single = replace(inertia, samples=(replace(inertia.samples[0], fraction=0.5),))
    singular = replace(chain, shaft=replace(chain.shaft, inertias=(single,)), grips=())
    with pytest.raises(ValueError, match="mass"):
        moving.moving_chain_response(singular, state, controls)


def test_observer_node_count_and_numeric_overflow_are_refused() -> None:
    chain, state, controls = _fixture()
    with pytest.raises(ValueError, match="observer"):
        moving.moving_chain_response(
            chain, replace(state, observer_id="other"), controls
        )
    shortened = replace(state, poses=state.poses[:1], twists=state.twists[:1])
    with pytest.raises(ValueError, match="node"):
        moving.moving_chain_response(chain, shortened, controls)
    with pytest.raises(ValueError, match="finite|numerical"):
        moving.moving_chain_response(
            chain, replace(state, twists=np.full((2, 6), 1e200)), controls
        )


@pytest.mark.parametrize("node", [-1, True, np.bool_(True), 0.5, "0"])
def test_grip_node_contract_is_strict(node: object) -> None:
    chain, _, _ = _fixture()
    with pytest.raises((TypeError, ValueError)):
        replace(chain.grips[0], node=node)


def test_topology_and_anchor_observer_are_checked() -> None:
    chain, _, _ = _fixture()
    port = chain.grips[0]
    with pytest.raises(ValueError, match="node"):
        replace(chain, grips=(replace(port, node=2),))
    with pytest.raises(ValueError, match="observer"):
        replace(
            chain,
            grips=(replace(port, anchor=replace(port.anchor, observer_id="other")),),
        )
    assert replace(port, node=np.int64(0)).node == 0


def test_state_owns_inputs_and_evaluation_preserves_them() -> None:
    chain, state, controls = _axial_case()
    poses, twists = np.asarray(state.poses).copy(), np.asarray(state.twists).copy()
    owned = replace(state, poses=poses, twists=twists)
    poses[:] = 99
    twists[:] = 99
    np.testing.assert_array_equal(owned.poses, state.poses)
    np.testing.assert_array_equal(owned.twists, state.twists)
    response = moving.moving_chain_response(chain, owned, controls)
    assert np.any(np.asarray(response.twist_rates) != 0)
    np.testing.assert_array_equal(owned.twists, state.twists)
