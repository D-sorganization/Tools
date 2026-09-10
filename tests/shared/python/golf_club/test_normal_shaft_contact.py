"""Independent axial three-mass and finite-work controls for contact coupling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from shared.python.swing_sim.impact._normal_shaft_contact import NormalShaftContact
from shared.python.swing_sim.impact._rigid_body_response import RigidBodyInertia
from shared.python.swing_sim.impact._spatial_contact_kinematics import (
    ContactBodyState,
    PlaneSphereGeometry,
)
from shared.python.swing_sim.impact.contact import KelvinVoigtContactLaw

from .test_shaft_inertia import _hat
from .test_shaft_moving_chain import (
    _advance,
    _advance_anchor,
    _axial_case,
    _nonplanar_case,
)


def _case(gap: float = -0.001, ball_velocity: float = -0.4) -> tuple:
    chain, state, controls = _axial_case()
    inertia = RigidBodyInertia("ball", 0.046, (0, 0, 0), np.eye(3) * 8e-6)
    model = NormalShaftContact(
        chain,
        1,
        inertia,
        PlaneSphereGeometry(0.02, (0, 0, 0), (0, 0, 1)),
        KelvinVoigtContactLaw(2e4, 3),
        controls,
    )
    pose = np.eye(4)
    pose[2, 3] = np.asarray(state.poses)[1, 2, 3] + 0.02 + gap
    ball = ContactBodyState(pose, [0, 0, ball_velocity, 0, 0, 0], "observer")
    return model, state, ball


@pytest.mark.parametrize(
    "gap,ball_velocity", [(-0.001, -0.4), (0.001, -0.4), (-0.001, 10.0)]
)
def test_normal_response_matches_independent_three_mass_equations_and_work(
    gap: float,
    ball_velocity: float,
) -> None:
    model, state, ball = _case(gap, ball_velocity)
    result = model.evaluate(state, ball, "ball")
    q, v = np.array([0.01, 0.03]), np.array([0.2, -0.1])
    compression, rate = max(-gap, 0), v[1] - ball_velocity
    force = max(0, 2e4 * compression + 3 * rate) if compression else 0
    mass = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    elastic = 1000 * np.array([[1, -1], [-1, 1]]) @ q
    grip = np.array([400 * (q[0] + 0.01) + 2 * (v[0] - 0.3) - 0.03 * 0.4, 0])
    exact = np.linalg.solve(mass, np.array([0, 0.7 - force]) - elastic - grip)
    np.testing.assert_allclose(
        np.asarray(result.shaft.twist_rates)[:, 2], exact, rtol=2e-12
    )
    assert result.ball.twist_rate[2] == pytest.approx(force / 0.046, rel=2e-12)
    np.testing.assert_allclose(
        np.asarray(result.shaft.twist_rates)[:, [0, 1, 3, 4, 5]], 0, atol=2e-10
    )
    assert result.normal.force_n == pytest.approx(force, rel=2e-12)
    effort = 0.03 * (exact[0] - 0.4) + 2 * (v[0] - 0.3) + 400 * (q[0] + 0.01)
    shaft_mass = mass - np.diag([0.03, 0])
    energy = (
        v @ shaft_mass @ v / 2
        + 1000 * (q[1] - q[0]) ** 2 / 2
        + 0.03 * (v[0] - 0.3) ** 2 / 2
        + 400 * (q[0] + 0.01) ** 2 / 2
        + 0.046 * ball_velocity**2 / 2
        + 2e4 * compression**2 / 2
    )
    assert result.energy.total_energy_j == pytest.approx(energy, rel=2e-12)
    assert result.energy.external_power_w == pytest.approx(0.7 * v[1], abs=1e-12)
    assert result.energy.anchor_power_w == pytest.approx(effort * 0.3, rel=2e-12)
    assert result.energy.grip_dissipated_power_w == pytest.approx(2 * (v[0] - 0.3) ** 2)
    viscous = 3 * rate**2 if compression and force > 0 else 0
    cutoff = -2e4 * compression * rate if compression and force == 0 else 0
    assert result.energy.viscous_power_w == pytest.approx(viscous)
    assert result.energy.cutoff_power_w == pytest.approx(cutoff, rel=2e-12)
    assert abs(result.energy.power_residual_w) < 2e-10


def test_each_evaluation_recomputes_contact_and_preserves_original_loads() -> None:
    model, state, ball = _case()
    original = model.chain
    first = model.evaluate(state, ball, "ball")
    shifted = np.asarray(ball.pose).copy()
    shifted[2, 3] += 0.0005
    second = model.evaluate(state, replace(ball, pose=shifted), "ball")
    assert first.normal.force_n - second.normal.force_n == pytest.approx(10)
    assert model.chain is original
    assert len(original.shaft.elastic.loads) == 1
    assert model.evaluate(state, ball, "ball") == first


@pytest.mark.parametrize("node", [-1, 2, True])
def test_face_node_must_identify_one_existing_material_node(node: object) -> None:
    model, _, _ = _case()
    with pytest.raises((TypeError, ValueError)):
        replace(model, face_node=node)


def test_mismatched_observer_or_ball_material_frame_is_refused() -> None:
    model, state, ball = _case()
    with pytest.raises(ValueError, match="observer"):
        model.evaluate(state, replace(ball, observer_id="other"), "ball")
    with pytest.raises(ValueError, match="frame"):
        model.evaluate(state, ball, "other")


def test_force_ceiling_is_not_silently_clipped() -> None:
    model, state, ball = _case()
    model = replace(model, law=KelvinVoigtContactLaw(2e4, 3, 10))
    with pytest.raises(ValueError, match="ceiling"):
        model.evaluate(state, ball, "ball")


def _spatial_case() -> tuple:
    model, _, _ = _case()
    chain, state, controls = _nonplanar_case()
    model = replace(model, chain=chain, controls=controls)
    face = np.asarray(state.poses)[1]
    pose = np.eye(4)
    pose[:3, :3] = expm(_hat(np.array([0, 0, 0, 0.2, -0.1, 0.3])))[:3, :3]
    pose[:3, 3] = face[:3, 3] + face[:3, :3] @ [0.03, -0.015, 0.019]
    ball = ContactBodyState(pose, [0.3, -0.1, -0.6, 20, -15, 30], "observer")
    return model, state, ball


def test_nonplanar_offset_contact_energy_derivative_with_moving_grip() -> None:
    model, state, ball = _spatial_case()
    response = model.evaluate(state, ball, "ball")
    assert response.normal.force_n > 0
    ball_twist, ball_rate = np.asarray(ball.twist), np.asarray(response.ball.twist_rate)
    # A central normal force on a centered isotropic sphere creates no spin.
    np.testing.assert_allclose(ball_rate[3:], 0, atol=2e-9)
    for step in (4e-7, 2e-7):
        energies = []
        for time in (-step, step):
            chain = model.chain
            advanced_chain = replace(
                chain, grips=tuple(_advance_anchor(port, time) for port in chain.grips)
            )
            advanced_model = replace(model, chain=advanced_chain)
            advanced_shaft = _advance(
                state, np.asarray(response.shaft.twist_rates), time
            )
            pose = np.asarray(ball.pose) @ expm(
                _hat(time * ball_twist + time**2 * ball_rate / 2)
            )
            advanced_ball = replace(
                ball, pose=pose, twist=ball_twist + time * ball_rate
            )
            energies.append(
                advanced_model.evaluate(
                    advanced_shaft, advanced_ball, "ball"
                ).energy.total_energy_j
            )
        derivative = (energies[1] - energies[0]) / (2 * step)
        assert derivative == pytest.approx(
            response.energy.energy_rate_w, rel=2e-6, abs=2e-6
        )
    assert abs(response.energy.power_residual_w) < 2e-10


def test_contact_response_preserves_common_observer_invariance() -> None:
    model, state, ball = _spatial_case()
    before = model.evaluate(state, ball, "ball")
    observer = expm(_hat(np.array([0.5, -0.3, 0.2, 0.4, -0.3, 0.5])))
    rotation = observer[:3, :3]
    chain = model.chain
    shaft = chain.shaft
    elastic = shaft.elastic
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
    changed_chain = replace(
        chain, shaft=replace(shaft, elastic=replace(elastic, loads=loads)), grips=grips
    )
    changed = replace(model, chain=changed_chain)
    after = changed.evaluate(
        replace(state, poses=observer @ state.poses),
        replace(ball, pose=observer @ ball.pose),
        "ball",
    )
    np.testing.assert_allclose(
        after.shaft.twist_rates, before.shaft.twist_rates, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        after.ball.twist_rate, before.ball.twist_rate, rtol=2e-9, atol=2e-9
    )
    for name in (
        "total_energy_j",
        "energy_rate_w",
        "external_power_w",
        "anchor_power_w",
        "grip_dissipated_power_w",
        "viscous_power_w",
        "cutoff_power_w",
    ):
        assert getattr(after.energy, name) == pytest.approx(
            getattr(before.energy, name), rel=2e-9, abs=2e-9
        )


def test_incomplete_shaft_state_is_refused_before_indexing_the_face_node() -> None:
    model, state, ball = _case()
    incomplete = replace(state, poses=state.poses[:1], twists=state.twists[:1])
    with pytest.raises(ValueError, match="node|shape"):
        model.evaluate(incomplete, ball, "ball")
