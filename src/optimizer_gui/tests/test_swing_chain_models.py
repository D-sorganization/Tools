"""TDD coverage for swingset and chain motion models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optimizer_gui.models.chain_model import (
    ChainConfig,
    ChainState,
    initial_catenary_angles,
    link_midpoints,
    simulate_chain,
)
from optimizer_gui.models.swingset_model import (
    HumanSegmentSpec,
    SwingControlAction,
    SwingPose,
    SwingSetConfig,
    SwingSetState,
    build_swingset_snapshot,
    heuristic_pumping_policy,
    simulate_swingset,
)


def test_chain_config_validates_positive_inputs() -> None:
    """Chain configuration rejects non-physical dimensions."""
    with pytest.raises(ValueError, match="segment_count"):
        ChainConfig(segment_count=0)
    with pytest.raises(ValueError, match="segment_length_m"):
        ChainConfig(segment_length_m=0.0)
    with pytest.raises(ValueError, match="link_mass_kg"):
        ChainConfig(link_mass_kg=-1.0)
    with pytest.raises(ValueError, match="damping"):
        ChainConfig(damping=-0.1)
    with pytest.raises(ValueError, match="coupling"):
        ChainConfig(coupling=-0.1)


def test_chain_state_validates_array_contracts() -> None:
    """Chain state validation catches malformed generalized coordinates."""
    config = ChainConfig(segment_count=2)

    with pytest.raises(ValueError, match="1-D"):
        ChainState(
            angles_rad=np.zeros((1, 2)),
            angular_velocities_rad_s=np.zeros(2),
        ).validated(config)
    with pytest.raises(ValueError, match="2 values"):
        ChainState(
            angles_rad=np.zeros(3),
            angular_velocities_rad_s=np.zeros(2),
        ).validated(config)
    with pytest.raises(ValueError, match="finite"):
        ChainState(
            angles_rad=np.array([0.0, np.nan]),
            angular_velocities_rad_s=np.zeros(2),
        ).validated(config)


def test_chain_positions_include_anchor_and_tip() -> None:
    """A two-link vertical chain has deterministic side-view coordinates."""
    config = ChainConfig(segment_count=2, segment_length_m=0.5)
    state = ChainState.stationary(config, angle_rad=0.0)

    positions = state.node_positions(config)

    np.testing.assert_allclose(positions, [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])

    with pytest.raises(ValueError, match="anchor"):
        state.node_positions(config, anchor_xy_m=(0.0, float("nan")))


def test_chain_midpoints_and_tip_speed_are_reported() -> None:
    """Chain analysis exposes geometry and whip-style motion metrics."""
    config = ChainConfig(segment_count=2, segment_length_m=1.0)
    state = ChainState(
        angles_rad=np.array([0.0, math.pi / 2.0]),
        angular_velocities_rad_s=np.array([0.0, 2.0]),
    )

    midpoints = link_midpoints(state.node_positions(config))
    metrics = state.metrics(config)

    assert midpoints.shape == (2, 2)
    assert metrics.tip_speed_m_s == pytest.approx(2.0)
    assert metrics.max_curvature_rad == pytest.approx(math.pi / 2.0)

    single = ChainState.stationary(ChainConfig(segment_count=1))
    assert single.metrics(ChainConfig(segment_count=1)).max_curvature_rad == 0.0
    with pytest.raises(ValueError, match="node_positions"):
        link_midpoints(np.zeros((1, 2)))


def test_chain_simulation_damps_energy() -> None:
    """A damped chain rollout should not increase total mechanical energy."""
    config = ChainConfig(segment_count=3, segment_length_m=0.4, damping=2.0)
    state = ChainState.stationary(config, angle_rad=0.4)

    rollout = simulate_chain(config, state, steps=24, dt_s=0.02)

    assert rollout.positions.shape == (25, 4, 2)
    assert rollout.energy_j[-1] < rollout.energy_j[0]
    assert rollout.tip_speed_m_s.shape == (25,)


def test_initial_catenary_angles_validate_segment_count() -> None:
    """Initial chain shape helper validates its public API."""
    with pytest.raises(ValueError, match="segment_count"):
        initial_catenary_angles(segment_count=1, sag_rad=0.2)
    with pytest.raises(ValueError, match="sag_rad"):
        initial_catenary_angles(segment_count=2, sag_rad=-0.1)

    angles = initial_catenary_angles(segment_count=5, sag_rad=0.2)

    assert angles.shape == (5,)
    assert angles[0] == pytest.approx(-angles[-1])


def test_chain_simulation_validates_rollout_inputs() -> None:
    """Chain rollout validates step count, dt, and torque history shape."""
    config = ChainConfig(segment_count=2)
    state = ChainState.stationary(config)

    with pytest.raises(ValueError, match="steps"):
        simulate_chain(config, state, steps=0, dt_s=0.01)
    with pytest.raises(ValueError, match="dt_s"):
        simulate_chain(config, state, steps=1, dt_s=0.0)
    with pytest.raises(ValueError, match="torque_history"):
        simulate_chain(
            config,
            state,
            steps=2,
            dt_s=0.01,
            torque_history_nm=np.zeros((1, 2)),
        )

    rollout = simulate_chain(
        config,
        state,
        steps=2,
        dt_s=0.01,
        torque_history_nm=np.zeros((2, 2)),
    )
    assert len(rollout.states) == 3


def test_swingset_config_exposes_shared_chain_config() -> None:
    """Swingset and chain analysis share the same chain configuration type."""
    config = SwingSetConfig(
        chain_segments=8,
        chain_length_m=2.4,
        chain_link_mass_kg=0.2,
    )

    chain_config = config.chain_config()

    assert isinstance(chain_config, ChainConfig)
    assert chain_config.segment_count == 8
    assert chain_config.segment_length_m == pytest.approx(0.3)


def test_swingset_config_rejects_invalid_body_segments() -> None:
    """Human segment specs enforce positive length and mass contracts."""
    with pytest.raises(ValueError, match="length_m"):
        HumanSegmentSpec(length_m=0.0, mass_kg=5.0)
    with pytest.raises(ValueError, match="mass_kg"):
        HumanSegmentSpec(length_m=0.5, mass_kg=0.0)
    with pytest.raises(ValueError, match="chain_segments"):
        SwingSetConfig(chain_segments=0)
    with pytest.raises(ValueError, match="damping"):
        SwingSetConfig(damping=-0.1)
    with pytest.raises(ValueError, match="pump_gain"):
        SwingSetConfig(pump_gain=-0.1)


def test_swingset_snapshot_contains_required_degrees_of_freedom() -> None:
    """Snapshot includes torso, kicking legs, and two arm segments holding chain."""
    config = SwingSetConfig()
    pose = SwingPose(
        swing_angle_rad=0.1,
        torso_lean_rad=-0.2,
        hip_angle_rad=0.35,
        knee_angle_rad=-0.45,
        shoulder_angle_rad=-0.3,
        elbow_angle_rad=0.55,
    )

    snapshot = build_swingset_snapshot(config, pose)

    for key in ("seat", "hip", "shoulder", "knee", "foot", "elbow", "hand"):
        assert key in snapshot.points
    assert snapshot.chain_nodes.shape == (config.chain_segments + 1, 2)
    assert snapshot.hand_chain_error_m >= 0.0
    assert snapshot.center_of_mass_m.shape == (2,)


def test_swingset_static_position_remains_static_without_control() -> None:
    """No control from the vertical rest pose should stay near rest."""
    config = SwingSetConfig(damping=0.0)
    state = SwingSetState.rest()

    rollout = simulate_swingset(
        config,
        state,
        steps=10,
        dt_s=0.02,
        policy=lambda _state, _time: SwingControlAction(),
    )

    assert rollout.swing_angles_rad[-1] == pytest.approx(0.0, abs=1e-12)
    assert rollout.metrics.max_abs_swing_angle_rad == pytest.approx(0.0)


def test_swingset_heuristic_policy_builds_amplitude() -> None:
    """The built-in pumping policy is suitable as a training baseline."""
    config = SwingSetConfig(pump_gain=0.9, damping=0.01)
    state = SwingSetState(
        pose=SwingPose(swing_angle_rad=0.08),
        swing_angular_velocity_rad_s=0.0,
    )

    passive = simulate_swingset(
        config,
        state,
        steps=180,
        dt_s=0.02,
        policy=lambda _state, _time: SwingControlAction(),
    )
    active = simulate_swingset(
        config,
        state,
        steps=180,
        dt_s=0.02,
        policy=heuristic_pumping_policy,
    )

    assert (
        active.metrics.max_abs_swing_angle_rad > passive.metrics.max_abs_swing_angle_rad
    )
    assert active.controls.shape == (180, 5)


def test_swingset_simulation_validates_rollout_inputs() -> None:
    """Swingset rollout validates step count and integration interval."""
    config = SwingSetConfig()
    state = SwingSetState.rest()

    with pytest.raises(ValueError, match="steps"):
        simulate_swingset(
            config,
            state,
            steps=0,
            dt_s=0.01,
            policy=heuristic_pumping_policy,
        )
    with pytest.raises(ValueError, match="dt_s"):
        simulate_swingset(
            config,
            state,
            steps=1,
            dt_s=0.0,
            policy=heuristic_pumping_policy,
        )


def test_optimizer_package_extends_to_nested_python_sources() -> None:
    """Standalone launch can import nested optimizer_gui UI modules."""
    import optimizer_gui

    nested = Path(optimizer_gui.__file__).parent / "python" / "optimizer_gui"

    assert str(nested) in list(optimizer_gui.__path__)
