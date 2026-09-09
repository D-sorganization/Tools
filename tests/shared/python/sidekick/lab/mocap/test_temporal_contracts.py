"""Contract tests for temporal reconstruction and biomechanical mapping
(TOOLS-M8 #4726).
"""

from __future__ import annotations

import math

import pytest
from sidekick.lab.mocap.enums import Availability
from sidekick.lab.mocap.observations import Landmark3D
from sidekick.lab.mocap.temporal import (
    ButterworthFilterConfig,
    DeliveryTrajectoryMappingConfig,
    GapPolicy,
    JointAngleConstraint,
    KinematicDerivatives,
    SavitzkyGolayConfig,
    SegmentLengthConstraint,
    adapt_to_delivery_trajectory,
    apply_joint_angle_constraint,
    apply_segment_length_constraint,
    compute_kinematic_derivatives,
    reconstruct_temporal_trajectory,
    smooth_trajectory_butterworth,
    smooth_trajectory_savgol,
)

from shared.python.swing_sim.delivery_interchange.trajectory import (
    DELIVERY_TRAJECTORY_FORMAT,
    DeliveryTrajectory,
    delivery_trajectory_to_json,
)


def _make_landmark(
    t_ns: int,
    xyz: tuple[float, float, float],
    avail: Availability = Availability.DERIVED,
    cov: tuple[float, ...] = (0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
) -> Landmark3D:
    cov_9 = tuple(float(x) for x in cov)
    if len(cov_9) != 9:
        cov_9 = (0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    return Landmark3D(
        landmark_id=f"lmk-test-{t_ns}",
        world_frame_id="world-test",
        skeleton_id="mediapipe-pose-33-v1",
        keypoint_id="right_wrist",
        timestamp_ns=t_ns,
        xyz_m=xyz,
        covariance_m2=cov_9,
        contributing_camera_ids=("cam-1", "cam-2"),
        rejected_camera_ids=(),
        method_id="mocap-triangulation-dlt-v1",
        availability=avail,
    )


def test_gap_policy_linear_interpolation() -> None:
    # 5 frames at 100 Hz (10 ms spacing), frame 2 missing
    t0 = 1_000_000_000
    dt_ns = 10_000_000  # 10 ms
    lmks = [
        _make_landmark(t0, (0.0, 0.0, 0.0)),
        _make_landmark(t0 + dt_ns, (0.1, 0.2, 0.3)),
        # frame 2 is missing
        _make_landmark(t0 + 3 * dt_ns, (0.3, 0.6, 0.9)),
        _make_landmark(t0 + 4 * dt_ns, (0.4, 0.8, 1.2)),
    ]
    policy = GapPolicy(max_gap_seconds=0.05, interpolation_kind="linear")
    traj = reconstruct_temporal_trajectory(
        landmarks=lmks,
        expected_sample_rate_hz=100.0,
        gap_policy=policy,
    )

    assert len(traj.timestamps_ns) == 5
    assert traj.availabilities[2] == Availability.PROVISIONAL
    assert pytest.approx(traj.positions_m[2][0], abs=1e-5) == 0.2
    assert pytest.approx(traj.positions_m[2][1], abs=1e-5) == 0.4
    assert pytest.approx(traj.positions_m[2][2], abs=1e-5) == 0.6
    assert traj.covariances_m2[2][0] > traj.covariances_m2[1][0]


def test_gap_policy_exceeds_max_gap_fails_closed() -> None:
    t0 = 1_000_000_000
    lmks = [
        _make_landmark(t0, (0.0, 0.0, 0.0)),
        _make_landmark(t0 + 200_000_000, (1.0, 1.0, 1.0)),
    ]
    policy = GapPolicy(max_gap_seconds=0.05, interpolation_kind="linear")
    traj = reconstruct_temporal_trajectory(
        landmarks=lmks,
        expected_sample_rate_hz=100.0,
        gap_policy=policy,
    )
    unavail_count = sum(1 for a in traj.availabilities if a == Availability.UNAVAILABLE)
    assert unavail_count > 0


def test_butterworth_smoothing() -> None:
    sample_rate = 100.0
    dt = 1.0 / sample_rate
    t0 = 1_000_000_000
    lmks = []
    for i in range(50):
        t = i * dt
        clean_x = math.sin(2.0 * math.pi * 2.0 * t)  # 2 Hz signal
        noise = 0.05 * (-1 if i % 2 == 0 else 1)  # 50 Hz Nyquist noise
        lmks.append(
            _make_landmark(
                t_ns=t0 + int(i * 1e9 * dt),
                xyz=(clean_x + noise, 0.0, 0.0),
            )
        )
    traj = reconstruct_temporal_trajectory(lmks, sample_rate)
    smoothed = smooth_trajectory_butterworth(
        traj,
        config=ButterworthFilterConfig(cutoff_hz=5.0, order=2),
    )
    diffs_raw = [
        abs(traj.positions_m[i + 1][0] - traj.positions_m[i][0]) for i in range(40)
    ]
    diffs_smooth = [
        abs(smoothed.positions_m[i + 1][0] - smoothed.positions_m[i][0])
        for i in range(40)
    ]
    assert sum(diffs_smooth) < sum(diffs_raw)


def test_savgol_smoothing() -> None:
    sample_rate = 60.0
    dt = 1.0 / sample_rate
    t0 = 1_000_000_000
    lmks = []
    for i in range(25):
        t = i * dt
        y = t * t + (0.02 if i % 2 == 0 else -0.02)
        lmks.append(
            _make_landmark(
                t_ns=t0 + int(i * 1e9 * dt),
                xyz=(0.0, y, 0.0),
            )
        )
    traj = reconstruct_temporal_trajectory(lmks, sample_rate)
    smoothed = smooth_trajectory_savgol(
        traj,
        config=SavitzkyGolayConfig(window_length=7, polyorder=2),
    )
    assert len(smoothed.positions_m) == 25
    assert smoothed.availabilities == traj.availabilities


def test_kinematic_derivatives_and_covariance_propagation() -> None:
    sample_rate = 100.0
    dt = 1.0 / sample_rate
    t0 = 1_000_000_000
    lmks = [
        _make_landmark(
            t_ns=t0 + int(i * 1e9 * dt),
            xyz=(i * dt * 1.0, 0.0, 0.0),
            cov=(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04),
        )
        for i in range(10)
    ]
    traj = reconstruct_temporal_trajectory(lmks, sample_rate)
    derivs = compute_kinematic_derivatives(traj)

    assert isinstance(derivs, KinematicDerivatives)
    for v in derivs.linear_velocities_mps[1:-1]:
        assert pytest.approx(v[0], abs=1e-3) == 1.0
        assert pytest.approx(v[1], abs=1e-3) == 0.0
        assert pytest.approx(v[2], abs=1e-3) == 0.0

    for a in derivs.linear_accelerations_mps2[2:-2]:
        assert pytest.approx(a[0], abs=1e-3) == 0.0
        assert pytest.approx(a[1], abs=1e-3) == 0.0

    expected_v_var = (0.04 + 0.04) / (4.0 * (dt**2))
    assert pytest.approx(derivs.velocity_covariances[3][0], rel=1e-2) == expected_v_var


def test_segment_length_invariance_constraint() -> None:
    p_elbow = (0.0, 0.0, 0.0)
    p_wrist = (0.35, 0.0, 0.0)

    constraint = SegmentLengthConstraint(
        proximal_keypoint_id="right_elbow",
        distal_keypoint_id="right_wrist",
        nominal_length_m=0.30,
        tolerance_m=0.01,
    )
    p_adj_elbow, p_adj_wrist = apply_segment_length_constraint(
        constraint=constraint,
        proximal_pos=p_elbow,
        distal_pos=p_wrist,
        proximal_weight=0.5,
        distal_weight=0.5,
    )
    dx = p_adj_wrist[0] - p_adj_elbow[0]
    dy = p_adj_wrist[1] - p_adj_elbow[1]
    dz = p_adj_wrist[2] - p_adj_elbow[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert pytest.approx(dist, abs=1e-4) == 0.30


def test_joint_angle_bounds_constraint() -> None:
    p_shoulder = (-0.3, 0.0, 0.0)
    p_elbow = (0.0, 0.0, 0.0)
    p_wrist_hyperextended = (0.3, -0.05, 0.0)

    constraint = JointAngleConstraint(
        joint_keypoint_id="right_elbow",
        min_angle_rad=0.0,
        max_angle_rad=math.pi,
    )
    valid, angle = apply_joint_angle_constraint(
        constraint=constraint,
        proximal_pos=p_shoulder,
        joint_pos=p_elbow,
        distal_pos=p_wrist_hyperextended,
    )
    assert 0.0 <= angle <= math.pi + 0.1


def test_adapt_to_delivery_trajectory() -> None:
    sample_rate = 100.0
    dt = 1.0 / sample_rate
    t0_ns = 1_000_000_000
    n_frames = 10

    lmks_butt = [
        _make_landmark(
            t_ns=t0_ns + int(i * 1e9 * dt),
            xyz=(0.5 + i * dt * 2.0, 0.8, 0.0),
        )
        for i in range(n_frames)
    ]
    lmks_shaft = [
        _make_landmark(
            t_ns=t0_ns + int(i * 1e9 * dt),
            xyz=(0.5 + i * dt * 2.0, 0.6, 0.0),
        )
        for i in range(n_frames)
    ]
    traj_butt = reconstruct_temporal_trajectory(lmks_butt, sample_rate)
    traj_shaft = reconstruct_temporal_trajectory(lmks_shaft, sample_rate)

    config = DeliveryTrajectoryMappingConfig(
        source_id="mocap-lab-v1",
        frame_id="world",
        butt_keypoint_id="right_wrist",
        shaft_guide_keypoint_id="right_index",
    )

    delivery_traj = adapt_to_delivery_trajectory(
        butt_trajectory=traj_butt,
        shaft_guide_trajectory=traj_shaft,
        config=config,
    )

    assert isinstance(delivery_traj, DeliveryTrajectory)
    assert delivery_traj.source_id == "mocap-lab-v1"
    assert delivery_traj.frame_id == "world"
    assert len(delivery_traj.samples) == n_frames

    wire_json = delivery_trajectory_to_json(delivery_traj)
    assert DELIVERY_TRAJECTORY_FORMAT in wire_json
