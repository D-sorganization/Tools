"""Temporal reconstruction, filtering, constraints, and delivery mapping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from shared.python.swing_sim.delivery_interchange.trajectory import (
    DeliveryTrajectory,
    TrajectorySample,
)

from ._validation import require_finite, require_text
from .enums import Availability
from .observations import Landmark3D

__all__: list[str] = []
Mat9 = tuple[float, float, float, float, float, float, float, float, float]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GapPolicy:
    max_gap_seconds: float = 0.1
    interpolation_kind: Literal["linear", "cubic"] = "linear"

    def __post_init__(self) -> None:
        if require_finite(self.max_gap_seconds, "max_gap_seconds") <= 0.0:
            raise ValueError("max_gap_seconds must be positive")
        if self.interpolation_kind not in ("linear", "cubic"):
            raise ValueError("interpolation_kind must be 'linear' or 'cubic'")


@dataclass(frozen=True, slots=True)
class TemporalTrajectory:
    skeleton_id: str
    keypoint_id: str
    timestamps_ns: tuple[int, ...]
    positions_m: tuple[Vec3, ...]
    covariances_m2: tuple[Mat9, ...]
    availabilities: tuple[Availability, ...]

    def __post_init__(self) -> None:
        require_text(self.skeleton_id, "skeleton_id")
        require_text(self.keypoint_id, "keypoint_id")
        n = len(self.timestamps_ns)
        if (
            len(self.positions_m) != n
            or len(self.covariances_m2) != n
            or len(self.availabilities) != n
        ):
            raise ValueError("All trajectory sequence lengths must match")


@dataclass(frozen=True, slots=True)
class ButterworthFilterConfig:
    cutoff_hz: float = 6.0
    order: int = 2

    def __post_init__(self) -> None:
        if require_finite(self.cutoff_hz, "cutoff_hz") <= 0.0:
            raise ValueError("cutoff_hz must be positive")
        if self.order < 1:
            raise ValueError("order must be >= 1")


@dataclass(frozen=True, slots=True)
class SavitzkyGolayConfig:
    window_length: int = 7
    polyorder: int = 2

    def __post_init__(self) -> None:
        if self.window_length % 2 == 0 or self.window_length < 3:
            raise ValueError("window_length must be an odd integer >= 3")
        if self.polyorder < 1 or self.polyorder >= self.window_length:
            raise ValueError("polyorder must satisfy 1 <= polyorder < window_length")


@dataclass(frozen=True, slots=True)
class KinematicDerivatives:
    timestamps_ns: tuple[int, ...]
    linear_velocities_mps: tuple[Vec3, ...]
    linear_accelerations_mps2: tuple[Vec3, ...]
    velocity_covariances: tuple[Mat9, ...]
    acceleration_covariances: tuple[Mat9, ...]


@dataclass(frozen=True, slots=True)
class SegmentLengthConstraint:
    proximal_keypoint_id: str
    distal_keypoint_id: str
    nominal_length_m: float
    tolerance_m: float = 0.02

    def __post_init__(self) -> None:
        require_text(self.proximal_keypoint_id, "proximal_keypoint_id")
        require_text(self.distal_keypoint_id, "distal_keypoint_id")
        if require_finite(self.nominal_length_m, "nominal_length_m") <= 0.0:
            raise ValueError("nominal_length_m must be positive")
        if require_finite(self.tolerance_m, "tolerance_m") < 0.0:
            raise ValueError("tolerance_m must be non-negative")


@dataclass(frozen=True, slots=True)
class JointAngleConstraint:
    joint_keypoint_id: str
    min_angle_rad: float = 0.0
    max_angle_rad: float = math.pi

    def __post_init__(self) -> None:
        require_text(self.joint_keypoint_id, "joint_keypoint_id")
        require_finite(self.min_angle_rad, "min_angle_rad")
        require_finite(self.max_angle_rad, "max_angle_rad")
        if self.min_angle_rad > self.max_angle_rad:
            raise ValueError("min_angle_rad cannot exceed max_angle_rad")


@dataclass(frozen=True, slots=True)
class DeliveryTrajectoryMappingConfig:
    source_id: str
    frame_id: str
    butt_keypoint_id: str
    shaft_guide_keypoint_id: str

    def __post_init__(self) -> None:
        require_text(self.source_id, "source_id")
        require_text(self.frame_id, "frame_id")
        require_text(self.butt_keypoint_id, "butt_keypoint_id")
        require_text(self.shaft_guide_keypoint_id, "shaft_guide_keypoint_id")


def reconstruct_temporal_trajectory(
    landmarks: Sequence[Landmark3D],
    expected_sample_rate_hz: float,
    gap_policy: GapPolicy | None = None,
) -> TemporalTrajectory:
    if not landmarks:
        raise ValueError("landmarks sequence cannot be empty")
    if require_finite(expected_sample_rate_hz, "expected_sample_rate_hz") <= 0.0:
        raise ValueError("expected_sample_rate_hz must be positive")
    cfg = gap_policy or GapPolicy()
    sorted_lmks = sorted(landmarks, key=lambda lm: lm.timestamp_ns)
    t0_ns, t_end_ns = sorted_lmks[0].timestamp_ns, sorted_lmks[-1].timestamp_ns
    dt_ns = int(round(1e9 / expected_sample_rate_hz))
    grid_times = list(range(t0_ns, t_end_ns + 1, dt_ns))
    if not grid_times or grid_times[-1] < t_end_ns:
        grid_times.append(t_end_ns)

    lmk_by_idx = {
        min(
            range(len(grid_times)), key=lambda i: abs(grid_times[i] - lm.timestamp_ns)
        ): lm
        for lm in sorted_lmks
    }
    positions, covs, avails = [], [], []
    known = sorted(lmk_by_idx.keys())
    max_gap_ns = int(round(cfg.max_gap_seconds * 1e9))

    for idx, t_ns in enumerate(grid_times):
        if idx in lmk_by_idx:
            lm = lmk_by_idx[idx]
            positions.append(lm.xyz_m)
            covs.append(lm.covariance_m2)
            avails.append(lm.availability)
        else:
            p_idx = max((i for i in known if i < idx), default=None)
            n_idx = min((i for i in known if i > idx), default=None)
            if (
                p_idx is None
                or n_idx is None
                or (grid_times[n_idx] - grid_times[p_idx]) > max_gap_ns
            ):
                positions.append((0.0, 0.0, 0.0))
                covs.append((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                avails.append(Availability.UNAVAILABLE)
            else:
                alpha = (t_ns - grid_times[p_idx]) / float(
                    grid_times[n_idx] - grid_times[p_idx]
                )
                p0, p1 = lmk_by_idx[p_idx].xyz_m, lmk_by_idx[n_idx].xyz_m
                positions.append(
                    (
                        p0[0] + alpha * (p1[0] - p0[0]),
                        p0[1] + alpha * (p1[1] - p0[1]),
                        p0[2] + alpha * (p1[2] - p0[2]),
                    )
                )
                c0, c1 = (
                    lmk_by_idx[p_idx].covariance_m2,
                    lmk_by_idx[n_idx].covariance_m2,
                )
                covs.append(
                    cast(
                        Mat9,
                        tuple(
                            (c0[k] * (1.0 - alpha) + c1[k] * alpha) * 2.0
                            for k in range(9)
                        ),
                    )
                )
                avails.append(Availability.PROVISIONAL)

    return TemporalTrajectory(
        skeleton_id=sorted_lmks[0].skeleton_id,
        keypoint_id=sorted_lmks[0].keypoint_id,
        timestamps_ns=tuple(grid_times),
        positions_m=tuple(positions),
        covariances_m2=tuple(covs),
        availabilities=tuple(avails),
    )


def _apply_iir_filter(data: list[float], b: list[float], a: list[float]) -> list[float]:
    y = [0.0] * len(data)
    for i in range(len(data)):
        val = b[0] * data[i]
        for j in range(1, len(b)):
            if i - j >= 0:
                val += b[j] * data[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                val -= a[j] * y[i - j]
        y[i] = val / a[0]
    return y


def smooth_trajectory_butterworth(
    trajectory: TemporalTrajectory,
    config: ButterworthFilterConfig | None = None,
) -> TemporalTrajectory:
    cfg = config or ButterworthFilterConfig()
    n = len(trajectory.positions_m)
    if n < 4:
        return trajectory
    dt = (trajectory.timestamps_ns[-1] - trajectory.timestamps_ns[0]) / (1e9 * (n - 1))
    fs = 1.0 / dt
    k = math.tan(math.pi * min(cfg.cutoff_hz, fs * 0.49) / fs)
    k2 = k * k
    denom = 1.0 + math.sqrt(2.0) * k + k2
    b = [k2 / denom, 2.0 * k2 / denom, k2 / denom]
    a = [1.0, 2.0 * (k2 - 1.0) / denom, (1.0 - math.sqrt(2.0) * k + k2) / denom]

    xs = [p[0] for p in trajectory.positions_m]
    ys = [p[1] for p in trajectory.positions_m]
    zs = [p[2] for p in trajectory.positions_m]
    fx = _apply_iir_filter(_apply_iir_filter(xs, b, a)[::-1], b, a)[::-1]
    fy = _apply_iir_filter(_apply_iir_filter(ys, b, a)[::-1], b, a)[::-1]
    fz = _apply_iir_filter(_apply_iir_filter(zs, b, a)[::-1], b, a)[::-1]

    return TemporalTrajectory(
        skeleton_id=trajectory.skeleton_id,
        keypoint_id=trajectory.keypoint_id,
        timestamps_ns=trajectory.timestamps_ns,
        positions_m=tuple((fx[i], fy[i], fz[i]) for i in range(n)),
        covariances_m2=trajectory.covariances_m2,
        availabilities=trajectory.availabilities,
    )


def smooth_trajectory_savgol(
    trajectory: TemporalTrajectory,
    config: SavitzkyGolayConfig | None = None,
) -> TemporalTrajectory:
    cfg = config or SavitzkyGolayConfig()
    n, w = len(trajectory.positions_m), cfg.window_length
    if n < w:
        return trajectory
    half = w // 2
    c7 = [
        -2.0 / 21.0,
        3.0 / 21.0,
        6.0 / 21.0,
        7.0 / 21.0,
        6.0 / 21.0,
        3.0 / 21.0,
        -2.0 / 21.0,
    ]
    coeffs = c7 if w == 7 else [1.0 / w] * w

    def smooth_1d(arr: list[float]) -> list[float]:
        res = list(arr)
        for i in range(half, n - half):
            res[i] = sum(coeffs[k] * arr[i - half + k] for k in range(w))
        return res

    xs = smooth_1d([p[0] for p in trajectory.positions_m])
    ys = smooth_1d([p[1] for p in trajectory.positions_m])
    zs = smooth_1d([p[2] for p in trajectory.positions_m])

    return TemporalTrajectory(
        skeleton_id=trajectory.skeleton_id,
        keypoint_id=trajectory.keypoint_id,
        timestamps_ns=trajectory.timestamps_ns,
        positions_m=tuple((xs[i], ys[i], zs[i]) for i in range(n)),
        covariances_m2=trajectory.covariances_m2,
        availabilities=trajectory.availabilities,
    )


def compute_kinematic_derivatives(
    trajectory: TemporalTrajectory,
) -> KinematicDerivatives:
    n = len(trajectory.timestamps_ns)
    if n < 2:
        raise ValueError("Trajectory must contain at least two points")
    pos, cov = trajectory.positions_m, trajectory.covariances_m2
    times = trajectory.timestamps_ns

    vels: list[Vec3] = []
    v_covs: list[Mat9] = []
    for i in range(n):
        if i == 0:
            dt = (times[1] - times[0]) * 1e-9
            dx = (pos[1][0] - pos[0][0], pos[1][1] - pos[0][1], pos[1][2] - pos[0][2])
            c_factor = 1.0 / (dt * dt)
            c = tuple((cov[0][k] + cov[1][k]) * c_factor for k in range(9))
        elif i == n - 1:
            dt = (times[-1] - times[-2]) * 1e-9
            dx = (
                pos[-1][0] - pos[-2][0],
                pos[-1][1] - pos[-2][1],
                pos[-1][2] - pos[-2][2],
            )
            c_factor = 1.0 / (dt * dt)
            c = tuple((cov[-2][k] + cov[-1][k]) * c_factor for k in range(9))
        else:
            dt = (times[i + 1] - times[i - 1]) * 1e-9
            dx = (
                pos[i + 1][0] - pos[i - 1][0],
                pos[i + 1][1] - pos[i - 1][1],
                pos[i + 1][2] - pos[i - 1][2],
            )
            c_factor = 1.0 / (dt * dt)
            c = tuple((cov[i - 1][k] + cov[i + 1][k]) * c_factor for k in range(9))
        vels.append((dx[0] / dt, dx[1] / dt, dx[2] / dt))
        v_covs.append(c)  # type: ignore[arg-type]

    accs: list[Vec3] = []
    a_covs: list[Mat9] = []
    for i in range(n):
        if i == 0 or i == n - 1:
            accs.append((0.0, 0.0, 0.0))
            a_covs.append((0.0,) * 9)
        else:
            dt = (times[i + 1] - times[i - 1]) * 1e-9
            dv = (
                vels[i + 1][0] - vels[i - 1][0],
                vels[i + 1][1] - vels[i - 1][1],
                vels[i + 1][2] - vels[i - 1][2],
            )
            c_factor = 1.0 / (dt * dt)
            c = tuple(
                (v_covs[i - 1][k] + v_covs[i + 1][k]) * c_factor for k in range(9)
            )
            accs.append((dv[0] / dt, dv[1] / dt, dv[2] / dt))
            a_covs.append(c)  # type: ignore[arg-type]

    return KinematicDerivatives(
        timestamps_ns=times,
        linear_velocities_mps=tuple(vels),
        linear_accelerations_mps2=tuple(accs),
        velocity_covariances=tuple(v_covs),
        acceleration_covariances=tuple(a_covs),
    )


def apply_segment_length_constraint(
    constraint: SegmentLengthConstraint,
    proximal_pos: Vec3,
    distal_pos: Vec3,
    proximal_weight: float = 0.5,
    distal_weight: float = 0.5,
) -> tuple[Vec3, Vec3]:
    dx = distal_pos[0] - proximal_pos[0]
    dy = distal_pos[1] - proximal_pos[1]
    dz = distal_pos[2] - proximal_pos[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d < 1e-9:
        return proximal_pos, distal_pos
    delta_d = constraint.nominal_length_m - d
    w_sum = proximal_weight + distal_weight
    w_prox, w_dist = proximal_weight / w_sum, distal_weight / w_sum
    ux, uy, uz = dx / d, dy / d, dz / d
    return (
        (
            proximal_pos[0] - w_prox * delta_d * ux,
            proximal_pos[1] - w_prox * delta_d * uy,
            proximal_pos[2] - w_prox * delta_d * uz,
        ),
        (
            distal_pos[0] + w_dist * delta_d * ux,
            distal_pos[1] + w_dist * delta_d * uy,
            distal_pos[2] + w_dist * delta_d * uz,
        ),
    )


def apply_joint_angle_constraint(
    constraint: JointAngleConstraint,
    proximal_pos: Vec3,
    joint_pos: Vec3,
    distal_pos: Vec3,
) -> tuple[bool, float]:
    v1 = (
        proximal_pos[0] - joint_pos[0],
        proximal_pos[1] - joint_pos[1],
        proximal_pos[2] - joint_pos[2],
    )
    v2 = (
        distal_pos[0] - joint_pos[0],
        distal_pos[1] - joint_pos[1],
        distal_pos[2] - joint_pos[2],
    )
    d1, d2 = math.sqrt(sum(x * x for x in v1)), math.sqrt(sum(x * x for x in v2))
    if d1 < 1e-9 or d2 < 1e-9:
        return True, 0.0
    cos_ang = max(-1.0, min(1.0, sum(v1[k] * v2[k] for k in range(3)) / (d1 * d2)))
    angle = math.acos(cos_ang)
    return constraint.min_angle_rad <= angle <= constraint.max_angle_rad, angle


def _mat_to_quat(x: Vec3, y: Vec3, z: Vec3) -> tuple[float, float, float, float]:
    r00, r01, r02 = x[0], y[0], z[0]
    r10, r11, r12 = x[1], y[1], z[1]
    r20, r21, r22 = x[2], y[2], z[2]
    tr = r00 + r11 + r22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
    elif (r00 > r11) and (r00 > r22):
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        q = ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        q = ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        q = ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)
    norm = math.sqrt(sum(c * c for c in q))
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def adapt_to_delivery_trajectory(
    butt_trajectory: TemporalTrajectory,
    shaft_guide_trajectory: TemporalTrajectory,
    config: DeliveryTrajectoryMappingConfig,
) -> DeliveryTrajectory:
    n = len(butt_trajectory.timestamps_ns)
    if n < 2 or len(shaft_guide_trajectory.timestamps_ns) != n:
        raise ValueError("Trajectories must have equal length >= 2")

    derivs_butt = compute_kinematic_derivatives(butt_trajectory)
    samples: list[TrajectorySample] = []
    t0_ns = butt_trajectory.timestamps_ns[0]

    for i in range(n):
        t_s = (butt_trajectory.timestamps_ns[i] - t0_ns) * 1e-9
        pos = butt_trajectory.positions_m[i]
        guide = shaft_guide_trajectory.positions_m[i]
        sz = (guide[0] - pos[0], guide[1] - pos[1], guide[2] - pos[2])
        norm_z = math.sqrt(sum(x * x for x in sz))
        z_axis = (
            (0.0, -1.0, 0.0)
            if norm_z < 1e-6
            else (sz[0] / norm_z, sz[1] / norm_z, sz[2] / norm_z)
        )
        up = (0.0, 1.0, 0.0) if abs(z_axis[1]) < 0.9 else (1.0, 0.0, 0.0)
        sx = (
            up[1] * z_axis[2] - up[2] * z_axis[1],
            up[2] * z_axis[0] - up[0] * z_axis[2],
            up[0] * z_axis[1] - up[1] * z_axis[0],
        )
        norm_x = math.sqrt(sum(x * x for x in sx))
        x_axis = (sx[0] / norm_x, sx[1] / norm_x, sx[2] / norm_x)
        y_axis = (
            z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
            z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
            z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
        )

        quat = _mat_to_quat(x_axis, y_axis, z_axis)

        samples.append(
            TrajectorySample(
                time_s=t_s,
                position_m=pos,
                quaternion_wxyz=quat,
                linear_velocity_mps=derivs_butt.linear_velocities_mps[i],
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
            )
        )

    return DeliveryTrajectory(
        source_id=config.source_id,
        frame_id=config.frame_id,
        samples=tuple(samples),
    )
