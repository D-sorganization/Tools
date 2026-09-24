# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Optimized swing source adapting movement-optimizer motion results (epic #4103).

Satisfies the :class:`~shared.python.swing_sim.swing_source.SwingSource` protocol
by wrapping an :class:`OptimizedSwingResult` or duck-typed optimizer result,
validating its SE(3) kinematics and trajectory bounds, and performing quaternion
SLERP orientation interpolation and linear position/twist interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rate_of_closure._contracts import require
from shared.python.swing_sim.types import SwingSample

if TYPE_CHECKING:
    pass

__all__ = [
    "OptimizedSwingResult",
    "OptimizedSwingSource",
]


@dataclass(frozen=True)
class OptimizedSwingResult:
    """Validated result container for optimized golf swing motion.

    Attributes:
        time_s: Monotonically increasing time grid [s], shape ``(N,)``.
        clubhead_poses: World/swing-frame SE(3) transforms, shape ``(N, 4, 4)``.
        clubhead_twists: Spatial twist [wx, wy, wz, vx, vy, vz], shape ``(N, 6)``.
        frame_convention: Frame identifier (``"swing_frame"`` or ``"app_frame"``).
        joint_positions_m: Optional articulated joint trajectories, shape ``(N, J, 3)``.
        joint_ids: Optional tuple of semantic joint identifiers, length ``J``.
        joint_torques_nm: Optional joint torques, shape ``(N, J)``.
        success: Optimization outcome flag.
    """

    time_s: np.ndarray
    clubhead_poses: np.ndarray
    clubhead_twists: np.ndarray
    frame_convention: str = "swing_frame"
    joint_positions_m: np.ndarray | None = None
    joint_ids: tuple[str, ...] | None = None
    joint_torques_nm: np.ndarray | None = None
    success: bool = True

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=float)
        require(
            times.ndim == 1 and len(times) >= 2,
            "time_s must be a 1D array of at least 2 points",
            times.shape,
        )
        require(bool(np.all(np.isfinite(times))), "time_s must be finite")
        require(abs(times[0]) < 1e-9, "time_s must start at 0.0", times[0])
        diffs = np.diff(times)
        require(bool(np.all(diffs > 0.0)), "time_s must be strictly increasing", diffs)

        n = len(times)
        poses = np.asarray(self.clubhead_poses, dtype=float)
        require(
            poses.shape == (n, 4, 4),
            f"clubhead_poses must have shape ({n}, 4, 4)",
            poses.shape,
        )
        require(bool(np.all(np.isfinite(poses))), "clubhead_poses must be finite")

        # Orthonormality check for rotation block of each sample
        for i in range(n):
            r = poses[i, :3, :3]
            eye = r.T @ r
            require(
                bool(np.allclose(eye, np.eye(3), atol=1e-3)),
                f"clubhead_poses[{i}] rotation block must be orthonormal",
                eye,
            )
            require(
                abs(np.linalg.det(r) - 1.0) < 1e-3,
                f"clubhead_poses[{i}] rotation block must have det +1",
            )

        twists = np.asarray(self.clubhead_twists, dtype=float)
        require(
            twists.shape == (n, 6),
            f"clubhead_twists must have shape ({n}, 6)",
            twists.shape,
        )
        require(bool(np.all(np.isfinite(twists))), "clubhead_twists must be finite")

        require(
            self.frame_convention in ("swing_frame", "app_frame"),
            f"frame_convention must be 'swing_frame' or 'app_frame', got "
            f"{self.frame_convention!r}",
            self.frame_convention,
        )

        if self.joint_positions_m is not None:
            joints = np.asarray(self.joint_positions_m, dtype=float)
            require(
                joints.ndim == 3 and joints.shape[0] == n and joints.shape[2] == 3,
                "joint_positions_m must have shape (N, J, 3)",
                joints.shape,
            )
            require(
                bool(np.all(np.isfinite(joints))),
                "joint_positions_m must be finite",
            )

        if self.joint_torques_nm is not None:
            torques = np.asarray(self.joint_torques_nm, dtype=float)
            require(
                torques.ndim == 2 and torques.shape[0] == n,
                "joint_torques_nm must have shape (N, J)",
                torques.shape,
            )
            require(
                bool(np.all(np.isfinite(torques))),
                "joint_torques_nm must be finite",
            )


class OptimizedSwingSource:
    """Adapts an :class:`OptimizedSwingResult` to the :class:`SwingSource` protocol."""

    def __init__(self, result: OptimizedSwingResult | Any) -> None:
        if not isinstance(result, OptimizedSwingResult):
            # Check duck-typing for OptimizationResult containing required fields
            require(
                hasattr(result, "clubhead_poses") and result.clubhead_poses is not None,
                "result must contain clubhead_poses",
            )
            require(
                hasattr(result, "clubhead_twists")
                and result.clubhead_twists is not None,
                "result must contain clubhead_twists",
            )
            raw_time = getattr(result, "t", getattr(result, "time_s", None))
            require(raw_time is not None, "result must contain time_s or t")
            result = OptimizedSwingResult(
                time_s=np.asarray(raw_time, dtype=float),
                clubhead_poses=np.asarray(result.clubhead_poses, dtype=float),
                clubhead_twists=np.asarray(result.clubhead_twists, dtype=float),
                frame_convention=str(
                    getattr(result, "frame_convention", "swing_frame") or "swing_frame"
                ),
                joint_positions_m=getattr(result, "joint_positions_m", None),
                joint_ids=getattr(result, "joint_ids", None),
                joint_torques_nm=getattr(result, "joint_torques_nm", None),
                success=bool(getattr(result, "success", True)),
            )
        self._result = result
        require(
            result.success,
            "OptimizedSwingResult indicates solver failure",
            result.success,
        )
        self._times = np.asarray(result.time_s, dtype=float)
        self._poses = np.asarray(result.clubhead_poses, dtype=float)
        self._twists = np.asarray(result.clubhead_twists, dtype=float)
        self._duration = float(self._times[-1])
        self._frame_convention = str(result.frame_convention)

        # Precompute rotations and Slerp interpolator
        rotations = Rotation.from_matrix(self._poses[:, :3, :3])
        self._slerp = Slerp(self._times, rotations)

    @property
    def duration(self) -> float:
        """Total duration [s] of the swing."""
        return self._duration

    @property
    def frame_convention(self) -> str:
        """Declared frame convention of the samples."""
        return str(self._frame_convention)

    def sample(self, t: float) -> SwingSample:
        """Return the clubhead sample at time ``t`` in ``[0, duration]``."""
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            f"t ({t}) must be within [0, duration ({self._duration})]",
            t,
        )
        t_clamped = min(max(t, 0.0), self._duration)

        # Interpolate position linearly
        pos_x = np.interp(t_clamped, self._times, self._poses[:, 0, 3])
        pos_y = np.interp(t_clamped, self._times, self._poses[:, 1, 3])
        pos_z = np.interp(t_clamped, self._times, self._poses[:, 2, 3])
        position = np.array([pos_x, pos_y, pos_z])

        # Interpolate orientation using SLERP
        rot_matrix = self._slerp(t_clamped).as_matrix()

        pose = np.eye(4, dtype=float)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = position

        # Interpolate twist linearly
        twist = np.zeros(6, dtype=float)
        for i in range(6):
            twist[i] = np.interp(t_clamped, self._times, self._twists[:, i])

        return SwingSample(t=t_clamped, pose=pose, twist=twist)

    def joint_positions(self, t: float) -> np.ndarray:
        """Articulated joint positions at time ``t``."""
        if self._result.joint_positions_m is None:
            return np.zeros((0, 3), dtype=float)
        joints_arr = np.asarray(self._result.joint_positions_m, dtype=float)
        t_clamped = min(max(t, 0.0), self._duration)
        n_joints = joints_arr.shape[1]
        interp_joints = np.zeros((n_joints, 3), dtype=float)
        for j in range(n_joints):
            for axis in range(3):
                interp_joints[j, axis] = np.interp(
                    t_clamped, self._times, joints_arr[:, j, axis]
                )
        return interp_joints

    @property
    def joint_ids(self) -> tuple[str, ...]:
        """Stable semantic joint identifiers."""
        return self._result.joint_ids or ()

    def joint_torques_at(self, t: float) -> dict[str, float]:
        """Generalized joint torques at time ``t``."""
        if not self.joint_ids:
            return {}
        if self._result.joint_torques_nm is None:
            return {j_id: 0.0 for j_id in self.joint_ids}
        torques_arr = np.asarray(self._result.joint_torques_nm, dtype=float)
        t_clamped = min(max(t, 0.0), self._duration)
        out: dict[str, float] = {}
        for idx, j_id in enumerate(self.joint_ids):
            if idx < torques_arr.shape[1]:
                out[j_id] = float(
                    np.interp(t_clamped, self._times, torques_arr[:, idx])
                )
            else:
                out[j_id] = 0.0
        return out
