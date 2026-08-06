"""Private validation and interpolation support for retained wedge sweeps."""

from __future__ import annotations

import numpy as np

from ._validation import require_rotation


def validated_sweep_arrays(
    times_s: object, poses: object, twists: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy and validate retained time, pose, and twist arrays."""
    try:
        times = np.array(times_s, dtype=float, copy=True)
        pose_array = np.array(poses, dtype=float, copy=True)
        twist_array = np.array(twists, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError("times, poses, and twists must be numeric arrays") from error
    if times.ndim != 1 or len(times) < 2 or not bool(np.all(np.isfinite(times))):
        raise ValueError(
            "times_s must be a finite one-dimensional array with >= 2 samples"
        )
    if not bool(np.all(np.diff(times) > 0.0)):
        raise ValueError("times_s must be strictly increasing")
    pose_shape_valid = pose_array.shape == (len(times), 4, 4)
    if not pose_shape_valid or not bool(np.all(np.isfinite(pose_array))):
        raise ValueError("poses must have finite shape (sample, 4, 4)")
    twist_shape_valid = twist_array.shape == (len(times), 6)
    if not twist_shape_valid or not bool(np.all(np.isfinite(twist_array))):
        raise ValueError("twists must have finite shape (sample, 6)")
    for pose in pose_array:
        require_rotation(pose[:3, :3])
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("every pose must be a homogeneous rigid transform")
    return times, pose_array, twist_array


def swept_times(times: np.ndarray, subdivisions: int) -> np.ndarray:
    """Insert a fixed number of audit samples within every retained interval."""
    values = [float(times[0])]
    for start, end in zip(times[:-1], times[1:], strict=True):
        values.extend(
            float(start + (end - start) * step / subdivisions)
            for step in range(1, subdivisions + 1)
        )
    return np.asarray(values)


def _interval(times: np.ndarray, time_s: float) -> tuple[int, float]:
    index = min(int(np.searchsorted(times, time_s, side="right")) - 1, len(times) - 2)
    index = max(index, 0)
    alpha = (time_s - float(times[index])) / float(times[index + 1] - times[index])
    return index, min(max(float(alpha), 0.0), 1.0)


def interpolated_pose(
    times: np.ndarray, poses: np.ndarray, time_s: float
) -> np.ndarray:
    """Interpolate translation and project the blended rotation onto SO(3)."""
    index, alpha = _interval(times, time_s)
    if alpha <= 0.0:
        result: np.ndarray = poses[index].copy()
        return result
    if alpha >= 1.0:
        result = poses[index + 1].copy()
        return result
    blended = (1.0 - alpha) * poses[index, :3, :3] + alpha * poses[index + 1, :3, :3]
    left, _, right = np.linalg.svd(blended)
    rotation = left @ right
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = (1.0 - alpha) * poses[index, :3, 3] + alpha * poses[index + 1, :3, 3]
    return pose


def interpolated_twist(
    times: np.ndarray, twists: np.ndarray, time_s: float
) -> np.ndarray:
    """Linearly interpolate an inertial-frame angular/linear twist."""
    index, alpha = _interval(times, time_s)
    result: np.ndarray = (1.0 - alpha) * twists[index] + alpha * twists[index + 1]
    return result


__all__ = [
    "interpolated_pose",
    "interpolated_twist",
    "swept_times",
    "validated_sweep_arrays",
]
