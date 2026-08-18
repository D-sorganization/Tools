"""Private validation and interpolation support for retained wedge sweeps."""

from __future__ import annotations

import math

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
    result: np.ndarray = np.asarray(values, dtype=float)
    return result


def _interval(times: np.ndarray, time_s: float) -> tuple[int, float]:
    index = min(int(np.searchsorted(times, time_s, side="right")) - 1, len(times) - 2)
    index = max(index, 0)
    alpha = (time_s - float(times[index])) / float(times[index + 1] - times[index])
    return index, min(max(float(alpha), 0.0), 1.0)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a scalar-first unit quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        first = axis
        second = (axis + 1) % 3
        third = (axis + 2) % 3
        scale = 2.0 * math.sqrt(
            max(
                1.0
                + float(rotation[first, first])
                - float(rotation[second, second])
                - float(rotation[third, third]),
                0.0,
            )
        )
        vector = np.zeros(3)
        vector[first] = 0.25 * scale
        vector[second] = (rotation[second, first] + rotation[first, second]) / scale
        vector[third] = (rotation[third, first] + rotation[first, third]) / scale
        scalar = (rotation[third, second] - rotation[second, third]) / scale
        values = (scalar, vector[0], vector[1], vector[2])
    quaternion = np.asarray(values, dtype=float)
    result: np.ndarray = quaternion / float(np.linalg.norm(quaternion))
    return result


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    scalar, x_value, y_value, z_value = quaternion
    result: np.ndarray = np.array(
        [
            [
                1.0 - 2.0 * (y_value * y_value + z_value * z_value),
                2.0 * (x_value * y_value - scalar * z_value),
                2.0 * (x_value * z_value + scalar * y_value),
            ],
            [
                2.0 * (x_value * y_value + scalar * z_value),
                1.0 - 2.0 * (x_value * x_value + z_value * z_value),
                2.0 * (y_value * z_value - scalar * x_value),
            ],
            [
                2.0 * (x_value * z_value - scalar * y_value),
                2.0 * (y_value * z_value + scalar * x_value),
                1.0 - 2.0 * (x_value * x_value + y_value * y_value),
            ],
        ]
    )
    return result


def _slerp_rotation(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate proper rotations along the shortest constant-rate arc."""
    start_quaternion = _rotation_to_quaternion(start)
    end_quaternion = _rotation_to_quaternion(end)
    dot = float(np.dot(start_quaternion, end_quaternion))
    if dot < 0.0:
        end_quaternion = -end_quaternion
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 1.0 - 1e-10:
        blended = (1.0 - alpha) * start_quaternion + alpha * end_quaternion
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        blended = (
            math.sin((1.0 - alpha) * angle) / sine * start_quaternion
            + math.sin(alpha * angle) / sine * end_quaternion
        )
    blended /= float(np.linalg.norm(blended))
    return _quaternion_to_rotation(blended)


def interpolated_pose(
    times: np.ndarray, poses: np.ndarray, time_s: float
) -> np.ndarray:
    """Interpolate translation and orientation at constant interval rates."""
    index, alpha = _interval(times, time_s)
    if alpha <= 0.0:
        result: np.ndarray = poses[index].copy()
        return result
    if alpha >= 1.0:
        result = poses[index + 1].copy()
        return result
    pose: np.ndarray = np.eye(4, dtype=float)
    pose[:3, :3] = _slerp_rotation(
        poses[index, :3, :3], poses[index + 1, :3, :3], alpha
    )
    pose[:3, 3] = (1.0 - alpha) * poses[index, :3, 3] + alpha * poses[index + 1, :3, 3]
    result = np.asarray(pose, dtype=float)
    return result


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
