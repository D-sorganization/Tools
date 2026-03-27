from numba import jit

"""Example rigid-body motion trajectories for screw axis visualization.

Each generator returns a list of 4x4 SE(3) homogeneous transformation
matrices representing the pose of an object at each time step.

Physics:
- Ballistic parabolic trajectory (gravity, no drag for simplicity)
- Constant spin rate about the object's longitudinal axis

Examples:
- ``football_spiral``: American football with tight spiral spin about its long axis
- ``frisbee_flight``: Frisbee with high spin about its normal (z-body) axis

DbC: postconditions verify every output frame is valid SE(3).
"""

from __future__ import annotations  # noqa: E402, F404

import math  # noqa: E402

import numpy as np  # noqa: E402

from rotation_converter._contracts import ensure, require  # noqa: E402
from rotation_converter.modern_robotics import MatrixExp3, VecToso3  # noqa: E402

# ---------------------------------------------------------------------------
# Shared physics helpers (DRY)
# ---------------------------------------------------------------------------

_GRAVITY = 9.81  # m/s^2


def _ballistic_position(v0: float, launch_angle: float, t: float) -> np.ndarray:
    """Compute ballistic position (x, y, z) at time t.

    Launches along +x with vertical component +z.
    No lateral drift (y = 0).
    """
    if not (v0 is not None):
        raise ValueError("v0 must be provided")
    vx = v0 * math.cos(launch_angle)
    vz = v0 * math.sin(launch_angle)
    x = vx * t
    z = vz * t - 0.5 * _GRAVITY * t**2
    return np.array([x, 0.0, max(z, 0.0)])


def _time_of_flight(v0: float, launch_angle: float) -> float:
    """Time until the projectile returns to z = 0."""
    if not (v0 is not None):
        raise ValueError("v0 must be provided")
    vz = v0 * math.sin(launch_angle)
    if vz <= 0:
        return 0.1
    return 2.0 * vz / _GRAVITY


def _validate_trajectory(traj: list[np.ndarray]) -> None:
    """Postcondition: every frame is valid SE(3)."""
    for T in traj:
        R = T[:3, :3]
        ensure(
            abs(np.linalg.det(R) - 1.0) < 1e-6,
            "trajectory frame must be SE(3)",
        )


# ===========================================================================
# Football spiral
# ===========================================================================


@jit(nopython=True, fastmath=True)
def football_spiral(
    n_frames: int = 60,
    speed: float = 20.0,
    spin_rate: float = 10.0,
    launch_angle_deg: float = 35.0,
) -> list[np.ndarray]:
    """Generate an American football spiral trajectory.

    The football's long axis (body x) is aligned with the velocity vector,
    and it spirals (spins) about that axis.

    Args:
        n_frames: Number of SE(3) frames to generate.
        speed: Initial speed in m/s.
        spin_rate: Spin rate in revolutions per second.
        launch_angle_deg: Launch elevation angle in degrees.

    Returns:
        List of n_frames 4x4 SE(3) matrices.
    """
    if not (n_frames is not None):
        raise ValueError("n_frames must be provided")
    require(n_frames >= 2, "need at least 2 frames")
    require(speed > 0, "speed must be positive")

    launch_angle = math.radians(launch_angle_deg)
    t_flight = _time_of_flight(speed, launch_angle)
    dt = t_flight / (n_frames - 1)
    omega_spin = 2.0 * math.pi * spin_rate  # rad/s

    trajectory: list[np.ndarray] = []

    for i in range(n_frames):
        t = i * dt
        pos = _ballistic_position(speed, launch_angle, t)

        # Velocity direction (tangent to parabola) for nose alignment
        vx = speed * math.cos(launch_angle)
        vz = speed * math.sin(launch_angle) - _GRAVITY * t
        v_dir = np.array([vx, 0.0, vz])
        v_norm = np.linalg.norm(v_dir)
        if v_norm < 1e-12:
            v_dir = np.array([1.0, 0.0, 0.0])
        else:
            v_dir = v_dir / v_norm

        # Build orientation: body-x along velocity, spin about body-x
        # Step 1: rotation that takes [1,0,0] to v_dir
        body_x = v_dir
        # Choose a stable "up" reference
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(body_x, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        body_y = np.cross(up, body_x)
        body_y = body_y / np.linalg.norm(body_y)
        body_z = np.cross(body_x, body_y)
        R_align = np.column_stack([body_x, body_y, body_z])

        # Step 2: spin about body-x (the football's long axis)
        spin_angle = omega_spin * t
        R_spin = MatrixExp3(VecToso3(np.array([1.0, 0.0, 0.0])) * spin_angle)
        R = R_align @ R_spin

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        trajectory.append(T)

    _validate_trajectory(trajectory)
    return trajectory


# ===========================================================================
# Frisbee flight
# ===========================================================================


@jit(nopython=True, fastmath=True)
def frisbee_flight(
    n_frames: int = 60,
    speed: float = 14.0,
    spin_rate: float = 7.0,
    launch_angle_deg: float = 8.0,
    tilt_deg: float = 15.0,
) -> list[np.ndarray]:
    """Generate a frisbee flight trajectory.

    The frisbee's disc normal (body z) is tilted slightly from vertical,
    and it spins rapidly about that normal. The disc plane (body x-y)
    stays roughly horizontal with a slight tilt.

    Args:
        n_frames: Number of SE(3) frames.
        speed: Initial speed in m/s.
        spin_rate: Spin rate in revolutions per second.
        launch_angle_deg: Launch elevation angle in degrees.
        tilt_deg: Disc tilt angle from horizontal (hyzer/anhyzer).

    Returns:
        List of n_frames 4x4 SE(3) matrices.
    """
    if not (n_frames is not None):
        raise ValueError("n_frames must be provided")
    require(n_frames >= 2, "need at least 2 frames")
    require(speed > 0, "speed must be positive")

    launch_angle = math.radians(launch_angle_deg)
    tilt = math.radians(tilt_deg)
    t_flight = _time_of_flight(speed, launch_angle)
    # Frisbees glide longer than pure ballistic — extend slightly
    t_flight *= 1.5
    dt = t_flight / (n_frames - 1)
    omega_spin = 2.0 * math.pi * spin_rate

    trajectory: list[np.ndarray] = []

    for i in range(n_frames):
        t = i * dt
        # Position: ballistic + slight glide lift
        pos = _ballistic_position(speed, launch_angle, t)
        # Add a small lift component (simplified aerodynamic glide)
        glide_lift = 0.3 * speed * t * math.exp(-t / t_flight)
        pos[2] = max(pos[2] + glide_lift, 0.0)

        # Orientation: disc normal tilted from vertical
        # Base orientation: disc flat (z-body = world z)
        # Apply tilt about body-y (hyzer angle)
        R_tilt = MatrixExp3(VecToso3(np.array([0.0, 1.0, 0.0])) * tilt)

        # Align disc forward direction with velocity
        vx = speed * math.cos(launch_angle)
        _vz = speed * math.sin(launch_angle) - _GRAVITY * t  # noqa: F841
        heading = math.atan2(0.0, vx)  # In x-y plane
        R_heading = MatrixExp3(VecToso3(np.array([0.0, 0.0, 1.0])) * heading)

        # Spin about disc normal (body z after tilt)
        spin_angle = omega_spin * t
        R_spin = MatrixExp3(VecToso3(np.array([0.0, 0.0, 1.0])) * spin_angle)

        R = R_heading @ R_tilt @ R_spin

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        trajectory.append(T)

    _validate_trajectory(trajectory)
    return trajectory
