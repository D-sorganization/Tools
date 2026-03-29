"""Screw axis visualization for rigid-body motions.

Extracts the instantaneous screw axis between consecutive SE(3) poses
and builds animation frame data for matplotlib 3D rendering.

Components:
- ``extract_screw_axes_from_trajectory``: SE(3) pairs -> screw parameters
- ``build_animation_frames``: trajectory -> renderable frame dicts
- ``ScrewAxisAnimator``: configures and runs the 3D animation

The animator displays:
- Object position trail (3D curve)
- Body-frame axes (RGB = xyz) at each frame
- Instantaneous screw axis (magenta arrow through space)
- Screw axis pitch indicator

DbC: validates SE(3) inputs, ensures finite outputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require
from rotation_converter.modern_robotics import (
    MatrixLog6,
    TransInv,
    TransToRp,
    se3ToVec,
)

# ===========================================================================
# Screw axis extraction
# ===========================================================================


def extract_screw_axes_from_trajectory(
    trajectory: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Extract the screw axis between each consecutive pair of SE(3) poses.

    For each pair (T_i, T_{i+1}), computes the relative transform
    dT = T_i^{-1} * T_{i+1}, then extracts the screw axis from its
    matrix logarithm.

    Args:
        trajectory: List of N 4x4 SE(3) matrices.

    Returns:
        List of N-1 dicts, each with keys:
        - ``axis``: 3-vector unit screw axis direction
        - ``point``: 3-vector point on the screw axis (in world frame)
        - ``pitch``: scalar (translation per radian, inf for pure translation)
        - ``theta``: scalar rotation angle (radians)
        - ``midpoint``: 3-vector midpoint between consecutive positions
    """
    require(len(trajectory) >= 2, "need at least 2 frames")

    screw_axes: list[dict[str, Any]] = []

    for i in range(len(trajectory) - 1):
        T1 = trajectory[i]
        T2 = trajectory[i + 1]

        # Relative motion in world frame
        dT = TransInv(T1) @ T2

        # Extract twist via matrix logarithm
        se3_mat = MatrixLog6(dT)
        V = se3ToVec(se3_mat)
        omega = V[:3]
        v = V[3:]

        omega_norm = np.linalg.norm(omega)
        v_norm = np.linalg.norm(v)

        # Midpoint between consecutive positions (for rendering location)
        p1 = T1[:3, 3]
        p2 = T2[:3, 3]
        midpoint = (p1 + p2) / 2.0

        if omega_norm < 1e-12 and v_norm < 1e-12:
            # No motion
            screw_axes.append(
                {
                    "axis": np.array([1.0, 0.0, 0.0]),
                    "point": midpoint,
                    "pitch": 0.0,
                    "theta": 0.0,
                    "midpoint": midpoint,
                }
            )
            continue

        if omega_norm < 1e-12:
            # Pure translation
            axis = v / v_norm
            screw_axes.append(
                {
                    "axis": axis,
                    "point": midpoint,
                    "pitch": float("inf"),
                    "theta": 0.0,
                    "midpoint": midpoint,
                }
            )
            continue

        # General screw motion
        theta = omega_norm
        axis_body = omega / omega_norm
        pitch = float(np.dot(omega, v) / (omega_norm**2))

        # Point on screw axis (in body frame of T1)
        q_body = np.cross(omega, v) / (omega_norm**2)

        # Transform axis and point to world frame
        R1 = T1[:3, :3]
        axis_world = R1 @ axis_body
        point_world = R1 @ q_body + p1

        screw_axes.append(
            {
                "axis": axis_world,
                "point": point_world,
                "pitch": pitch,
                "theta": theta,
                "midpoint": midpoint,
            }
        )

    return screw_axes


# ===========================================================================
# Animation frame builder
# ===========================================================================

_BODY_AXIS_LENGTH = 0.3  # Length of body-frame axis arrows


def build_animation_frames(
    trajectory: list[np.ndarray],
    body_axis_length: float = _BODY_AXIS_LENGTH,
) -> list[dict[str, Any]]:
    """Build renderable frame data for each time step.

    For each frame, produces:
    - position: 3-vector world position
    - orientation: 3x3 rotation matrix
    - body_axes: list of 3 dicts {origin, direction} for x/y/z body axes
    - screw_axis: dict with screw axis data (None for first frame)

    Args:
        trajectory: List of N SE(3) matrices.
        body_axis_length: Length of body-frame axis arrows for display.

    Returns:
        List of N frame dicts.
    """
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    require(len(trajectory) >= 1, "need at least 1 frame")

    screw_axes = (
        extract_screw_axes_from_trajectory(trajectory) if len(trajectory) > 1 else []
    )
    frames: list[dict[str, Any]] = []

    for i, T in enumerate(trajectory):
        R, p = TransToRp(T)
        # Body frame axes in world coordinates
        body_axes = [
            {
                "origin": p.copy(),
                "direction": R[:, col] * body_axis_length,
            }
            for col in range(3)
        ]

        frame: dict[str, Any] = {
            "position": p,
            "orientation": R,
            "body_axes": body_axes,
            "screw_axis": screw_axes[i - 1] if i > 0 else None,
        }
        frames.append(frame)

    ensure(len(frames) == len(trajectory), "must produce one frame per pose")
    return frames


# ===========================================================================
# ScrewAxisAnimator
# ===========================================================================


class ScrewAxisAnimator:
    """Configures and runs a 3D matplotlib animation of screw axes.

    Usage::

        from rotation_converter.motion_examples import football_spiral
        traj = football_spiral(n_frames=60)
        animator = ScrewAxisAnimator(traj, title="Football Spiral")
        animator.show()  # Opens interactive matplotlib window
        # or
        animator.save("football.gif", fps=30)

    The animation shows:
    - A fading position trail (white curve)
    - Body-frame coordinate axes (R=x, G=y, B=z)
    - The instantaneous screw axis (magenta arrow)
    - Screw axis pitch as arrow thickness
    """

    def __init__(
        self,
        trajectory: list[np.ndarray],
        title: str = "Screw Axis Visualization",
        screw_axis_length: float = 1.5,
        trail_length: int = 20,
    ) -> None:
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        require(len(trajectory) >= 2, "need at least 2 frames")
        self._trajectory = trajectory
        self._title = title
        self._screw_axis_length = screw_axis_length
        self._trail_length = trail_length
        self._frames = build_animation_frames(trajectory)
        self.show_screw_axis = True
        self.show_euler = False
        self.show_quaternion = False

    @property
    def n_frames(self) -> int:
        return len(self._trajectory)

    @property
    def title(self) -> str:
        return self._title

    @property
    def frames(self) -> list[dict[str, Any]]:
        return self._frames

    @property
    def trajectory_path(self) -> np.ndarray:
        """Nx3 array of position points along the trajectory."""
        return np.array([T[:3, 3] for T in self._trajectory])

    def get_plot_bounds(self) -> dict[str, tuple[float, float]]:
        """Compute axis-aligned bounding box with padding."""
        path = self.trajectory_path
        margin = 2.0
        return {
            "x": (float(path[:, 0].min() - margin), float(path[:, 0].max() + margin)),
            "y": (float(path[:, 1].min() - margin), float(path[:, 1].max() + margin)),
            "z": (float(path[:, 2].min() - margin), float(path[:, 2].max() + margin)),
        }

    def _setup_axes(self, ax: Any) -> None:
        """Configure the 3D axes appearance."""
        bounds = self.get_plot_bounds()
        ax.set_xlim(*bounds["x"])
        ax.set_ylim(*bounds["y"])
        ax.set_zlim(*bounds["z"])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(self._title)
        # Equal aspect ratio
        max_range = max(
            bounds["x"][1] - bounds["x"][0],
            bounds["y"][1] - bounds["y"][0],
            bounds["z"][1] - bounds["z"][0],
        )
        mid_x = sum(bounds["x"]) / 2
        mid_y = sum(bounds["y"]) / 2
        mid_z = sum(bounds["z"]) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    def _draw_frame(self, ax: Any, frame_idx: int) -> None:
        """Draw a single animation frame onto the axes."""
        if not (frame_idx is not None):
            raise ValueError("frame_idx must be provided")
        ax.cla()
        self._setup_axes(ax)

        frame = self._frames[frame_idx]
        path = self.trajectory_path

        # Trail: draw past positions
        trail_start = max(0, frame_idx - self._trail_length)
        trail = path[trail_start : frame_idx + 1]
        if len(trail) > 1:
            ax.plot(
                trail[:, 0],
                trail[:, 1],
                trail[:, 2],
                color="white",
                alpha=0.6,
                linewidth=1.5,
            )
        # Full path (faint)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="gray",
            alpha=0.15,
            linewidth=0.5,
            linestyle="--",
        )

        # Current position marker
        pos = frame["position"]
        ax.scatter(*pos, color="white", s=40, zorder=5)

        # Body-frame axes (RGB = XYZ)
        colors = ["red", "green", "blue"]
        labels = ["x", "y", "z"]
        for i, ax_data in enumerate(frame["body_axes"]):
            origin = ax_data["origin"]
            direction = ax_data["direction"]
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=colors[i],
                linewidth=2.0,
                arrow_length_ratio=0.15,
                label=f"body-{labels[i]}" if frame_idx == 1 else None,
            )

        # Screw axis
        screw = frame.get("screw_axis")
        if self.show_screw_axis and screw is not None and abs(screw["theta"]) > 1e-8:
            s_axis = screw["axis"]
            s_point = screw["point"]
            s_len = self._screw_axis_length

            # Draw screw axis as a long line through the point
            p1 = s_point - s_axis * s_len
            p2 = s_point + s_axis * s_len
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="magenta",
                linewidth=2.5,
                alpha=0.8,
            )
            # Arrow head showing rotation direction
            ax.quiver(
                s_point[0],
                s_point[1],
                s_point[2],
                s_axis[0] * s_len * 0.6,
                s_axis[1] * s_len * 0.6,
                s_axis[2] * s_len * 0.6,
                color="magenta",
                linewidth=2.0,
                arrow_length_ratio=0.2,
                alpha=0.9,
            )

            # Pitch indicator: if finite pitch, show translation component
            pitch = screw["pitch"]
            if pitch != float("inf") and abs(pitch) > 0.01:
                # Small offset arrows along the axis showing helical advance
                for sign in [-0.3, 0.3]:
                    offset_pt = s_point + s_axis * sign * s_len
                    ax.scatter(
                        offset_pt[0],
                        offset_pt[1],
                        offset_pt[2],
                        color="cyan",
                        s=15,
                        alpha=0.6,
                    )

        # Frame counter
        ax.text2D(
            0.02,
            0.95,
            f"Frame {frame_idx + 1}/{self.n_frames}",
            transform=ax.transAxes,
            color="white",
            fontsize=9,
        )
        y_text = 0.90
        if self.show_screw_axis and screw is not None:
            theta_deg = math.degrees(screw["theta"])
            pitch_str = (
                "pitch=inf"
                if screw["pitch"] == float("inf")
                else f"pitch={screw['pitch']:.3f}"
            )
            ax.text2D(
                0.02,
                y_text,
                f"theta={theta_deg:.1f} deg  {pitch_str}",
                transform=ax.transAxes,
                color="magenta",
                fontsize=8,
            )
            y_text -= 0.05

        if self.show_euler or self.show_quaternion:
            import rotation_converter.core as rc_core
            from rotation_converter.converter import Rotation

            rot = Rotation.from_rotation_matrix(frame["orientation"])
            if self.show_euler:
                eu = rot.as_euler("xyz")
                ax.text2D(
                    0.02,
                    y_text,
                    f"Euler (xyz): [{eu[0]:.2f}, {eu[1]:.2f}, {eu[2]:.2f}] rad",
                    transform=ax.transAxes,
                    color="cyan",
                    fontsize=8,
                )
                y_text -= 0.05
            if self.show_quaternion:
                q = rc_core.rotation_matrix_to_quaternion(frame["orientation"])
                ax.text2D(
                    0.02,
                    y_text,
                    f"Quat (wxyz): [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
                    transform=ax.transAxes,
                    color="yellow",
                    fontsize=8,
                )
                y_text -= 0.05

    def show(self, interval: int = 50) -> None:
        """Display the animation in an interactive matplotlib window.

        Args:
            interval: Milliseconds between frames.
        """
        if not (interval is not None):
            raise ValueError("interval must be provided")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        ax.tick_params(colors="gray")

        def update(frame_idx: int) -> None:
            self._draw_frame(ax, frame_idx)

        _anim = FuncAnimation(  # noqa: F841 — must keep reference to prevent GC
            fig,
            update,  # type: ignore[arg-type]
            frames=self.n_frames,
            interval=interval,
            repeat=True,
        )
        plt.tight_layout()
        plt.show()

    def save(
        self,
        filepath: str,
        fps: int = 30,
        dpi: int = 100,
    ) -> None:
        """Save the animation to a file (GIF, MP4, etc.).

        Args:
            filepath: Output file path (e.g. "football.gif").
            fps: Frames per second.
            dpi: Resolution.
        """
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        ax.tick_params(colors="gray")

        def update(frame_idx: int) -> None:
            self._draw_frame(ax, frame_idx)

        anim = FuncAnimation(
            fig,
            update,  # type: ignore[arg-type]
            frames=self.n_frames,
            interval=1000 // fps,
            repeat=False,
        )
        anim.save(filepath, fps=fps, dpi=dpi)
        plt.close(fig)

    def save_snapshot(
        self,
        filepath: str,
        frame_idx: int = 0,
        dpi: int = 150,
    ) -> None:
        """Save a single frame as a static image.

        Args:
            filepath: Output image path (e.g. "frame_10.png").
            frame_idx: Which frame to render.
            dpi: Resolution.
        """
        if not (filepath is not None):
            raise ValueError("filepath must be provided")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        ax.tick_params(colors="gray")
        self._draw_frame(ax, frame_idx)
        fig.savefig(filepath, dpi=dpi, facecolor="black", bbox_inches="tight")
        plt.close(fig)
