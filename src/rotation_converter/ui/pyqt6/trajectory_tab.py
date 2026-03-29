"""Trajectory Plots Tab — screw axes, Euler angles, quaternions, angular velocity.

Extracted from the former monolithic main_window.py for god-class decomposition.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import rotation_converter as rc
from rotation_converter.converter import Rotation
from rotation_converter.rigid_transform import RigidTransform
from rotation_converter.ui.pyqt6.plot_helpers import (
    get_plot_colors,
    style_figure,
)


class TrajectoryPlotsTab(QWidget):
    """Generate trajectory plots: screw axes, Euler angles, quaternions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._traj: list[np.ndarray] = []
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Example:"))
        self._example_combo = QComboBox()
        self._example_combo.addItems(["Football Spiral", "Frisbee Flight"])
        ctrl.addWidget(self._example_combo)

        ctrl.addWidget(QLabel("Frames:"))
        self._n_frames = QSpinBox()
        self._n_frames.setRange(5, 500)
        self._n_frames.setValue(60)
        ctrl.addWidget(self._n_frames)

        ctrl.addWidget(QLabel("Speed (m/s):"))
        self._speed_input = QDoubleSpinBox()
        self._speed_input.setRange(1.0, 100.0)
        self._speed_input.setValue(20.0)
        ctrl.addWidget(self._speed_input)

        ctrl.addWidget(QLabel("Spin (rev/s):"))
        self._spin_input = QDoubleSpinBox()
        self._spin_input.setRange(-50.0, 50.0)
        self._spin_input.setValue(10.0)
        ctrl.addWidget(self._spin_input)

        ctrl.addWidget(QLabel("Angle (deg):"))
        self._angle_input = QDoubleSpinBox()
        self._angle_input.setRange(-90.0, 90.0)
        self._angle_input.setValue(35.0)
        ctrl.addWidget(self._angle_input)

        self._gen_btn = QPushButton("Generate")
        ctrl.addWidget(self._gen_btn)

        ctrl2 = QHBoxLayout()
        ctrl2.addWidget(QLabel("Plot:"))
        self._plot_combo = QComboBox()
        self._plot_combo.addItems(
            [
                "Screw Axis Parameters",
                "Euler Angles Over Time",
                "Quaternion Components",
                "Position Trajectory",
                "Angular Velocity",
                "Body vs Space Twist",
            ]
        )
        ctrl.addWidget(self._plot_combo)

        self._plot_btn = QPushButton("Plot")
        ctrl2.addWidget(self._plot_btn)
        ctrl2.addStretch()

        layout.addLayout(ctrl)
        layout.addLayout(ctrl2)

        # Plot area
        self._fig = Figure(figsize=(10, 5), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

    def _connect_signals(self) -> None:
        self._gen_btn.clicked.connect(self._generate)
        self._plot_btn.clicked.connect(self._plot)

    def _generate(self) -> None:
        n = self._n_frames.value()
        speed = self._speed_input.value()
        spin = self._spin_input.value()
        angle = self._angle_input.value()

        if self._example_combo.currentIndex() == 0:
            self._traj = rc.football_spiral(
                n_frames=n, speed=speed, spin_rate=spin, launch_angle_deg=angle
            )
        else:
            self._traj = rc.frisbee_flight(
                n_frames=n,
                speed=speed,
                spin_rate=spin,
                launch_angle_deg=angle,
                tilt_deg=15.0,
            )
        self._plot()

    def _plot(self) -> None:
        if not self._traj:
            return
        idx = self._plot_combo.currentIndex()
        self._fig.clear()

        if idx == 0:
            self._plot_screw_params()
        elif idx == 1:
            self._plot_euler()
        elif idx == 2:
            self._plot_quaternions()
        elif idx == 3:
            self._plot_position()
        elif idx == 4:
            self._plot_angular_velocity()
        elif idx == 5:
            self._plot_body_space_twist()

        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_screw_params(self) -> None:
        axes_data = rc.extract_screw_axes_from_trajectory(self._traj)
        t = np.arange(len(axes_data))
        thetas = [a["theta"] for a in axes_data]
        pitches = [a["pitch"] if a["pitch"] != float("inf") else 0 for a in axes_data]
        axis_dirs = np.array([a["axis"] for a in axes_data])

        ax1 = self._fig.add_subplot(131)
        ax2 = self._fig.add_subplot(132)
        ax3 = self._fig.add_subplot(133)
        style_figure(self._fig, [ax1, ax2, ax3])

        c = get_plot_colors()
        ax1.plot(t, thetas, color=c["accent"], linewidth=1.5)
        ax1.set_title("Rotation Angle (\u03b8)", fontsize=9)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("rad")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, pitches, color=c["axes"][1], linewidth=1.5)
        ax2.set_title("Screw Pitch", fontsize=9)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("m/rad")
        ax2.grid(True, alpha=0.3)

        for i, lbl in enumerate(["\u03c9x", "\u03c9y", "\u03c9z"]):
            ax3.plot(t, axis_dirs[:, i], color=c["axes"][i], linewidth=1.5, label=lbl)
        ax3.set_title("Screw Axis Direction", fontsize=9)
        ax3.set_xlabel("Step")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

    def _plot_euler(self) -> None:
        n = len(self._traj)
        t = np.arange(n)
        angles = np.zeros((n, 3))
        for i, T in enumerate(self._traj):
            rot = Rotation.from_rotation_matrix(T[:3, :3])
            angles[i] = rot.as_euler("xyz")

        ax = self._fig.add_subplot(111)
        style_figure(self._fig, ax)
        c = get_plot_colors()
        for j, lbl in enumerate(["Roll (X)", "Pitch (Y)", "Yaw (Z)"]):
            ax.plot(
                t, np.degrees(angles[:, j]),
                color=c["axes"][j], linewidth=1.5, label=lbl,
            )
        ax.set_title("Euler Angles (XYZ) Over Time", fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Degrees")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_quaternions(self) -> None:
        n = len(self._traj)
        t = np.arange(n)
        quats = np.zeros((n, 4))
        for i, T in enumerate(self._traj):
            quats[i] = rc.rotation_matrix_to_quaternion(T[:3, :3])

        ax = self._fig.add_subplot(111)
        style_figure(self._fig, ax)
        c = get_plot_colors()
        labels = ["w", "x", "y", "z"]
        qcolors = [c["accent"], c["axes"][0], c["axes"][1], c["axes"][2]]
        for j in range(4):
            ax.plot(t, quats[:, j], color=qcolors[j], linewidth=1.5, label=labels[j])
        ax.set_title("Quaternion Components Over Time", fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_position(self) -> None:
        n = len(self._traj)
        pos = np.array([T[:3, 3] for T in self._traj])

        ax1 = self._fig.add_subplot(121)
        ax2 = self._fig.add_subplot(122, projection="3d")
        style_figure(self._fig, [ax1, ax2])
        c = get_plot_colors()

        t = np.arange(n)
        for j, lbl in enumerate(["X", "Y", "Z"]):
            ax1.plot(t, pos[:, j], color=c["axes"][j], linewidth=1.5, label=lbl)
        ax1.set_title("Position vs Frame", fontsize=10)
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Position (m)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=c["accent"], linewidth=1.5)
        ax2.scatter(  # Axes3D
            pos[0, 0], pos[0, 1], pos[0, 2], color=c["axes"][1], s=40, label="Start"
        )
        ax2.scatter(  # Axes3D
            pos[-1, 0], pos[-1, 1], pos[-1, 2], color=c["axes"][0], s=40, label="End"
        )
        ax2.set_title("3D Trajectory", fontsize=10)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")  # Axes3D
        ax2.legend(fontsize=7)

    def _plot_angular_velocity(self) -> None:
        axes_data = rc.extract_screw_axes_from_trajectory(self._traj)
        t = np.arange(len(axes_data))
        omega = np.array([a["axis"] * a["theta"] for a in axes_data])

        ax = self._fig.add_subplot(111)
        style_figure(self._fig, ax)
        c = get_plot_colors()
        for j, lbl in enumerate(["\u03c9x", "\u03c9y", "\u03c9z"]):
            ax.plot(t, omega[:, j], color=c["axes"][j], linewidth=1.5, label=lbl)
        ax.set_title("Angular Velocity Components", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("rad/step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_body_space_twist(self) -> None:
        n = min(len(self._traj) - 1, len(self._traj))
        t = np.arange(n - 1) if n > 1 else np.array([0])
        body_tw = np.zeros((max(n - 1, 1), 6))
        space_tw = np.zeros_like(body_tw)

        for i in range(min(n - 1, len(self._traj) - 1)):
            T1 = self._traj[i]
            T2 = self._traj[i + 1]
            dT = RigidTransform.from_matrix(
                np.linalg.inv(T1) @ T2, source="b", target="a"
            )
            body_tw[i] = dT.body_twist()
            space_tw[i] = dT.space_twist()

        ax1 = self._fig.add_subplot(121)
        ax2 = self._fig.add_subplot(122)
        style_figure(self._fig, [ax1, ax2])
        c = get_plot_colors()

        labels_w = ["\u03c9x", "\u03c9y", "\u03c9z"]
        labels_v = ["vx", "vy", "vz"]
        for j in range(3):
            ax1.plot(
                t, body_tw[: len(t), j],
                color=c["axes"][j], linewidth=1.5, label=f"body {labels_w[j]}",
            )
            ax1.plot(
                t, space_tw[: len(t), j],
                color=c["axes"][j], linewidth=1.5, linestyle="--",
                label=f"space {labels_w[j]}",
            )
        ax1.set_title("Angular: Body (solid) vs Space (dashed)", fontsize=9)
        ax1.set_xlabel("Step")
        ax1.legend(fontsize=6, ncol=2)
        ax1.grid(True, alpha=0.3)

        for j in range(3):
            ax2.plot(
                t, body_tw[: len(t), 3 + j],
                color=c["axes"][j], linewidth=1.5, label=f"body {labels_v[j]}",
            )
            ax2.plot(
                t, space_tw[: len(t), 3 + j],
                color=c["axes"][j], linewidth=1.5, linestyle="--",
                label=f"space {labels_v[j]}",
            )
        ax2.set_title("Linear: Body (solid) vs Space (dashed)", fontsize=9)
        ax2.set_xlabel("Step")
        ax2.legend(fontsize=6, ncol=2)
        ax2.grid(True, alpha=0.3)
