"""Screw Axis Visualiser Tab — interactive 3D screw axis animation.

Extracted from the former monolithic main_window.py for god-class decomposition.
"""

from __future__ import annotations

import rotation_converter as rc
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rotation_converter.ui.pyqt6.plot_helpers import get_plot_colors


class ScrewVisualiserTab(QWidget):
    """Interactive 3D screw axis animation (frame-by-frame)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._animator: rc.ScrewAxisAnimator | None = None
        self._frame_idx: int = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._advance_frame)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Example:"))
        self._vis_combo = QComboBox()
        self._vis_combo.addItems(["Football Spiral", "Frisbee Flight"])
        ctrl.addWidget(self._vis_combo)

        ctrl.addWidget(QLabel("Frames:"))
        self._vis_frames = QSpinBox()
        self._vis_frames.setRange(5, 500)
        self._vis_frames.setValue(60)
        ctrl.addWidget(self._vis_frames)

        ctrl.addWidget(QLabel("Speed:"))
        self._vis_speed = QDoubleSpinBox()
        self._vis_speed.setRange(1.0, 100.0)
        self._vis_speed.setValue(14.0)
        ctrl.addWidget(self._vis_speed)

        ctrl.addWidget(QLabel("Spin:"))
        self._vis_spin = QDoubleSpinBox()
        self._vis_spin.setRange(-50.0, 50.0)
        self._vis_spin.setValue(7.0)
        ctrl.addWidget(self._vis_spin)

        ctrl.addWidget(QLabel("Angle:"))
        self._vis_angle = QDoubleSpinBox()
        self._vis_angle.setRange(-90.0, 90.0)
        self._vis_angle.setValue(8.0)
        ctrl.addWidget(self._vis_angle)

        self._vis_gen_btn = QPushButton("Generate")
        ctrl.addWidget(self._vis_gen_btn)

        ctrl2 = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._stop_btn = QPushButton("Stop")
        self._prev_btn = QPushButton("\u25c0 Prev")
        self._next_btn = QPushButton("Next \u25b6")
        ctrl2.addWidget(self._play_btn)
        ctrl2.addWidget(self._stop_btn)
        ctrl2.addWidget(self._prev_btn)
        ctrl2.addWidget(self._next_btn)

        self._chk_screw = QCheckBox("Screw Axis")
        self._chk_screw.setChecked(True)
        self._chk_euler = QCheckBox("Euler")
        self._chk_quat = QCheckBox("Quaternion")
        ctrl2.addWidget(self._chk_screw)
        ctrl2.addWidget(self._chk_euler)
        ctrl2.addWidget(self._chk_quat)

        self._frame_label = QLabel("Frame: 0/0")
        ctrl2.addWidget(self._frame_label)
        ctrl2.addStretch()
        layout.addLayout(ctrl)
        layout.addLayout(ctrl2)

        self._fig = Figure(figsize=(10, 7), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

    def _connect_signals(self) -> None:
        self._vis_gen_btn.clicked.connect(self._generate)
        self._play_btn.clicked.connect(self._play)
        self._stop_btn.clicked.connect(self._timer.stop)
        self._prev_btn.clicked.connect(self._prev_frame)
        self._next_btn.clicked.connect(self._advance_frame)
        self._chk_screw.stateChanged.connect(self._draw_frame)
        self._chk_euler.stateChanged.connect(self._draw_frame)
        self._chk_quat.stateChanged.connect(self._draw_frame)

    def _generate(self) -> None:
        n = self._vis_frames.value()
        speed = self._vis_speed.value()
        spin = self._vis_spin.value()
        angle = self._vis_angle.value()

        if self._vis_combo.currentIndex() == 0:
            traj = rc.football_spiral(
                n_frames=n, speed=speed, spin_rate=spin, launch_angle_deg=angle
            )
            title = "Football Spiral"
        else:
            traj = rc.frisbee_flight(
                n_frames=n,
                speed=speed,
                spin_rate=spin,
                launch_angle_deg=angle,
                tilt_deg=15.0,
            )
            title = "Frisbee Flight"

        self._animator = rc.ScrewAxisAnimator(traj, title=title)
        self._frame_idx = 0
        self._draw_frame()

    def _play(self) -> None:
        if self._animator:
            self._timer.start()

    def _advance_frame(self) -> None:
        if self._animator is None:
            return
        self._frame_idx = (self._frame_idx + 1) % self._animator.n_frames
        self._draw_frame()

    def _prev_frame(self) -> None:
        if self._animator is None:
            return
        self._frame_idx = (self._frame_idx - 1) % self._animator.n_frames
        self._draw_frame()

    def _draw_frame(self) -> None:
        if self._animator is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection="3d")
        c = get_plot_colors()
        ax.set_facecolor(c["bg"])
        self._fig.set_facecolor(c["bg"])
        ax.tick_params(colors=c["fg"], labelsize=7)

        # Delegate to the animator's draw method
        self._animator.show_screw_axis = self._chk_screw.isChecked()
        self._animator.show_euler = self._chk_euler.isChecked()
        self._animator.show_quaternion = self._chk_quat.isChecked()
        self._animator._draw_frame(ax, self._frame_idx)

        # Override text colours with theme
        for text_obj in ax.texts:
            text_obj.set_color(c["fg"])

        self._fig.tight_layout()
        self._canvas.draw()
        self._frame_label.setText(
            f"Frame: {self._frame_idx + 1}/{self._animator.n_frames}"
        )
