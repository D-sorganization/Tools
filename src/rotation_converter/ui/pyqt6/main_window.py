# ruff: noqa: E501
# mypy: disable-error-code="attr-defined, misc"
"""Rotation Converter Main Window -- PyQt6 GUI.

Tabbed interface providing:
1. Rotation Converter -- live pairwise conversion between all representations
2. Rigid Transform -- frame-aware SE(3) with body/space twist conversion
3. Trajectory Plots -- screw axis, Euler, quaternion, and body-frame plots
4. 3D Visualiser -- interactive coordinate-frame and screw-axis rendering

Each tab is implemented in its own module (god-class decomposition):
- rotation_tab.py       -> RotationConverterTab
- rigid_transform_tab.py -> RigidTransformTab
- trajectory_tab.py     -> TrajectoryPlotsTab
- screw_visualiser_tab.py -> ScrewVisualiserTab
- console_tab.py        -> CommandConsoleTab
- reference_frame_tab.py -> ReferenceFrameTab

Shared plotting helpers live in plot_helpers.py.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import rotation_converter as rc
from rotation_converter.ui.pyqt6.console_tab import CommandConsoleTab
from rotation_converter.ui.pyqt6.plot_helpers import (
    EULER_CONVENTIONS,
)
from rotation_converter.ui.pyqt6.plot_helpers import (
    fmt_mat as _fmt_mat,
)
from rotation_converter.ui.pyqt6.plot_helpers import (
    fmt_vec as _fmt_vec,
)
from rotation_converter.ui.pyqt6.plot_helpers import (
    get_plot_colors as _get_plot_colors,
)
from rotation_converter.ui.pyqt6.plot_helpers import (
    parse_vec as _parse_vec,
)
from rotation_converter.ui.pyqt6.plot_helpers import (
    style_figure as _style_figure,
)
from rotation_converter.ui.pyqt6.reference_frame_tab import ReferenceFrameTab
from shared.python.theme.integration import ThemedWindowMixin, get_theme_manager
from shared.python.theme.matplotlib_style import apply_plot_theme

logger = logging.getLogger(__name__)

# ── Rotation and Transform imports ──────────────────────────────────
from rotation_converter import Rotation  # noqa: E402
from rotation_converter.rigid_transform import RigidTransform  # noqa: E402

# Re-export tab classes for backward compatibility
__all__ = [
    "RotationConverterMainWindow",
    "RotationConverterTab",
    "RigidTransformTab",
    "TrajectoryPlotsTab",
    "ScrewVisualiserTab",
]


# =====================================================================
# Rotation Converter Tab
# =====================================================================


class RotationConverterTab(QWidget):
    """Live conversion between all rotation representations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()
        self._update_outputs()

    def _build_input_group(self) -> QGroupBox:
        """Build the left-side input group with representation selector and value entry."""  # noqa: E501
        input_group = QGroupBox("Input Rotation")
        input_layout = QVBoxLayout(input_group)
        form = QFormLayout()
        self._repr_combo = QComboBox()
        self._repr_combo.addItems(
            [
                "Quaternion (w,x,y,z)",
                "Euler Angles (rad)",
                "Axis-Angle",
                "Rodrigues Vector",
                "Rotation Matrix (row-major)",
            ]
        )
        form.addRow("Representation:", self._repr_combo)
        self._euler_conv = QComboBox()
        self._euler_conv.addItems(EULER_CONVENTIONS)
        form.addRow("Euler Convention:", self._euler_conv)
        self._input_edit = QTextEdit()
        self._input_edit.setPlaceholderText("Enter values (space or comma separated)")
        self._input_edit.setMaximumHeight(100)
        self._input_edit.setText("1.0 0.0 0.0 0.0")
        form.addRow("Values:", self._input_edit)
        self._convert_btn = QPushButton("Convert")
        form.addRow(self._convert_btn)
        input_layout.addLayout(form)
        return input_group

    def _build_output_section(self) -> QVBoxLayout:
        """Build the output section with target combo, result display, and all-repr text."""  # noqa: E501
        right = QVBoxLayout()
        target_group = QGroupBox("Main Output Extraction")
        target_layout = QFormLayout(target_group)
        self._target_repr = QComboBox()
        self._target_repr.addItems(
            [
                "Quaternion (w,x,y,z)",
                "Euler Angles (rad)",
                "Axis-Angle",
                "Rodrigues Vector",
                "Rotation Matrix (row-major)",
            ]
        )
        self._target_euler_conv = QComboBox()
        self._target_euler_conv.addItems(EULER_CONVENTIONS)
        target_layout.addRow("Convert To:", self._target_repr)
        target_layout.addRow("Euler Convention:", self._target_euler_conv)
        self._main_result = QLineEdit()
        self._main_result.setReadOnly(True)
        self._main_result.setStyleSheet("font-size: 16px; font-weight: bold;")
        target_layout.addRow("Result:", self._main_result)
        right.addWidget(target_group, 1)
        output_group = QGroupBox("All Representations")
        output_layout = QVBoxLayout(output_group)
        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        output_layout.addWidget(self._output_text)
        right.addWidget(output_group, 2)
        return right

    def _build_plot_section(self) -> QGroupBox:
        """Build the 3D rotation axes matplotlib plot group."""
        plot_group = QGroupBox("3D Rotation Axes")
        plot_layout = QVBoxLayout(plot_group)
        self._fig = Figure(figsize=(4, 3), dpi=100)
        _tm = get_theme_manager()
        apply_plot_theme(self._fig, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self._fig, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setMaximumHeight(30)
        plot_layout.addWidget(self._toolbar)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout.addWidget(self._canvas)
        return plot_group

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.addWidget(self._build_input_group(), 1)
        right = self._build_output_section()
        right.addWidget(self._build_plot_section(), 3)
        layout.addLayout(right, 2)

    def _connect_signals(self) -> None:
        self._convert_btn.clicked.connect(self._update_outputs)
        self._repr_combo.currentIndexChanged.connect(self._on_repr_changed)
        self._target_repr.currentIndexChanged.connect(self._update_outputs)
        self._target_euler_conv.currentIndexChanged.connect(self._update_outputs)

    def _on_repr_changed(self) -> None:
        idx = self._repr_combo.currentIndex()
        self._euler_conv.setEnabled(idx == 1)
        placeholders = [
            "w x y z  (e.g. 1 0 0 0)",
            "a b c  in radians (e.g. 0.1 0.2 0.3)",
            "ax ay az angle  (e.g. 0 0 1 1.5708)",
            "rx ry rz  (e.g. 0 0 0.5)",
            "r00 r01 r02 r10 r11 r12 r20 r21 r22  (row-major)",
        ]
        self._input_edit.setPlaceholderText(placeholders[idx])

    def _update_outputs(self) -> None:
        text = self._input_edit.toPlainText().strip()
        v = _parse_vec(text)
        if v is None:
            self._output_text.setPlainText("Invalid input — enter numeric values")
            return
        idx = self._repr_combo.currentIndex()
        conv = self._euler_conv.currentText()

        try:
            if idx == 0:  # Quaternion
                rot = Rotation.from_quaternion(v)
            elif idx == 1:  # Euler
                if len(v) != 3:
                    raise ValueError("Euler angles need 3 values")
                rot = Rotation.from_euler(v[0], v[1], v[2], conv)
            elif idx == 2:  # Axis-angle
                if len(v) != 4:
                    raise ValueError("Axis-angle needs 4 values (ax ay az angle)")
                axis = v[:3] / np.linalg.norm(v[:3])
                rot = Rotation.from_axis_angle(axis, v[3])
            elif idx == 3:  # Rodrigues
                rot = Rotation.from_rodrigues(v)
            elif idx == 4:  # Rotation matrix
                if len(v) != 9:
                    raise ValueError("Rotation matrix needs 9 values (row-major)")
                R = v.reshape(3, 3)
                rot = Rotation.from_rotation_matrix(R)
            else:
                return
        except Exception as e:  # noqa: BLE001 — user input can raise any error; display it
            self._output_text.setPlainText(f"Error: {e}")
            return

        self._display_all(rot, conv)
        self._update_main_result(rot)
        self._draw_rotation(rot)

    def _update_main_result(self, rot: Rotation) -> None:
        if rot is None:
            raise ValueError("rot must be provided")
        idx = self._target_repr.currentIndex()
        conv = self._target_euler_conv.currentText()
        try:
            self._target_euler_conv.setEnabled(idx == 1)
            if idx == 0:
                res = _fmt_vec(rot.as_quaternion())
            elif idx == 1:
                res = _fmt_vec(np.array(rot.as_euler(conv)))
            elif idx == 2:
                ax, ang = rot.as_axis_angle()
                res = f"{_fmt_vec(ax)}  {ang:.6f}"
            elif idx == 3:
                res = _fmt_vec(rot.as_rodrigues())
            elif idx == 4:
                R = rot.as_rotation_matrix()
                res = _fmt_vec(R.flatten())
            else:
                res = ""
            self._main_result.setText(res)
        except Exception as e:  # noqa: BLE001 — rotation conversion may raise any arithmetic error
            self._main_result.setText(f"Error: {e}")

    def _display_all(self, rot: Rotation, conv: str) -> None:
        if rot is None:
            raise ValueError("rot must be provided")
        q = rot.as_quaternion()
        R = rot.as_rotation_matrix()
        axis, angle = rot.as_axis_angle()
        rod = rot.as_rodrigues()

        lines = [
            "━━━ Quaternion (w, x, y, z) ━━━",
            _fmt_vec(q),
            "",
            f"━━━ Euler Angles [{conv}] (rad) ━━━",
        ]
        for c in EULER_CONVENTIONS:
            try:
                e = rot.as_euler(c)
                marker = " ◀" if c == conv else ""
                lines.append(f"  {c}: {e[0]: .6f}  {e[1]: .6f}  {e[2]: .6f}{marker}")
            except Exception:  # noqa: BLE001 — Euler conversion may fail for degenerate rotations
                lines.append(f"  {c}: (error)")
        lines += [
            "",
            "━━━ Axis-Angle ━━━",
            f"  axis:  {_fmt_vec(axis)}",
            f"  angle: {angle:.6f} rad ({math.degrees(angle):.2f} deg)",
            "",
            "━━━ Rodrigues Vector ━━━",
            f"  {_fmt_vec(rod)}",
            "",
            "━━━ Rotation Matrix ━━━",
            _fmt_mat(R),
        ]
        self._output_text.setPlainText("\n".join(lines))

    def _draw_rotation(self, rot: Rotation) -> None:
        if rot is None:
            raise ValueError("rot must be provided")
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection="3d")
        _style_figure(self._fig, ax)

        R = rot.as_rotation_matrix()
        origin = np.zeros(3)
        colors = _get_plot_colors()["axes"]
        labels = ["X", "Y", "Z"]

        for i in range(3):
            d = R[:, i]
            ax.quiver(
                *origin,
                *d,
                color=colors[i],
                linewidth=2.5,
                arrow_length_ratio=0.12,
                label=f"body-{labels[i]}",
            )
            # World frame (faint)
            w = np.zeros(3)
            w[i] = 1.0
            ax.quiver(
                *origin,
                *w,
                color=colors[i],
                linewidth=1.0,
                alpha=0.25,
                arrow_length_ratio=0.08,
                linestyle="--",
            )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)  # Axes3D
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # Axes3D
        ax.set_title("Body Frame (solid) vs World (dashed)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        self._fig.tight_layout()
        self._canvas.draw()


# =====================================================================
# Rigid Transform Tab
# =====================================================================


class RigidTransformTab(QWidget):
    """Frame-aware SE(3) conversions with body/space twist display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left: Input
        input_group = QGroupBox("Input Transform")
        inp = QVBoxLayout(input_group)

        form = QFormLayout()
        self._tf_repr = QComboBox()
        self._tf_repr.addItems(
            [
                "Quaternion + Translation",
                "Euler (xyz) + Translation",
                "Axis-Angle + Translation",
                "4x4 SE(3) Matrix (row-major)",
            ]
        )
        form.addRow("Representation:", self._tf_repr)

        self._source_edit = QLineEdit("body")
        self._target_edit = QLineEdit("world")
        form.addRow("Source Frame:", self._source_edit)
        form.addRow("Target Frame:", self._target_edit)

        self._tf_input = QTextEdit()
        self._tf_input.setPlaceholderText(
            "Quaternion: w x y z tx ty tz\n(e.g. 1 0 0 0 1.0 2.0 3.0)"
        )
        self._tf_input.setMaximumHeight(80)
        self._tf_input.setText("1 0 0 0 1.0 2.0 3.0")
        form.addRow("Values:", self._tf_input)

        self._tf_btn = QPushButton("Convert")
        form.addRow(self._tf_btn)
        inp.addLayout(form)
        layout.addWidget(input_group, 1)

        # Right: Outputs + 3D Plot
        right = QVBoxLayout()
        out_group = QGroupBox("All Representations")
        out_layout = QVBoxLayout(out_group)
        self._tf_output = QTextEdit()
        self._tf_output.setReadOnly(True)
        self._tf_output.setStyleSheet("font-family: monospace; font-size: 11px;")
        out_layout.addWidget(self._tf_output)
        right.addWidget(out_group, 2)

        plot_group = QGroupBox("3D Transform Visualisation")
        plot_layout = QVBoxLayout(plot_group)
        self._tf_fig = Figure(figsize=(4, 3), dpi=100)
        _tm = get_theme_manager()
        apply_plot_theme(self._tf_fig, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self._tf_fig, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )
        self._tf_canvas = FigureCanvas(self._tf_fig)
        self._tf_toolbar = NavigationToolbar(self._tf_canvas, self)
        self._tf_toolbar.setMaximumHeight(30)
        plot_layout.addWidget(self._tf_toolbar)
        self._tf_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout.addWidget(self._tf_canvas)
        right.addWidget(plot_group, 3)
        layout.addLayout(right, 2)

    def _connect_signals(self) -> None:
        self._tf_btn.clicked.connect(self._update)

    def _update(self) -> None:
        text = self._tf_input.toPlainText().strip()
        v = _parse_vec(text)
        if v is None:
            self._tf_output.setPlainText("Invalid input")
            return
        idx = self._tf_repr.currentIndex()
        src = self._source_edit.text().strip() or "body"
        tgt = self._target_edit.text().strip() or "world"

        try:
            if idx == 0:  # Quaternion + translation
                if len(v) != 7:
                    raise ValueError("Need 7 values: w x y z tx ty tz")
                T = RigidTransform.from_quaternion_translation(
                    v[:4], v[4:], source=src, target=tgt
                )
            elif idx == 1:  # Euler + translation
                if len(v) != 6:
                    raise ValueError("Need 6 values: a b c tx ty tz")
                T = RigidTransform.from_euler_translation(
                    v[0], v[1], v[2], v[3:], convention="xyz", source=src, target=tgt
                )
            elif idx == 2:  # Axis-angle + translation
                if len(v) != 7:
                    raise ValueError("Need 7 values: ax ay az angle tx ty tz")
                axis = v[:3] / np.linalg.norm(v[:3])
                T = RigidTransform.from_axis_angle_translation(
                    axis, v[3], v[4:], source=src, target=tgt
                )
            elif idx == 3:  # 4x4 matrix
                if len(v) != 16:
                    raise ValueError("Need 16 values (row-major 4x4)")
                T = RigidTransform.from_matrix(v.reshape(4, 4), source=src, target=tgt)
            else:
                return
        except Exception as e:  # noqa: BLE001 — user input can raise any error; display it
            self._tf_output.setPlainText(f"Error: {e}")
            return

        self._display_transform(T)
        self._draw_transform(T)

    def _display_transform(self, T: RigidTransform) -> None:
        if T is None:
            raise ValueError("T must be provided")
        q, p = T.as_quaternion_translation()
        R, p2 = T.as_rotation_translation()
        euler, _ = T.as_euler_translation("xyz")
        ax, ang, _ = T.as_axis_angle_translation()
        rod, _ = T.as_rodrigues_translation()
        Vb = T.body_twist()
        Vs = T.space_twist()

        lines = [
            f"━━━ Frame: {T.source_frame} → {T.target_frame} ━━━",
            "",
            "━━━ Quaternion + Translation ━━━",
            f"  q: {_fmt_vec(q)}",
            f"  p: {_fmt_vec(p)}",
            "",
            "━━━ Euler (xyz) + Translation ━━━",
            f"  angles: {euler[0]: .6f}  {euler[1]: .6f}  {euler[2]: .6f} rad",
            f"  p:      {_fmt_vec(p)}",
            "",
            "━━━ Axis-Angle + Translation ━━━",
            f"  axis:  {_fmt_vec(ax)}",
            f"  angle: {ang:.6f} rad ({math.degrees(ang):.2f} deg)",
            f"  p:     {_fmt_vec(p)}",
            "",
            "━━━ Rodrigues + Translation ━━━",
            f"  r: {_fmt_vec(rod)}",
            f"  p: {_fmt_vec(p)}",
            "",
            "━━━ 4x4 SE(3) Matrix ━━━",
            _fmt_mat(T.as_matrix()),
            "",
            "━━━ Body-Frame Twist [ω_b; v_b] ━━━",
            f"  {_fmt_vec(Vb)}",
            "",
            "━━━ Space-Frame Twist [ω_s; v_s] ━━━",
            f"  {_fmt_vec(Vs)}",
        ]

        try:
            screw = T.as_screw()
            lines += [
                "",
                "━━━ Screw Axis ━━━",
                f"  axis:  {_fmt_vec(screw['axis'])}",
                f"  point: {_fmt_vec(screw['point'])}",
                f"  pitch: {screw['pitch']:.6f}",
                f"  theta: {screw['theta']:.6f} rad",
            ]
        except Exception:  # noqa: BLE001 — screw decomposition is optional display; skip on error
            pass

        self._tf_output.setPlainText("\n".join(lines))

    def _draw_transform(self, T: RigidTransform) -> None:
        if T is None:
            raise ValueError("T must be provided")
        self._tf_fig.clear()
        ax = self._tf_fig.add_subplot(111, projection="3d")
        _style_figure(self._tf_fig, ax)

        colors = _get_plot_colors()["axes"]
        labels = ["X", "Y", "Z"]

        # World frame at origin
        for i in range(3):
            d = np.zeros(3)
            d[i] = 1.0
            ax.quiver(
                0,
                0,
                0,
                d[0],
                d[1],
                d[2],
                color=colors[i],
                linewidth=1,
                alpha=0.3,
                arrow_length_ratio=0.08,
                linestyle="--",
            )

        # Transformed frame
        R = T.rotation_matrix
        p = T.translation
        scale = 0.5
        for i in range(3):
            d = R[:, i] * scale
            ax.quiver(
                p[0],
                p[1],
                p[2],
                d[0],
                d[1],
                d[2],
                color=colors[i],
                linewidth=2.5,
                arrow_length_ratio=0.12,
                label=f"{labels[i]}'",
            )

        # Connection line
        c = _get_plot_colors()
        ax.plot(
            [0, p[0]],
            [0, p[1]],
            [0, p[2]],
            color=c["accent"],
            linewidth=1.5,
            alpha=0.5,
            linestyle=":",
        )

        ax.scatter(*p, color=c["accent"], s=40, zorder=5)  # Axes3D

        margin = max(float(np.linalg.norm(p)) * 1.3, 2.0)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_zlim(-margin, margin)  # Axes3D
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # Axes3D
        ax.set_title(f"{T.source_frame} → {T.target_frame}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        self._tf_fig.tight_layout()
        self._tf_canvas.draw()


# =====================================================================
# Trajectory Plots Tab
# =====================================================================


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
        _tm = get_theme_manager()
        apply_plot_theme(self._fig, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self._fig, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )
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
        _style_figure(self._fig, [ax1, ax2, ax3])

        c = _get_plot_colors()
        ax1.plot(t, thetas, color=c["accent"], linewidth=1.5)
        ax1.set_title("Rotation Angle (θ)", fontsize=9)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("rad")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, pitches, color=c["axes"][1], linewidth=1.5)
        ax2.set_title("Screw Pitch", fontsize=9)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("m/rad")
        ax2.grid(True, alpha=0.3)

        for i, lbl in enumerate(["ωx", "ωy", "ωz"]):
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
        _style_figure(self._fig, ax)
        c = _get_plot_colors()
        for j, lbl in enumerate(["Roll (X)", "Pitch (Y)", "Yaw (Z)"]):
            ax.plot(
                t,
                np.degrees(angles[:, j]),
                color=c["axes"][j],
                linewidth=1.5,
                label=lbl,
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
        _style_figure(self._fig, ax)
        c = _get_plot_colors()
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
        _style_figure(self._fig, [ax1, ax2])
        c = _get_plot_colors()

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
        _style_figure(self._fig, ax)
        c = _get_plot_colors()
        for j, lbl in enumerate(["ωx", "ωy", "ωz"]):
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
        _style_figure(self._fig, [ax1, ax2])
        c = _get_plot_colors()

        labels_w = ["ωx", "ωy", "ωz"]
        labels_v = ["vx", "vy", "vz"]
        for j in range(3):
            ax1.plot(
                t,
                body_tw[: len(t), j],
                color=c["axes"][j],
                linewidth=1.5,
                label=f"body {labels_w[j]}",
            )
            ax1.plot(
                t,
                space_tw[: len(t), j],
                color=c["axes"][j],
                linewidth=1.5,
                linestyle="--",
                label=f"space {labels_w[j]}",
            )
        ax1.set_title("Angular: Body (solid) vs Space (dashed)", fontsize=9)
        ax1.set_xlabel("Step")
        ax1.legend(fontsize=6, ncol=2)
        ax1.grid(True, alpha=0.3)

        for j in range(3):
            ax2.plot(
                t,
                body_tw[: len(t), 3 + j],
                color=c["axes"][j],
                linewidth=1.5,
                label=f"body {labels_v[j]}",
            )
            ax2.plot(
                t,
                space_tw[: len(t), 3 + j],
                color=c["axes"][j],
                linewidth=1.5,
                linestyle="--",
                label=f"space {labels_v[j]}",
            )
        ax2.set_title("Linear: Body (solid) vs Space (dashed)", fontsize=9)
        ax2.set_xlabel("Step")
        ax2.legend(fontsize=6, ncol=2)
        ax2.grid(True, alpha=0.3)


# =====================================================================
# 3D Screw Axis Visualiser Tab
# =====================================================================


class ScrewVisualiserTab(QWidget):
    """Interactive 3D screw axis animation (frame-by-frame)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._animator: rc.ScrewAxisAnimator | None = None
        self._frame_idx = 0
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
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        ctrl2.addWidget(self._play_btn)
        ctrl2.addWidget(self._stop_btn)
        ctrl2.addWidget(self._prev_btn)
        ctrl2.addWidget(self._next_btn)

        from PyQt6.QtWidgets import QCheckBox

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
        _tm = get_theme_manager()
        apply_plot_theme(self._fig, _tm.get_current_colors())
        _tm.themeChanged.connect(
            lambda name: apply_plot_theme(
                self._fig, _tm.get_theme_colors(name) or _tm.get_current_colors()
            )
        )
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
        c = _get_plot_colors()
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


# =====================================================================
# Main Window
# =====================================================================


class RotationConverterMainWindow(ThemedWindowMixin, QMainWindow):
    """Main window with tabbed interface and theme integration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setup_theme_support()
        self.setWindowTitle("Rotation Converter")
        self.setMinimumSize(1200, 800)
        self._build_ui()
        self._build_menus()
        self._apply_theme()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._tabs.addTab(RotationConverterTab(), "Rotation Converter")
        self._tabs.addTab(RigidTransformTab(), "Rigid Transform")
        self._tabs.addTab(ReferenceFrameTab(), "Reference Frames & Lie Groups")
        self._tabs.addTab(TrajectoryPlotsTab(), "Trajectory Plots")
        self._tabs.addTab(ScrewVisualiserTab(), "3D Screw Visualiser")
        self._tabs.addTab(CommandConsoleTab(), "Python Console")
        layout.addWidget(self._tabs)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready -- select a tab to begin")

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        if menu_bar is None:
            raise RuntimeError("Rotation converter menu bar is unavailable")

        # File menu
        file_menu = menu_bar.addMenu("&File")
        if file_menu is None:
            raise RuntimeError("Rotation converter File menu is unavailable")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        if help_menu is None:
            raise RuntimeError("Rotation converter Help menu is unavailable")
        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _apply_theme(self) -> None:
        try:
            mgr = get_theme_manager()
            mgr.apply_theme_to_window(self)
            mgr.themeChanged.connect(self._on_theme_changed)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Theme system is optional; window still works without it.
            logger.exception("Failed to apply optional rotation converter theme")

    def _on_theme_changed(self, theme_name: str) -> None:
        """Refresh all plots when the theme changes."""
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            if tab is None:
                continue
            if hasattr(tab, "_update_outputs"):
                tab._update_outputs()
            elif hasattr(tab, "_update"):
                tab._update()
            elif hasattr(tab, "_plot"):
                tab._plot()
            elif hasattr(tab, "_draw_frame"):
                tab._draw_frame()

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "Rotation Converter",
            f"<b>Rotation Converter v{rc.__version__}</b><br><br>"
            "Comprehensive rotation and rigid-body transform converter "
            "with interactive 3D visualization.<br><br>"
            "Supports: quaternions, Euler angles, rotation matrices, "
            "axis-angle, Rodrigues vectors, SE(3), twists, screw axes, "
            "frame-aware transforms, and Modern Robotics kinematics.<br><br>"
            "Part of the D-sorganization Tools suite.",
        )
