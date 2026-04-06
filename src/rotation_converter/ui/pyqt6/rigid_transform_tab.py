"""Rigid Transform Tab — frame-aware SE(3) conversions with body/space twist display.

Extracted from the former monolithic main_window.py for god-class decomposition.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rotation_converter.rigid_transform import RigidTransform
from rotation_converter.ui.pyqt6.plot_helpers import (
    fmt_mat,
    fmt_vec,
    get_plot_colors,
    parse_vec,
    style_figure,
)


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
        v = parse_vec(text)
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
        except Exception as e:  # noqa: BLE001
            self._tf_output.setPlainText(f"Error: {e}")
            return

        self._display_transform(T)
        self._draw_transform(T)

    def _display_transform(self, T: RigidTransform) -> None:
        assert T is not None, "T must be provided"
        q, p = T.as_quaternion_translation()
        R, p2 = T.as_rotation_translation()
        euler, _ = T.as_euler_translation("xyz")
        ax, ang, _ = T.as_axis_angle_translation()
        rod, _ = T.as_rodrigues_translation()
        Vb = T.body_twist()
        Vs = T.space_twist()

        lines = [
            f"\u2501\u2501\u2501 Frame: {T.source_frame} \u2192 {T.target_frame} \u2501\u2501\u2501",
            "",
            "\u2501\u2501\u2501 Quaternion + Translation \u2501\u2501\u2501",
            f"  q: {fmt_vec(q)}",
            f"  p: {fmt_vec(p)}",
            "",
            "\u2501\u2501\u2501 Euler (xyz) + Translation \u2501\u2501\u2501",
            f"  angles: {euler[0]: .6f}  {euler[1]: .6f}  {euler[2]: .6f} rad",
            f"  p:      {fmt_vec(p)}",
            "",
            "\u2501\u2501\u2501 Axis-Angle + Translation \u2501\u2501\u2501",
            f"  axis:  {fmt_vec(ax)}",
            f"  angle: {ang:.6f} rad ({math.degrees(ang):.2f} deg)",
            f"  p:     {fmt_vec(p)}",
            "",
            "\u2501\u2501\u2501 Rodrigues + Translation \u2501\u2501\u2501",
            f"  r: {fmt_vec(rod)}",
            f"  p: {fmt_vec(p)}",
            "",
            "\u2501\u2501\u2501 4x4 SE(3) Matrix \u2501\u2501\u2501",
            fmt_mat(T.as_matrix()),
            "",
            "\u2501\u2501\u2501 Body-Frame Twist [\u03c9_b; v_b] \u2501\u2501\u2501",
            f"  {fmt_vec(Vb)}",
            "",
            "\u2501\u2501\u2501 Space-Frame Twist [\u03c9_s; v_s] \u2501\u2501\u2501",
            f"  {fmt_vec(Vs)}",
        ]

        try:
            screw = T.as_screw()
            lines += [
                "",
                "\u2501\u2501\u2501 Screw Axis \u2501\u2501\u2501",
                f"  axis:  {fmt_vec(screw['axis'])}",
                f"  point: {fmt_vec(screw['point'])}",
                f"  pitch: {screw['pitch']:.6f}",
                f"  theta: {screw['theta']:.6f} rad",
            ]
        except (ValueError, ArithmeticError, AttributeError):
            # Screw decomposition is undefined for pure translations (pitch → ∞);
            # silently omit the section rather than crashing the display.
            lines.append("  (screw axis undefined for this transform)")

        self._tf_output.setPlainText("\n".join(lines))

    def _draw_transform(self, T: RigidTransform) -> None:
        assert T is not None, "T must be provided"
        self._tf_fig.clear()
        ax = self._tf_fig.add_subplot(111, projection="3d")
        style_figure(self._tf_fig, ax)

        colors = get_plot_colors()["axes"]
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
        c = get_plot_colors()
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
        ax.set_title(f"{T.source_frame} \u2192 {T.target_frame}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        self._tf_fig.tight_layout()
        self._tf_canvas.draw()
