"""Rotation Converter Tab — live pairwise conversion between all representations.

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

from rotation_converter.converter import Rotation
from rotation_converter.ui.pyqt6.plot_helpers import (
    EULER_CONVENTIONS,
    fmt_mat,
    fmt_vec,
    get_plot_colors,
    parse_vec,
    style_figure,
)


class RotationConverterTab(QWidget):
    """Live conversion between all rotation representations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()
        self._update_outputs()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left: Input
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
        layout.addWidget(input_group, 1)

        # Right: Outputs + Plot
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

        # 3D rotation axes plot
        plot_group = QGroupBox("3D Rotation Axes")
        plot_layout = QVBoxLayout(plot_group)
        self._fig = Figure(figsize=(4, 3), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._toolbar.setMaximumHeight(30)
        plot_layout.addWidget(self._toolbar)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout.addWidget(self._canvas)
        right.addWidget(plot_group, 3)

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
        v = parse_vec(text)
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
        except Exception as e:  # noqa: BLE001
            self._output_text.setPlainText(f"Error: {e}")
            return

        self._display_all(rot, conv)
        self._update_main_result(rot)
        self._draw_rotation(rot)

    def _update_main_result(self, rot: Rotation) -> None:
        assert rot is not None, "rot must be provided"
        idx = self._target_repr.currentIndex()
        conv = self._target_euler_conv.currentText()
        try:
            self._target_euler_conv.setEnabled(idx == 1)
            if idx == 0:
                res = fmt_vec(rot.as_quaternion())
            elif idx == 1:
                res = fmt_vec(np.array(rot.as_euler(conv)))
            elif idx == 2:
                ax, ang = rot.as_axis_angle()
                res = f"{fmt_vec(ax)}  {ang:.6f}"
            elif idx == 3:
                res = fmt_vec(rot.as_rodrigues())
            elif idx == 4:
                R = rot.as_rotation_matrix()
                res = fmt_vec(R.flatten())
            else:
                res = ""
            self._main_result.setText(res)
        except Exception as e:  # noqa: BLE001
            self._main_result.setText(f"Error: {e}")

    def _display_all(self, rot: Rotation, conv: str) -> None:
        assert rot is not None, "rot must be provided"
        q = rot.as_quaternion()
        R = rot.as_rotation_matrix()
        axis, angle = rot.as_axis_angle()
        rod = rot.as_rodrigues()

        lines = [
            "\u2501\u2501\u2501 Quaternion (w, x, y, z) \u2501\u2501\u2501",
            fmt_vec(q),
            "",
            f"\u2501\u2501\u2501 Euler Angles [{conv}] (rad) \u2501\u2501\u2501",
        ]
        for c in EULER_CONVENTIONS:
            try:
                e = rot.as_euler(c)
                marker = " \u25c0" if c == conv else ""
                lines.append(f"  {c}: {e[0]: .6f}  {e[1]: .6f}  {e[2]: .6f}{marker}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"  {c}: (error)")
        lines += [
            "",
            "\u2501\u2501\u2501 Axis-Angle \u2501\u2501\u2501",
            f"  axis:  {fmt_vec(axis)}",
            f"  angle: {angle:.6f} rad ({math.degrees(angle):.2f} deg)",
            "",
            "\u2501\u2501\u2501 Rodrigues Vector \u2501\u2501\u2501",
            f"  {fmt_vec(rod)}",
            "",
            "\u2501\u2501\u2501 Rotation Matrix \u2501\u2501\u2501",
            fmt_mat(R),
        ]
        self._output_text.setPlainText("\n".join(lines))

    def _draw_rotation(self, rot: Rotation) -> None:
        assert rot is not None, "rot must be provided"
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection="3d")
        style_figure(self._fig, ax)

        R = rot.as_rotation_matrix()
        origin = np.zeros(3)
        colors = get_plot_colors()["axes"]
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
