"""PyQt6 tabs for swingset and chain dynamics analysis."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optimizer_gui.models.chain_model import (
    ChainConfig,
    ChainState,
    initial_catenary_angles,
    simulate_chain,
)
from optimizer_gui.models.swingset_model import (
    HumanSegmentSpec,
    SwingPose,
    SwingSetConfig,
    SwingSetState,
    build_swingset_snapshot,
    heuristic_pumping_policy,
    simulate_swingset,
)

ACCENT = QColor("#89b4fa")
CHAIN = QColor("#bac2de")
BODY = QColor("#a6e3a1")
LEG = QColor("#f9e2af")
ARM = QColor("#f5c2e7")
SURFACE = QColor("#313244")


class MotionCanvas(QWidget):
    """Small side-view renderer for chain and swingset snapshots."""

    def __init__(self) -> None:
        """Initialize an empty motion canvas."""
        super().__init__()
        self.setMinimumHeight(300)
        self._chain_nodes: list[tuple[float, float]] = []
        self._body_points: dict[str, tuple[float, float]] = {}

    def set_scene(
        self,
        chain_nodes: list[tuple[float, float]],
        body_points: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Update rendered geometry."""
        self._chain_nodes = chain_nodes
        self._body_points = body_points or {}
        self.update()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        """Paint the current side-view scene."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SURFACE)
        if not self._chain_nodes:
            return
        points = self._project_points()
        self._draw_polyline(painter, points, CHAIN, 3)
        self._draw_body(painter, points)
        painter.setBrush(ACCENT)
        for point in points[:1] + points[-1:]:
            painter.drawEllipse(point, 5, 5)

    def _project_points(self) -> list[QPointF]:
        """Project model coordinates into widget coordinates."""
        all_points = self._chain_nodes + list(self._body_points.values())
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        span_x = max(max_x - min_x, 0.5)
        span_y = max(max_y - min_y, 0.5)
        scale = 0.82 * min(self.width() / span_x, self.height() / span_y)
        offset_x = 0.5 * (self.width() - scale * (min_x + max_x))
        offset_y = 0.5 * (self.height() - scale * (min_y + max_y))
        return [
            QPointF(offset_x + scale * x, offset_y + scale * y)
            for x, y in self._chain_nodes
        ]

    def _body_point(self, name: str) -> QPointF:
        """Project one body point using the current scene bounds."""
        previous_nodes = self._chain_nodes
        self._chain_nodes = [self._body_points[name]]
        point = self._project_points()[0]
        self._chain_nodes = previous_nodes
        return point

    def _draw_polyline(
        self,
        painter: QPainter,
        points: list[QPointF],
        color: QColor,
        width: int,
    ) -> None:
        """Draw connected line segments."""
        painter.setPen(QPen(color, width))
        for start, end in zip(points[:-1], points[1:], strict=False):
            painter.drawLine(start, end)

    def _draw_body(self, painter: QPainter, _chain_points: list[QPointF]) -> None:
        """Draw rider body links when a swingset scene is active."""
        if not self._body_points:
            return
        pairs = [
            ("hip", "shoulder", BODY, 5),
            ("hip", "knee", LEG, 4),
            ("knee", "foot", LEG, 4),
            ("shoulder", "elbow", ARM, 4),
            ("elbow", "hand", ARM, 4),
        ]
        for start, end, color, width in pairs:
            painter.setPen(QPen(color, width))
            painter.drawLine(self._body_point(start), self._body_point(end))


class SwingsetTab(QWidget):
    """Interactive swingset model tab."""

    def __init__(self) -> None:
        """Create controls and initial rendering."""
        super().__init__()
        self.canvas = MotionCanvas()
        self.metric_label = QLabel()
        self._spinboxes: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        """Build the tab layout."""
        layout = QGridLayout(self)
        layout.addWidget(self.canvas, 0, 0, 3, 1)
        layout.addWidget(self._build_chain_group(), 0, 1)
        layout.addWidget(self._build_body_group(), 1, 1)
        run_button = QPushButton("Run Baseline Policy")
        run_button.clicked.connect(self._run_policy)
        layout.addWidget(run_button, 2, 1)
        layout.addWidget(self.metric_label, 3, 0, 1, 2)

    def _build_chain_group(self) -> QGroupBox:
        """Build chain and seat controls."""
        group = QGroupBox("Swingset")
        form = QFormLayout(group)
        self._add_spin(form, "segments", "Chain segments", 3, 40, 14, integer=True)
        self._add_spin(form, "chain_length", "Chain length m", 1.0, 5.0, 2.4)
        self._add_spin(form, "link_mass", "Link mass kg", 0.01, 2.0, 0.16)
        self._add_spin(form, "seat_mass", "Seat mass kg", 0.5, 25.0, 4.5)
        return group

    def _build_body_group(self) -> QGroupBox:
        """Build rider segment controls."""
        group = QGroupBox("Rider")
        form = QFormLayout(group)
        self._add_spin(form, "torso_len", "Torso length m", 0.2, 1.2, 0.62)
        self._add_spin(form, "torso_mass", "Torso mass kg", 5.0, 80.0, 28.0)
        self._add_spin(form, "thigh_len", "Thigh length m", 0.15, 0.9, 0.46)
        self._add_spin(form, "thigh_mass", "Thigh mass kg", 1.0, 25.0, 8.0)
        self._add_spin(form, "shank_len", "Shank length m", 0.15, 0.9, 0.45)
        self._add_spin(form, "shank_mass", "Shank mass kg", 1.0, 20.0, 5.5)
        self._add_spin(form, "arm_len", "Arm segment m", 0.1, 0.8, 0.30)
        self._add_spin(form, "arm_mass", "Arm segment kg", 0.2, 10.0, 2.0)
        return group

    def _add_spin(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        lower: float,
        upper: float,
        value: float,
        *,
        integer: bool = False,
    ) -> None:
        """Create and register one spin box."""
        if integer:
            spin = QSpinBox()
            spin.setRange(int(lower), int(upper))
            spin.setValue(int(value))
        else:
            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setValue(value)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
        spin.valueChanged.connect(self._refresh)
        self._spinboxes[key] = spin
        form.addRow(label, spin)

    def _config(self) -> SwingSetConfig:
        """Return config from current controls."""
        arm = HumanSegmentSpec(self._value("arm_len"), self._value("arm_mass"))
        return SwingSetConfig(
            chain_segments=int(self._value("segments")),
            chain_length_m=self._value("chain_length"),
            chain_link_mass_kg=self._value("link_mass"),
            seat_mass_kg=self._value("seat_mass"),
            torso=HumanSegmentSpec(self._value("torso_len"), self._value("torso_mass")),
            thigh=HumanSegmentSpec(self._value("thigh_len"), self._value("thigh_mass")),
            shank=HumanSegmentSpec(self._value("shank_len"), self._value("shank_mass")),
            upper_arm=arm,
            forearm=arm,
        )

    def _value(self, key: str) -> float:
        """Return a spin-box value by key."""
        return float(self._spinboxes[key].value())

    def _refresh(self) -> None:
        """Render the current static model."""
        config = self._config()
        pose = SwingPose(swing_angle_rad=0.18, hip_angle_rad=0.25, elbow_angle_rad=0.35)
        snapshot = build_swingset_snapshot(config, pose)
        self.canvas.set_scene(
            [tuple(point) for point in snapshot.chain_nodes],
            {key: tuple(value) for key, value in snapshot.points.items()},
        )
        self.metric_label.setText(
            f"Rider mass {config.rider_mass_kg:.1f} kg | "
            f"hand-chain error {snapshot.hand_chain_error_m:.3f} m"
        )

    def _run_policy(self) -> None:
        """Run and display the deterministic baseline policy."""
        rollout = simulate_swingset(
            self._config(),
            SwingSetState(pose=SwingPose(swing_angle_rad=0.08)),
            steps=180,
            dt_s=0.02,
            policy=heuristic_pumping_policy,
        )
        snapshot = rollout.snapshots[-1]
        self.canvas.set_scene(
            [tuple(point) for point in snapshot.chain_nodes],
            {key: tuple(value) for key, value in snapshot.points.items()},
        )
        self.metric_label.setText(
            f"Peak angle {rollout.metrics.max_abs_swing_angle_rad:.3f} rad | "
            f"energy proxy {rollout.metrics.final_energy_proxy_j:.1f} J"
        )


class ChainDynamicsTab(QWidget):
    """Interactive chain whip-motion analysis tab."""

    def __init__(self) -> None:
        """Create chain controls and rendering."""
        super().__init__()
        self.canvas = MotionCanvas()
        self.metric_label = QLabel()
        self._spinboxes: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        """Build the tab layout."""
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        controls = QGroupBox("Chain")
        form = QFormLayout(controls)
        self._add_spin(form, "segments", "Segments", 2, 60, 16, integer=True)
        self._add_spin(form, "length", "Link length m", 0.03, 1.0, 0.18)
        self._add_spin(form, "mass", "Link mass kg", 0.01, 4.0, 0.12)
        self._add_spin(form, "sag", "Initial sag rad", 0.0, 1.2, 0.35)
        layout.addWidget(controls)
        run_button = QPushButton("Simulate Whip")
        run_button.clicked.connect(self._simulate)
        layout.addWidget(run_button)
        layout.addWidget(self.metric_label)

    def _add_spin(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        lower: float,
        upper: float,
        value: float,
        *,
        integer: bool = False,
    ) -> None:
        """Create one chain-control spin box."""
        if integer:
            spin = QSpinBox()
            spin.setRange(int(lower), int(upper))
            spin.setValue(int(value))
        else:
            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setValue(value)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
        spin.valueChanged.connect(self._refresh)
        self._spinboxes[key] = spin
        form.addRow(label, spin)

    def _config(self) -> ChainConfig:
        """Return chain config from controls."""
        return ChainConfig(
            segment_count=int(self._value("segments")),
            segment_length_m=self._value("length"),
            link_mass_kg=self._value("mass"),
        )

    def _state(self) -> ChainState:
        """Return initial chain state from controls."""
        config = self._config()
        angles = initial_catenary_angles(config.segment_count, self._value("sag"))
        velocities = 0.6 * np.sin(np.linspace(0.0, np.pi, config.segment_count))
        return ChainState(angles, velocities)

    def _value(self, key: str) -> float:
        """Return a spin-box value by key."""
        return float(self._spinboxes[key].value())

    def _refresh(self) -> None:
        """Render the initial chain."""
        state = self._state()
        positions = state.node_positions(self._config())
        self.canvas.set_scene([tuple(point) for point in positions])
        metrics = state.metrics(self._config())
        self.metric_label.setText(
            f"Tip speed {metrics.tip_speed_m_s:.3f} m/s | "
            f"curvature {metrics.max_curvature_rad:.3f} rad"
        )

    def _simulate(self) -> None:
        """Run and display a short chain dynamics rollout."""
        rollout = simulate_chain(self._config(), self._state(), steps=120, dt_s=0.01)
        self.canvas.set_scene([tuple(point) for point in rollout.positions[-1]])
        self.metric_label.setText(
            f"Peak tip speed {rollout.tip_speed_m_s.max():.3f} m/s | "
            f"final energy {rollout.energy_j[-1]:.2f} J"
        )


def create_swingset_tab() -> QWidget:
    """Create the swingset analysis tab widget."""
    return SwingsetTab()


def create_chain_tab() -> QWidget:
    """Create the chain dynamics tab widget."""
    return ChainDynamicsTab()
