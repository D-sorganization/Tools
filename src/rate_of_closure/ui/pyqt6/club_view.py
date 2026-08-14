"""Animated 3D clubhead view.

The view renders procedural or STL geometry under the delivery transform,
with playback, fixed/moving display modes, velocity vectors, hosel/shaft
alignment, and an engineering-style reference marker.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club_camera import (
    DEFAULT_CLUB_CAMERA,
    ClubCamera,
    ClubCameraAction,
    camera_status,
)
from rate_of_closure.club_mesh_source import (
    generated_mesh_source,
    imported_mesh_source,
    procedural_mesh_source,
)
from rate_of_closure.mesh import HeadMesh
from rate_of_closure.model import ImpactResult, ImpactScenario, solve
from rate_of_closure.ui.pyqt6.club_view_controls import build_playback_bar
from rate_of_closure.ui.pyqt6.club_view_lifecycle import (
    TIMER_INTERVAL_MS,
    ClubViewLifecycleMixin,
)
from rate_of_closure.ui.pyqt6.club_view_render import VIEW_MODES
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)

logger = logging.getLogger(__name__)

__all__ = ["VIEW_MODES", "Club3DView"]


class Club3DView(ClubViewLifecycleMixin, QWidget):
    """Animated 3D rendering of the rotating clubhead at impact."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(build_playback_bar(self))
        self._status = QLabel()
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._error = QLabel()
        self._error.setTextFormat(Qt.TextFormat.PlainText)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #ff6b6b")
        self._error.hide()
        self._error_kind: str | None = None
        layout.addWidget(self._error)
        layout.addWidget(self._canvas)

        self._scenario: ImpactScenario | None = None
        self._result: ImpactResult | None = None
        self._source = procedural_mesh_source()
        self._mesh: HeadMesh | None = None
        self._hosel: np.ndarray | None = None
        self._cog: np.ndarray | None = None
        self._phase = 0.0
        self._speed = 1.0
        self._camera = DEFAULT_CLUB_CAMERA
        self._zoom = self._camera.zoom
        self._resume_after_orbit = False
        self._timer = QTimer(self)
        self._timer.setInterval(TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        # Scroll-to-zoom; drag-to-orbit is native to Axes3D and the view
        # angles are captured/restored across animation redraws.
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_orbit_started)
        self._canvas.mpl_connect("button_release_event", self._on_orbit_finished)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setAccessibleName("Interactive 3D clubhead camera")
        self._canvas.setAccessibleDescription(
            "Arrow keys orbit, plus and minus zoom, and Home resets the camera."
        )
        self._canvas.installEventFilter(self)
        self._update_status()

    # ── construction ────────────────────────────────────────────────
    # ── public API ──────────────────────────────────────────────────
    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt a new scenario without starting background animation."""
        result = solve(scenario)
        prior = (self._scenario, self._result, self._phase)
        self._scenario = scenario
        self._result = result
        self._phase = 0.0
        try:
            self._draw()
        except Exception:
            self._scenario, self._result, self._phase = prior
            try:
                self._draw()
            except Exception as rollback_error:
                logger.error("club scenario rollback redraw failed: %s", rollback_error)
            raise
        if self._error_kind == "render":
            self._error.hide()
            self._error_kind = None

    def try_set_scenario(self, scenario: ImpactScenario) -> bool:
        """Adopt editor scenario through the UI render-error boundary."""
        try:
            self.set_scenario(scenario)
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
            return False
        return True

    def set_playback_speed(self, multiplier: float) -> None:
        """Set the animation speed multiplier (0.1-3.0)."""
        clamped = max(0.1, min(3.0, multiplier))
        self._speed_slider.setValue(round(clamped * 100))

    def playback_speed(self) -> float:
        """Current animation speed multiplier."""
        return self._speed

    def is_playing(self) -> bool:
        """Whether the clubhead animation timer is running."""
        return self._timer.isActive()

    def set_view_mode(self, mode: str) -> None:
        """Select a display mode by name (see :data:`VIEW_MODES`)."""
        if mode not in VIEW_MODES:
            logger.warning("unknown view mode requested: %s", mode)
            return
        self._mode_combo.setCurrentText(mode)

    def view_mode(self) -> str:
        """The active display mode name."""
        return str(self._mode_combo.currentText())

    def set_zoom(self, factor: float) -> None:
        """Set the camera zoom factor (0.3-4.0; larger = closer)."""
        candidate = ClubCamera(
            self._camera.azimuth_deg, self._camera.elevation_deg, factor
        )
        self._adopt_camera(candidate)

    def zoom(self) -> float:
        """Current camera zoom factor."""
        return float(self._zoom)

    def load_mesh(self, path: str) -> None:
        """Synchronously validate and atomically adopt a bounded STL source."""
        self._apply_source(imported_mesh_source(path, self._source.generation + 1))

    def try_load_mesh(self, path: str) -> bool:
        """Load through the nonmodal UI error boundary."""
        try:
            candidate = imported_mesh_source(path, self._source.generation + 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("STL load failed: %s", exc)
            self._set_error(f"STL load failed: {str(exc)[:512]}", "import")
            return False
        try:
            self._apply_source(candidate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("clubhead render failed: %s", exc)
            self._set_error(f"Clubhead render failed: {str(exc)[:512]}", "render")
            return False
        return True

    def set_head_mesh(
        self,
        mesh: HeadMesh,
        hosel_point: tuple[float, float, float] | None = None,
        cog_point: tuple[float, float, float] | None = None,
        label: str = "clubhead",
    ) -> None:
        """Render a prepared head mesh (STL or parametric) in the view.

        The Club group's "Generate Representative Head" passes its
        per-type hosel point (where the shaft line attaches) and its
        volumetric COG; STL loads leave them ``None``.
        """
        if hosel_point is None or cog_point is None:
            raise ValueError(
                "generated mesh requires exact hosel and geometric centroid"
            )
        hosel = hosel_point
        centroid = cog_point
        self._apply_source(
            generated_mesh_source(
                mesh,
                label,
                self._source.generation + 1,
                hosel=hosel,
                geometric_centroid=centroid,
            )
        )

    def clear_mesh(self) -> None:
        """Discard any loaded STL mesh and restore the procedural head."""
        self._apply_source(procedural_mesh_source(self._source.generation + 1))

    def try_clear_mesh(self) -> bool:
        """Reset through the UI render-error boundary."""
        try:
            self.clear_mesh()
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
            return False
        return True

    def try_set_head_mesh(
        self,
        mesh: HeadMesh,
        hosel_point: tuple[float, float, float],
        cog_point: tuple[float, float, float],
        label: str,
    ) -> bool:
        """Adopt generated geometry through the UI render-error boundary."""
        try:
            self.set_head_mesh(mesh, hosel_point, cog_point, label)
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
            return False
        return True

    def show_cg_check(self) -> QCheckBox:
        """The neutral reference-marker checkbox (test seam)."""
        return cast(QCheckBox, self._show_cg_check)

    def shaft_attachment(self) -> np.ndarray | None:
        """Model-frame shaft attachment: the generated head's hosel
        shifted with the mesh; ``None`` for the wireframe hosel."""
        if self._scenario is None or self._mesh is None or self._hosel is None:
            return None
        return np.asarray(self._hosel + self._head_shift(self._mesh, self._scenario))

    def cg_marker_point(self) -> np.ndarray | None:
        """Model-frame CG marker location, or ``None`` when hidden."""
        if self._scenario is None or not self._show_cg_check.isChecked():
            return None
        if self._mesh is None or self._cog is None:
            return np.zeros(3)  # scenario reference datum, not a mass CG
        return np.asarray(self._cog + self._head_shift(self._mesh, self._scenario))

    @staticmethod
    def _head_shift(mesh: HeadMesh, scenario: ImpactScenario) -> np.ndarray:
        """+x shift placing the mesh's face plane at GC-to-face."""
        d = scenario.com_to_face_mm / 1000.0
        return np.array([d - float(mesh.triangles[..., 0].max()), 0.0, 0.0])

    def has_mesh(self) -> bool:
        """Whether an STL mesh is currently rendered."""
        return self._mesh is not None

    def stop(self) -> None:
        """Stop the animation timer (used on window close and in tests)."""
        self._timer.stop()
        self._play_button.setChecked(False)

    # ── internals ──────────────────────────────────────────────────
    def _on_load_mesh_clicked(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Clubhead STL", "", "STL meshes (*.stl);;All files (*)"
        )
        if not path:
            return
        self.try_load_mesh(path)

    def _on_scroll(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button not in {"up", "down"}:
            return
        factor = self._zoom * (1.1 if event.button == "up" else 1.0 / 1.1)
        try:
            self.set_zoom(factor)
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")

    def _on_orbit_started(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        self._resume_after_orbit = self._timer.isActive()
        self._timer.stop()

    def _on_orbit_finished(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return
        candidate = ClubCamera(
            90.0 - float(self._axes.azim), float(self._axes.elev), self._zoom
        )
        resume = self._resume_after_orbit
        self._resume_after_orbit = False
        try:
            self._adopt_camera(candidate)
        except Exception as error:
            self.stop()
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
            return
        if resume:
            self._timer.start()

    def eventFilter(  # noqa: N802
        self,
        watched: QObject | None,
        event: QEvent | None,
    ) -> bool:
        """Apply keyboard camera actions without moving focus."""
        if (
            watched is self._canvas
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
        ):
            key = event.key()
            actions: dict[int, ClubCameraAction] = {
                Qt.Key.Key_Left: ClubCameraAction.LEFT,
                Qt.Key.Key_Right: ClubCameraAction.RIGHT,
                Qt.Key.Key_Up: ClubCameraAction.UP,
                Qt.Key.Key_Down: ClubCameraAction.DOWN,
                Qt.Key.Key_Plus: ClubCameraAction.ZOOM_IN,
                Qt.Key.Key_Equal: ClubCameraAction.ZOOM_IN,
                Qt.Key.Key_Minus: ClubCameraAction.ZOOM_OUT,
                Qt.Key.Key_Home: ClubCameraAction.HOME,
            }
            action = actions.get(key)
            if action is not None:
                self._try_camera_action(action)
                return True
        return super().eventFilter(watched, event)

    def _update_status(self) -> None:
        self._status.setText(camera_status(self._camera, self._source.status))

    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._scenario is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed_changed(self, value: int) -> None:
        self._speed = value / 100.0
        self._speed_label.setText(f"{self._speed:.1f}x")

    def _draw(self) -> None:
        from rate_of_closure.ui.pyqt6.club_view_render import draw_club_view

        draw_club_view(self)
