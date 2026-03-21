"""Matplotlib-backed 3D canvas for vessel drafter previews."""

from __future__ import annotations

from math import fmod
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vessel_drafter.preview.vessel_drafter_scene import Vessel3DScene
from vessel_drafter.preview.vessel_drafter_view_options import (
    Vessel3DViewOptions,
)

DEFAULT_VIEW = (25.0, -55.0)
SECTION_VIEW_ELEVATION = 0.0
SECTION_VIEW_AZIMUTH_OFFSET_DEGREES = -90.0


class VesselDrafterThreeDCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 6.0))
        super().__init__(self.figure)
        self._scene: Vessel3DScene | None = None
        self._view_state = DEFAULT_VIEW
        self._view_options = Vessel3DViewOptions()
        self._pending_view_state: tuple[float, float] | None = None
        self.current_labels: tuple[str, ...] = ()
        self.current_face_count = 0

    @property
    def view_state(self) -> tuple[float, float]:
        return self._view_state

    def draw_scene(
        self,
        scene: Vessel3DScene,
        view_options: Vessel3DViewOptions | None = None,
    ) -> None:
        assert scene is not None, "scene must be provided"
        resolved_view_options = view_options or self._view_options
        self._prepare_view_state_for_draw(resolved_view_options)
        self._scene = scene
        self._view_options = resolved_view_options
        self.current_labels = tuple(mesh.label for mesh in scene.meshes)
        self.current_face_count = sum(len(mesh.faces) for mesh in scene.meshes)
        self.figure.clear()
        axes = self.figure.add_subplot(111, projection="3d")
        meshes = scene.meshes
        if resolved_view_options.split_enabled:
            meshes = tuple(reversed(meshes))
        for index, mesh in enumerate(meshes):
            axes.add_collection3d(
                Poly3DCollection(
                    mesh.polygons,
                    facecolors=mesh.color_hex,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=mesh.alpha,
                    shade=False,
                    zorder=index + 1,
                )
            )
        self._apply_axes_formatting(axes, scene, resolved_view_options)
        self.draw_idle()

    def queue_default_view(self, view_options: Vessel3DViewOptions) -> None:
        self._pending_view_state = _default_view_state(view_options)

    def reset_view(self) -> None:
        self.queue_default_view(self._view_options)
        if self._scene is not None:
            self.draw_scene(self._scene, self._view_options)

    def _prepare_view_state_for_draw(
        self,
        view_options: Vessel3DViewOptions,
    ) -> None:
        assert view_options is not None, "view_options must be provided"
        if self._pending_view_state is not None:
            self._view_state = self._pending_view_state
            self._pending_view_state = None
            return
        if self._scene is None:
            self._view_state = _default_view_state(view_options)
            return
        self._capture_view_state()

    def _capture_view_state(self) -> None:
        if not self.figure.axes:
            return
        axes: Any = self.figure.axes[0]
        self._view_state = (axes.elev, axes.azim)

    def _apply_axes_formatting(
        self,
        axes: Any,
        scene: Vessel3DScene,
        view_options: Vessel3DViewOptions,
    ) -> None:
        assert scene is not None, "scene must be provided"
        min_x, max_x, min_y, max_y, min_z, max_z = scene.bounds
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        span_z = max(max_z - min_z, 1.0)
        axes.set_xlim(*_expand_limits(min_x, max_x, span_x))
        axes.set_ylim(*_expand_limits(min_y, max_y, span_y))
        axes.set_zlim(*_expand_limits(min_z, max_z, span_z))
        axes.set_box_aspect((span_x, span_y, span_z))
        axes.set_proj_type("ortho" if view_options.split_enabled else "persp")
        axes.set_axis_off()
        axes.view_init(elev=self._view_state[0], azim=self._view_state[1])


def _expand_limits(minimum: float, maximum: float, span: float) -> tuple[float, float]:
    padding = span * 0.12
    return minimum - padding, maximum + padding


def _default_view_state(
    view_options: Vessel3DViewOptions,
) -> tuple[float, float]:
    if not view_options.split_enabled:
        return DEFAULT_VIEW
    return (
        SECTION_VIEW_ELEVATION,
        _normalized_azimuth(
            view_options.normalized_split_angle_degrees
            + SECTION_VIEW_AZIMUTH_OFFSET_DEGREES
        ),
    )


def _normalized_azimuth(angle_degrees: float) -> float:
    normalized = fmod(angle_degrees + 180.0, 360.0)
    if normalized < 0.0:
        normalized += 360.0
    return normalized - 180.0
