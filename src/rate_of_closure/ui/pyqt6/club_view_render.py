"""Rendering helpers for the PyQt clubhead view."""

from typing import Any, cast

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rate_of_closure.club_camera import matplotlib_view
from rate_of_closure.mesh import HeadMesh
from rate_of_closure.model import ImpactScenario
from rate_of_closure.ui.pyqt6.engineering_markers import draw_cg_marker

_COL_FACE, _COL_BODY, _COL_SHAFT = "#0A84FF", "#8b949e", "#AC8E68"
_COL_V_REF, _COL_V_POINT, _COL_IMPACT = "#30D158", "#FF375F", "#FFD60A"
_COL_GROUND, _COL_COG = "#8b949e", "#FF9F0A"
_FACE_HALF_WIDTH, _FACE_HALF_HEIGHT, _BODY_DEPTH, _SHAFT_STUB = 0.058, 0.028, 0.11, 0.35
_LIGHT_DIR = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])
_MESH_BASE_RGB = np.array([0.56, 0.62, 0.70])
_MESH_AMBIENT, _MESH_SPECULAR, _ANIMATION_SPAN_MS = 0.22, 0.32, 8.0
VIEW_MODES = ("Head Fixed in Place", "Head Moving Through Space")


def _rodrigues(axis_omega: np.ndarray, dt: float) -> np.ndarray:
    theta = float(np.linalg.norm(axis_omega)) * dt
    if abs(theta) < 1e-12:
        return cast(np.ndarray, np.eye(3))
    axis = axis_omega / np.linalg.norm(axis_omega)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return cast(
        np.ndarray, np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    )


def _head_wireframe(scenario: ImpactScenario) -> dict[str, np.ndarray]:
    d = scenario.com_to_face_mm / 1000.0
    w, h = _FACE_HALF_WIDTH, _FACE_HALF_HEIGHT
    face = np.array([[d, -h, -w], [d, -h, w], [d, h, w], [d, h, -w], [d, -h, -w]])
    back = face - np.array([_BODY_DEPTH, 0.0, 0.0])
    lie = np.radians(scenario.lie_angle_deg)
    hosel = np.array([d - 0.02, h, -w])
    shaft = np.vstack(
        [hosel, hosel + np.array([0.0, np.sin(lie), -np.cos(lie)]) * _SHAFT_STUB]
    )
    impact = np.array(
        [
            d,
            scenario.impact_offset_high_mm / 1000.0,
            scenario.impact_offset_toe_mm / 1000.0,
        ]
    )
    return {"face": face, "back": back, "shaft": shaft, "impact": impact}


def _display(points: np.ndarray) -> np.ndarray:
    return np.asarray(points)[..., [2, 0, 1]]


def draw_club_view(view: Any) -> None:
    if view._scenario is None or view._result is None:
        return
    scenario = view._scenario
    result = view._result
    omega = np.radians(np.array(result.omega_dps))
    time_s = (view._phase - 0.5) * _ANIMATION_SPAN_MS / 1000.0
    rotation = _rodrigues(omega, time_s)
    moving = view._mode_combo.currentText() == VIEW_MODES[1]
    speed_mps = result.reference_speed_mph * 0.44704
    offset = np.array([speed_mps * time_s, 0.0, 0.0]) if moving else np.zeros(3)

    parts = _head_wireframe(scenario)
    axes = view._axes
    elev, azim = matplotlib_view(view._camera)
    axes.clear()
    axes.set_proj_type("ortho")
    if view._mesh is not None:
        draw_club_mesh(view, view._mesh, scenario, rotation, offset)
        attachment = view.shaft_attachment()
        if attachment is not None:
            # Hosel-true shaft: attach at the per-type hosel point.
            lie = np.radians(scenario.lie_angle_deg)
            shaft_dir = np.array([0.0, np.sin(lie), -np.cos(lie)])
            shaft = np.vstack([attachment, attachment + shaft_dir * _SHAFT_STUB])
            pts = _display(shaft @ rotation.T + offset)
            axes.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=_COL_SHAFT, lw=2.0)
    else:
        for key, color, width in (
            ("face", _COL_FACE, 2.2),
            ("back", _COL_BODY, 1.2),
            ("shaft", _COL_SHAFT, 2.0),
        ):
            pts = _display(parts[key] @ rotation.T + offset)
            axes.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=width)
        for a, b in zip(parts["face"], parts["back"], strict=True):
            seg = _display(np.vstack([a, b]) @ rotation.T + offset)
            axes.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=_COL_BODY, lw=0.8)

    impact = parts["impact"] @ rotation.T + offset
    axes.scatter(*_display(impact), color=_COL_IMPACT, s=45, zorder=5)
    axes.scatter(*_display(offset), color=_COL_BODY, s=30)
    cg_point = view.cg_marker_point()
    if cg_point is not None:
        placed = _display(cg_point @ rotation.T + offset)
        generated = view._source.kind == "generated"
        draw_cg_marker(
            axes,
            placed,
            _COL_COG,
            label="geometric centroid" if generated else "scenario reference",
            abbreviation="GC" if generated else "REF",
        )

    if moving:
        # Target line on the ground plane, for spatial reference.
        line = _display(np.array([[-0.4, -0.05, 0.0], [0.4, -0.05, 0.0]]))
        axes.plot(
            line[:, 0],
            line[:, 1],
            line[:, 2],
            color=_COL_GROUND,
            lw=0.8,
            ls=":",
        )

    scale = 0.0035  # m per (m/s): keeps arrows inside the box
    v_ref = np.array([result.reference_speed_mph, 0.0, 0.0]) * 0.44704
    v_point = np.array(result.point_velocity_mps)
    for origin, vec, color, label in (
        (offset, v_ref, _COL_V_REF, "reference (GC) path"),
        (impact, v_point, _COL_V_POINT, "impact-point path"),
    ):
        axes.quiver(
            *_display(origin),
            *_display(vec * scale),
            color=color,
            lw=2.0,
            arrow_length_ratio=0.22,
            label=label,
        )

    limit = (0.24 if not moving else 0.42) / view._zoom
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit * 0.6, limit * 1.4)
    axes.set_zlim(-limit * 0.6, limit * 1.4)
    axes.view_init(elev=elev, azim=azim)
    axes.set_xlabel("z — right of target [m]")
    axes.set_ylabel("x — target line [m]")
    axes.set_zlabel("y — up [m]")
    axes.set_title(
        f"Path Δ {result.path_deviation_deg:+.2f}°   "
        f"AoA Δ {result.aoa_deviation_deg:+.2f}°   "
        f"t = {time_s * 1000.0:+.1f} ms"
    )
    axes.legend(loc="upper left", fontsize=8)
    view._canvas.draw()


def draw_club_mesh(
    view: Any,
    mesh: HeadMesh,
    scenario: ImpactScenario,
    rotation: np.ndarray,
    offset: np.ndarray,
) -> None:
    """Shaded STL head under the same transform as the wireframe.

    The mesh is shifted by :meth:`_head_shift` so its face plane
    sits at ``com_to_face``, then rotated about the reference point
    and translated with the head. Shading is flat lambert-ish:
    ``ambient + (1 - ambient) * |n . L|`` with a fixed world light
    on the rotated normals; depth ordering is Poly3DCollection's
    native painter's-algorithm z-sort.
    """
    tris = (mesh.triangles + view._head_shift(mesh, scenario)) @ rotation.T + offset
    normals = mesh.normals @ rotation.T
    lambert = np.abs(normals @ _LIGHT_DIR)
    diffuse = (1.0 - _MESH_AMBIENT - _MESH_SPECULAR) * lambert
    specular = _MESH_SPECULAR * lambert**20
    intensity = _MESH_AMBIENT + diffuse + specular
    colors = np.clip(intensity[:, None] * _MESH_BASE_RGB[None, :], 0.0, 1.0)
    collection = Poly3DCollection(
        _display(tris), facecolors=colors, edgecolors="none", linewidths=0.0
    )
    view._axes.add_collection3d(collection)
