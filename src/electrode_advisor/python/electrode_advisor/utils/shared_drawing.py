"""Shared 3D drawing helpers for electrode visualization.

Extracted from VisualizationUpdateMixin, PathsMixin, and ViaMetalMixin
to eliminate code duplication (DRY).  Every mixin now delegates to
these pure / static helper functions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers (pure functions -- no widget state)
# ---------------------------------------------------------------------------


def compute_wall_position(
    electrode_pos: dict[str, Any],
    bath_radius: float,
) -> np.ndarray:
    """Compute glass bath wall intersection for an electrode."""
    angle = electrode_pos["angle"]
    tip_z = electrode_pos["tip"][2]
    return np.array(
        [
            bath_radius * np.cos(angle),
            bath_radius * np.sin(angle),
            tip_z,
        ]
    )


def build_trapezoidal_prism(
    wall1: np.ndarray,
    tip1: np.ndarray,
    tip2: np.ndarray,
    wall2: np.ndarray,
    electrode_z: float,
    effective_height: float,
) -> list[list[list[float]]]:
    """Build 6-face trapezoidal prism vertices from wall/tip positions."""
    z_top = electrode_z + effective_height / 2
    z_bottom = electrode_z - effective_height / 2

    # 8 vertices: bottom face (0-3), top face (4-7)
    v = [
        [wall1[0], wall1[1], z_bottom],
        [tip1[0], tip1[1], z_bottom],
        [tip2[0], tip2[1], z_bottom],
        [wall2[0], wall2[1], z_bottom],
        [wall1[0], wall1[1], z_top],
        [tip1[0], tip1[1], z_top],
        [tip2[0], tip2[1], z_top],
        [wall2[0], wall2[1], z_top],
    ]

    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # E1 side
        [v[1], v[2], v[6], v[5]],  # tip-to-tip
        [v[2], v[3], v[7], v[6]],  # E2 side
        [v[3], v[0], v[4], v[7]],  # wall-to-wall
    ]


def build_extrusion_faces(
    wall_pos: np.ndarray,
    tip_pos: np.ndarray,
    perp_scaled: np.ndarray,
    z_start: float,
    z_end: float,
) -> list[list[np.ndarray]]:
    """Build the 6 faces for a rectangular electrode extrusion box.

    Parameters
    ----------
    wall_pos : np.ndarray
        Glass wall intersection position.
    tip_pos : np.ndarray
        Electrode tip position.
    perp_scaled : np.ndarray
        Perpendicular direction vector scaled by effective radius.
    z_start, z_end : float
        Vertical extents of the extrusion.

    Returns
    -------
    list[list[np.ndarray]]
        Six faces (bottom, top, 4 sides) for ``Poly3DCollection``.
    """
    vertices: list[np.ndarray] = []

    # Bottom face (at z_start)
    vertices.append(wall_pos + perp_scaled + np.array([0, 0, z_start - wall_pos[2]]))
    vertices.append(wall_pos - perp_scaled + np.array([0, 0, z_start - wall_pos[2]]))
    vertices.append(tip_pos - perp_scaled + np.array([0, 0, z_start - tip_pos[2]]))
    vertices.append(tip_pos + perp_scaled + np.array([0, 0, z_start - tip_pos[2]]))

    # Top face (at z_end)
    vertices.append(wall_pos + perp_scaled + np.array([0, 0, z_end - wall_pos[2]]))
    vertices.append(wall_pos - perp_scaled + np.array([0, 0, z_end - wall_pos[2]]))
    vertices.append(tip_pos - perp_scaled + np.array([0, 0, z_end - tip_pos[2]]))
    vertices.append(tip_pos + perp_scaled + np.array([0, 0, z_end - tip_pos[2]]))

    faces: list[list[np.ndarray]] = []
    faces.append([vertices[0], vertices[1], vertices[2], vertices[3]])  # Bottom
    faces.append([vertices[4], vertices[5], vertices[6], vertices[7]])  # Top
    for i in range(4):
        j = (i + 1) % 4
        faces.append([vertices[i], vertices[j], vertices[j + 4], vertices[i + 4]])

    return faces


# ---------------------------------------------------------------------------
# Annotation helpers (read widget state via *owner* parameter)
# ---------------------------------------------------------------------------


def annotate_path_value(
    owner: Any,
    ax: Any,
    mid_x: float,
    mid_y: float,
    mid_z: float,
    value: float,
    checkbox_name: str,
    fmt: str,
    bg_color: str,
    text_color: str,
) -> None:
    """Annotate a path with a formatted value label."""
    if not (
        hasattr(owner, checkbox_name)
        and getattr(owner, checkbox_name).isChecked()
        and value > 0
    ):
        return

    ax.text(
        mid_x,
        mid_y,
        mid_z,
        fmt.format(value),
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": bg_color,
            "alpha": 0.8,
        },
        fontsize=8,
        ha="center",
        va="center",
        color=text_color,
    )


def annotate_resistance_value(
    owner: Any,
    ax: Any,
    mid_x: float,
    mid_y: float,
    electrode_z: float,
    resistance_value: float,
    current_value: float,
    bg_color: str,
    text_color: str,
) -> None:
    """Annotate a path with a resistance value label."""
    if not (
        hasattr(owner, "show_resistance_values_checkbox")
        and owner.show_resistance_values_checkbox.isChecked()
        and resistance_value > 0
    ):
        return

    # Offset below current annotation when both visible
    offset = (
        -2.0
        if (
            hasattr(owner, "show_current_values_checkbox")
            and owner.show_current_values_checkbox.isChecked()
            and current_value > 0
        )
        else 1.5
    )
    mid_z = electrode_z + offset

    if resistance_value == float("inf"):
        text = "\u221e\u03a9"
    elif resistance_value >= 1.0:
        text = f"{resistance_value:.2f}\u03a9"
    else:
        text = f"{resistance_value:.3f}\u03a9"

    ax.text(
        mid_x,
        mid_y,
        mid_z,
        text,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": bg_color,
            "alpha": 0.8,
        },
        fontsize=8,
        ha="center",
        va="center",
        color=text_color,
    )


# ---------------------------------------------------------------------------
# Drawing helpers (render to a matplotlib axis)
# ---------------------------------------------------------------------------


def draw_electrode_length_extrusion(
    ax: Any,
    electrode_pos: dict[str, Any],
    metal_height: float,
    electrode_radius: float,
    bath_radius: float,
    direction: str,
    color: str,
    alpha: float,
    horizontal_spreading_factor: float,
) -> None:
    """Draw rectangular extrusion along electrode length within glass bath.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Target axis for rendering.
    electrode_pos : dict
        Electrode position dict with ``tip``, ``angle``, and ``depth`` keys.
    metal_height, electrode_radius, bath_radius : float
        Geometry dimensions.
    direction : str
        ``"down"`` or ``"up"`` -- the extrusion direction.
    color, alpha : str, float
        Visual styling.
    horizontal_spreading_factor : float
        Multiplier from ``config.horizontal_spreading_factor``.
    """
    try:
        # Get glass wall position for this electrode
        angle = electrode_pos["angle"]
        wall_pos = np.array(
            [
                bath_radius * np.cos(angle),
                bath_radius * np.sin(angle),
                electrode_pos["tip"][2],
            ]
        )

        tip_pos = electrode_pos["tip"]
        effective_radius = electrode_radius * horizontal_spreading_factor
        electrode_z = electrode_pos["tip"][2]

        if direction == "down":
            z_start = electrode_z - electrode_radius
            z_end = metal_height
        else:  # up
            z_start = metal_height
            z_end = electrode_z - electrode_radius

        electrode_dir = tip_pos - wall_pos
        electrode_length = np.linalg.norm(electrode_dir[:2])

        if electrode_length > 0:
            electrode_unit = electrode_dir[:2] / electrode_length
            perp = np.array([-electrode_unit[1], electrode_unit[0], 0])
            perp_scaled = perp * effective_radius

            faces = build_extrusion_faces(
                wall_pos, tip_pos, perp_scaled, z_start, z_end
            )

            face_collection = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolors=color,
                edgecolor="darkred",
                linewidth=0.5,
            )
            ax.add_collection3d(face_collection)

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.exception("Error drawing electrode length extrusion: %s", e)


def draw_via_metal_path(
    owner: Any,
    ax: Any,
    electrode1_pos: dict[str, Any],
    electrode2_pos: dict[str, Any],
    metal_height: float,
    electrode_radius: float,
    bath_radius: float,
    color: str,
    alpha: float,
    current_value: float,
    resistance_value: float,
    horizontal_spreading_factor: float,
) -> None:
    """Draw the correct 3-segment via-metal path with vertical extrusions.

    Renders two vertical extrusions (down from electrode 1, up to electrode 2)
    and annotates current / resistance values when their checkboxes are ticked.
    """
    try:
        # Segment 1: down from E1
        draw_electrode_length_extrusion(
            ax,
            electrode1_pos,
            metal_height,
            electrode_radius,
            bath_radius,
            direction="down",
            color=color,
            alpha=alpha,
            horizontal_spreading_factor=horizontal_spreading_factor,
        )

        # Segment 2: through metal layer -- implied by metal layer itself

        # Segment 3: up to E2
        draw_electrode_length_extrusion(
            ax,
            electrode2_pos,
            metal_height,
            electrode_radius,
            bath_radius,
            direction="up",
            color=color,
            alpha=alpha,
            horizontal_spreading_factor=horizontal_spreading_factor,
        )

        # Annotate current
        if (
            hasattr(owner, "show_current_values_checkbox")
            and owner.show_current_values_checkbox.isChecked()
            and current_value > 0
        ):
            e1_tip = electrode1_pos["tip"]
            e2_tip = electrode2_pos["tip"]
            mid_x = (e1_tip[0] + e2_tip[0]) / 2
            mid_y = (e1_tip[1] + e2_tip[1]) / 2
            mid_z = metal_height + 0.5

            ax.text(
                mid_x,
                mid_y,
                mid_z,
                f"{current_value:.0f}A",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "lightcoral",
                    "alpha": 0.8,
                },
                fontsize=8,
                ha="center",
                va="center",
                color="darkred",
            )

        # Annotate resistance
        if (
            hasattr(owner, "show_resistance_values_checkbox")
            and owner.show_resistance_values_checkbox.isChecked()
            and resistance_value > 0
        ):
            e1_tip = electrode1_pos["tip"]
            e2_tip = electrode2_pos["tip"]
            mid_x = (e1_tip[0] + e2_tip[0]) / 2
            mid_y = (e1_tip[1] + e2_tip[1]) / 2

            offset = (
                -1.0
                if (
                    hasattr(owner, "show_current_values_checkbox")
                    and owner.show_current_values_checkbox.isChecked()
                    and current_value > 0
                )
                else 0.5
            )
            mid_z = metal_height + offset

            if resistance_value == float("inf"):
                resistance_text = "\u221e\u03a9"
            elif resistance_value >= 1.0:
                resistance_text = f"{resistance_value:.2f}\u03a9"
            else:
                resistance_text = f"{resistance_value:.3f}\u03a9"

            ax.text(
                mid_x,
                mid_y,
                mid_z,
                resistance_text,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "lightpink",
                    "alpha": 0.8,
                },
                fontsize=8,
                ha="center",
                va="center",
                color="darkred",
            )

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.exception("Error drawing correct via-metal path: %s", e)


def draw_trapezoidal_path(
    owner: Any,
    ax: Any,
    electrode1_pos: dict[str, Any],
    electrode2_pos: dict[str, Any],
    conductive_height: float,
    bath_radius: float,
    vertical_spreading_factor: float,
    color: str = "blue",
    alpha: float = 0.4,
    current_value: float = 0.0,
    resistance_value: float = 0.0,
) -> None:
    """Draw the correct trapezoidal prism path between two electrodes.

    The trapezoid is formed ONLY within the glass bath area (not in refractory).
    Conductive paths start at the glass bath wall, not at electrode bases.
    """
    try:
        e1_tip = electrode1_pos["tip"]
        e2_tip = electrode2_pos["tip"]

        e1_wall = compute_wall_position(electrode1_pos, bath_radius)
        e2_wall = compute_wall_position(electrode2_pos, bath_radius)

        effective_height = conductive_height * vertical_spreading_factor
        electrode_z = (e1_tip[2] + e2_tip[2]) / 2

        faces = build_trapezoidal_prism(
            e1_wall, e1_tip, e2_tip, e2_wall, electrode_z, effective_height
        )

        face_collection = Poly3DCollection(
            faces,
            alpha=alpha,
            facecolors=color,
            edgecolor="darkblue",
            linewidth=0.5,
        )
        ax.add_collection3d(face_collection)

        # Boundary lines
        if alpha > 0.3:
            for wall, tip in [(e1_wall, e1_tip), (e2_wall, e2_tip)]:
                ax.plot(
                    [wall[0], tip[0]],
                    [wall[1], tip[1]],
                    [electrode_z, electrode_z],
                    "k-",
                    linewidth=2,
                    alpha=0.8,
                )

        # Annotate current and resistance
        mid_x = (e1_wall[0] + e2_wall[0]) / 2
        mid_y = (e1_wall[1] + e2_wall[1]) / 2

        annotate_path_value(
            owner,
            ax,
            mid_x,
            mid_y,
            electrode_z + 1.5,
            current_value,
            "show_current_values_checkbox",
            "{:.0f}A",
            "lightyellow",
            "darkblue",
        )
        annotate_resistance_value(
            owner,
            ax,
            mid_x,
            mid_y,
            electrode_z,
            resistance_value,
            current_value,
            "lightgreen",
            "darkgreen",
        )

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.exception("Error drawing correct trapezoidal path: %s", e)
