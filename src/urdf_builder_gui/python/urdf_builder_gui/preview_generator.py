"""Preview text generator — pure-Python, GUI-independent.

Generates human-readable model structure previews from a URDFConfig.
Extracted from main_window.py for TDD compliance and LoD.
"""

from __future__ import annotations

import logging

from urdf_builder_gui.anthropometric_model import (
    HEIGHT_RATIOS,
    URDFConfig,
    get_template_segments,
)
from urdf_builder_gui.contracts import require

logger = logging.getLogger(__name__)


def generate_preview_text(config: URDFConfig) -> str:
    """Generate a human-readable preview of the model structure.

    **Pre-conditions** (DbC):
      - ``config.robot_name`` must not be empty.
      - ``config.height_m`` must be > 0.
      - ``config.mass_kg`` must be > 0.

    Returns:
        Multi-line preview string.
    """
    require(bool(config.robot_name), "robot_name must not be empty", config.robot_name)
    require(config.height_m > 0, "height_m must be positive", config.height_m)
    require(config.mass_kg > 0, "mass_kg must be positive", config.mass_kg)

    logger.info("Generating preview for '%s'", config.robot_name)

    lines: list[str] = [
        "Model Structure Preview",
        "=" * 50,
        f"\nRobot Name: {config.robot_name}",
        f"Template: {config.template}",
        "\nBody Parameters:",
        f"  Height: {config.height_m:.2f} m",
        f"  Mass: {config.mass_kg:.1f} kg",
        f"  Gender Factor: {config.gender_factor:.2f}",
    ]

    # Proportions
    lines.append("\nSegment Proportions:")
    for key, value in config.proportions.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value * 100:.0f}%")

    # Estimated segment sizes — uses shared HEIGHT_RATIOS (DRY)
    lines.append("\nEstimated Segment Sizes:")
    segment_labels = {
        "pelvis": "Pelvis Height",
        "torso": "Torso Height",
        "head": "Head Diameter",
        "thigh": "Thigh Length",
        "shin": "Shin Length",
        "upper_arm": "Upper Arm Length",
        "forearm": "Forearm Length",
    }
    for key, label in segment_labels.items():
        length = config.height_m * HEIGHT_RATIOS[key]
        lines.append(f"  {label}: {length:.3f} m")

    # Template segments
    segments = get_template_segments(config.template)
    lines.append(f"\nTemplate Segments ({len(segments)}):")
    for seg in segments:
        lines.append(f"  • {seg}")

    # Options
    lines.append("\nOptions:")
    lines.append(f"  Default Geometry: {config.geometry_type}")
    lines.append(f"  Collision Geometry: {config.collision_geometry}")
    lines.append(f"  Joint Damping: {config.damping:.2f}")
    lines.append(f"  Joint Friction: {config.friction:.2f}")
    lines.append(f"  Inertia Mode: {config.inertia_mode}")
    lines.append(f"  Density: {config.density:.0f} kg/m³")

    return "\n".join(lines)


__all__ = [
    "generate_preview_text",
]
