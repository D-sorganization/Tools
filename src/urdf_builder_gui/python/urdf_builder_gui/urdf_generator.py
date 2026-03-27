from numba import jit

"""URDF XML generator — pure-Python, GUI-independent.

Generates well-formed URDF XML from a URDFConfig. This module has
NO dependency on PyQt6 or any GUI framework, making it fully testable
and reusable across the fleet (web viewer, CLI, etc.).

Addresses:
  - Issue #1342: God class extraction
  - Issue #1343: Template selection dead code
  - Issue #1344: Standalone generator ignores settings
  - Issue #1348: Hardcoded inertia placeholders
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
import re  # noqa: E402
import xml.etree.ElementTree as ET  # nosec B405 — input is self-generated  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from urdf_builder_gui.contracts import require  # noqa: E402

from .anthropometric_model import (  # noqa: E402
    URDFConfig,
    compute_box_inertia,
    compute_segment_length,
    compute_segment_mass,
    get_template_segments,
)

logger = logging.getLogger(__name__)


# ── Segment definition table ────────────────────────────────────────────


@dataclass(frozen=True)
class _SegmentDef:
    """Internal definition for a body segment's connection."""

    base_key: str  # Key into HEIGHT_RATIOS / MASS_RATIOS
    parent: str  # Parent link name in URDF
    origin_z: float  # Z offset from parent (fraction of height)
    width_frac: float  # Width as fraction of segment length
    depth_frac: float  # Depth as fraction of segment length
    joint_type: str = "revolute"
    axis: str = "0 1 0"
    limit_lower: float = -1.0
    limit_upper: float = 1.0
    proportion_key: str | None = None  # Maps to URDFConfig.proportions


_SEGMENT_DEFS: dict[str, _SegmentDef] = {
    "pelvis": _SegmentDef("pelvis", "", 0.0, 2.0, 1.2, "fixed", proportion_key=None),
    "torso": _SegmentDef(
        "torso", "pelvis", 0.078, 1.5, 1.0, proportion_key="torso_length"
    ),
    "head": _SegmentDef(
        "head",
        "torso",
        0.278,
        1.0,
        1.0,
        limit_lower=-0.5,
        limit_upper=0.5,
        proportion_key="head_size",
    ),
    "upper_arm_l": _SegmentDef(
        "upper_arm",
        "torso",
        0.25,
        0.4,
        0.4,
        axis="1 0 0",
        limit_lower=-3.14,
        limit_upper=3.14,
        proportion_key="arm_length",
    ),
    "forearm_l": _SegmentDef(
        "forearm",
        "upper_arm_l",
        0.186,
        0.35,
        0.35,
        limit_lower=-2.5,
        limit_upper=0.0,
        proportion_key="arm_length",
    ),
    "hand_l": _SegmentDef(
        "hand", "forearm_l", 0.146, 0.5, 0.3, proportion_key="arm_length"
    ),
    "upper_arm_r": _SegmentDef(
        "upper_arm",
        "torso",
        0.25,
        0.4,
        0.4,
        axis="1 0 0",
        limit_lower=-3.14,
        limit_upper=3.14,
        proportion_key="arm_length",
    ),
    "forearm_r": _SegmentDef(
        "forearm",
        "upper_arm_r",
        0.186,
        0.35,
        0.35,
        limit_lower=-2.5,
        limit_upper=0.0,
        proportion_key="arm_length",
    ),
    "hand_r": _SegmentDef(
        "hand", "forearm_r", 0.146, 0.5, 0.3, proportion_key="arm_length"
    ),
    "thigh_l": _SegmentDef(
        "thigh",
        "pelvis",
        0.0,
        0.5,
        0.5,
        axis="1 0 0",
        limit_lower=-1.5,
        limit_upper=1.5,
        proportion_key="leg_length",
    ),
    "shin_l": _SegmentDef(
        "shin",
        "thigh_l",
        -0.245,
        0.4,
        0.4,
        limit_lower=-2.6,
        limit_upper=0.0,
        proportion_key="leg_length",
    ),
    "foot_l": _SegmentDef(
        "foot", "shin_l", -0.246, 1.5, 0.6, "fixed", proportion_key="leg_length"
    ),
    "thigh_r": _SegmentDef(
        "thigh",
        "pelvis",
        0.0,
        0.5,
        0.5,
        axis="1 0 0",
        limit_lower=-1.5,
        limit_upper=1.5,
        proportion_key="leg_length",
    ),
    "shin_r": _SegmentDef(
        "shin",
        "thigh_r",
        -0.245,
        0.4,
        0.4,
        limit_lower=-2.6,
        limit_upper=0.0,
        proportion_key="leg_length",
    ),
    "foot_r": _SegmentDef(
        "foot", "shin_r", -0.246, 1.5, 0.6, "fixed", proportion_key="leg_length"
    ),
}


# ── Public API ──────────────────────────────────────────────────────────


@jit(nopython=True, fastmath=True)
def generate_urdf_xml(config: URDFConfig) -> str:
    """Generate a complete URDF XML string from configuration.

    **Pre-conditions** (DbC):
      - ``config.robot_name`` must be a valid XML NCName (non-empty, no spaces).
      - ``config.height_m`` must be > 0.
      - ``config.mass_kg`` must be > 0.

    **Post-conditions**:
      - Returns well-formed XML starting with ``<?xml``.
      - Contains a ``<robot>`` root element.

    Returns:
        A URDF XML string.
    """
    require(
        bool(config.robot_name) and _is_valid_xml_name(config.robot_name),
        "robot_name must be a valid XML name (no spaces or special chars)",
        config.robot_name,
    )
    require(config.height_m > 0, "height_m must be positive", config.height_m)
    require(config.mass_kg > 0, "mass_kg must be positive", config.mass_kg)

    logger.info(
        "Generating URDF: name=%s height=%.2f mass=%.1f template=%s",
        config.robot_name,
        config.height_m,
        config.mass_kg,
        config.template,
    )

    segments = get_template_segments(config.template)

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<robot name="{config.robot_name}">',
        "  <!-- Generated by Parametric URDF Builder -->",
        f"  <!-- Height: {config.height_m}m, Mass: {config.mass_kg}kg -->",
        "",
    ]

    for seg_name in segments:
        seg_def = _SEGMENT_DEFS.get(seg_name)
        if seg_def is None:
            logger.warning("Unknown segment: %s, skipping", seg_name)
            continue

        prop_factor = config.proportions.get(seg_def.proportion_key or "", 1.0)
        seg_length = compute_segment_length(
            config.height_m, seg_def.base_key, prop_factor
        )
        seg_mass = compute_segment_mass(config.mass_kg, seg_def.base_key)
        seg_width = seg_length * seg_def.width_frac
        seg_depth = seg_length * seg_def.depth_frac

        ixx, iyy, izz = compute_box_inertia(seg_mass, seg_width, seg_length, seg_depth)

        # Link
        lines.append(f'  <link name="{seg_name}">')
        lines.append("    <visual>")
        lines.append("      <geometry>")
        lines.append(
            f'        <box size="{seg_width:.4f} {seg_depth:.4f} {seg_length:.4f}"/>'
        )
        lines.append("      </geometry>")
        lines.append('      <material name="skin">')
        lines.append('        <color rgba="0.8 0.6 0.5 1.0"/>')
        lines.append("      </material>")
        lines.append("    </visual>")

        if config.collision_geometry != "None":
            lines.append("    <collision>")
            lines.append("      <geometry>")
            lines.append(
                f'        <box size="{seg_width:.4f} {seg_depth:.4f}'
                f' {seg_length:.4f}"/>'
            )
            lines.append("      </geometry>")
            lines.append("    </collision>")

        lines.append("    <inertial>")
        lines.append(f'      <mass value="{seg_mass:.4f}"/>')
        lines.append(
            f'      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0"'
            f' iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>'
        )
        lines.append("    </inertial>")
        lines.append("  </link>")
        lines.append("")

        # Joint (skip for root segment)
        if seg_def.parent:
            joint_name = f"{seg_def.parent}_to_{seg_name}"
            origin_z = config.height_m * seg_def.origin_z
            lines.append(f'  <joint name="{joint_name}" type="{seg_def.joint_type}">')
            lines.append(f'    <parent link="{seg_def.parent}"/>')
            lines.append(f'    <child link="{seg_name}"/>')
            lines.append(f'    <origin xyz="0 0 {origin_z:.4f}"/>')
            if seg_def.joint_type == "revolute":
                lines.append(f'    <axis xyz="{seg_def.axis}"/>')
                lines.append(
                    f'    <limit lower="{seg_def.limit_lower:.2f}"'
                    f' upper="{seg_def.limit_upper:.2f}"'
                    f' effort="100" velocity="2.0"/>'
                )
                lines.append(
                    f"    <dynamics"
                    f' damping="{config.damping:.2f}"'
                    f' friction="{config.friction:.2f}"/>'
                )
            lines.append("  </joint>")
            lines.append("")

    lines.append("</robot>")

    urdf_xml = "\n".join(lines)
    logger.info("Generated URDF with %d lines", len(lines))
    return urdf_xml


def validate_urdf_structure(urdf_xml: str) -> tuple[bool, list[str]]:
    """Validate basic URDF structural integrity.

    Checks:
      - Well-formed XML
      - Root element is ``<robot>``
      - All joints reference existing links
      - No duplicate link names

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: list[str] = []
    try:
        root = ET.fromstring(urdf_xml)  # nosec B314
    except ET.ParseError as e:
        return False, [f"XML parse error: {e}"]

    if root.tag != "robot":
        errors.append(f"Root element must be <robot>, got <{root.tag}>")

    # Collect link names
    link_names: set[str] = set()
    for link in root.findall("link"):
        name = link.get("name", "")
        if name in link_names:
            errors.append(f"Duplicate link name: {name}")
        link_names.add(name)

    # Validate joint references
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is not None:
            parent_link = parent_el.get("link", "")
            if parent_link not in link_names:
                errors.append(
                    f"Joint '{joint.get('name')}' references unknown"
                    f" parent link: {parent_link}"
                )
        if child_el is not None:
            child_link = child_el.get("link", "")
            if child_link not in link_names:
                errors.append(
                    f"Joint '{joint.get('name')}' references unknown"
                    f" child link: {child_link}"
                )

    return len(errors) == 0, errors


def _is_valid_xml_name(name: str) -> bool:
    """Check if a string is a valid XML NCName."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9._-]*$", name))


__all__ = [
    "generate_urdf_xml",
    "validate_urdf_structure",
]
