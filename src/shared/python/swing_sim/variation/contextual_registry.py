"""Built-in variables that require an execution-context adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ..localized_torque import SHOULDER_JOINT_ID, WRIST_JOINT_ID


@dataclass(frozen=True)
class ContextualVariable:
    """Registry source data for a context-gated variable."""

    name: str
    label: str
    unit: str
    default: float
    typical_scale: float
    guidance: str
    applicability: str
    point_id: str


LOCALIZED_TORQUE_VARIABLES = (
    ContextualVariable(
        "shoulder_commanded_torque_offset_nm",
        "Shoulder Commanded Torque Offset",
        "N·m",
        0.0,
        2.0,
        "Additive double-pendulum command over a required half-open time "
        f"window at {SHOULDER_JOINT_ID}.",
        "localized_torque_only",
        SHOULDER_JOINT_ID,
    ),
    ContextualVariable(
        "wrist_commanded_torque_offset_nm",
        "Wrist Commanded Torque Offset",
        "N·m",
        0.0,
        1.0,
        "Additive double-pendulum command over a required half-open time "
        f"window at {WRIST_JOINT_ID}.",
        "localized_torque_only",
        WRIST_JOINT_ID,
    ),
)

__all__ = ["LOCALIZED_TORQUE_VARIABLES", "ContextualVariable"]
