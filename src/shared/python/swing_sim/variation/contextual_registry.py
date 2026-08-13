"""Built-in variables that require an execution-context adapter."""

from __future__ import annotations

from dataclasses import dataclass


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


LOCALIZED_TORQUE_VARIABLES = (
    ContextualVariable(
        "shoulder_commanded_torque_offset_nm",
        "Shoulder Commanded Torque Offset",
        "N·m",
        0.0,
        2.0,
        "Additive double-pendulum command over a required half-open time "
        "window at joint.shoulder.",
        "localized_torque_only",
    ),
    ContextualVariable(
        "wrist_commanded_torque_offset_nm",
        "Wrist Commanded Torque Offset",
        "N·m",
        0.0,
        1.0,
        "Additive double-pendulum command over a required half-open time "
        "window at joint.wrist.",
        "localized_torque_only",
    ),
)

__all__ = ["LOCALIZED_TORQUE_VARIABLES", "ContextualVariable"]
