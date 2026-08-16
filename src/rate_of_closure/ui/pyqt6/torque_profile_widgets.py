"""Shared widget metadata for prescribed joint-torque profile authoring."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from rate_of_closure.ui.pyqt6.torque_profile_controller import RunMode

MODEL_JOINTS = {
    "model.double_pendulum.v1": (
        ("joint.shoulder", "Shoulder"),
        ("joint.wrist", "Wrist"),
    ),
    "model.triple_pendulum.v1": (
        ("joint.shoulder", "Shoulder"),
        ("joint.wrist", "Wrist"),
        ("joint.club", "Club"),
    ),
}

MODE_DESCRIPTIONS = {
    RunMode.OPTIMIZED_DEFAULT: (
        "Uses the selected simulator source and its default or solver-configured "
        "motion without applying a prescribed joint-torque profile."
    ),
    RunMode.PRESCRIBED_TORQUE: (
        "Executes a complete double-pendulum profile in the time-aware Python "
        "dynamics kernel. Triple-pendulum profiles can be authored and exchanged, "
        "but are not yet executable."
    ),
}


def clickable_button(button: QPushButton, tooltip: str) -> QPushButton:
    """Give an action button the shared pointer and guidance treatment."""
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setToolTip(tooltip)
    return button


__all__ = ["MODE_DESCRIPTIONS", "MODEL_JOINTS", "clickable_button"]
