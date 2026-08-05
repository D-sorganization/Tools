"""Typed run configuration and stable IDs for shared swing dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shared.python.contracts import require

from ._torque_profile_validation import stable_id

DOUBLE_PENDULUM_MODEL_ID = "model.double_pendulum.v1"
SHOULDER_JOINT_ID = "joint.shoulder"
WRIST_JOINT_ID = "joint.wrist"
DOUBLE_PENDULUM_JOINT_IDS = (SHOULDER_JOINT_ID, WRIST_JOINT_ID)


class SwingRunMode(StrEnum):
    """Supported execution modes for the shared double-pendulum source."""

    PASSIVE = "passive"
    PRESCRIBED = "prescribed"


@dataclass(frozen=True)
class DoublePendulumRunConfig:
    """Select passive dynamics or one library-backed prescribed profile."""

    mode: SwingRunMode = SwingRunMode.PASSIVE
    prescribed_profile_id: str | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.mode, SwingRunMode), "invalid swing run mode", self.mode
        )
        if self.mode is SwingRunMode.PASSIVE:
            require(
                self.prescribed_profile_id is None,
                "passive mode must not specify a prescribed profile",
            )
            return
        require(
            self.prescribed_profile_id is not None,
            "prescribed mode requires a profile_id",
        )
        stable_id(self.prescribed_profile_id, "prescribed_profile_id")

    @classmethod
    def prescribed(cls, profile_id: str) -> DoublePendulumRunConfig:
        """Build a prescribed-mode configuration for a stable profile ID."""
        return cls(
            mode=SwingRunMode.PRESCRIBED,
            prescribed_profile_id=profile_id,
        )


__all__ = [
    "DOUBLE_PENDULUM_JOINT_IDS",
    "DOUBLE_PENDULUM_MODEL_ID",
    "SHOULDER_JOINT_ID",
    "WRIST_JOINT_ID",
    "DoublePendulumRunConfig",
    "SwingRunMode",
]
