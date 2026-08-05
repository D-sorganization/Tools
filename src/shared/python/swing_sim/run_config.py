"""Typed run configuration and stable IDs for shared swing dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shared.python.contracts import require

from ._torque_profile_validation import stable_id

DOUBLE_PENDULUM_MODEL_ID = "model.double_pendulum.v1"
SHOULDER_JOINT_ID = "joint.shoulder"
WRIST_JOINT_ID = "joint.wrist"
DOUBLE_PENDULUM_JOINT_IDS = (SHOULDER_JOINT_ID, WRIST_JOINT_ID)


class SwingRunMode(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Supported execution modes for the shared double-pendulum source."""

    PASSIVE = "passive"
    PRESCRIBED = "prescribed"


@dataclass(frozen=True)
class JointLockConfig:
    """Immutable ideal locks keyed by canonical double-pendulum joint IDs."""

    locked_joint_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifiers = tuple(self.locked_joint_ids)
        for joint_id in identifiers:
            stable_id(joint_id, "locked joint_id")
        require(
            len(set(identifiers)) == len(identifiers),
            "locked joint IDs must be unique",
            identifiers,
        )
        unknown = set(identifiers) - set(DOUBLE_PENDULUM_JOINT_IDS)
        require(
            not unknown,
            "locked joint IDs must belong to the double-pendulum model",
            sorted(unknown),
        )
        canonical = tuple(
            joint_id
            for joint_id in DOUBLE_PENDULUM_JOINT_IDS
            if joint_id in identifiers
        )
        object.__setattr__(self, "locked_joint_ids", canonical)

    @property
    def has_locks(self) -> bool:
        """Whether at least one coordinate is locked."""
        return bool(self.locked_joint_ids)

    @property
    def mask(self) -> tuple[bool, bool]:
        """Return shoulder/wrist lock flags in kernel coordinate order."""
        return (
            SHOULDER_JOINT_ID in self.locked_joint_ids,
            WRIST_JOINT_ID in self.locked_joint_ids,
        )

    def is_locked(self, joint_id: str) -> bool:
        """Return whether one stable joint ID is locked."""
        stable_id(joint_id, "joint_id")
        return joint_id in self.locked_joint_ids


@dataclass(frozen=True)
class DoublePendulumRunConfig:
    """Configure passive/prescribed dynamics and optional ideal joint locks."""

    mode: SwingRunMode = SwingRunMode.PASSIVE
    prescribed_profile_id: str | None = None
    joint_locks: JointLockConfig = field(default_factory=JointLockConfig)

    def __post_init__(self) -> None:
        require(
            isinstance(self.mode, SwingRunMode), "invalid swing run mode", self.mode
        )
        require(
            isinstance(self.joint_locks, JointLockConfig),
            "joint_locks must be a JointLockConfig",
            self.joint_locks,
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
    def prescribed(
        cls,
        profile_id: str,
        *,
        joint_locks: JointLockConfig | None = None,
    ) -> DoublePendulumRunConfig:
        """Build a prescribed-mode configuration for a stable profile ID."""
        return cls(
            mode=SwingRunMode.PRESCRIBED,
            prescribed_profile_id=profile_id,
            joint_locks=joint_locks or JointLockConfig(),
        )


__all__ = [
    "DOUBLE_PENDULUM_JOINT_IDS",
    "DOUBLE_PENDULUM_MODEL_ID",
    "SHOULDER_JOINT_ID",
    "WRIST_JOINT_ID",
    "DoublePendulumRunConfig",
    "JointLockConfig",
    "SwingRunMode",
]
