"""Strict workspace selection for the canonical torque-profile library."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shared.python.swing_sim.run_config import (
    DoublePendulumRunConfig,
    JointLockConfig,
    SwingRunMode,
)
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile

TORQUE_WORKSPACE_SCHEMA = "rate_of_closure.torque_workspace_selection"
TORQUE_WORKSPACE_SCHEMA_VERSION = 1

_ENVELOPE_FIELDS = frozenset({"schema", "schema_version", "data"})
_DATA_FIELDS = frozenset({"active_profile_id", "run_config", "selection_provenance"})
_RUN_FIELDS = frozenset({"mode", "prescribed_profile_id", "locked_joint_ids"})
_PROVENANCE_FIELDS = frozenset({"kind", "profile_source"})


class LegacyTorqueMigrationRequired(ValueError):
    """A legacy explorer session needs an explicit torque-state fallback."""


@dataclass(frozen=True)
class TorqueWorkspaceState:
    """Immutable profile library, active selection, and existing run contract."""

    profiles: tuple[PrescribedTorqueProfile, ...]
    active_profile_id: str | None
    run_config: DoublePendulumRunConfig

    def __post_init__(self) -> None:
        """Validate identity relationships and normalize deterministic order."""
        profiles = tuple(self.profiles)
        if any(
            not isinstance(profile, PrescribedTorqueProfile) for profile in profiles
        ):
            raise TypeError("profiles must contain PrescribedTorqueProfile values")
        identifiers = tuple(profile.profile_id for profile in profiles)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("torque profile IDs must be unique")
        if not isinstance(self.run_config, DoublePendulumRunConfig):
            raise TypeError("run_config must be a DoublePendulumRunConfig")
        if profiles and self.active_profile_id is None:
            raise ValueError("a non-empty torque library requires an active profile")
        if (
            self.active_profile_id is not None
            and self.active_profile_id not in identifiers
        ):
            raise ValueError("active torque profile is not present in the library")
        selected = self.run_config.prescribed_profile_id
        if self.run_config.mode is SwingRunMode.PRESCRIBED:
            if selected != self.active_profile_id:
                raise ValueError("prescribed run profile must be the active profile")
        elif selected is not None:
            raise ValueError("passive run mode cannot prescribe a profile")
        object.__setattr__(
            self,
            "profiles",
            tuple(sorted(profiles, key=lambda profile: profile.profile_id)),
        )

    def active_profile(self) -> PrescribedTorqueProfile | None:
        """Return the selected canonical profile without label-based lookup."""
        if self.active_profile_id is None:
            return None
        return next(
            profile
            for profile in self.profiles
            if profile.profile_id == self.active_profile_id
        )


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TypeError(f"{context} has invalid fields")
    return value


def _selection_provenance(state: TorqueWorkspaceState) -> dict[str, object]:
    profile = state.active_profile()
    return {
        "kind": "none" if profile is None else "library_profile",
        "profile_source": None if profile is None else profile.source.value,
    }


def torque_workspace_to_payload(state: TorqueWorkspaceState) -> dict[str, object]:
    """Serialize selection only; canonical profiles remain at the workspace root."""
    if not isinstance(state, TorqueWorkspaceState):
        raise TypeError("state must be a TorqueWorkspaceState")
    config = state.run_config
    return {
        "schema": TORQUE_WORKSPACE_SCHEMA,
        "schema_version": TORQUE_WORKSPACE_SCHEMA_VERSION,
        "data": {
            "active_profile_id": state.active_profile_id,
            "run_config": {
                "mode": config.mode.value,
                "prescribed_profile_id": config.prescribed_profile_id,
                "locked_joint_ids": list(config.joint_locks.locked_joint_ids),
            },
            "selection_provenance": _selection_provenance(state),
        },
    }


def _run_config(value: object) -> DoublePendulumRunConfig:
    data = _exact_mapping(value, _RUN_FIELDS, "torque run_config")
    mode = SwingRunMode(data["mode"])
    locked = data["locked_joint_ids"]
    if not isinstance(locked, (list, tuple)):
        raise TypeError("locked_joint_ids must be a JSON array")
    locks = JointLockConfig(tuple(locked))
    profile_id = data["prescribed_profile_id"]
    if mode is SwingRunMode.PASSIVE:
        if profile_id is not None:
            raise ValueError("passive run mode cannot prescribe a profile")
        return DoublePendulumRunConfig(joint_locks=locks)
    if not isinstance(profile_id, str):
        raise TypeError("prescribed_profile_id must be a stable string")
    return DoublePendulumRunConfig.prescribed(profile_id, joint_locks=locks)


def torque_workspace_from_payload(
    value: object,
    profiles: Sequence[PrescribedTorqueProfile],
) -> TorqueWorkspaceState:
    """Parse selection and validate its provenance against the root library."""
    envelope = _exact_mapping(value, _ENVELOPE_FIELDS, "torque workspace")
    if (
        envelope["schema"] != TORQUE_WORKSPACE_SCHEMA
        or envelope["schema_version"] != TORQUE_WORKSPACE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported torque workspace selection payload")
    data = _exact_mapping(envelope["data"], _DATA_FIELDS, "torque workspace.data")
    active = data["active_profile_id"]
    if active is not None and not isinstance(active, str):
        raise TypeError("active_profile_id must be a stable string or null")
    state = TorqueWorkspaceState(
        tuple(profiles), active, _run_config(data["run_config"])
    )
    provenance = _exact_mapping(
        data["selection_provenance"],
        _PROVENANCE_FIELDS,
        "torque selection_provenance",
    )
    if dict(provenance) != _selection_provenance(state):
        raise ValueError("torque selection provenance does not match profile source")
    return state


def migrate_legacy_torque_fallback(
    fallback: TorqueWorkspaceState,
    document_profiles: Sequence[PrescribedTorqueProfile],
) -> TorqueWorkspaceState:
    """Preserve explicit live state unless a legacy root library conflicts."""
    if not isinstance(fallback, TorqueWorkspaceState):
        raise TypeError("legacy torque fallback must be complete")
    profiles = tuple(sorted(document_profiles, key=lambda profile: profile.profile_id))
    if profiles and profiles != fallback.profiles:
        raise LegacyTorqueMigrationRequired(
            "legacy workspace torque library conflicts with the explicit fallback"
        )
    return fallback


__all__ = [
    "LegacyTorqueMigrationRequired",
    "TORQUE_WORKSPACE_SCHEMA",
    "TORQUE_WORKSPACE_SCHEMA_VERSION",
    "TorqueWorkspaceState",
    "migrate_legacy_torque_fallback",
    "torque_workspace_from_payload",
    "torque_workspace_to_payload",
]
