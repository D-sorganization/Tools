"""Immutable in-memory library for reusable prescribed torque profiles."""

from __future__ import annotations

from dataclasses import dataclass

from shared.python.contracts import require

from ._torque_profile_validation import stable_id
from .torque_profiles import PrescribedTorqueProfile


@dataclass(frozen=True)
class TorqueProfileLibrary:
    """Validated collection indexed by stable profile identifier."""

    profiles: tuple[PrescribedTorqueProfile, ...] = ()

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        require(
            all(isinstance(profile, PrescribedTorqueProfile) for profile in profiles),
            "profiles must contain PrescribedTorqueProfile values",
        )
        profile_ids = tuple(profile.profile_id for profile in profiles)
        require(len(profile_ids) == len(set(profile_ids)), "profile IDs must be unique")
        object.__setattr__(self, "profiles", profiles)

    def get(self, profile_id: str) -> PrescribedTorqueProfile:
        """Resolve a profile or fail with an actionable stable-ID error."""
        selected_id = stable_id(profile_id, "profile_id")
        for profile in self.profiles:
            if profile.profile_id == selected_id:
                return profile
        require(False, "torque profile_id not found in library", selected_id)
        raise AssertionError("unreachable")

    def for_model(self, model_id: str) -> tuple[PrescribedTorqueProfile, ...]:
        """Return profiles compatible with one stable model identifier."""
        selected_id = stable_id(model_id, "model_id")
        return tuple(
            profile for profile in self.profiles if profile.model_id == selected_id
        )

    def with_profile(self, profile: PrescribedTorqueProfile) -> TorqueProfileLibrary:
        """Return a new library with the supplied profile added or replaced."""
        require(
            isinstance(profile, PrescribedTorqueProfile),
            "profile must be a PrescribedTorqueProfile",
            profile,
        )
        retained = tuple(
            current
            for current in self.profiles
            if current.profile_id != profile.profile_id
        )
        return TorqueProfileLibrary((*retained, profile))


__all__ = ["TorqueProfileLibrary"]
