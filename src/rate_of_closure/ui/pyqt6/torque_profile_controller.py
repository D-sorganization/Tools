"""UI-neutral seam between Rate of Closure and canonical torque profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote

from rate_of_closure.simulation.records import SimulationRun
from rate_of_closure.simulation.torque_history import fit_run_torque_profile
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_JOINT_IDS,
    DOUBLE_PENDULUM_MODEL_ID,
)
from shared.python.swing_sim.torque_library import (
    TorqueProfileLibrary as CanonicalTorqueProfileLibrary,
)
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)


class RunMode(StrEnum):
    """User-selected simulation input mode."""

    OPTIMIZED_DEFAULT = "optimized_default"
    PRESCRIBED_TORQUE = "prescribed_torque"


@dataclass(frozen=True)
class ProfileDraft:
    """Editable metadata needed to construct one canonical profile."""

    profile_id: str
    model_id: str
    name: str
    description: str
    time_domain_s: tuple[float, float]


@dataclass(frozen=True)
class TorqueExecutionSelection:
    """Validated UI-to-simulation prescribed-torque selection."""

    mode: RunMode
    profile: PrescribedTorqueProfile | None
    execution_ready: bool
    validation_message: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _profile_filename(profile_id: str) -> str:
    return f"{quote(profile_id, safe='._-')}.json"


def _write_profile(profile: PrescribedTorqueProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(profile.dumps(), encoding="utf-8")
    temporary.replace(path)


class TorqueProfileLibraryAdapter:
    """Persist the shared immutable library as canonical profile JSON files."""

    def __init__(self) -> None:
        self._library = CanonicalTorqueProfileLibrary()
        self._active_profile_id: str | None = None

    def profiles(self) -> tuple[PrescribedTorqueProfile, ...]:
        """Return profiles in stable identifier order."""
        return tuple(sorted(self._library.profiles, key=lambda item: item.profile_id))

    def active_profile(self) -> PrescribedTorqueProfile | None:
        """Return the selected profile, if any."""
        if self._active_profile_id is None:
            return None
        return self._library.get(self._active_profile_id)

    def canonical_library(self) -> CanonicalTorqueProfileLibrary:
        """Return the immutable library for a simulation request."""
        return self._library

    def set_active(self, profile_id: str) -> PrescribedTorqueProfile:
        """Select an existing profile."""
        profile = self._library.get(profile_id)
        self._active_profile_id = profile_id
        return profile

    def assign(
        self, draft: ProfileDraft, joint_id: str, coefficients: list[float]
    ) -> PrescribedTorqueProfile:
        """Create or replace one joint assignment using canonical c0-first values."""
        previous = next(
            (
                profile
                for profile in self._library.profiles
                if profile.profile_id == draft.profile_id
            ),
            None,
        )
        if previous is not None and previous.model_id != draft.model_id:
            raise ValueError("A profile model cannot change after assignments exist")
        assignment = JointTorqueAssignment(
            joint_id=joint_id,
            polynomial=TorquePolynomial(tuple(coefficients)),
        )
        profile = self._updated_profile(draft, previous, assignment)
        self._library = self._library.with_profile(profile)
        self._active_profile_id = profile.profile_id
        return profile

    def _updated_profile(
        self,
        draft: ProfileDraft,
        previous: PrescribedTorqueProfile | None,
        assignment: JointTorqueAssignment,
    ) -> PrescribedTorqueProfile:
        timestamp = _utc_now()
        assignments = (
            {}
            if previous is None
            else {item.joint_id: item for item in previous.assignments}
        )
        assignments[assignment.joint_id] = assignment
        return PrescribedTorqueProfile(
            profile_id=draft.profile_id,
            model_id=draft.model_id,
            name=draft.name,
            description=draft.description,
            source=TorqueProfileSource.DRAWN,
            source_metadata={
                "application": "rate_of_closure",
                "editor": "polynomial_generator",
            },
            created_at_utc=(timestamp if previous is None else previous.created_at_utc),
            modified_at_utc=timestamp,
            time_domain_s=draft.time_domain_s,
            assignments=tuple(assignments[key] for key in sorted(assignments)),
        )

    def import_profile(self, path: Path) -> PrescribedTorqueProfile:
        """Validate and add one canonical profile JSON file."""
        profile = PrescribedTorqueProfile.loads(path.read_text(encoding="utf-8"))
        self._library = self._library.with_profile(profile)
        self._active_profile_id = profile.profile_id
        return profile

    def fit_run(
        self,
        draft: ProfileDraft,
        run: SimulationRun,
        degree: int,
    ) -> PrescribedTorqueProfile:
        """Fit retained applied torques and select the canonical result."""
        previous = next(
            (
                profile
                for profile in self._library.profiles
                if profile.profile_id == draft.profile_id
            ),
            None,
        )
        timestamp = _utc_now()
        lock_ids = run.config.swing_run_config.joint_locks.locked_joint_ids
        profile = fit_run_torque_profile(
            run,
            profile_id=draft.profile_id,
            name=draft.name,
            description=draft.description,
            degree=degree,
            source_metadata={
                "application": "rate_of_closure",
                "source_kind": run.config.source_kind,
                "contact_outcome": run.impact_outcome.status.value,
                "joint_locks": ",".join(lock_ids) if lock_ids else "none",
            },
            created_at_utc=(timestamp if previous is None else previous.created_at_utc),
            modified_at_utc=timestamp,
        )
        self._library = self._library.with_profile(profile)
        self._active_profile_id = profile.profile_id
        return profile

    def export_profile(self, profile_id: str, path: Path) -> None:
        """Export one profile without a UI-specific wrapper schema."""
        _write_profile(self._library.get(profile_id), path)

    def save_library(self, directory: Path) -> int:
        """Save every profile as an independently portable canonical JSON file."""
        directory.mkdir(parents=True, exist_ok=True)
        for profile in self.profiles():
            _write_profile(profile, directory / _profile_filename(profile.profile_id))
        return len(self._library.profiles)

    def load_library(self, directory: Path) -> int:
        """Atomically replace the library from canonical JSON files in a directory."""
        loaded = [
            PrescribedTorqueProfile.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]
        self._library = CanonicalTorqueProfileLibrary(tuple(loaded))
        self._active_profile_id = loaded[0].profile_id if loaded else None
        return len(loaded)


def execution_selection(
    mode: RunMode,
    profile: PrescribedTorqueProfile | None,
) -> TorqueExecutionSelection:
    """Validate what the current shared dynamics kernel can execute."""
    if mode is RunMode.OPTIMIZED_DEFAULT:
        return TorqueExecutionSelection(mode, profile, True, "Default execution ready.")
    if profile is None:
        return TorqueExecutionSelection(
            mode,
            None,
            False,
            "Author or load a prescribed-torque profile before running.",
        )
    if profile.model_id != DOUBLE_PENDULUM_MODEL_ID:
        return TorqueExecutionSelection(
            mode,
            profile,
            False,
            "Prescribed execution currently supports the double-pendulum model; "
            "the triple-pendulum profile remains portable for future execution.",
        )
    joint_ids = {assignment.joint_id for assignment in profile.assignments}
    if joint_ids != set(DOUBLE_PENDULUM_JOINT_IDS):
        missing = sorted(set(DOUBLE_PENDULUM_JOINT_IDS) - joint_ids)
        return TorqueExecutionSelection(
            mode,
            profile,
            False,
            "Assign every double-pendulum joint before running. Missing: "
            + ", ".join(missing),
        )
    return TorqueExecutionSelection(
        mode,
        profile,
        True,
        "Prescribed double-pendulum execution ready.",
    )


__all__ = [
    "ProfileDraft",
    "RunMode",
    "TorqueExecutionSelection",
    "TorqueProfileLibraryAdapter",
    "execution_selection",
]
