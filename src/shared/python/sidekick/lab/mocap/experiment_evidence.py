"""Structural comparison evidence receipts without scientific qualification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ._validation import require_text, require_unique_text
from .experiment_contracts import (
    ActuationInputKind,
    CapabilityAvailability,
    CapabilitySupport,
    ReplayMode,
    _require_sha256,
)
from .experiment_replay import ExperimentReplayBundle

COMPARISON_EVIDENCE_SCHEMA_VERSION = "comparison-evidence/1.0.0"
_OPAQUE_REFERENCE = re.compile(r"^opaque:[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")


def _require_opaque_reference(value: str) -> str:
    reference = require_text(value, "reference_id")
    if not _OPAQUE_REFERENCE.fullmatch(reference):
        raise ValueError("reference_id must be an opaque token, not a path or URL")
    return reference


class ComparisonLevel(StrEnum):
    """Kind of evidence comparison; values align with UpstreamDrift F01."""

    IDENTITY = "identity"
    TRANSCRIPTION_FEASIBILITY = "transcription_feasibility"
    WITHIN_ENGINE_REPLAY = "within_engine_replay"
    SAME_INPUT = "same_input"
    OBSERVATION_ACCURACY = "observation_accuracy"
    BIOMECHANICAL_EQUIVALENCE = "biomechanical_equivalence"


class DriveMode(StrEnum):
    """Coarse package drive class used by the UpstreamDrift F01 inventory."""

    TORQUE = "torque"
    MUSCLE_EXCITATION = "muscle_excitation"


class ImplementationEvidenceKind(StrEnum):
    """Implementation identities carried with each engine evidence row."""

    DYNAMICS = "dynamics"
    CONTACT = "contact"
    INTEGRATOR = "integrator"
    ACTUATOR = "actuator"
    RESTART = "restart"


class EvidenceArtifactKind(StrEnum):
    """Opaque, digest-bound artifacts referenced by an evidence row."""

    DYNAMICS = "dynamics"
    CONTACT = "contact"
    INTEGRATOR = "integrator"
    ACTUATOR = "actuator"
    RESTART = "restart"
    OBSERVATION = "observation"
    FORCE = "force"
    MUSCLE_STATE = "muscle_state"
    TRANSCRIPTION = "transcription"
    OBSERVATION_SCORE = "observation_score"


@dataclass(frozen=True, slots=True)
class EvidenceArtifactReference:
    """A private or public artifact pointer that contains no path or payload."""

    kind: EvidenceArtifactKind
    reference_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceArtifactKind):
            raise TypeError("kind must be an EvidenceArtifactKind")
        object.__setattr__(
            self, "reference_id", _require_opaque_reference(self.reference_id)
        )
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class ImplementationEvidence:
    """Declared support, availability and immutable identity for one implementation."""

    kind: ImplementationEvidenceKind
    implementation_id: str | None
    version: str | None
    sha256: str | None
    required: bool
    support: CapabilitySupport
    availability: CapabilityAvailability
    reason: str | None = None
    evidence_reference_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ImplementationEvidenceKind):
            raise TypeError("kind must be an ImplementationEvidenceKind")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        if not isinstance(self.support, CapabilitySupport):
            raise TypeError("support must be a CapabilitySupport")
        if not isinstance(self.availability, CapabilityAvailability):
            raise TypeError("availability must be a CapabilityAvailability")
        for name in ("implementation_id", "version", "reason"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_text(value, name))
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", _require_sha256(self.sha256, "sha256"))
        if self.evidence_reference_id is not None:
            if self.sha256 is None:
                raise ValueError(
                    "an implementation evidence reference requires a sha256"
                )
            object.__setattr__(
                self,
                "evidence_reference_id",
                _require_opaque_reference(self.evidence_reference_id),
            )
        if self.is_available and not all(
            (self.implementation_id, self.version, self.sha256)
        ):
            raise ValueError(
                "available implementation evidence requires id, version and hash"
            )
        if not self.is_available and self.reason is None:
            raise ValueError(
                "unknown or unavailable implementation evidence requires a reason"
            )

    @property
    def is_available(self) -> bool:
        """Return whether the implementation is declared supported and available."""
        return (
            self.support is CapabilitySupport.SUPPORTED
            and self.availability is CapabilityAvailability.AVAILABLE
        )


@dataclass(frozen=True, slots=True)
class ComparisonRowRequirement:
    """Caller-supplied required row and minimum evidence class."""

    row_id: str
    required: bool
    replay_mode: ReplayMode | None = None
    required_implementation_kinds: tuple[ImplementationEvidenceKind, ...] = ()
    required_artifact_kinds: tuple[EvidenceArtifactKind, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", require_text(self.row_id, "row_id"))
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        if self.replay_mode is not None:
            if not isinstance(self.replay_mode, ReplayMode):
                raise TypeError("replay_mode must be a ReplayMode")
        implementations = tuple(self.required_implementation_kinds)
        artifacts = tuple(self.required_artifact_kinds)
        if any(
            not isinstance(item, ImplementationEvidenceKind) for item in implementations
        ):
            raise TypeError("required_implementation_kinds contains an invalid kind")
        if any(not isinstance(item, EvidenceArtifactKind) for item in artifacts):
            raise TypeError("required_artifact_kinds contains an invalid kind")
        object.__setattr__(self, "required_implementation_kinds", implementations)
        object.__setattr__(self, "required_artifact_kinds", artifacts)
        require_unique_text(
            tuple(item.value for item in implementations), "implementation kinds"
        )
        require_unique_text(tuple(item.value for item in artifacts), "artifact kinds")


@dataclass(frozen=True, slots=True)
class ComparisonEvidenceRow:
    """One engine/model row with an optional T01 replay payload and receipts."""

    row_id: str
    package_id: str
    variant_id: str
    drive_mode: DriveMode
    replay_bundle: ExperimentReplayBundle | None
    support: CapabilitySupport
    availability: CapabilityAvailability
    implementation_evidence: tuple[ImplementationEvidence, ...] = ()
    artifacts: tuple[EvidenceArtifactReference, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("row_id", "package_id", "variant_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        if not isinstance(self.drive_mode, DriveMode):
            raise TypeError("drive_mode must be a DriveMode")
        if self.replay_bundle is not None and not isinstance(
            self.replay_bundle, ExperimentReplayBundle
        ):
            raise TypeError("replay_bundle must be an ExperimentReplayBundle or None")
        self._validate_drive_mode_compatibility()
        if not isinstance(self.support, CapabilitySupport):
            raise TypeError("support must be a CapabilitySupport")
        if not isinstance(self.availability, CapabilityAvailability):
            raise TypeError("availability must be a CapabilityAvailability")
        if self.reason is not None:
            object.__setattr__(self, "reason", require_text(self.reason, "reason"))
        if (
            self.support is not CapabilitySupport.SUPPORTED
            or self.availability is not CapabilityAvailability.AVAILABLE
        ) and self.reason is None:
            raise ValueError("unknown or unavailable evidence rows require a reason")
        if self.replay_bundle is None and self.reason is None:
            raise ValueError(
                "a row without a replay bundle requires an explicit reason"
            )
        evidence = tuple(self.implementation_evidence)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ImplementationEvidence) for item in evidence):
            raise TypeError("implementation_evidence contains an invalid value")
        if any(not isinstance(item, EvidenceArtifactReference) for item in artifacts):
            raise TypeError("artifacts contains an invalid value")
        object.__setattr__(self, "implementation_evidence", evidence)
        object.__setattr__(self, "artifacts", artifacts)
        require_unique_text(
            tuple(item.kind.value for item in evidence), "implementation_evidence"
        )
        require_unique_text(tuple(item.reference_id for item in artifacts), "artifacts")

    def _validate_drive_mode_compatibility(self) -> None:
        """Keep F01's coarse drive class distinct from T01's physical input kind."""
        if self.replay_bundle is None:
            return
        input_kind = self.replay_bundle.input_history.input_kind
        if self.drive_mode is DriveMode.TORQUE:
            compatible = {
                ActuationInputKind.ACTUATOR_COMMAND,
                ActuationInputKind.ACTUATOR_TORQUE,
                ActuationInputKind.ACTUATOR_FORCE,
                ActuationInputKind.GENERALIZED_EFFORT,
            }
        else:
            compatible = {ActuationInputKind.MUSCLE_EXCITATION}
        if input_kind not in compatible:
            raise ValueError(
                f"drive_mode {self.drive_mode.value!r} is incompatible with "
                f"T01 input_kind {input_kind.value!r}"
            )

    @property
    def replay_mode(self) -> ReplayMode | None:
        """Return the actual mode declared by the linked replay policy."""
        if self.replay_bundle is None:
            return None
        return self.replay_bundle.policy.replay_mode


@dataclass(frozen=True, slots=True)
class ComparisonEvidenceReceipt:
    """Versioned structural evidence matrix; it never emits a qualification result."""

    receipt_id: str
    comparison_level: ComparisonLevel
    requirements: tuple[ComparisonRowRequirement, ...]
    rows: tuple[ComparisonEvidenceRow, ...]
    schema_version: str = COMPARISON_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", require_text(self.receipt_id, "receipt_id")
        )
        if self.schema_version != COMPARISON_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {COMPARISON_EVIDENCE_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.comparison_level, ComparisonLevel):
            raise TypeError("comparison_level must be a ComparisonLevel")
        requirements = tuple(self.requirements)
        rows = tuple(self.rows)
        if not requirements or any(
            not isinstance(item, ComparisonRowRequirement) for item in requirements
        ):
            raise ValueError(
                "requirements must contain ComparisonRowRequirement values"
            )
        if any(not isinstance(item, ComparisonEvidenceRow) for item in rows):
            raise TypeError("rows contains an invalid value")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "rows", rows)
        requirement_ids = tuple(item.row_id for item in requirements)
        row_ids = tuple(item.row_id for item in rows)
        require_unique_text(requirement_ids, "requirements")
        require_unique_text(row_ids, "rows")
        if set(row_ids) - set(requirement_ids):
            raise ValueError("rows contain ids absent from requirements")
        self._validate_row_modes()
        self._validate_comparison_modes()

    @property
    def missing_required_rows(self) -> tuple[str, ...]:
        """Return required rows lacking an executable replay payload."""
        by_id = {item.row_id: item for item in self.rows}
        return tuple(
            item.row_id
            for item in self.requirements
            if item.required
            and (item.row_id not in by_id or by_id[item.row_id].replay_bundle is None)
        )

    @property
    def missing_required_evidence(self) -> tuple[str, ...]:
        """Return required implementation or artifact evidence not present."""
        by_id = {item.row_id: item for item in self.rows}
        missing: set[str] = set()
        for requirement in self.requirements:
            row = by_id.get(requirement.row_id)
            if not requirement.required or row is None:
                continue
            implementation_by_kind = {
                item.kind: item for item in row.implementation_evidence
            }
            artifact_kinds = {item.kind for item in row.artifacts}
            missing.update(
                f"{row.row_id}:implementation:{kind.value}"
                for kind in requirement.required_implementation_kinds
                if kind not in implementation_by_kind
                or not implementation_by_kind[kind].is_available
            )
            missing.update(
                f"{row.row_id}:implementation:{item.kind.value}"
                for item in row.implementation_evidence
                if item.required and not item.is_available
            )
            missing.update(
                f"{row.row_id}:artifact:{kind.value}"
                for kind in requirement.required_artifact_kinds
                if kind not in artifact_kinds
            )
        return tuple(sorted(missing))

    def _validate_row_modes(self) -> None:
        requirements = {item.row_id: item for item in self.requirements}
        for row in self.rows:
            expected = requirements[row.row_id].replay_mode
            if (
                expected is not None
                and row.replay_mode is not None
                and row.replay_mode != expected
            ):
                raise ValueError(
                    f"row {row.row_id!r} replay_mode {row.replay_mode.value!r} "
                    "does not satisfy "
                    f"required replay_mode {expected.value!r}"
                )

    def _validate_comparison_modes(self) -> None:
        replay_rows = tuple(row for row in self.rows if row.replay_bundle is not None)
        if (
            self.comparison_level is not ComparisonLevel.IDENTITY
            and len(replay_rows) > 1
        ):
            drive_modes = {row.drive_mode for row in replay_rows}
            if len(drive_modes) != 1:
                raise ValueError("comparison cannot mix drive modes")
            modes = {row.replay_mode for row in replay_rows}
            if len(modes) != 1:
                raise ValueError("comparison cannot mix replay modes")
        if self.comparison_level is ComparisonLevel.SAME_INPUT and len(replay_rows) > 1:
            comparison_keys = set()
            for row in replay_rows:
                bundle = row.replay_bundle
                assert bundle is not None, "replay row must contain a bundle"
                comparison_keys.add(
                    (
                        bundle.applied_input_sha256,
                        bundle.time_grid_sha256,
                        bundle.input_channel_schema_sha256,
                        bundle.policy_sha256,
                        bundle.state_schema_sha256,
                    )
                )
            if len(comparison_keys) != 1:
                raise ValueError(
                    "same_input comparison requires matching input, time-grid, "
                    "channel, "
                    "policy, and state-schema hashes"
                )


def build_comparison_evidence_receipt(
    receipt_id: str,
    comparison_level: ComparisonLevel,
    requirements: tuple[ComparisonRowRequirement, ...],
    rows: tuple[ComparisonEvidenceRow, ...],
) -> ComparisonEvidenceReceipt:
    """Build an immutable comparison evidence receipt after structural checks."""
    return ComparisonEvidenceReceipt(receipt_id, comparison_level, requirements, rows)


__all__ = [
    "COMPARISON_EVIDENCE_SCHEMA_VERSION",
    "ComparisonEvidenceReceipt",
    "ComparisonEvidenceRow",
    "ComparisonLevel",
    "ComparisonRowRequirement",
    "DriveMode",
    "EvidenceArtifactKind",
    "EvidenceArtifactReference",
    "ImplementationEvidence",
    "ImplementationEvidenceKind",
    "build_comparison_evidence_receipt",
]
