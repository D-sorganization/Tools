"""Immutable value contracts for qualified ground material profile libraries."""

from __future__ import annotations

from dataclasses import dataclass, field

from .profile_enums import (
    CANONICAL_GROUND_PARAMETER_IDS,
    GROUND_MATERIAL_PROFILE_SCHEMA_VERSION,
    GROUND_PROFILE_LIBRARY_SCHEMA_VERSION,
    GroundEvidenceKind,
    GroundModelUseStatus,
    GroundParameterId,
    GroundQualificationGateId,
    GroundQualificationStatus,
)
from .profile_validation import (
    ProfileDocument,
    bounded_number,
    calibrated_model_use,
    canonical_enum_subset,
    exact_record,
    exact_records,
    nonnegative_number,
    parameter_unit,
    parameter_validity,
    parameter_value,
    positive_number,
    qualification_decisions,
    sha256_digest,
    sorted_unique_texts,
    strict_boolean,
    strict_text,
    validity_source_ids,
)


@dataclass(frozen=True)
class GroundMaterialParameter:
    """One SI value with uncertainty and evidence-linked validity bounds."""

    parameter_id: GroundParameterId
    value_si: float
    standard_uncertainty_si: float
    coverage_factor: float
    confidence_level: float
    validity_lower_si: float
    validity_upper_si: float
    validity_lower_evidence_ids: tuple[str, ...]
    validity_upper_evidence_ids: tuple[str, ...]
    unit_si: str = ""

    def __post_init__(self) -> None:
        parameter_id = GroundParameterId(self.parameter_id)
        expected_unit = parameter_unit(str(parameter_id))
        unit = (
            expected_unit
            if self.unit_si == ""
            else strict_text(self.unit_si, "unit_si")
        )
        if unit != expected_unit:
            raise ValueError(f"unit_si for {parameter_id} must be {expected_unit}")
        object.__setattr__(self, "parameter_id", parameter_id)
        object.__setattr__(self, "unit_si", unit)
        object.__setattr__(
            self, "value_si", parameter_value(str(parameter_id), self.value_si)
        )
        object.__setattr__(
            self,
            "standard_uncertainty_si",
            nonnegative_number(self.standard_uncertainty_si, "standard_uncertainty_si"),
        )
        object.__setattr__(
            self,
            "coverage_factor",
            positive_number(self.coverage_factor, "coverage_factor"),
        )
        if self.coverage_factor < 1.0:
            raise ValueError("coverage_factor must be at least 1")
        object.__setattr__(
            self,
            "confidence_level",
            bounded_number(self.confidence_level, "confidence_level", (0.0, 1.0)),
        )
        if self.confidence_level == 0.0:
            raise ValueError("confidence_level must be positive")
        lower, upper = parameter_validity(
            str(parameter_id),
            self.value_si,
            self.validity_lower_si,
            self.validity_upper_si,
        )
        object.__setattr__(self, "validity_lower_si", lower)
        object.__setattr__(self, "validity_upper_si", upper)
        for name in ("validity_lower_evidence_ids", "validity_upper_evidence_ids"):
            object.__setattr__(
                self,
                name,
                sorted_unique_texts(getattr(self, name), name),
            )


@dataclass(frozen=True)
class GroundProfileEvidence:
    """Immutable evidence identity and its parameter coverage."""

    evidence_id: str
    kind: GroundEvidenceKind
    citation: str
    source_uri: str
    source_sha256: str
    parameter_ids: tuple[GroundParameterId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", strict_text(self.evidence_id, "evidence_id")
        )
        object.__setattr__(self, "kind", GroundEvidenceKind(self.kind))
        for name in ("citation", "source_uri"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        object.__setattr__(
            self, "source_sha256", sha256_digest(self.source_sha256, "source_sha256")
        )
        object.__setattr__(
            self,
            "parameter_ids",
            canonical_enum_subset(
                self.parameter_ids,
                GroundParameterId,
                CANONICAL_GROUND_PARAMETER_IDS,
                "evidence parameter_ids",
            ),
        )


@dataclass(frozen=True)
class GroundProfileRights:
    """Machine-readable rights needed to decide safe reuse."""

    license_id: str
    rights_holder: str
    redistribution_allowed: bool
    derivative_use_allowed: bool

    def __post_init__(self) -> None:
        for name in ("license_id", "rights_holder"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        for name in ("redistribution_allowed", "derivative_use_allowed"):
            object.__setattr__(self, name, strict_boolean(getattr(self, name), name))


@dataclass(frozen=True)
class GroundApplicability:
    """Bounded environment and declared surface classes for one profile."""

    surface_classes: tuple[str, ...]
    temperature_min_k: float
    temperature_max_k: float
    moisture_min_fraction: float
    moisture_max_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_classes",
            sorted_unique_texts(self.surface_classes, "surface_classes"),
        )
        minimum = positive_number(self.temperature_min_k, "temperature_min_k")
        maximum = positive_number(self.temperature_max_k, "temperature_max_k")
        if maximum < minimum:
            raise ValueError("temperature maximum must not be below minimum")
        object.__setattr__(self, "temperature_min_k", minimum)
        object.__setattr__(self, "temperature_max_k", maximum)
        moisture_min = bounded_number(
            self.moisture_min_fraction, "moisture_min_fraction", (0.0, 1.0)
        )
        moisture_max = bounded_number(
            self.moisture_max_fraction, "moisture_max_fraction", (0.0, 1.0)
        )
        if moisture_max < moisture_min:
            raise ValueError("moisture maximum must not be below minimum")
        object.__setattr__(self, "moisture_min_fraction", moisture_min)
        object.__setattr__(self, "moisture_max_fraction", moisture_max)


@dataclass(frozen=True)
class GroundCalibrationRecord:
    """Calibration method and exact evidence/parameter dependencies."""

    calibration_id: str
    method: str
    evidence_ids: tuple[str, ...]
    parameter_ids: tuple[GroundParameterId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "calibration_id", strict_text(self.calibration_id, "calibration_id")
        )
        object.__setattr__(
            self, "method", strict_text(self.method, "calibration method")
        )
        object.__setattr__(
            self,
            "evidence_ids",
            sorted_unique_texts(self.evidence_ids, "calibration evidence_ids"),
        )
        object.__setattr__(
            self,
            "parameter_ids",
            canonical_enum_subset(
                self.parameter_ids,
                GroundParameterId,
                CANONICAL_GROUND_PARAMETER_IDS,
                "calibration parameter_ids",
            ),
        )


@dataclass(frozen=True)
class GroundProfileProvenance:
    """Reproducible producer, source revision, and immutable source digest."""

    producer: str
    producer_version: str
    source_revision: str
    source_sha256: str

    def __post_init__(self) -> None:
        for name in ("producer", "producer_version", "source_revision"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        object.__setattr__(
            self, "source_sha256", sha256_digest(self.source_sha256, "source_sha256")
        )


@dataclass(frozen=True)
class GroundQualificationGate:
    """One stable, derived qualification decision."""

    gate_id: GroundQualificationGateId
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", GroundQualificationGateId(self.gate_id))
        object.__setattr__(self, "passed", strict_boolean(self.passed, "gate passed"))


@dataclass(frozen=True)
class GroundProfileQualification:
    """Ordered qualification gates and their coherent aggregate status."""

    status: GroundQualificationStatus
    gates: tuple[GroundQualificationGate, ...]

    def __post_init__(self) -> None:
        status = GroundQualificationStatus(self.status)
        gates = exact_records(self.gates, GroundQualificationGate, "qualification gate")
        if tuple(gate.gate_id for gate in gates) != tuple(GroundQualificationGateId):
            raise ValueError("qualification gates must use canonical gate order")
        expected = (
            GroundQualificationStatus.QUALIFIED
            if all(gate.passed for gate in gates)
            else GroundQualificationStatus.UNQUALIFIED
        )
        if status is not expected:
            raise ValueError("qualification status does not match gate results")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "gates", gates)

    def gate_passed(self, gate_id: GroundQualificationGateId | str) -> bool:
        """Return the derived result for one stable gate identifier."""
        normalized = GroundQualificationGateId(gate_id)
        return next(gate.passed for gate in self.gates if gate.gate_id is normalized)


@dataclass(frozen=True)
class GroundMaterialProfile(ProfileDocument):
    """Strict v1 profile with derived qualification and model-use status."""

    profile_id: str
    display_name: str
    revision: str
    parameters: tuple[GroundMaterialParameter, ...]
    evidence: tuple[GroundProfileEvidence, ...]
    rights: GroundProfileRights
    applicability: GroundApplicability
    calibration: GroundCalibrationRecord
    provenance: GroundProfileProvenance
    qualification: GroundProfileQualification = field(init=False)
    model_use_status: GroundModelUseStatus = field(init=False)
    schema_version: str = GROUND_MATERIAL_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("profile_id", "display_name", "revision"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        if self.schema_version != GROUND_MATERIAL_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        parameters = exact_records(
            self.parameters, GroundMaterialParameter, "material parameter"
        )
        if (
            tuple(item.parameter_id for item in parameters)
            != CANONICAL_GROUND_PARAMETER_IDS
        ):
            raise ValueError("parameters must use exact canonical parameter order")
        if self.parameter_value("kinetic_friction") > self.parameter_value(
            "static_friction"
        ):
            raise ValueError("kinetic_friction must not exceed static_friction")
        evidence = exact_records(
            self.evidence, GroundProfileEvidence, "profile evidence"
        )
        exact_record(self.rights, GroundProfileRights, "rights")
        exact_record(self.applicability, GroundApplicability, "applicability")
        exact_record(self.calibration, GroundCalibrationRecord, "calibration")
        exact_record(self.provenance, GroundProfileProvenance, "provenance")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if not evidence or evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("evidence must be nonempty, sorted, and unique")
        unknown = set(self.calibration.evidence_ids) - set(evidence_ids)
        if unknown:
            raise ValueError("calibration references an unknown evidence_id")
        unknown_validity = validity_source_ids(parameters) - set(evidence_ids)
        if unknown_validity:
            raise ValueError("parameter references an unknown validity evidence_id")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "evidence", evidence)
        qualification = _qualification(self)
        object.__setattr__(self, "qualification", qualification)
        model_status = (
            GroundModelUseStatus.CALIBRATED
            if calibrated_model_use(self, CANONICAL_GROUND_PARAMETER_IDS)
            else GroundModelUseStatus.ILLUSTRATIVE
        )
        object.__setattr__(self, "model_use_status", model_status)

    def parameter_value(self, parameter_id: GroundParameterId | str) -> float:
        """Return one SI parameter value by stable identifier."""
        normalized = GroundParameterId(parameter_id)
        return next(
            item.value_si for item in self.parameters if item.parameter_id is normalized
        )


def _qualification(profile: GroundMaterialProfile) -> GroundProfileQualification:
    decisions = qualification_decisions(profile, CANONICAL_GROUND_PARAMETER_IDS)
    gates = tuple(
        GroundQualificationGate(GroundQualificationGateId(gate_id), passed)
        for gate_id, passed in zip(GroundQualificationGateId, decisions, strict=True)
    )
    status = (
        GroundQualificationStatus.QUALIFIED
        if all(decisions)
        else GroundQualificationStatus.UNQUALIFIED
    )
    return GroundProfileQualification(status, gates)


@dataclass(frozen=True)
class GroundProfileLibrary(ProfileDocument):
    """Deterministic, immutable collection of material profile revisions."""

    library_id: str
    revision: str
    profiles: tuple[GroundMaterialProfile, ...]
    provenance: GroundProfileProvenance
    schema_version: str = GROUND_PROFILE_LIBRARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("library_id", "revision"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        if self.schema_version != GROUND_PROFILE_LIBRARY_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        profiles = exact_records(
            self.profiles, GroundMaterialProfile, "material profile"
        )
        exact_record(self.provenance, GroundProfileProvenance, "provenance")
        identities = tuple((item.profile_id, item.revision) for item in profiles)
        if not profiles:
            raise ValueError("profiles must not be empty")
        if len(set(identities)) != len(identities):
            raise ValueError("profile identities must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError("profiles must be sorted by profile_id and revision")
        object.__setattr__(self, "profiles", profiles)

    def profile(
        self, profile_id: str, revision: str | None = None
    ) -> GroundMaterialProfile:
        """Return exactly one identified profile, rejecting ambiguous revisions."""
        matches = tuple(
            item
            for item in self.profiles
            if item.profile_id == profile_id
            and (revision is None or item.revision == revision)
        )
        if len(matches) != 1:
            raise KeyError("profile lookup must resolve exactly one revision")
        return matches[0]
