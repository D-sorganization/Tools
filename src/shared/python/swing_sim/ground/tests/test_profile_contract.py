"""Value-contract tests for ground material profiles and libraries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from shared.python.swing_sim.ground.profile_types import (
    CANONICAL_GROUND_PARAMETER_IDS,
    GROUND_MATERIAL_PROFILE_SCHEMA_VERSION,
    GROUND_PROFILE_LIBRARY_SCHEMA_VERSION,
    GroundApplicability,
    GroundCalibrationRecord,
    GroundEvidenceKind,
    GroundMaterialParameter,
    GroundMaterialProfile,
    GroundModelUseStatus,
    GroundParameterId,
    GroundProfileEvidence,
    GroundProfileLibrary,
    GroundProfileProvenance,
    GroundProfileRights,
    GroundQualificationGateId,
    GroundQualificationStatus,
)
from shared.python.swing_sim.ground.profile_wire import (
    library_from_json,
    profile_from_json,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _parameters() -> tuple[GroundMaterialParameter, ...]:
    values = (
        0.42,
        0.61,
        0.48,
        0.035,
        1_250_000.0,
        0.72,
        0.012,
        0.18,
        0.31,
        890.0,
        0.24,
    )
    uncertainties = (
        0.02,
        0.03,
        0.03,
        0.004,
        25_000.0,
        0.04,
        0.001,
        0.02,
        0.03,
        20.0,
        0.02,
    )
    lower_bounds = (0.2, 0.4, 0.3, 0.01, 900_000.0, 0.5, 0.005, 0.1, 0.2, 700.0, 0.1)
    upper_bounds = (0.6, 0.8, 0.7, 0.06, 1_600_000.0, 0.9, 0.02, 0.3, 0.5, 1_100.0, 0.4)
    return tuple(
        GroundMaterialParameter(
            cast(GroundParameterId, parameter_id),
            value,
            uncertainty,
            coverage_factor=2.0,
            confidence_level=0.95,
            validity_lower_si=lower_bound,
            validity_upper_si=upper_bound,
            validity_lower_evidence_ids=("evidence-001",),
            validity_upper_evidence_ids=("evidence-001",),
        )
        for parameter_id, value, uncertainty, lower_bound, upper_bound in zip(
            CANONICAL_GROUND_PARAMETER_IDS,
            values,
            uncertainties,
            lower_bounds,
            upper_bounds,
            strict=True,
        )
    )


def _provenance() -> GroundProfileProvenance:
    return GroundProfileProvenance("lab-import", "1.2.0", "dataset-r7", _SHA_A)


def _profile(
    *,
    rights: GroundProfileRights | None = None,
    evidence_parameter_ids: tuple[str, ...] | None = None,
    evidence_kind: GroundEvidenceKind = GroundEvidenceKind.MEASURED_DATASET,
) -> GroundMaterialProfile:
    parameter_ids = cast(
        tuple[GroundParameterId, ...],
        evidence_parameter_ids or tuple(CANONICAL_GROUND_PARAMETER_IDS),
    )
    evidence = GroundProfileEvidence(
        "evidence-001",
        evidence_kind,
        "Synthetic contract fixture; not a production material claim.",
        "urn:fixture:ground-profile:001",
        _SHA_B,
        parameter_ids,
    )
    calibration = GroundCalibrationRecord(
        "calibration-001",
        "bounded least-squares fixture",
        (evidence.evidence_id,),
        cast(tuple[GroundParameterId, ...], tuple(CANONICAL_GROUND_PARAMETER_IDS)),
    )
    return GroundMaterialProfile(
        "fixture-fairway",
        "Synthetic fairway contract fixture",
        "1.0.0",
        _parameters(),
        (evidence,),
        rights or GroundProfileRights("MIT", "Synthetic fixture authors", True, True),
        GroundApplicability(("fairway",), 270.0, 315.0, 0.0, 1.0),
        calibration,
        _provenance(),
    )


def _library() -> GroundProfileLibrary:
    return GroundProfileLibrary(
        "fixture-library",
        "1.0.0",
        (_profile(),),
        _provenance(),
    )


def test_profile_normalizes_exact_ordered_si_parameter_contract() -> None:
    profile = _profile()

    assert profile.schema_version == GROUND_MATERIAL_PROFILE_SCHEMA_VERSION
    assert tuple(item.parameter_id for item in profile.parameters) == tuple(
        CANONICAL_GROUND_PARAMETER_IDS
    )
    assert tuple(item.unit_si for item in profile.parameters) == (
        "1",
        "1",
        "1",
        "1",
        "Pa",
        "1",
        "m",
        "1",
        "1",
        "kg/m^3",
        "1",
    )
    assert profile.parameter_value("firmness_pa") == 1_250_000.0


def test_profile_qualification_and_model_use_are_derived_from_ordered_gates() -> None:
    profile = _profile()

    assert profile.qualification.status is GroundQualificationStatus.QUALIFIED
    assert tuple(gate.gate_id for gate in profile.qualification.gates) == tuple(
        GroundQualificationGateId
    )
    assert all(gate.passed for gate in profile.qualification.gates)
    assert profile.model_use_status is GroundModelUseStatus.CALIBRATED

    restricted = _profile(
        rights=GroundProfileRights(
            "LicenseRef-internal", "Fixture authors", False, False
        )
    )
    assert restricted.qualification.status is GroundQualificationStatus.UNQUALIFIED
    assert not restricted.qualification.gate_passed(
        GroundQualificationGateId.RIGHTS_REUSABLE
    )
    assert restricted.model_use_status is GroundModelUseStatus.CALIBRATED
    incomplete = _profile(evidence_parameter_ids=("normal_restitution",))
    assert incomplete.qualification.status is GroundQualificationStatus.UNQUALIFIED
    assert not incomplete.qualification.gate_passed(
        GroundQualificationGateId.EVIDENCE_TRACEABLE
    )
    assert incomplete.model_use_status is GroundModelUseStatus.ILLUSTRATIVE

    estimated = _profile(evidence_kind=GroundEvidenceKind.ENGINEERING_ESTIMATE)
    assert estimated.qualification.status is GroundQualificationStatus.QUALIFIED
    assert estimated.model_use_status is GroundModelUseStatus.ILLUSTRATIVE


def test_parameter_validity_bounds_are_ordered_sourced_and_value_enclosing() -> None:
    parameter = _parameters()[0]

    assert parameter.validity_lower_si == 0.2
    assert parameter.validity_upper_si == 0.6
    assert parameter.validity_lower_evidence_ids == ("evidence-001",)
    assert parameter.validity_upper_evidence_ids == ("evidence-001",)

    with pytest.raises(ValueError, match="validity bounds must enclose value_si"):
        replace(parameter, validity_lower_si=0.5)
    with pytest.raises(ValueError, match="validity bounds must enclose value_si"):
        replace(parameter, validity_upper_si=0.3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(parameter, validity_upper_si=1.1)
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(
            parameter,
            validity_lower_evidence_ids=("evidence-002", "evidence-001"),
        )


def test_bound_sources_drive_validity_gate_and_model_use_status() -> None:
    profile = _profile()
    first = replace(
        profile.parameters[0],
        validity_lower_evidence_ids=("missing-evidence",),
    )
    with pytest.raises(ValueError, match="unknown validity evidence_id"):
        replace(profile, parameters=(first,) + profile.parameters[1:])

    noncovering_evidence = replace(
        profile.evidence[0],
        evidence_id="evidence-002",
        parameter_ids=cast(tuple[GroundParameterId, ...], ("static_friction",)),
    )
    first = replace(
        profile.parameters[0],
        validity_upper_evidence_ids=("evidence-002",),
    )
    noncovering_source = replace(
        profile,
        parameters=(first,) + profile.parameters[1:],
        evidence=(profile.evidence[0], noncovering_evidence),
    )
    assert noncovering_source.qualification.gate_passed(
        GroundQualificationGateId.EVIDENCE_TRACEABLE
    )
    assert not noncovering_source.qualification.gate_passed(
        GroundQualificationGateId.VALIDITY_BOUNDS_TRACEABLE
    )
    assert noncovering_source.model_use_status is GroundModelUseStatus.ILLUSTRATIVE


def test_profile_rejects_parameter_order_units_bounds_and_numeric_bools() -> None:
    parameters = list(_parameters())
    parameters[0], parameters[1] = parameters[1], parameters[0]
    with pytest.raises(ValueError, match="canonical parameter order"):
        replace(_profile(), parameters=tuple(parameters))

    with pytest.raises(ValueError, match="unit_si"):
        replace(_parameters()[4], unit_si="kPa")
    with pytest.raises(ValueError, match="value_si"):
        replace(_parameters()[0], value_si=True)
    with pytest.raises(ValueError, match="finite"):
        replace(_parameters()[0], value_si=float("nan"))
    with pytest.raises(ValueError, match="kinetic_friction"):
        replace(
            _profile(),
            parameters=tuple(
                replace(item, value_si=0.9, validity_upper_si=1.0)
                if item.parameter_id == "kinetic_friction"
                else item
                for item in _parameters()
            ),
        )


def test_metadata_requires_canonical_unique_references_and_bounded_ranges() -> None:
    with pytest.raises(ValueError, match="canonical parameter order"):
        replace(
            _profile().evidence[0],
            parameter_ids=cast(
                tuple[GroundParameterId, ...],
                ("static_friction", "normal_restitution"),
            ),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        GroundApplicability(("green", "green"), 270.0, 300.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="temperature"):
        replace(_profile().applicability, temperature_min_k=320.0)
    with pytest.raises(ValueError, match="unknown evidence_id"):
        replace(
            _profile(),
            calibration=replace(
                _profile().calibration, evidence_ids=("missing-evidence",)
            ),
        )
    with pytest.raises(ValueError, match="control characters"):
        replace(_profile(), display_name="unsafe\u0001name")


def test_profile_wire_is_exact_canonical_and_digest_stable() -> None:
    profile = _profile()
    canonical = profile.to_json()

    assert profile_from_json(canonical) == profile
    assert profile.canonical_sha256() == profile_from_json(canonical).canonical_sha256()
    assert len(profile.canonical_sha256()) == 64
    assert (
        profile.canonical_sha256()
        == "e5377325a77b2b7a195a51d4715fb4f333bf4a49ac1992e743371102ada01537"  # pragma: allowlist secret  # noqa: E501
    )

    with pytest.raises(ValueError, match="canonical"):
        profile_from_json(canonical.replace(":", ": ", 1))
    with pytest.raises(ValueError, match="duplicate"):
        profile_from_json(
            canonical.replace(
                '"display_name":', '"display_name":"duplicate","display_name":', 1
            )
        )
    with pytest.raises(ValueError, match="non-finite"):
        profile_from_json(canonical.replace('"value_si":0.42', '"value_si":NaN'))

    payload = profile.to_dict()
    payload["unknown"] = "rejected"
    with pytest.raises(ValueError, match="fields"):
        GroundMaterialProfile.from_dict(payload)

    incoherent = profile.to_dict()
    incoherent["model_use_status"] = "illustrative"
    with pytest.raises(ValueError, match="model_use_status must equal the derived"):
        GroundMaterialProfile.from_dict(incoherent)


def test_library_requires_sorted_unique_profiles_and_round_trips() -> None:
    library = _library()

    assert library.schema_version == GROUND_PROFILE_LIBRARY_SCHEMA_VERSION
    assert library_from_json(library.to_json()) == library
    assert (
        library.canonical_sha256()
        == "9e0bb8dff1e4de3e7cde72ebe879105c016344049c540ea0920590ecd4963550"  # pragma: allowlist secret  # noqa: E501
    )
    assert library.profile("fixture-fairway") == _profile()

    second = replace(_profile(), profile_id="fixture-bunker")
    with pytest.raises(ValueError, match="sorted"):
        replace(library, profiles=(library.profiles[0], second))
    with pytest.raises(ValueError, match="unique"):
        replace(library, profiles=(library.profiles[0], library.profiles[0]))


def test_numeric_contract_rejects_precanonical_bounds_and_unsafe_integers() -> None:
    with pytest.raises(ValueError, match="safe range"):
        replace(_parameters()[4], value_si=9_007_199_254_740_992)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(_parameters()[0], value_si=1.000000000004)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(_parameters()[6], value_si=-0.000000000004)


def test_qualification_requires_referenced_coverage_and_applicable_moisture() -> None:
    profile = _profile()
    referenced = replace(
        profile.evidence[0],
        evidence_id="evidence-001",
        parameter_ids=cast(tuple[GroundParameterId, ...], ("normal_restitution",)),
    )
    unreferenced = replace(
        profile.evidence[0],
        evidence_id="evidence-002",
        parameter_ids=cast(
            tuple[GroundParameterId, ...],
            tuple(CANONICAL_GROUND_PARAMETER_IDS[1:]),
        ),
    )
    incomplete_calibration = replace(
        profile,
        evidence=(referenced, unreferenced),
        calibration=replace(profile.calibration, evidence_ids=("evidence-001",)),
    )
    assert incomplete_calibration.qualification.gate_passed(
        GroundQualificationGateId.EVIDENCE_TRACEABLE
    )
    assert not incomplete_calibration.qualification.gate_passed(
        GroundQualificationGateId.CALIBRATION_TRACEABLE
    )

    out_of_range = replace(
        profile,
        applicability=replace(
            profile.applicability,
            moisture_min_fraction=0.5,
            moisture_max_fraction=0.9,
        ),
    )
    assert not out_of_range.qualification.gate_passed(
        GroundQualificationGateId.APPLICABILITY_BOUNDED
    )


@dataclass(frozen=True)
class _SpoofParameter:
    parameter_id: object
    value_si: float
    standard_uncertainty_si: float
    coverage_factor: float
    confidence_level: float
    unit_si: str
    injected: str = "must-not-serialize"


@dataclass(frozen=True)
class _ExtendedProfile(GroundMaterialProfile):
    injected: str = "must-not-serialize"


def test_runtime_rejects_spoofed_nested_records_and_document_subclasses() -> None:
    parameter = _parameters()[0]
    spoof = _SpoofParameter(
        parameter.parameter_id,
        parameter.value_si,
        parameter.standard_uncertainty_si,
        parameter.coverage_factor,
        parameter.confidence_level,
        parameter.unit_si,
    )
    with pytest.raises(TypeError, match="GroundMaterialParameter"):
        replace(
            _profile(),
            parameters=cast(
                tuple[GroundMaterialParameter, ...],
                cast(Any, (spoof,) + _parameters()[1:]),
            ),
        )

    profile = _profile()
    extended = _ExtendedProfile(
        profile.profile_id,
        profile.display_name,
        profile.revision,
        profile.parameters,
        profile.evidence,
        profile.rights,
        profile.applicability,
        profile.calibration,
        profile.provenance,
    )
    with pytest.raises(TypeError, match="exact document type"):
        extended.to_dict()
