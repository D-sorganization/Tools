"""Versioned study wire schema for impact studies (IA-T6, #5075)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

STUDY_WIRE_FORMAT = "swing_sim.impact_study/1"

#: Pinned schema formats this parser accepts.
SUPPORTED_WIRE_FORMATS = (STUDY_WIRE_FORMAT,)


class EvidenceTier(Enum):
    """Truthful evidence level of a study's results.

    ``ILLUSTRATIVE`` results come from unqualified models or hand-built
    fixtures.  ``VERIFIED`` results carry demonstrated convergence.  Only
    ``MEASURED_VALIDATED`` results are backed by calibrated measurements.
    """

    ILLUSTRATIVE = "illustrative"
    VERIFIED = "verified"
    MEASURED_VALIDATED = "measured_validated"


@runtime_checkable
class EvidenceResolver(Protocol):
    """Protocol for resolving and authenticating qualification evidence."""

    def resolve(self, identifier: str) -> bool:
        """Return True if identifier resolves to authentic evidence."""
        ...


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Provenance:
    """Exact code, data and calibration identities behind a study.

    Attributes:
        code_id: Engine/model identity, e.g. ``module/x@commit``.
        data_ids: Content hashes of raw data records backing the study.
        calibration_id: Calibration record identity, required for the
            measured tier.
        coordinate_frame: Frame label every metric is expressed in.
        qualification_id: Optional qualification record identity.
        paired_control_id: Optional study identifier of paired control case.
    """

    code_id: str
    data_ids: tuple[str, ...] = ()
    calibration_id: str | None = None
    coordinate_frame: str = "lab_right_handed_z_up"
    qualification_id: str | None = None
    paired_control_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code_id", _identifier(self.code_id, "code_id"))
        data_ids = tuple(_identifier(item, "data_id") for item in self.data_ids)
        object.__setattr__(self, "data_ids", data_ids)
        if self.calibration_id is not None:
            object.__setattr__(
                self,
                "calibration_id",
                _identifier(self.calibration_id, "calibration_id"),
            )
        object.__setattr__(
            self,
            "coordinate_frame",
            _identifier(self.coordinate_frame, "coordinate_frame"),
        )
        if self.qualification_id is not None:
            object.__setattr__(
                self,
                "qualification_id",
                _identifier(self.qualification_id, "qualification_id"),
            )
        if self.paired_control_id is not None:
            object.__setattr__(
                self,
                "paired_control_id",
                _identifier(self.paired_control_id, "paired_control_id"),
            )


@dataclass(frozen=True)
class MetricRecord:
    """One named metric with unit, optional uncertainty and evidence tier.

    Attributes:
        name: Stable metric identifier, e.g. ``ball_speed_mps``.
        value: Finite metric value in SI or explicitly labelled units.
        unit: Unit label for the value.
        uncertainty_convention: Convention label, e.g. ``one_sigma``.
        confidence_level: Optional confidence level in (0, 1) for coverage intervals.
    """

    name: str
    value: float
    unit: str
    uncertainty: float | None = None
    uncertainty_convention: str | None = None
    confidence_level: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "metric name"))
        object.__setattr__(self, "value", _finite(self.value, f"{self.name} value"))
        object.__setattr__(self, "unit", _identifier(self.unit, f"{self.name} unit"))
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, f"{self.name} uncertainty")
            if uncertainty < 0.0:
                raise ValueError(f"{self.name} uncertainty must be >= 0")
            object.__setattr__(self, "uncertainty", uncertainty)
            if self.uncertainty_convention is None:
                object.__setattr__(self, "uncertainty_convention", "one_sigma")
            else:
                object.__setattr__(
                    self,
                    "uncertainty_convention",
                    _identifier(
                        self.uncertainty_convention,
                        f"{self.name} uncertainty_convention",
                    ),
                )
            if self.confidence_level is not None:
                confidence = _finite(
                    self.confidence_level, f"{self.name} confidence_level"
                )
                if not (0.0 < confidence < 1.0):
                    raise ValueError(f"{self.name} confidence_level must be in (0, 1)")
                object.__setattr__(self, "confidence_level", confidence)
        else:
            if self.uncertainty_convention is not None:
                raise ValueError(
                    f"{self.name}: uncertainty_convention requires uncertainty"
                )
            if self.confidence_level is not None:
                raise ValueError(f"{self.name}: confidence_level requires uncertainty")


@dataclass(frozen=True)
class CompletenessChecks:
    """Energy and convergence completeness gates for a study."""

    energy_closure_residual_fraction: float | None = None
    convergence_demonstrated: bool = False

    def __post_init__(self) -> None:
        if self.energy_closure_residual_fraction is not None:
            residual = _finite(
                self.energy_closure_residual_fraction, "energy closure residual"
            )
            if residual < 0.0:
                raise ValueError("energy closure residual must be >= 0")
            object.__setattr__(self, "energy_closure_residual_fraction", residual)
        if not isinstance(self.convergence_demonstrated, bool):
            raise ValueError("convergence_demonstrated must be a bool")


@dataclass(frozen=True)
class InvalidCase:
    """One rejected case with the explicit reason it was rejected."""

    case_id: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))


@dataclass(frozen=True)
class ImpactStudyV1:
    """Versioned impact study wire (``swing_sim.impact_study/1``).

    Ball launch, contact, vibration and acoustic metrics travel in
    separate sections so launch never masquerades as sound.  Absent
    acoustics are serialized as an absent key, never as zeros.
    """

    study_id: str
    model_tier: EvidenceTier
    provenance: Provenance
    launch_metrics: tuple[MetricRecord, ...] = ()
    contact_metrics: tuple[MetricRecord, ...] = ()
    vibration_metrics: tuple[MetricRecord, ...] = ()
    acoustic_metrics: tuple[MetricRecord, ...] | None = None
    completeness: CompletenessChecks = field(default_factory=CompletenessChecks)
    invalid_cases: tuple[InvalidCase, ...] = ()
    v1_coupling_report: str | None = None
    schema_format: str = STUDY_WIRE_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _identifier(self.study_id, "study_id"))
        if not isinstance(self.model_tier, EvidenceTier):
            raise ValueError("model_tier must be an EvidenceTier")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")
        if self.schema_format not in SUPPORTED_WIRE_FORMATS:
            raise ValueError(f"unsupported schema_format {self.schema_format!r}")
        for name in (
            "launch_metrics",
            "contact_metrics",
            "vibration_metrics",
        ):
            section = getattr(self, name)
            if not isinstance(section, tuple) or not all(
                isinstance(record, MetricRecord) for record in section
            ):
                raise ValueError(f"{name} must be a tuple of MetricRecord")
        if self.acoustic_metrics is not None:
            if not (
                isinstance(self.acoustic_metrics, tuple)
                and all(
                    isinstance(record, MetricRecord) for record in self.acoustic_metrics
                )
            ):
                raise ValueError(
                    "acoustic_metrics must be a tuple of MetricRecord or None"
                )
            if len(self.acoustic_metrics) == 0:
                raise ValueError(
                    "acoustic_metrics must be None or a non-empty tuple of "
                    "MetricRecord; use None when acoustics are absent"
                )
        if not isinstance(self.completeness, CompletenessChecks):
            raise ValueError("completeness must be CompletenessChecks")
        if not isinstance(self.invalid_cases, tuple) or not all(
            isinstance(case, InvalidCase) for case in self.invalid_cases
        ):
            raise ValueError("invalid_cases must be a tuple of InvalidCase")
        if self.v1_coupling_report is not None:
            if not isinstance(self.v1_coupling_report, str):
                raise ValueError("v1_coupling_report must be a str or None")
            try:
                legacy_doc = json.loads(self.v1_coupling_report)
            except Exception as err:
                raise ValueError(
                    f"v1_coupling_report must be valid JSON: {err}"
                ) from err
            if not isinstance(legacy_doc, Mapping):
                raise ValueError("v1_coupling_report JSON must be an object")
            fmt = legacy_doc.get("format") or legacy_doc.get("report_format")
            if fmt != "golf_club.impact_coupling_report/1":
                raise ValueError(
                    "v1_coupling_report format must be "
                    f"'golf_club.impact_coupling_report/1', got {fmt!r}"
                )
        if (
            self.provenance.paired_control_id is not None
            and self.provenance.paired_control_id == self.study_id
        ):
            raise ValueError(
                "paired_control_id cannot equal study_id "
                "(a study cannot be its own paired control)"
            )
        self._enforce_tier_truthfulness()

    def _enforce_tier_truthfulness(self) -> None:
        if self.model_tier is EvidenceTier.VERIFIED and (
            not self.completeness.convergence_demonstrated
        ):
            raise ValueError("VERIFIED tier requires demonstrated convergence")
        if self.model_tier is not EvidenceTier.MEASURED_VALIDATED:
            return
        if self.provenance.calibration_id is None:
            raise ValueError("MEASURED_VALIDATED tier requires a calibration_id")
        if not self.provenance.data_ids:
            raise ValueError("MEASURED_VALIDATED tier requires data ids")


_ALLOWED_STUDY_KEYS = {
    "schema_format",
    "study_id",
    "model_tier",
    "provenance",
    "launch_metrics",
    "contact_metrics",
    "vibration_metrics",
    "acoustic_metrics",
    "completeness",
    "invalid_cases",
    "v1_coupling_report",
}

_ALLOWED_PROVENANCE_KEYS = {
    "code_id",
    "data_ids",
    "calibration_id",
    "coordinate_frame",
    "qualification_id",
    "paired_control_id",
}

_ALLOWED_COMPLETENESS_KEYS = {
    "energy_closure_residual_fraction",
    "convergence_demonstrated",
}

_ALLOWED_METRIC_KEYS = {
    "name",
    "value",
    "unit",
    "uncertainty",
    "uncertainty_convention",
    "confidence_level",
}

_ALLOWED_INVALID_CASE_KEYS = {
    "case_id",
    "reason",
}


def _check_allowed_keys(
    actual: Mapping[str, Any], allowed: set[str], scope: str
) -> None:
    extra = set(actual.keys()) - allowed
    if extra:
        sorted_extra = sorted(extra)
        raise ValueError(f"unknown field(s) in {scope}: {sorted_extra}")


def _is_resolvable(resolver: object, identifier: str) -> bool:
    if hasattr(resolver, "resolve") and callable(resolver.resolve):
        return bool(resolver.resolve(identifier))
    if callable(resolver):
        return bool(resolver(identifier))
    if isinstance(resolver, Mapping):
        return identifier in resolver
    if isinstance(resolver, (set, frozenset, list, tuple)):
        return identifier in resolver
    raise TypeError(f"unsupported resolver type: {type(resolver).__name__}")


def verify_study_qualification(
    study: ImpactStudyV1,
    *,
    resolver: (
        EvidenceResolver | Callable[[str], bool] | Mapping[str, Any] | None
    ) = None,
) -> None:
    """Verify that a study's evidence tier is truthful and authentic.

    For ``MEASURED_VALIDATED`` tier:
    - An explicit evidence resolver is required; metadata labels alone do not
      authenticate evidence.
    - Calibration ID must resolve.
    - Every data ID must resolve.
    - If qualification_id is declared, it must resolve.
    - If paired_control_id is declared, it must resolve and not match study_id.

    Raises:
        TypeError: If study is not an ImpactStudyV1.
        ValueError: If evidence cannot be resolved or tier requirements are violated.
    """
    if not isinstance(study, ImpactStudyV1):
        raise TypeError("study must be ImpactStudyV1")
    if study.model_tier is EvidenceTier.VERIFIED:
        if not study.completeness.convergence_demonstrated:
            raise ValueError("VERIFIED tier requires demonstrated convergence")
        if resolver is not None:
            if study.provenance.qualification_id and not _is_resolvable(
                resolver, study.provenance.qualification_id
            ):
                raise ValueError(
                    "unresolvable qualification_id: "
                    f"{study.provenance.qualification_id!r}"
                )
            if study.provenance.paired_control_id and not _is_resolvable(
                resolver, study.provenance.paired_control_id
            ):
                raise ValueError(
                    "unresolvable paired_control_id: "
                    f"{study.provenance.paired_control_id!r}"
                )
        return
    if study.model_tier is not EvidenceTier.MEASURED_VALIDATED:
        return
    if resolver is None:
        raise ValueError(
            "MEASURED_VALIDATED tier requires an evidence resolver; "
            "metadata labels do not authenticate evidence"
        )
    if study.provenance.calibration_id is None:
        raise ValueError("MEASURED_VALIDATED tier requires a calibration_id")
    if not _is_resolvable(resolver, study.provenance.calibration_id):
        raise ValueError(
            f"unresolvable calibration_id: {study.provenance.calibration_id!r}"
        )
    if not study.provenance.data_ids:
        raise ValueError("MEASURED_VALIDATED tier requires data ids")
    for data_id in study.provenance.data_ids:
        if not _is_resolvable(resolver, data_id):
            raise ValueError(f"unresolvable data_id: {data_id!r}")
    if study.provenance.qualification_id and not _is_resolvable(
        resolver, study.provenance.qualification_id
    ):
        raise ValueError(
            f"unresolvable qualification_id: {study.provenance.qualification_id!r}"
        )
    if study.provenance.paired_control_id:
        if study.provenance.paired_control_id == study.study_id:
            raise ValueError("paired_control_id cannot equal study_id")
        if not _is_resolvable(resolver, study.provenance.paired_control_id):
            raise ValueError(
                "unresolvable paired_control_id: "
                f"{study.provenance.paired_control_id!r}"
            )


def _metrics_payload(
    records: tuple[MetricRecord, ...] | None,
) -> list[dict[str, Any]] | None:
    if records is None:
        return None
    return [
        {
            "name": r.name,
            "value": r.value,
            "unit": r.unit,
            "uncertainty": r.uncertainty,
            "uncertainty_convention": r.uncertainty_convention,
            "confidence_level": r.confidence_level,
        }
        for r in records
    ]


def _metrics_from_payload(
    payload: object, name: str
) -> tuple[MetricRecord, ...] | None:
    if payload is None:
        return None
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list or absent")
    if name == "acoustic_metrics" and len(payload) == 0:
        raise ValueError("acoustic_metrics in wire must be absent or non-empty")
    records: list[MetricRecord] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"metric in {name} must be a JSON object")
        _check_allowed_keys(item, _ALLOWED_METRIC_KEYS, f"metric payload in {name}")
        records.append(
            MetricRecord(
                name=item["name"],
                value=item["value"],
                unit=item["unit"],
                uncertainty=item.get("uncertainty"),
                uncertainty_convention=item.get("uncertainty_convention"),
                confidence_level=item.get("confidence_level"),
            )
        )
    return tuple(records)


def serialize_study_wire(study: ImpactStudyV1) -> str:
    """Serialize ``study`` to a deterministic JSON document.

    Sorted keys, compact separators and ``allow_nan=False``: identical
    inputs produce byte-identical wires.  Unavailable acoustics are
    serialized as an absent ``acoustic_metrics`` key.
    """
    if not isinstance(study, ImpactStudyV1):
        raise TypeError("study must be ImpactStudyV1")
    payload: dict[str, Any] = {
        "schema_format": study.schema_format,
        "study_id": study.study_id,
        "model_tier": study.model_tier.value,
        "provenance": {
            "code_id": study.provenance.code_id,
            "data_ids": list(study.provenance.data_ids),
            "calibration_id": study.provenance.calibration_id,
            "coordinate_frame": study.provenance.coordinate_frame,
            "qualification_id": study.provenance.qualification_id,
            "paired_control_id": study.provenance.paired_control_id,
        },
        "launch_metrics": _metrics_payload(study.launch_metrics),
        "contact_metrics": _metrics_payload(study.contact_metrics),
        "vibration_metrics": _metrics_payload(study.vibration_metrics),
        "completeness": {
            "energy_closure_residual_fraction": (
                study.completeness.energy_closure_residual_fraction
            ),
            "convergence_demonstrated": (study.completeness.convergence_demonstrated),
        },
        "invalid_cases": [
            {"case_id": case.case_id, "reason": case.reason}
            for case in study.invalid_cases
        ],
        "v1_coupling_report": study.v1_coupling_report,
    }
    acoustics = _metrics_payload(study.acoustic_metrics)
    if acoustics is not None:
        payload["acoustic_metrics"] = acoustics
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def parse_study_wire(document: str) -> ImpactStudyV1:
    """Parse a study wire document back into an :class:`ImpactStudyV1`."""
    if not isinstance(document, str):
        raise TypeError("document must be a str")
    payload = json.loads(document)
    if not isinstance(payload, Mapping):
        raise ValueError("study wire must be a JSON object")
    _check_allowed_keys(payload, _ALLOWED_STUDY_KEYS, "study wire")
    if payload.get("schema_format") not in SUPPORTED_WIRE_FORMATS:
        raise ValueError(
            f"unsupported study wire format {payload.get('schema_format')!r}"
        )
    provenance_payload = payload["provenance"]
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("provenance must be a JSON object")
    _check_allowed_keys(
        provenance_payload, _ALLOWED_PROVENANCE_KEYS, "provenance payload"
    )

    completeness_payload = payload["completeness"]
    if not isinstance(completeness_payload, Mapping):
        raise ValueError("completeness must be a JSON object")
    _check_allowed_keys(
        completeness_payload, _ALLOWED_COMPLETENESS_KEYS, "completeness payload"
    )

    invalid_cases_payload = payload.get("invalid_cases", [])
    if not isinstance(invalid_cases_payload, list):
        raise ValueError("invalid_cases must be a list")
    for item in invalid_cases_payload:
        if not isinstance(item, Mapping):
            raise ValueError("invalid_case must be a JSON object")
        _check_allowed_keys(item, _ALLOWED_INVALID_CASE_KEYS, "invalid case payload")

    v1_coupling_report = payload.get("v1_coupling_report")
    if v1_coupling_report is not None and not isinstance(v1_coupling_report, str):
        raise ValueError("v1_coupling_report in study wire must be a str or null")

    return ImpactStudyV1(
        study_id=payload["study_id"],
        model_tier=EvidenceTier(payload["model_tier"]),
        provenance=Provenance(
            code_id=provenance_payload["code_id"],
            data_ids=tuple(provenance_payload["data_ids"]),
            calibration_id=provenance_payload.get("calibration_id"),
            coordinate_frame=provenance_payload.get(
                "coordinate_frame", "lab_right_handed_z_up"
            ),
            qualification_id=provenance_payload.get("qualification_id"),
            paired_control_id=provenance_payload.get("paired_control_id"),
        ),
        launch_metrics=_metrics_from_payload(
            payload.get("launch_metrics"), "launch_metrics"
        )
        or (),
        contact_metrics=_metrics_from_payload(
            payload.get("contact_metrics"), "contact_metrics"
        )
        or (),
        vibration_metrics=_metrics_from_payload(
            payload.get("vibration_metrics"), "vibration_metrics"
        )
        or (),
        acoustic_metrics=_metrics_from_payload(
            payload.get("acoustic_metrics"), "acoustic_metrics"
        ),
        completeness=CompletenessChecks(
            energy_closure_residual_fraction=(
                completeness_payload.get("energy_closure_residual_fraction")
            ),
            convergence_demonstrated=completeness_payload.get(
                "convergence_demonstrated", False
            ),
        ),
        invalid_cases=tuple(
            InvalidCase(case_id=item["case_id"], reason=item["reason"])
            for item in invalid_cases_payload
        ),
        v1_coupling_report=v1_coupling_report,
    )
