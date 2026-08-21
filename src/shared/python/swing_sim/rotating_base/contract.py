"""Fail-closed wire contract for the qualified rotating-base study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_ID = "swing-sim/rotating-base-provider-result"
SCHEMA_VERSION = 1
STUDY_SCHEMA_VERSION = "rotating-base-torso-velocity-study-v1"
STUDY_ID = "registered-rotating-base-two-hand-torso-velocity-grid"
MODEL_TIER = "planar_rotating_base_two_hand_compliant_club"
EXPECTED_UPSTREAM_SOURCE_REVISION = "967c40f54cc03f8cae89cde09268d62771d220fe"
MATCHING_RULES = ("relative_club_rate", "absolute_club_rate")
TORSO_PROFILES = ("accelerate", "constant_rate", "decelerate")
KILLSWITCH_CHANNELS = ("torso", "bilateral_arm", "bilateral_wrist")


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact(name: str, value: object, expected: object) -> None:
    if value != expected:
        raise ValueError(f"{name} must equal {expected!r}")


@dataclass(frozen=True, slots=True)
class RotatingBaseCaseMetrics:
    """Scalar outcome and numerical-gate values for one retained case."""

    initial_club_rate_rad_s: float
    final_torso_rate_rad_s: float
    impact_speed_m_s: float
    clubhead_speed_gain_m_s: float
    contact_work_on_club_j: float
    braking_grip_work_j: float
    force_couple_work_j: float
    negative_along_path_impulse_ns: float
    bilateral_wrist_work_j: float
    total_control_work_j: float
    distal_energy_gain_j: float
    peak_grip_force_n: float
    maximum_constraint_residual_m: float
    maximum_velocity_constraint_residual_m_s: float
    maximum_contact_power_identity_residual_w: float
    work_energy_closure_j: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RotatingBaseCaseMetrics:
        """Parse finite case metrics from one wire row."""
        names = tuple(cls.__dataclass_fields__)
        parsed = {name: _finite(name, value.get(name)) for name in names}
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class RotatingBaseCase:
    """One attempted design row, including adverse exclusions."""

    case_index: int
    torso_profile: str
    matching_rule: str
    initial_torso_rate_rad_s: float
    metrics: RotatingBaseCaseMetrics
    valid: bool
    exclusion_reasons: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> RotatingBaseCase:
        """Parse a retained design row and enforce validity semantics."""
        row = _mapping("case", value)
        profile = _string("torso_profile", row.get("torso_profile"))
        rule = _string("matching_rule", row.get("matching_rule"))
        if profile not in TORSO_PROFILES:
            raise ValueError("torso_profile is outside the registered design")
        if rule not in MATCHING_RULES:
            raise ValueError("matching_rule is outside the registered design")
        valid = row.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("valid must be Boolean")
        reasons = tuple(
            _string("exclusion reason", reason)
            for reason in _sequence("exclusion_reasons", row.get("exclusion_reasons"))
        )
        if valid and reasons:
            raise ValueError("valid case cannot carry exclusion reasons")
        if not valid and not reasons:
            raise ValueError("invalid case must retain an exclusion reason")
        return cls(
            case_index=_integer("case_index", row.get("case_index")),
            torso_profile=profile,
            matching_rule=rule,
            initial_torso_rate_rad_s=_finite(
                "initial_torso_rate_rad_s", row.get("initial_torso_rate_rad_s")
            ),
            metrics=RotatingBaseCaseMetrics.from_mapping(row),
            valid=valid,
            exclusion_reasons=reasons,
        )


@dataclass(frozen=True, slots=True)
class KillswitchChannel:
    """One exact same-state continuation comparison."""

    pre_branch_state_max_abs_difference: float
    delivery_speed_difference_m_s: float
    post_branch_contact_work_difference_j: float

    @classmethod
    def from_mapping(cls, value: object) -> KillswitchChannel:
        """Parse one finite killswitch result."""
        row = _mapping("killswitch channel", value)
        return cls(
            pre_branch_state_max_abs_difference=_finite(
                "pre_branch_state_max_abs_difference",
                row.get("pre_branch_state_max_abs_difference"),
            ),
            delivery_speed_difference_m_s=_finite(
                "delivery_speed_difference_m_s",
                row.get("delivery_speed_difference_m_s"),
            ),
            post_branch_contact_work_difference_j=_finite(
                "post_branch_contact_work_difference_j",
                row.get("post_branch_contact_work_difference_j"),
            ),
        )


@dataclass(frozen=True, slots=True)
class SameStateKillswitch:
    """Registered torso, arm, and wrist same-state comparisons."""

    branch_time_s: float
    pre_branch_state_max_abs_difference: float
    channels: tuple[tuple[str, KillswitchChannel], ...]

    @classmethod
    def from_mapping(cls, value: object) -> SameStateKillswitch:
        """Parse the complete registered channel set without favorable filtering."""
        row = _mapping("same_state_killswitch", value)
        channels = _mapping("killswitch channels", row.get("channels"))
        if tuple(channels) != KILLSWITCH_CHANNELS:
            raise ValueError("killswitch channels must match the registered order")
        parsed = tuple(
            (name, KillswitchChannel.from_mapping(channels[name]))
            for name in KILLSWITCH_CHANNELS
        )
        return cls(
            branch_time_s=_finite("branch_time_s", row.get("branch_time_s")),
            pre_branch_state_max_abs_difference=_finite(
                "pre_branch_state_max_abs_difference",
                row.get("pre_branch_state_max_abs_difference"),
            ),
            channels=parsed,
        )

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the immutable registered channel order."""
        return tuple(name for name, _channel in self.channels)


@dataclass(frozen=True, slots=True)
class RotatingBaseStudy:
    """Qualified study-level design and scientific boundaries."""

    attempted_case_count: int
    valid_case_count: int
    cases: tuple[RotatingBaseCase, ...]
    same_state_killswitch: SameStateKillswitch
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> RotatingBaseStudy:
        """Parse the study and reconcile design counts and claim boundaries."""
        study = _mapping("study", value)
        _exact(
            "study.schema_version", study.get("schema_version"), STUDY_SCHEMA_VERSION
        )
        _exact("study.study_id", study.get("study_id"), STUDY_ID)
        _exact("study.model_tier", study.get("model_tier"), MODEL_TIER)
        rules = _mapping("matching_rules", study.get("matching_rules"))
        if tuple(rules) != MATCHING_RULES:
            raise ValueError("matching_rules must match the registered order")
        cases = tuple(
            RotatingBaseCase.from_mapping(row)
            for row in _sequence("cases", study.get("cases"))
        )
        attempted = _integer("attempted_case_count", study.get("attempted_case_count"))
        valid = _integer("valid_case_count", study.get("valid_case_count"))
        if attempted != len(cases):
            raise ValueError("attempted_case_count does not match retained cases")
        if valid != sum(case.valid for case in cases):
            raise ValueError("valid_case_count does not match retained cases")
        if tuple(case.case_index for case in cases) != tuple(range(attempted)):
            raise ValueError("case_index must be contiguous and ordered")
        _validate_claims(study.get("claims"))
        limitations = tuple(
            _string("limitation", item)
            for item in _sequence("limitations", study.get("limitations"))
        )
        if not limitations:
            raise ValueError("limitations must not be empty")
        return cls(
            attempted_case_count=attempted,
            valid_case_count=valid,
            cases=cases,
            same_state_killswitch=SameStateKillswitch.from_mapping(
                study.get("same_state_killswitch")
            ),
            limitations=limitations,
        )

    @property
    def human_coaching_supported(self) -> bool:
        """Return the fixed scientific promotion boundary."""
        return False


def _validate_claims(value: object) -> None:
    claims = _mapping("claims", value)
    universal = claims.get("universal_high_torso_velocity_strategy")
    if universal not in {"not_supported", "rejected"}:
        raise ValueError("universal torso-velocity strategy must remain unsupported")
    if claims.get("human_coaching_strategy") != "unsupported":
        raise ValueError("human coaching must remain unsupported")


@dataclass(frozen=True, slots=True)
class RotatingBaseProviderResult:
    """Source-pinned rotating-base study result for Python and web consumers."""

    source_revision: str
    study: RotatingBaseStudy

    @classmethod
    def from_mapping(cls, value: object) -> RotatingBaseProviderResult:
        """Parse an exact provider envelope and reject source or schema drift."""
        payload = _mapping("provider result", value)
        _exact("schema_id", payload.get("schema_id"), SCHEMA_ID)
        _exact("schema_version", payload.get("schema_version"), SCHEMA_VERSION)
        _exact(
            "source_revision",
            payload.get("source_revision"),
            EXPECTED_UPSTREAM_SOURCE_REVISION,
        )
        return cls(
            source_revision=EXPECTED_UPSTREAM_SOURCE_REVISION,
            study=RotatingBaseStudy.from_mapping(payload.get("study")),
        )


__all__ = [
    "EXPECTED_UPSTREAM_SOURCE_REVISION",
    "KILLSWITCH_CHANNELS",
    "MATCHING_RULES",
    "MODEL_TIER",
    "RotatingBaseCase",
    "RotatingBaseCaseMetrics",
    "RotatingBaseProviderResult",
    "RotatingBaseStudy",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SameStateKillswitch",
]
