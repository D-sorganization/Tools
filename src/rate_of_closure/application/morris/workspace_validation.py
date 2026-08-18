"""Strict parsing and cross-invariants for Morris workspace v1."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict

from rate_of_closure.application._workspace_validation import (
    exact_mapping,
    freeze_object,
    thaw_json,
)
from shared.python.swing_sim.variation.spec import variable_registry

from ._response_types import MorrisSource
from .contracts import MorrisAuthorityRequest, parse_morris_request
from .request_document import CANONICAL_MORRIS_FACTOR_KEYS, spec_id_for_key
from .response_contract import parse_morris_job
from .workspace_types import (
    MorrisCompletedEvidence,
    MorrisWorkspace,
    MorrisWorkspaceFactorDraft,
    MorrisWorkspaceSetup,
)

MORRIS_WORKSPACE_SCHEMA_ID = "rate-of-closure/morris-workspace"
MORRIS_WORKSPACE_SCHEMA_VERSION = 1
MORRIS_WORKSPACE_EXPORT_SCOPE = "authority-base-and-morris-controls-only"
INVALID_BOUNDS_MESSAGE = "Bounds must be finite numbers with lower < upper."
MAX_BOUND_TEXT = 128
MAX_VALIDATION_TEXT = 256
MAX_BOUND_MAGNITUDE = 1_000_000_000.0
_DECIMAL_NUMBER = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)

_ROOT_FIELDS = frozenset({"schema_id", "schema_version", "setup", "completed_evidence"})
_SETUP_FIELDS = frozenset(
    {
        "export_scope",
        "base",
        "factor_drafts",
        "trajectories",
        "levels",
        "seed",
        "minimum_effects",
        "worker_count",
    }
)
_DRAFT_FIELDS = frozenset(
    {"variable_key", "enabled", "lower", "upper", "validation_error"}
)
_EVIDENCE_FIELDS = frozenset({"request", "job"})
_MAX_REPORT_ESTIMATES = 1_000
_MAX_REPORT_ASSUMPTIONS = 64


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


def _raw_text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
    ):
        raise ValueError(f"{name} must be a bounded text value")
    return value


def _numeric_bounds(lower: str, upper: str) -> tuple[float, float] | None:
    if (
        _DECIMAL_NUMBER.fullmatch(lower) is None
        or _DECIMAL_NUMBER.fullmatch(upper) is None
    ):
        return None
    try:
        parsed = (float(lower), float(upper))
    except ValueError:
        return None
    if (
        not all(math.isfinite(value) for value in parsed)
        or any(abs(value) > MAX_BOUND_MAGNITUDE for value in parsed)
        or parsed[0] >= parsed[1]
    ):
        return None
    return parsed


def _parse_draft(value: object, expected_key: str) -> MorrisWorkspaceFactorDraft:
    item = exact_mapping(value, _DRAFT_FIELDS, "Morris workspace factor draft")
    if item["variable_key"] != expected_key:
        raise ValueError("factor drafts must retain canonical keys and order")
    if type(item["enabled"]) is not bool:
        raise TypeError("factor draft enabled must be boolean")
    lower = _raw_text(item["lower"], "factor lower", MAX_BOUND_TEXT)
    upper = _raw_text(item["upper"], "factor upper", MAX_BOUND_TEXT)
    bounds = _numeric_bounds(lower, upper)
    error = item["validation_error"]
    if error is not None:
        error = _raw_text(error, "factor validation_error", MAX_VALIDATION_TEXT)
    expected_error = None if bounds is not None else INVALID_BOUNDS_MESSAGE
    if error != expected_error:
        raise ValueError("factor validation state does not match its raw bounds")
    if item["enabled"] and (bounds is None or expected_error is not None):
        raise ValueError("enabled factor bounds must be valid")
    return MorrisWorkspaceFactorDraft(
        expected_key, item["enabled"], lower, upper, error
    )


def _parse_setup(value: object) -> MorrisWorkspaceSetup:
    item = exact_mapping(value, _SETUP_FIELDS, "Morris workspace setup")
    if item["export_scope"] != MORRIS_WORKSPACE_EXPORT_SCOPE:
        raise ValueError("unsupported Morris workspace export scope")
    drafts = item["factor_drafts"]
    if not isinstance(drafts, list) or len(drafts) != len(CANONICAL_MORRIS_FACTOR_KEYS):
        raise ValueError("factor drafts must contain the complete canonical registry")
    base = freeze_object(item["base"], "Morris workspace base")
    parsed = tuple(
        _parse_draft(draft, key)
        for draft, key in zip(drafts, CANONICAL_MORRIS_FACTOR_KEYS, strict=True)
    )
    if base.get("support_mode") != "tee" and parsed[-1].enabled:
        raise ValueError("tee-height factor is unavailable for ground support")
    trajectories = _integer(item["trajectories"], "trajectories", 2, 5_000)
    levels = _integer(item["levels"], "levels", 4, 10_000)
    if levels % 2:
        raise ValueError("levels must be even")
    return MorrisWorkspaceSetup(
        MORRIS_WORKSPACE_EXPORT_SCOPE,
        base,
        parsed,
        trajectories,
        levels,
        _integer(item["seed"], "seed", 0, 2**31 - 1),
        _integer(item["minimum_effects"], "minimum_effects", 2, trajectories),
        _integer(item["worker_count"], "worker_count", 1, 32),
    )


def request_from_setup(
    setup: MorrisWorkspaceSetup, request_id: str
) -> MorrisAuthorityRequest:
    """Build the exact authority request implied by one validated setup."""
    registry = variable_registry()
    factors: list[dict[str, object]] = []
    for draft in setup.factor_drafts:
        if not draft.enabled:
            continue
        bounds = _numeric_bounds(draft.lower, draft.upper)
        assert bounds is not None
        factors.append(
            {
                "spec_id": spec_id_for_key(draft.variable_key),
                "variable_key": draft.variable_key,
                "lower": bounds[0],
                "upper": bounds[1],
                "unit": registry[draft.variable_key].unit,
            }
        )
    return parse_morris_request(
        {
            "schema_id": "rate-of-closure/morris-request",
            "schema_version": 1,
            "request_id": request_id,
            "base": thaw_json(setup.base),
            "factors": factors,
            "trajectories": setup.trajectories,
            "levels": setup.levels,
            "seed": setup.seed,
            "minimum_effects": setup.minimum_effects,
            "worker_count": setup.worker_count,
        }
    )


def base_config_from_setup(setup: MorrisWorkspaceSetup):  # type: ignore[no-untyped-def]
    """Validate and reconstruct the base without requiring a runnable factor."""
    from .contracts import _parse_base

    return _parse_base(thaw_json(setup.base)).simulation_config()


def _source_identity(source: MorrisSource) -> tuple[object, ...]:
    return (
        source.spec_id,
        source.variable_key,
        source.unit,
        source.bounds,
        source.time_window_s,
        source.point_ids,
    )


def _validate_evidence(
    setup: MorrisWorkspaceSetup, value: object
) -> MorrisCompletedEvidence:
    item = exact_mapping(value, _EVIDENCE_FIELDS, "Morris completed evidence")
    job_value = item["job"]
    if isinstance(job_value, Mapping):
        report_value = job_value.get("report")
        if isinstance(report_value, Mapping):
            estimates = report_value.get("estimates")
            assumptions = report_value.get("assumptions")
            if isinstance(estimates, list) and len(estimates) > _MAX_REPORT_ESTIMATES:
                raise ValueError("Morris report exceeds the estimate count limit")
            if (
                isinstance(assumptions, list)
                and len(assumptions) > _MAX_REPORT_ASSUMPTIONS
            ):
                raise ValueError("Morris report exceeds the assumption count limit")
    request = parse_morris_request(item["request"])
    expected = request_from_setup(setup, request.request_id)
    if request != expected:
        if request.base.values != expected.base.values:
            raise ValueError("completed evidence base differs from setup base")
        raise ValueError("completed evidence request design differs from setup design")
    job = parse_morris_job(item["job"])
    if job.status != "completed" or job.report is None:
        raise ValueError("workspace evidence must be completed")
    if job.request_id != request.request_id:
        raise ValueError("completed evidence request_id values disagree")
    if job.total_samples != request.total_samples:
        raise ValueError("completed evidence sample totals disagree")
    report = job.report
    design = (report.trajectories, report.levels, report.seed, report.total_samples)
    expected_design = (
        request.trajectories,
        request.levels,
        request.seed,
        request.total_samples,
    )
    if design != expected_design:
        raise ValueError("completed evidence report design differs from request")
    expected_sources = {
        (
            factor.spec_id,
            factor.variable_key,
            factor.unit,
            (factor.lower, factor.upper),
            None,
            (),
        )
        for factor in request.factors
    }
    actual_sources = {
        _source_identity(estimate.source) for estimate in report.estimates
    }
    if actual_sources != expected_sources:
        raise ValueError(
            "completed evidence report sources differ from request factors"
        )
    return MorrisCompletedEvidence(request, job)


def parse_morris_workspace(value: object) -> MorrisWorkspace:
    """Parse a strict v1 workspace with no ambient credentials or runtime state."""
    item = exact_mapping(value, _ROOT_FIELDS, "Morris workspace")
    if item["schema_id"] != MORRIS_WORKSPACE_SCHEMA_ID:
        raise ValueError("unsupported Morris workspace schema ID")
    if item["schema_version"] != MORRIS_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("unsupported Morris workspace schema version")
    setup = _parse_setup(item["setup"])
    evidence = (
        None
        if item["completed_evidence"] is None
        else _validate_evidence(setup, item["completed_evidence"])
    )
    return MorrisWorkspace(MORRIS_WORKSPACE_SCHEMA_ID, 1, setup, evidence)


def workspace_factor_dict(draft: MorrisWorkspaceFactorDraft) -> dict[str, object]:
    """Return one factor draft's exact stable wire representation."""
    return asdict(draft)


__all__ = [
    "INVALID_BOUNDS_MESSAGE",
    "MAX_BOUND_MAGNITUDE",
    "MORRIS_WORKSPACE_EXPORT_SCOPE",
    "MORRIS_WORKSPACE_SCHEMA_ID",
    "MORRIS_WORKSPACE_SCHEMA_VERSION",
    "parse_morris_workspace",
    "base_config_from_setup",
    "request_from_setup",
    "workspace_factor_dict",
]
