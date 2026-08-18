"""Strict immutable client-side Morris authority response contracts."""

from __future__ import annotations

import math

from rate_of_closure.application._workspace_validation import exact_mapping, stable_id

from ._metric_validation import validate_finite_metrics
from ._response_constants import (
    ADEQUACIES as _ADEQUACIES,
)
from ._response_constants import (
    API_PREFIX as _API_PREFIX,
)
from ._response_constants import (
    AVAILABILITIES as _AVAILABILITIES,
)
from ._response_constants import (
    CAPABILITY_FIELDS as _CAPABILITY_FIELDS,
)
from ._response_constants import (
    CAPABILITY_SCHEMA_ID as _CAPABILITY_SCHEMA_ID,
)
from ._response_constants import (
    DENOMINATOR_FIELDS as _DENOMINATOR_FIELDS,
)
from ._response_constants import (
    DESIGN_FIELDS as _DESIGN_FIELDS,
)
from ._response_constants import (
    EFFECT_FIELDS as _EFFECT_FIELDS,
)
from ._response_constants import (
    ESTIMATE_FIELDS as _ESTIMATE_FIELDS,
)
from ._response_constants import (
    JOB_FIELDS as _JOB_FIELDS,
)
from ._response_constants import (
    REPORT_FIELDS as _REPORT_FIELDS,
)
from ._response_constants import (
    REPORT_SCHEMA_ID as _REPORT_SCHEMA_ID,
)
from ._response_constants import (
    SOURCE_FIELDS as _SOURCE_FIELDS,
)
from ._response_constants import (
    TARGET_FIELDS as _TARGET_FIELDS,
)
from ._response_constants import (
    TARGET_KINDS as _TARGET_KINDS,
)
from ._response_types import (
    MorrisCapability,
    MorrisDenominator,
    MorrisEffects,
    MorrisResponseEstimate,
    MorrisResponseJob,
    MorrisResponseReport,
    MorrisSource,
    MorrisTarget,
)
from .contracts import MORRIS_JOB_SCHEMA_ID, MORRIS_REQUEST_SCHEMA_ID


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
        )
    ):
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nullable_finite(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def parse_morris_capability(value: object) -> MorrisCapability:
    """Parse the host's exact six-field capability document."""
    item = exact_mapping(value, _CAPABILITY_FIELDS, "Morris capability")
    expected = (
        _CAPABILITY_SCHEMA_ID,
        1,
        _API_PREFIX,
        MORRIS_REQUEST_SCHEMA_ID,
        MORRIS_JOB_SCHEMA_ID,
    )
    actual = (
        item["schema_id"],
        item["schema_version"],
        item["api_prefix"],
        item["request_schema_id"],
        item["job_schema_id"],
    )
    if actual != expected:
        raise ValueError("unsupported Morris capability contract")
    if type(item["available"]) is not bool:
        raise TypeError("capability available must be boolean")
    return MorrisCapability(item["available"], *expected[2:])


def _parse_source(value: object) -> MorrisSource:
    item = exact_mapping(value, _SOURCE_FIELDS, "Morris source")
    bounds = item["bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Morris source bounds must contain two values")
    lower = _nullable_finite(bounds[0], "source lower")
    upper = _nullable_finite(bounds[1], "source upper")
    if lower is None or upper is None or lower >= upper:
        raise ValueError("Morris source bounds must satisfy finite lower < upper")
    window = item["time_window_s"]
    time_window: tuple[float, float] | None = None
    if window is not None:
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("source time_window_s must contain two values")
        start = _nullable_finite(window[0], "source window start")
        end = _nullable_finite(window[1], "source window end")
        if start is None or end is None or start >= end:
            raise ValueError("source time window must satisfy start < end")
        time_window = (start, end)
    point_ids = item["point_ids"]
    if not isinstance(point_ids, list):
        raise TypeError("source point_ids must be an array")
    points = tuple(stable_id(point, "source point_id") for point in point_ids)
    if len(set(points)) != len(points):
        raise ValueError("source point_ids must be unique")
    return MorrisSource(
        stable_id(item["spec_id"], "source spec_id"),
        _text(item["variable_key"], "source variable_key"),
        _text(item["unit"], "source unit"),
        (lower, upper),
        time_window,
        points,
    )


def _parse_target(value: object) -> MorrisTarget:
    item = exact_mapping(value, _TARGET_FIELDS, "Morris target")
    target = MorrisTarget(
        stable_id(item["name"], "target name"),
        _text(item["unit"], "target unit"),
        _text(item["kind"], "target kind"),
        _nullable_finite(item["time_s"], "target time_s"),
        _optional_text(item["point_id"], "target point_id"),
        _optional_text(item["coordinate_frame"], "target coordinate_frame"),
    )
    if target.kind not in _TARGET_KINDS:
        raise ValueError("unsupported Morris target kind")
    if target.kind == "state-point" and (
        target.point_id is None or target.coordinate_frame is None
    ):
        raise ValueError("state-point target requires point and coordinate frame")
    return target


def _parse_effects(value: object) -> MorrisEffects:
    item = exact_mapping(value, _EFFECT_FIELDS, "Morris effects")
    effects = MorrisEffects(
        *(
            _nullable_finite(item[name], name)
            for name in ("mu", "mu_star", "mu_star_standard_error", "sigma")
        )
    )
    optional_values = (
        effects.mu,
        effects.mu_star,
        effects.mu_star_standard_error,
        effects.sigma,
    )
    if any(value is None for value in optional_values):
        if any(value is not None for value in optional_values):
            raise ValueError("Morris effects must be wholly available or unavailable")
        return effects
    assert effects.mu is not None and effects.mu_star is not None
    assert effects.mu_star_standard_error is not None and effects.sigma is not None
    if (
        effects.mu_star < abs(effects.mu)
        or effects.mu_star_standard_error < 0.0
        or effects.sigma < 0.0
    ):
        raise ValueError("Morris effect magnitudes are inconsistent")
    return effects


def _parse_denominator(value: object, trajectories: int) -> MorrisDenominator:
    item = exact_mapping(value, _DENOMINATOR_FIELDS, "Morris denominator")
    result = MorrisDenominator(
        *(
            _integer(item[name], name)
            for name in (
                "total_pairs",
                "valid_pairs",
                "typed_no_impact_pairs",
                "no_impact_unavailable_pairs",
                "failed_pairs",
                "nonfinite_pairs",
            )
        )
    )
    exclusive = (
        result.valid_pairs
        + result.no_impact_unavailable_pairs
        + result.failed_pairs
        + result.nonfinite_pairs
    )
    if result.total_pairs != trajectories or exclusive != trajectories:
        raise ValueError("Morris exclusive denominator categories are inconsistent")
    if result.typed_no_impact_pairs > result.total_pairs:
        raise ValueError("typed no-impact pairs exceed the total")
    if (
        result.no_impact_unavailable_pairs > result.typed_no_impact_pairs
        or result.typed_no_impact_pairs > result.total_pairs - result.failed_pairs
    ):
        raise ValueError("Morris typed no-impact denominator invariant failed")
    return result


def _parse_estimate(value: object, trajectories: int) -> MorrisResponseEstimate:
    item = exact_mapping(value, _ESTIMATE_FIELDS, "Morris estimate")
    availability = _text(item["availability"], "availability")
    adequacy = _text(item["sample_adequacy"], "sample_adequacy")
    if availability not in _AVAILABILITIES or adequacy not in _ADEQUACIES:
        raise ValueError("unsupported Morris scientific state")
    effects = _parse_effects(item["effects"])
    values = (
        effects.mu,
        effects.mu_star,
        effects.mu_star_standard_error,
        effects.sigma,
    )
    if availability == "insufficient-data" and any(
        value is not None for value in values
    ):
        raise ValueError("insufficient-data effects must be null")
    if availability == "constant-output" and any(value != 0.0 for value in values):
        raise ValueError("constant-output effects must be zero")
    if availability == "available" and any(value is None for value in values):
        raise ValueError("available effects must be finite")
    unavailable = availability == "insufficient-data"
    if unavailable != (adequacy == "insufficient"):
        raise ValueError("availability and sample adequacy disagree")
    denominator = _parse_denominator(item["denominator"], trajectories)
    if adequacy == "adequate" and denominator.valid_pairs < 10:
        raise ValueError("adequate Morris estimate requires ten valid pairs")
    if adequacy == "limited" and not 2 <= denominator.valid_pairs < 10:
        raise ValueError("limited Morris estimate requires two through nine pairs")
    validate_finite_metrics(effects, availability, denominator.valid_pairs)
    return MorrisResponseEstimate(
        _parse_source(item["source"]),
        _parse_target(item["target"]),
        effects,
        availability,
        adequacy,
        denominator,
    )


def _parse_report(value: object) -> MorrisResponseReport:
    item = exact_mapping(value, _REPORT_FIELDS, "Morris report")
    if (
        item["schema_id"] != _REPORT_SCHEMA_ID
        or item["schema_version"] != 1
        or item["method"] != "morris-elementary-effects"
    ):
        raise ValueError("unsupported Morris report schema or method")
    design = exact_mapping(item["design"], _DESIGN_FIELDS, "Morris design")
    trajectories = _integer(design["trajectories"], "trajectories")
    levels = _integer(design["levels"], "levels")
    seed = _integer(design["seed"], "seed")
    total_samples = _integer(design["total_samples"], "total_samples")
    step = _nullable_finite(design["normalized_step"], "normalized_step")
    if trajectories < 1 or levels < 4 or levels % 2 or step is None:
        raise ValueError("Morris design fields are invalid")
    expected_step = levels / (2.0 * (levels - 1))
    if not math.isclose(step, expected_step, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Morris normalized step does not match levels")
    estimates_value = item["estimates"]
    if not isinstance(estimates_value, list) or not estimates_value:
        raise TypeError("Morris estimates must be a nonempty array")
    estimates = tuple(_parse_estimate(value, trajectories) for value in estimates_value)
    pairs = tuple((value.source.spec_id, value.target.name) for value in estimates)
    sources = {value.source.spec_id for value in estimates}
    targets = {value.target.name for value in estimates}
    if len(set(pairs)) != len(pairs) or len(pairs) != len(sources) * len(targets):
        raise ValueError("Morris estimates must form a unique complete matrix")
    for source_id in sources:
        source_variants = {
            value.source for value in estimates if value.source.spec_id == source_id
        }
        if len(source_variants) != 1:
            raise ValueError("source provenance changes within report")
    for target_name in targets:
        target_variants = {
            value.target for value in estimates if value.target.name == target_name
        }
        if len(target_variants) != 1:
            raise ValueError("target provenance changes within report")
    if total_samples != trajectories * (len(sources) + 1):
        raise ValueError("Morris design total_samples is inconsistent")
    assumptions = item["assumptions"]
    if not isinstance(assumptions, list) or not assumptions:
        raise TypeError("Morris assumptions must be a nonempty array")
    parsed_assumptions = tuple(_text(value, "assumption") for value in assumptions)
    if len(set(parsed_assumptions)) != len(parsed_assumptions):
        raise ValueError("Morris assumptions must be unique")
    return MorrisResponseReport(
        trajectories,
        levels,
        seed,
        total_samples,
        step,
        parsed_assumptions,
        _text(item["interaction_caveat"], "interaction_caveat"),
        estimates,
    )


def parse_morris_job(value: object) -> MorrisResponseJob:
    """Parse an exact job envelope and its optional canonical report."""
    item = exact_mapping(value, _JOB_FIELDS, "Morris job")
    if item["schema_id"] != MORRIS_JOB_SCHEMA_ID or item["schema_version"] != 1:
        raise ValueError("unsupported Morris job schema")
    status = item["status"]
    allowed = {"queued", "running", "completed", "cancelled", "failed"}
    if status not in allowed:
        raise ValueError("unsupported Morris job status")
    completed = _integer(item["completed_samples"], "completed_samples")
    total = _integer(item["total_samples"], "total_samples")
    if total < 1 or completed > total or type(item["cancel_requested"]) is not bool:
        raise ValueError("Morris job progress fields are inconsistent")
    report = _parse_report(item["report"]) if item["report"] is not None else None
    error_code: str | None = None
    error_message: str | None = None
    if item["error"] is not None:
        error = exact_mapping(
            item["error"], frozenset({"code", "message"}), "Morris job error"
        )
        error_code = stable_id(error["code"], "error code")
        error_message = _text(error["message"], "error message")
    if (status == "completed") != (report is not None):
        raise ValueError("completed Morris job requires exactly one report")
    if status == "completed" and completed != total:
        raise ValueError("completed Morris job must report all samples")
    if (status == "failed") != (error_code is not None):
        raise ValueError("failed Morris job requires exactly one error")
    return MorrisResponseJob(
        stable_id(item["job_id"], "job_id"),
        stable_id(item["request_id"], "request_id"),
        status,
        completed,
        total,
        item["cancel_requested"],
        report,
        error_code,
        error_message,
    )


__all__ = [
    "MorrisCapability",
    "MorrisDenominator",
    "MorrisEffects",
    "MorrisResponseEstimate",
    "MorrisResponseJob",
    "MorrisResponseReport",
    "MorrisSource",
    "MorrisTarget",
    "parse_morris_capability",
    "parse_morris_job",
]
