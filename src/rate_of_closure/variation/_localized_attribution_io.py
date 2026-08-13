"""Wire I/O and selection operations for localized attribution authority."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from typing import Any, cast

from shared.python.contracts import require

from ._localized_attribution_types import (
    _STATUS_VALUES,
    AUTHORITY_SCHEMA_ID,
    AUTHORITY_SCHEMA_VERSION,
    VIEW_SCHEMA_ID,
    VIEW_SCHEMA_VERSION,
    AttributionAuthority,
    AttributionDenominator,
    AttributionObservation,
    AttributionSource,
    AttributionTarget,
    AttributionView,
    AttributionViewDefinition,
    Availability,
    TrialStatus,
    _exact,
    _finite,
    _index,
)


def _source(raw: object) -> AttributionSource:
    record = _exact(
        raw, {"spec_id", "variable_key", "joint_id", "time_window_s", "unit"}, "source"
    )
    window = record["time_window_s"]
    require(
        isinstance(window, list) and len(window) == 2, "window must contain two values"
    )
    values = cast(list[object], window)
    return AttributionSource(
        cast(str, record["spec_id"]),
        cast(str, record["variable_key"]),
        cast(str, record["joint_id"]),
        (_finite(values[0], "window start"), _finite(values[1], "window end")),
        cast(str, record["unit"]),
    )


def _target(raw: object) -> AttributionTarget:
    record = _exact(
        raw,
        {"target_id", "kind", "name", "unit", "time_s", "point_id", "coordinate_frame"},
        "target",
    )
    return AttributionTarget(
        cast(str, record["target_id"]),
        cast(str, record["kind"]),
        cast(str, record["name"]),
        cast(str, record["unit"]),
        None if record["time_s"] is None else _finite(record["time_s"], "target time"),
        cast(str | None, record["point_id"]),
        cast(str | None, record["coordinate_frame"]),
    )


def _availability(raw: object) -> Availability:
    require(
        isinstance(raw, str) and raw in {item.value for item in Availability},
        "invalid availability",
        raw,
    )
    return Availability(raw)


def _observation(raw: object) -> AttributionObservation:
    fields = {
        "source_spec_id",
        "target_id",
        "baseline_trial_index",
        "perturbed_trial_index",
        "baseline_status",
        "perturbed_status",
        "baseline_source_value",
        "perturbed_source_value",
        "baseline_target_value",
        "perturbed_target_value",
        "response",
        "availability",
    }
    record = _exact(raw, fields, "observation")

    def nullable(name: str) -> float | None:
        return None if record[name] is None else _finite(record[name], name)

    require(
        record["baseline_status"] in _STATUS_VALUES
        and record["perturbed_status"] in _STATUS_VALUES,
        "invalid trial status",
    )
    return AttributionObservation(
        cast(str, record["source_spec_id"]),
        cast(str, record["target_id"]),
        _index(record["baseline_trial_index"], "baseline_trial_index"),
        _index(record["perturbed_trial_index"], "perturbed_trial_index"),
        TrialStatus(cast(str, record["baseline_status"])),
        TrialStatus(cast(str, record["perturbed_status"])),
        _finite(record["baseline_source_value"], "baseline source value"),
        _finite(record["perturbed_source_value"], "perturbed source value"),
        nullable("baseline_target_value"),
        nullable("perturbed_target_value"),
        nullable("response"),
        _availability(record["availability"]),
    )


def attribution_authority_from_dict(raw: object) -> AttributionAuthority:
    """Parse one exact finite schema-v1 authority document."""
    record = _exact(
        raw,
        {
            "schema_id",
            "schema_version",
            "authority_id",
            "interpretation",
            "sources",
            "targets",
            "observations",
        },
        "authority",
    )
    require(record["schema_id"] == AUTHORITY_SCHEMA_ID, "invalid schema_id")
    require(
        isinstance(record["schema_version"], int)
        and not isinstance(record["schema_version"], bool)
        and record["schema_version"] == AUTHORITY_SCHEMA_VERSION,
        "invalid schema_version",
    )
    require(
        isinstance(record["sources"], list)
        and isinstance(record["targets"], list)
        and isinstance(record["observations"], list),
        "authority arrays are required",
    )
    sources = cast(list[object], record["sources"])
    targets = cast(list[object], record["targets"])
    observations = cast(list[object], record["observations"])
    return AttributionAuthority(
        cast(str, record["authority_id"]),
        tuple(_source(item) for item in sources),
        tuple(_target(item) for item in targets),
        tuple(_observation(item) for item in observations),
        cast(str, record["interpretation"]),
    )


def attribution_authority_to_dict(authority: AttributionAuthority) -> dict[str, object]:
    """Return the canonical snake-case wire document."""
    require(
        isinstance(authority, AttributionAuthority),
        "authority must be AttributionAuthority",
    )

    def wire(value: object) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(asdict(cast(Any, value)))))

    return {
        "schema_id": AUTHORITY_SCHEMA_ID,
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_id": authority.authority_id,
        "interpretation": authority.interpretation,
        "sources": [wire(source) for source in authority.sources],
        "targets": [wire(target) for target in authority.targets],
        "observations": [wire(observation) for observation in authority.observations],
    }


def build_attribution_view(
    authority: AttributionAuthority, definition: AttributionViewDefinition
) -> AttributionView:
    """Resolve a strict selection without estimating missing observations."""
    require(authority.authority_id == definition.authority_id, "authority_id mismatch")
    source = next(
        (
            item
            for item in authority.sources
            if item.spec_id == definition.source_spec_id
        ),
        None,
    )
    target = next(
        (item for item in authority.targets if item.target_id == definition.target_id),
        None,
    )
    require(source is not None, "unknown source_spec_id", definition.source_spec_id)
    require(target is not None, "unknown target_id", definition.target_id)
    rows = tuple(
        item
        for item in authority.observations
        if item.source_spec_id == definition.source_spec_id
        and item.target_id == definition.target_id
    )
    selected = next(
        (
            item
            for item in rows
            if item.baseline_trial_index == definition.baseline_trial_index
            and item.perturbed_trial_index == definition.perturbed_trial_index
        ),
        None,
    )
    require(selected is not None, "selected attribution pair is unavailable")
    typed_miss = sum(
        TrialStatus.EVALUATED_NO_IMPACT in {row.baseline_status, row.perturbed_status}
        for row in rows
    )
    denominator = AttributionDenominator(
        len(rows),
        sum(row.availability is Availability.AVAILABLE for row in rows),
        typed_miss,
        sum(row.availability is Availability.NO_IMPACT_UNAVAILABLE for row in rows),
        sum(row.availability is Availability.NUMERICAL_FAILURE for row in rows),
        sum(row.availability is Availability.NONFINITE_UNAVAILABLE for row in rows),
    )
    return AttributionView(
        cast(AttributionSource, source),
        cast(AttributionTarget, target),
        cast(AttributionObservation, selected),
        rows,
        denominator,
    )


def attribution_view_to_json(definition: AttributionViewDefinition) -> str:
    """Serialize one exact persisted selection."""
    payload = {
        "schema_id": VIEW_SCHEMA_ID,
        "schema_version": VIEW_SCHEMA_VERSION,
        **asdict(definition),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def attribution_view_from_json(text: str) -> AttributionViewDefinition:
    """Parse a strict persisted selection without numeric coercion."""
    require(isinstance(text, str), "view JSON must be text")
    try:
        raw: Any = json.loads(text)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid attribution view JSON") from error
    fields = {
        "schema_id",
        "schema_version",
        "authority_id",
        "source_spec_id",
        "target_id",
        "baseline_trial_index",
        "perturbed_trial_index",
    }
    record = _exact(raw, fields, "view definition")
    require(record["schema_id"] == VIEW_SCHEMA_ID, "invalid schema_id")
    require(
        isinstance(record["schema_version"], int)
        and not isinstance(record["schema_version"], bool)
        and record["schema_version"] == VIEW_SCHEMA_VERSION,
        "invalid schema_version",
    )
    return AttributionViewDefinition(
        cast(str, record["authority_id"]),
        cast(str, record["source_spec_id"]),
        cast(str, record["target_id"]),
        _index(record["baseline_trial_index"], "baseline_trial_index"),
        _index(record["perturbed_trial_index"], "perturbed_trial_index"),
    )


def attribution_observations_to_csv(authority: AttributionAuthority) -> str:
    """Export every raw observation with source/target provenance intact."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "interpretation",
            "source_spec_id",
            "joint_id",
            "window_start_s",
            "window_end_s",
            "target_id",
            "target_kind",
            "target_name",
            "target_time_s",
            "target_point_id",
            "baseline_trial",
            "perturbed_trial",
            "baseline_status",
            "perturbed_status",
            "baseline_source_value",
            "perturbed_source_value",
            "baseline_target_value",
            "perturbed_target_value",
            "response",
            "availability",
        )
    )
    sources = {item.spec_id: item for item in authority.sources}
    targets = {item.target_id: item for item in authority.targets}
    for row in authority.observations:
        source, target = sources[row.source_spec_id], targets[row.target_id]
        writer.writerow(
            (
                authority.interpretation,
                source.spec_id,
                source.joint_id,
                *source.time_window_s,
                target.target_id,
                target.kind,
                target.name,
                target.time_s,
                target.point_id,
                row.baseline_trial_index,
                row.perturbed_trial_index,
                row.baseline_status.value,
                row.perturbed_status.value,
                row.baseline_source_value,
                row.perturbed_source_value,
                row.baseline_target_value,
                row.perturbed_target_value,
                row.response,
                row.availability.value,
            )
        )
    return output.getvalue()
