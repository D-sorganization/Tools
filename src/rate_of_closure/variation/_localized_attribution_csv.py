"""Formula-safe row and CSV export for localized attribution authority."""

from __future__ import annotations

import csv
import io

from ._localized_attribution_types import (
    AUTHORITY_SCHEMA_ID,
    AUTHORITY_SCHEMA_VERSION,
    AttributionAuthority,
)

CSV_HEADER = (
    "schema_id",
    "schema_version",
    "authority_id",
    "interpretation",
    "source_spec_id",
    "source_variable",
    "source_unit",
    "joint_id",
    "window_start_s",
    "window_end_s",
    "target_id",
    "target_kind",
    "target_name",
    "target_unit",
    "target_frame",
    "target_convention",
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


def _cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if isinstance(value, str) and text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def attribution_observations_to_rows(
    authority: AttributionAuthority,
) -> list[list[str]]:
    """Return canonical parsed rows shared with the TypeScript exporter."""
    sources = {item.spec_id: item for item in authority.sources}
    targets = {item.target_id: item for item in authority.targets}
    rows = [list(CSV_HEADER)]
    for observation in authority.observations:
        source = sources[observation.source_spec_id]
        target = targets[observation.target_id]
        values = (
            AUTHORITY_SCHEMA_ID,
            AUTHORITY_SCHEMA_VERSION,
            authority.authority_id,
            authority.interpretation,
            source.spec_id,
            source.variable_key,
            source.unit,
            source.joint_id,
            *source.time_window_s,
            target.target_id,
            target.kind,
            target.name,
            target.unit,
            target.coordinate_frame,
            target.convention,
            target.time_s,
            target.point_id,
            observation.baseline_trial_index,
            observation.perturbed_trial_index,
            observation.baseline_status.value,
            observation.perturbed_status.value,
            observation.baseline_source_value,
            observation.perturbed_source_value,
            observation.baseline_target_value,
            observation.perturbed_target_value,
            observation.response,
            observation.availability.value,
        )
        rows.append([_cell(value) for value in values])
    return rows


def attribution_observations_to_csv(authority: AttributionAuthority) -> str:
    """Export canonical rows with spreadsheet-safe string cells."""
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(
        attribution_observations_to_rows(authority)
    )
    return output.getvalue()


__all__ = ["attribution_observations_to_csv", "attribution_observations_to_rows"]
