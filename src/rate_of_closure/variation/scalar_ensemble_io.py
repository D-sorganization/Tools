"""Lossless exports and honest summaries for scalar-ensemble datasets."""

from __future__ import annotations

import csv
import io

from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset


def _spreadsheet_safe(value: object) -> object:
    if isinstance(value, str) and value.lstrip("'").startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return "" if value is None else value


def scalar_ensemble_csv(dataset: ScalarEnsembleDataset) -> str:
    """Serialize every raw row, nullable variable, and attribute to CSV."""
    attribute_keys = sorted(
        {
            key
            for row in dataset.rows
            for key in (() if row.attributes is None else row.attributes)
        }
    )
    fixed = ["row_id", "trial_index", "series_id", "cohort"]
    variable_keys = [variable.key for variable in dataset.variables]
    attributes = [f"attribute:{key}" for key in attribute_keys]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [_spreadsheet_safe(item) for item in fixed + variable_keys + attributes]
    )
    for row in dataset.rows:
        row_attributes = {} if row.attributes is None else row.attributes
        values: list[object] = [
            row.row_id,
            row.trial_index,
            row.series_id,
            row.cohort,
            *(row.values[key] for key in variable_keys),
            *(row_attributes.get(key) for key in attribute_keys),
        ]
        writer.writerow([_spreadsheet_safe(value) for value in values])
    return output.getvalue().removesuffix("\n")


def non_complete_reason_summary(dataset: ScalarEnsembleDataset) -> str:
    """Report why non-complete rows ended so counts are not read as defects.

    A horizon nonconvergence is normalized into the ``failed`` cohort by the
    observation contract, which has no separate member for it. Naming the
    retained reason keeps the count truthful.
    """
    counts: dict[str, int] = {}
    for row in dataset.rows:
        if row.cohort == "complete":
            continue
        attributes = {} if row.attributes is None else row.attributes
        reason = attributes.get("reason_code") or "unspecified"
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    detail = "; ".join(f"{reason} x{count}" for reason, count in ordered)
    return f" Non-complete reasons: {detail}."


__all__ = ["non_complete_reason_summary", "scalar_ensemble_csv"]
