"""Stable, versioned exports for ranked capability alternatives."""

from __future__ import annotations

import csv
import io

from rate_of_closure.variation.canonical_numeric_json import canonical_numeric_json
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from shared.python.contracts import require
from shared.python.swing_sim.flight.capability_result import (
    OptimizationAlternative,
    OptimizationResult,
)

CAPABILITY_RESULT_EXPORT_SCHEMA = "capability-result-export/v1"
_HEADERS = (
    "rank",
    "club_id",
    "parameters",
    "score",
    "mean_carry_m",
    "expected_miss_m",
    "dispersion_rms_m",
    "target_hold_probability",
    "cvar_miss_m",
    "downside_carry_m",
    "sample_count",
    "successful_count",
    "no_impact_count",
    "failed_count",
    "failure_fraction",
    "confidence",
    "extrapolated",
    "pareto_efficient",
    "limiting_constraints",
)


def _parameter_units(dataset: ScalarEnsembleDataset) -> dict[str, str]:
    units = {
        variable.key.removeprefix("nominal."): variable.unit
        for variable in dataset.variables
        if variable.key.startswith("nominal.")
    }
    require(bool(units), "capability result export requires parameter units")
    return units


def _parameter_text(item: OptimizationAlternative, units: dict[str, str]) -> str:
    missing = [
        parameter_id
        for parameter_id, _value in item.parameters
        if parameter_id not in units
    ]
    require(not missing, "result parameters require declared units", tuple(missing))
    return "; ".join(
        f"{parameter_id}={value:.12g} {units[parameter_id]}"
        for parameter_id, value in item.parameters
    )


def _alternative_row(
    item: OptimizationAlternative, units: dict[str, str]
) -> tuple[object, ...]:
    return (
        item.rank,
        item.club_id,
        _parameter_text(item, units),
        item.score,
        item.mean_carry_m,
        item.expected_miss_m,
        item.dispersion_rms_m,
        item.target_hold_probability,
        item.cvar_miss_m,
        item.downside_carry_m,
        item.sample_count,
        item.successful_count,
        item.no_impact_count,
        item.failed_count,
        item.failure_fraction,
        item.confidence,
        item.extrapolated,
        item.pareto_efficient,
        "; ".join(item.limiting_constraints),
    )


def capability_alternatives_csv(
    result: OptimizationResult, dataset: ScalarEnsembleDataset
) -> str:
    """Export every ranked diagnostic with unambiguous parameter units."""
    require(dataset.result_id == result.problem_id, "result and dataset IDs must match")
    units = _parameter_units(dataset)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_HEADERS)
    writer.writerows(_alternative_row(item, units) for item in result.alternatives)
    return output.getvalue().removesuffix("\n")


def capability_result_export_json(
    result: OptimizationResult, dataset: ScalarEnsembleDataset
) -> str:
    """Export the strict result and its external unit declarations."""
    require(dataset.result_id == result.problem_id, "result and dataset IDs must match")
    units = _parameter_units(dataset)
    source = canonical_numeric_json(
        {
            "parameter_units": [
                {"parameter_id": parameter_id, "unit": unit}
                for parameter_id, unit in units.items()
            ],
            "result": result.to_dict(),
            "schema_version": CAPABILITY_RESULT_EXPORT_SCHEMA,
        }
    )
    return str(source)


__all__ = [
    "CAPABILITY_RESULT_EXPORT_SCHEMA",
    "capability_alternatives_csv",
    "capability_result_export_json",
]
