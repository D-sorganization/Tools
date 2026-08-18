"""Strict untrusted-value parser for the scalar-ensemble/v1 wire contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

MAX_SCALAR_ENSEMBLE_STAGES = 32
MAX_SCALAR_ENSEMBLE_CATEGORIES = 64
MAX_SCALAR_ENSEMBLE_VARIABLES = 256
MAX_SCALAR_ENSEMBLE_COHORTS = 32
MAX_SCALAR_ENSEMBLE_ROWS = 100_000

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "result_id",
        "provenance",
        "stages",
        "categories",
        "variables",
        "cohorts",
        "rows",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"adapter_id", "source_schema_version", "source_provenance"}
)
_LABEL_FIELDS = frozenset({"key", "label"})
_VARIABLE_FIELDS = frozenset({"key", "label", "unit", "stage_key", "category_key"})
_ROW_REQUIRED_FIELDS = frozenset({"row_id", "trial_index", "cohort", "values"})
_ROW_OPTIONAL_FIELDS = frozenset({"series_id", "attributes"})


def _exact_mapping(
    value: object, expected_fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object")
    item = cast(Mapping[str, Any], value)
    actual = frozenset(item)
    if actual != expected_fields:
        missing = sorted(expected_fields - actual)
        unknown = sorted(actual - expected_fields)
        raise ValueError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return item


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{name} must be nonblank text")
    canonical_numeric_json(value)
    return value


def _array(value: object, name: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} entries")
    return cast(Sequence[Any], value)


def _finite_or_null(value: object, name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise TypeError(f"{name} must be finite or null")
    canonical_numeric_json(value)
    return float(cast(int | float, value))


def _provenance(value: object) -> ScalarEnsembleProvenance:
    item = _exact_mapping(value, _PROVENANCE_FIELDS, "scalar ensemble provenance")
    return ScalarEnsembleProvenance(
        _text(item["adapter_id"], "provenance adapter_id"),
        _text(item["source_schema_version"], "provenance source_schema_version"),
        _text(item["source_provenance"], "provenance source_provenance"),
    )


def _stage(value: object, name: str) -> ScalarEnsembleStage:
    item = _exact_mapping(value, _LABEL_FIELDS, name)
    return ScalarEnsembleStage(
        _text(item["key"], f"{name} key"),
        _text(item["label"], f"{name} label"),
    )


def _category(value: object, name: str) -> ScalarVariableCategory:
    item = _exact_mapping(value, _LABEL_FIELDS, name)
    return ScalarVariableCategory(
        _text(item["key"], f"{name} key"),
        _text(item["label"], f"{name} label"),
    )


def _cohort(value: object, name: str) -> ScalarCohortDefinition:
    item = _exact_mapping(value, _LABEL_FIELDS, name)
    return ScalarCohortDefinition(
        _text(item["key"], f"{name} key"),
        _text(item["label"], f"{name} label"),
    )


def _variable(value: object, index: int) -> ScalarVariableDefinition:
    name = f"scalar ensemble variable[{index}]"
    item = _exact_mapping(value, _VARIABLE_FIELDS, name)
    return ScalarVariableDefinition(
        _text(item["key"], f"{name} key"),
        _text(item["label"], f"{name} label"),
        _text(item["unit"], f"{name} unit"),
        _text(item["stage_key"], f"{name} stage_key"),
        _text(item["category_key"], f"{name} category_key"),
    )


def _row_fields(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str for key in value.keys()
    ):
        raise TypeError(f"{name} must be a JSON object")
    fields = frozenset(cast(Mapping[str, Any], value))
    allowed = _ROW_REQUIRED_FIELDS | _ROW_OPTIONAL_FIELDS
    if not _ROW_REQUIRED_FIELDS <= fields or not fields <= allowed:
        raise ValueError(f"{name} fields do not match v1 schema")
    return cast(Mapping[str, Any], value)


def _values(value: object, name: str) -> dict[str, float | None]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return {
        _text(key, f"{name} key"): _finite_or_null(item, f"{name}[{key!r}]")
        for key, item in cast(Mapping[str, object], value).items()
    }


def _attributes(value: object, name: str) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be a JSON object")
    parsed: dict[str, str | None] = {}
    for key, item in cast(Mapping[str, object], value).items():
        parsed[_text(key, f"{name} key")] = (
            None if item is None else _text(item, f"{name}[{key!r}]")
        )
    return parsed


def _row(value: object, index: int) -> ScalarEnsembleRow:
    name = f"scalar ensemble row[{index}]"
    item = _row_fields(value, name)
    trial_index = item["trial_index"]
    if type(trial_index) is not int or trial_index < 0:
        raise TypeError(f"{name} trial_index must be a nonnegative integer")
    series_id = (
        _text(item["series_id"], f"{name} series_id") if "series_id" in item else None
    )
    attributes = (
        _attributes(item["attributes"], f"{name} attributes")
        if "attributes" in item
        else None
    )
    return ScalarEnsembleRow(
        _text(item["row_id"], f"{name} row_id"),
        trial_index,
        _text(item["cohort"], f"{name} cohort"),
        _values(item["values"], f"{name} values"),
        series_id,
        attributes,
    )


def scalar_ensemble_dataset_from_wire(
    value: object,
    *,
    max_rows: int = MAX_SCALAR_ENSEMBLE_ROWS,
) -> ScalarEnsembleDataset:
    """Parse one exact bounded scalar-ensemble/v1 object without model execution."""
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_SCALAR_ENSEMBLE_ROWS:
        raise ValueError("max_rows must be a bounded positive integer")
    item = _exact_mapping(value, _ROOT_FIELDS, "scalar ensemble result")
    if item["schema_version"] != SCALAR_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError("unsupported scalar ensemble schema")
    stages = _array(
        item["stages"], "scalar ensemble stages", MAX_SCALAR_ENSEMBLE_STAGES
    )
    categories = _array(
        item["categories"],
        "scalar ensemble categories",
        MAX_SCALAR_ENSEMBLE_CATEGORIES,
    )
    variables = _array(
        item["variables"], "scalar ensemble variables", MAX_SCALAR_ENSEMBLE_VARIABLES
    )
    cohorts = _array(
        item["cohorts"], "scalar ensemble cohorts", MAX_SCALAR_ENSEMBLE_COHORTS
    )
    rows = _array(item["rows"], "scalar ensemble rows", max_rows)
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        _text(item["result_id"], "scalar ensemble result_id"),
        _provenance(item["provenance"]),
        tuple(
            _stage(entry, f"scalar ensemble stage[{index}]")
            for index, entry in enumerate(stages)
        ),
        tuple(
            _category(entry, f"scalar ensemble category[{index}]")
            for index, entry in enumerate(categories)
        ),
        tuple(_variable(entry, index) for index, entry in enumerate(variables)),
        tuple(
            _cohort(entry, f"scalar ensemble cohort[{index}]")
            for index, entry in enumerate(cohorts)
        ),
        tuple(_row(entry, index) for index, entry in enumerate(rows)),
    )


__all__ = [
    "MAX_SCALAR_ENSEMBLE_ROWS",
    "scalar_ensemble_dataset_from_wire",
]
