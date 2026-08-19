"""Cross-runtime scalar ensemble v1 contract and scatter projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import quote

from shared.python.contracts import require

SCALAR_ENSEMBLE_SCHEMA_VERSION = "scalar-ensemble/v1"


def _nonempty(value: str, name: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty")


def _unique_keys(definitions: tuple[Any, ...], name: str) -> None:
    keys = tuple(str(item.key) for item in definitions)
    require(len(set(keys)) == len(keys), f"{name} keys must be unique", keys)


@dataclass(frozen=True)
class ScalarEnsembleProvenance:
    """Auditable adapter and upstream-source identity."""

    adapter_id: str
    source_schema_version: str
    source_provenance: str

    def __post_init__(self) -> None:
        """Require every provenance field to be nonempty."""
        for name in ("adapter_id", "source_schema_version", "source_provenance"):
            _nonempty(getattr(self, name), f"provenance.{name}")


@dataclass(frozen=True)
class ScalarEnsembleStage:
    """Labeled process stage used to organize variables."""

    key: str
    label: str

    def __post_init__(self) -> None:
        """Require nonempty stage identity and label."""
        _nonempty(self.key, "stage key")
        _nonempty(self.label, "stage label")


@dataclass(frozen=True)
class ScalarVariableCategory:
    """Labeled domain category used to group variables."""

    key: str
    label: str

    def __post_init__(self) -> None:
        """Require nonempty category identity and label."""
        _nonempty(self.key, "category key")
        _nonempty(self.label, "category label")


@dataclass(frozen=True)
class ScalarVariableDefinition:
    """One unit-bearing selectable scalar definition."""

    key: str
    label: str
    unit: str
    stage_key: str
    category_key: str

    def __post_init__(self) -> None:
        """Require all variable metadata fields to be nonempty."""
        for name in ("key", "label", "unit", "stage_key", "category_key"):
            _nonempty(getattr(self, name), f"variable {name}")


@dataclass(frozen=True)
class ScalarCohortDefinition:
    """Caller-defined, labeled row cohort."""

    key: str
    label: str

    def __post_init__(self) -> None:
        """Require nonempty cohort identity and label."""
        _nonempty(self.key, "cohort key")
        _nonempty(self.label, "cohort label")


def scalar_ensemble_row_id(trial_index: int, series_id: str | None = None) -> str:
    """Return the canonical RFC3986-encoded composite row identity."""
    require(
        type(trial_index) is int and trial_index >= 0,
        "trial_index must be a nonnegative integer",
        trial_index,
    )
    if series_id is None:
        return f"trial:{trial_index}"
    _nonempty(series_id, "series_id")
    return f"series:{quote(series_id, safe='-._~')}/trial:{trial_index}"


@dataclass(frozen=True)
class ScalarEnsembleRow:
    """One immutable trial/series row of nullable raw scalars."""

    row_id: str
    trial_index: int
    cohort: str
    values: Mapping[str, float | None] = field(repr=False)
    series_id: str | None = None
    attributes: Mapping[str, str | None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate canonical identity and finite-or-null raw values."""
        require(
            self.row_id == scalar_ensemble_row_id(self.trial_index, self.series_id),
            "row_id must equal its canonical composite identity",
        )
        _nonempty(self.cohort, "cohort")
        values = dict(self.values)
        require(
            all(value is None or math.isfinite(value) for value in values.values()),
            "row values must be finite or null",
        )
        object.__setattr__(self, "values", MappingProxyType(values))
        if self.attributes is not None:
            attributes = dict(self.attributes)
            for key, value in attributes.items():
                _nonempty(key, "attribute key")
                if value is not None:
                    _nonempty(value, "attribute value")
            object.__setattr__(self, "attributes", MappingProxyType(attributes))

    def value(self, key: str) -> float | None:
        """Return one raw scalar by declared key."""
        require(key in self.values, "unknown scalar value", key)
        return self.values[key]

    def to_wire(self) -> dict[str, Any]:
        """Return the exact scalar-ensemble/v1 row wire shape."""
        wire: dict[str, Any] = {
            "row_id": self.row_id,
            "trial_index": self.trial_index,
            "cohort": self.cohort,
            "values": dict(self.values),
        }
        if self.series_id is not None:
            wire["series_id"] = self.series_id
        if self.attributes is not None:
            wire["attributes"] = dict(self.attributes)
        return wire


@dataclass(frozen=True)
class ScalarAvailability:
    """Exact axis and paired-finite availability counts."""

    total_rows: int = 0
    x_finite: int = 0
    y_finite: int = 0
    paired_finite: int = 0
    unavailable: int = 0

    def __post_init__(self) -> None:
        """Require nonnegative and internally consistent counts."""
        counts = (self.total_rows, self.x_finite, self.y_finite, self.paired_finite)
        require(
            all(count >= 0 for count in counts),
            "availability counts must be nonnegative",
        )
        require(self.x_finite <= self.total_rows, "x_finite exceeds total_rows")
        require(self.y_finite <= self.total_rows, "y_finite exceeds total_rows")
        require(
            self.paired_finite <= min(self.x_finite, self.y_finite),
            "paired_finite is invalid",
        )
        require(
            self.unavailable == self.total_rows - self.paired_finite,
            "unavailable must equal total_rows minus paired_finite",
        )


@dataclass(frozen=True)
class ScalarScatterPoint:
    """One paired-finite scatter point retaining row identity."""

    row_id: str
    trial_index: int
    cohort: str
    x: float
    y: float
    series_id: str | None = None


@dataclass(frozen=True)
class ScalarScatterAvailability:
    """Overall and per-cohort scatter availability."""

    overall: ScalarAvailability
    by_cohort: Mapping[str, ScalarAvailability]

    def __post_init__(self) -> None:
        """Freeze the exact per-cohort availability map."""
        object.__setattr__(self, "by_cohort", MappingProxyType(dict(self.by_cohort)))


@dataclass(frozen=True)
class ScalarScatterData:
    """Paired-finite points and exact availability for two axes."""

    x_variable: ScalarVariableDefinition
    y_variable: ScalarVariableDefinition
    points: tuple[ScalarScatterPoint, ...]
    availability: ScalarScatterAvailability


@dataclass(frozen=True)
class ScalarEnsembleDataset:
    """Validated, immutable scalar-ensemble/v1 result."""

    schema_version: str
    result_id: str
    provenance: ScalarEnsembleProvenance
    stages: tuple[ScalarEnsembleStage, ...]
    categories: tuple[ScalarVariableCategory, ...]
    variables: tuple[ScalarVariableDefinition, ...]
    cohorts: tuple[ScalarCohortDefinition, ...]
    rows: tuple[ScalarEnsembleRow, ...]

    def __post_init__(self) -> None:
        """Validate definitions, relationships, rows, and uniqueness."""
        require(
            self.schema_version == SCALAR_ENSEMBLE_SCHEMA_VERSION,
            "unsupported scalar ensemble schema",
            self.schema_version,
        )
        _nonempty(self.result_id, "result_id")
        for name in ("stages", "categories", "variables", "cohorts", "rows"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        require(
            self.stages and self.categories and self.variables,
            "stages, categories, and variables must be nonempty",
        )
        require(bool(self.cohorts), "cohorts must be nonempty")
        for definitions, name in (
            (self.stages, "stage"),
            (self.categories, "category"),
            (self.variables, "variable"),
            (self.cohorts, "cohort"),
        ):
            _unique_keys(definitions, name)
        self._validate_variable_references()
        self._validate_rows()

    def _validate_variable_references(self) -> None:
        stages = {stage.key for stage in self.stages}
        categories = {category.key for category in self.categories}
        require(
            all(variable.stage_key in stages for variable in self.variables),
            "variable references an unknown stage",
        )
        require(
            all(variable.category_key in categories for variable in self.variables),
            "variable references an unknown category",
        )

    def _validate_rows(self) -> None:
        row_ids = tuple(row.row_id for row in self.rows)
        variable_keys = {variable.key for variable in self.variables}
        cohort_keys = {cohort.key for cohort in self.cohorts}
        require(len(set(row_ids)) == len(row_ids), "row_id values must be unique")
        require(
            all(row.cohort in cohort_keys for row in self.rows),
            "row references an unknown cohort",
        )
        require(
            all(set(row.values) == variable_keys for row in self.rows),
            "row values must contain exactly the declared variable keys",
        )

    def variable(self, key: str) -> ScalarVariableDefinition:
        """Return a declared variable by stable key."""
        match = next(
            (variable for variable in self.variables if variable.key == key), None
        )
        require(match is not None, "scatter axes must be declared variables", key)
        return cast(ScalarVariableDefinition, match)

    def scatter(self, x_key: str, y_key: str) -> ScalarScatterData:
        """Derive paired-finite points and complete availability counts."""
        x_variable = self.variable(x_key)
        y_variable = self.variable(y_key)
        cohort_keys = tuple(cohort.key for cohort in self.cohorts)
        points = _scatter_points(self.rows, x_key, y_key)
        overall = _availability(self.rows, x_key, y_key)
        by_cohort = {
            cohort: _availability(
                tuple(row for row in self.rows if row.cohort == cohort),
                x_key,
                y_key,
            )
            for cohort in cohort_keys
        }
        return ScalarScatterData(
            x_variable,
            y_variable,
            points,
            ScalarScatterAvailability(overall, by_cohort),
        )

    def to_wire(self) -> dict[str, Any]:
        """Return the exact snake-case scalar-ensemble/v1 wire representation."""
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "provenance": _record_wire(self.provenance),
            "stages": [_record_wire(item) for item in self.stages],
            "categories": [_record_wire(item) for item in self.categories],
            "variables": [_record_wire(item) for item in self.variables],
            "cohorts": [_record_wire(item) for item in self.cohorts],
            "rows": [row.to_wire() for row in self.rows],
        }


def _scatter_points(
    rows: tuple[ScalarEnsembleRow, ...],
    x_key: str,
    y_key: str,
) -> tuple[ScalarScatterPoint, ...]:
    points: list[ScalarScatterPoint] = []
    for row in rows:
        x_value = row.value(x_key)
        y_value = row.value(y_key)
        if x_value is None or y_value is None:
            continue
        points.append(
            ScalarScatterPoint(
                row.row_id,
                row.trial_index,
                row.cohort,
                x_value,
                y_value,
                row.series_id,
            )
        )
    return tuple(points)


def _availability(
    rows: tuple[ScalarEnsembleRow, ...],
    x_key: str,
    y_key: str,
) -> ScalarAvailability:
    x_finite = sum(row.value(x_key) is not None for row in rows)
    y_finite = sum(row.value(y_key) is not None for row in rows)
    paired = sum(
        row.value(x_key) is not None and row.value(y_key) is not None for row in rows
    )
    return ScalarAvailability(len(rows), x_finite, y_finite, paired, len(rows) - paired)


def _record_wire(record: Any) -> dict[str, Any]:
    return {name: getattr(record, name) for name in record.__dataclass_fields__}


__all__ = [
    "SCALAR_ENSEMBLE_SCHEMA_VERSION",
    "ScalarAvailability",
    "ScalarCohortDefinition",
    "ScalarEnsembleDataset",
    "ScalarEnsembleProvenance",
    "ScalarEnsembleRow",
    "ScalarEnsembleStage",
    "ScalarScatterAvailability",
    "ScalarScatterData",
    "ScalarScatterPoint",
    "ScalarVariableCategory",
    "ScalarVariableDefinition",
    "scalar_ensemble_row_id",
]
