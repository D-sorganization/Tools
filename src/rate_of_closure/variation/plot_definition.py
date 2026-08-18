"""Versioned, JSON-safe definitions for reproducible variation plots."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LOW_VARIABILITY_METRICS,
)
from shared.python.swing_sim.variation.dispersion_metric_types import (
    validated_confidence_level,
)

from ._plot_definition_contract import (
    _normalize_nullable_integer,
    _normalize_nullable_real,
    _strict_integer,
    _strict_nullable_boolean,
    _strict_nullable_integer,
    _strict_nullable_real,
    _strict_nullable_string,
    _strict_variable_keys,
    _validate_exact_fields,
    _validate_variable_keys_object,
)
from ._plot_definition_migration import migrate_v1, migrate_v2
from .simulation_types import APP_FRAME_ID

PLOT_DEFINITION_SCHEMA_VERSION = 3
PlotType = Literal[
    "scalar_scatter",
    "swing_arc_overlay",
    "geometric_variability",
    "distribution_matrix",
]
_PLOT_TYPES = {
    "scalar_scatter",
    "swing_arc_overlay",
    "geometric_variability",
    "distribution_matrix",
}
_OUTCOME_FILTERS = {
    "evaluated_hit",
    "evaluated_no_impact",
    "numerical_failure",
}
_PERTURBATION_BANDS = {
    "lower",
    "middle",
    "upper",
    "Lower Half",
    "Upper Half",
    "Lower Third",
    "Middle Third",
    "Upper Third",
}
_NULLABLE_STRING_FIELDS = {
    "coordinate_frame",
    "x_variable_key",
    "y_variable_key",
    "point_id",
    "position_unit",
    "alignment_basis",
    "dispersion_metric",
    "dispersion_unit",
    "outcome_filter",
    "perturbation_source_key",
    "perturbation_band",
}
_NULLABLE_REAL_FIELDS = {
    "quiet_threshold",
    "confidence_level",
    "min_quiet_duration_s",
    "camera_yaw_deg",
    "camera_pitch_deg",
    "camera_zoom",
    "phase_end_fraction",
}
_NULLABLE_INTEGER_FIELDS = {"min_quiet_samples", "selected_trial_index"}
_APPLICABLE_FIELDS = {
    "scalar_scatter": {"x_variable_key", "y_variable_key", "selected_trial_index"},
    "swing_arc_overlay": {
        "coordinate_frame",
        "point_id",
        "position_unit",
        "alignment_basis",
        "dispersion_metric",
        "dispersion_unit",
        "quiet_threshold",
        "confidence_level",
        "min_quiet_duration_s",
        "min_quiet_samples",
        "selected_trial_index",
        "camera_yaw_deg",
        "camera_pitch_deg",
        "camera_zoom",
        "outcome_filter",
        "phase_end_fraction",
        "perturbation_source_key",
        "perturbation_band",
        "show_confidence_ellipsoids",
    },
    "geometric_variability": {
        "coordinate_frame",
        "point_id",
        "position_unit",
        "alignment_basis",
        "dispersion_metric",
        "dispersion_unit",
        "quiet_threshold",
        "confidence_level",
        "min_quiet_duration_s",
        "min_quiet_samples",
        "outcome_filter",
        "phase_end_fraction",
        "perturbation_source_key",
        "perturbation_band",
    },
    "distribution_matrix": {"variable_keys"},
}


@dataclass(frozen=True)
class PlotDefinition:
    """Serializable plot state independent of a particular UI toolkit."""

    result_id: str
    plot_type: PlotType
    coordinate_frame: str | None = None
    x_variable_key: str | None = None
    y_variable_key: str | None = None
    point_id: str | None = None
    position_unit: str | None = None
    alignment_basis: str | None = None
    dispersion_metric: str | None = None
    dispersion_unit: str | None = None
    quiet_threshold: float | None = None
    confidence_level: float | None = None
    min_quiet_duration_s: float | None = None
    min_quiet_samples: int | None = None
    selected_trial_index: int | None = None
    camera_yaw_deg: float | None = None
    camera_pitch_deg: float | None = None
    camera_zoom: float | None = None
    outcome_filter: str | None = None
    phase_end_fraction: float | None = None
    perturbation_source_key: str | None = None
    perturbation_band: str | None = None
    variable_keys: tuple[str, ...] | None = None
    show_confidence_ellipsoids: bool | None = None

    def __post_init__(self) -> None:
        self._validate_full_object()

    def _validate_full_object(self) -> None:
        """Fail closed for direct construction and pre-serialization reuse."""
        result_id = _strict_nullable_string(self.result_id, "result_id")
        require(result_id is not None, "result_id must be non-empty")
        require(
            isinstance(self.plot_type, str) and self.plot_type in _PLOT_TYPES,
            "unknown plot_type",
            self.plot_type,
        )
        for name in _NULLABLE_STRING_FIELDS:
            _strict_nullable_string(getattr(self, name), name)
        for name in _NULLABLE_REAL_FIELDS:
            object.__setattr__(
                self, name, _normalize_nullable_real(getattr(self, name), name)
            )
        for name in _NULLABLE_INTEGER_FIELDS:
            object.__setattr__(
                self, name, _normalize_nullable_integer(getattr(self, name), name)
            )
        _validate_variable_keys_object(self.variable_keys)
        require(
            self.show_confidence_ellipsoids is None
            or type(self.show_confidence_ellipsoids) is bool,
            "show_confidence_ellipsoids must be null or boolean",
        )
        applicable = _APPLICABLE_FIELDS[cast(str, self.plot_type)]
        for item in fields(self):
            if item.name not in {"result_id", "plot_type"} | applicable:
                require(
                    getattr(self, item.name) is None,
                    f"{item.name} is not applicable to {self.plot_type}",
                )
        require(
            self.selected_trial_index is None
            or (
                type(self.selected_trial_index) is int
                and self.selected_trial_index >= 0
            ),
            "selected_trial_index must be non-negative",
        )
        pitch = self.camera_pitch_deg
        require(
            pitch is None or -90.0 <= pitch <= 90.0,
            "camera_pitch_deg must be in [-90, 90]",
            pitch,
        )
        require(
            self.camera_zoom is None or self.camera_zoom > 0,
            "camera_zoom must be finite and greater than zero",
        )
        require(
            self.phase_end_fraction is None or 0 < self.phase_end_fraction <= 1,
            "phase_end_fraction must be finite and in (0, 1]",
        )
        if self.plot_type == "scalar_scatter":
            require(self.x_variable_key is not None, "scatter requires x_variable_key")
            require(self.y_variable_key is not None, "scatter requires y_variable_key")
        geometric = self.plot_type in {
            "swing_arc_overlay",
            "geometric_variability",
        }
        if geometric:
            require(self.point_id is not None, "geometric plot requires point_id")
            require(
                self.coordinate_frame == APP_FRAME_ID,
                f"geometric coordinate_frame must equal {APP_FRAME_ID}",
            )
            require(self.position_unit == "m", "geometric position_unit must be m")
            require(
                self.alignment_basis == "common_simulation_time_s",
                "geometric alignment_basis must be common_simulation_time_s",
            )
            self._validate_dispersion_state()
            self._validate_geometric_filters()
        if self.plot_type == "swing_arc_overlay":
            require(
                type(self.show_confidence_ellipsoids) is bool,
                "swing arc requires show_confidence_ellipsoids",
            )
            require(
                not self.show_confidence_ellipsoids
                or self.dispersion_metric == ELLIPSOID_VOLUME,
                "confidence surfaces require confidence-ellipsoid volume",
            )
        if self.plot_type == "distribution_matrix":
            variable_keys = self.variable_keys
            require(
                variable_keys is not None and 2 <= len(variable_keys) <= 8,
                "distribution matrix requires 2 to 8 variable_keys",
            )
            if variable_keys is not None:
                require(
                    len(set(variable_keys)) == len(variable_keys),
                    "distribution matrix variable_keys must be unique",
                )
                require(
                    all(bool(key.strip()) for key in variable_keys),
                    "distribution matrix variable_keys must be non-empty",
                )

    def _validate_geometric_filters(self) -> None:
        """Validate stable filter values that can reproduce the selected cohort."""
        require(
            self.outcome_filter is None or self.outcome_filter in _OUTCOME_FILTERS,
            "unknown outcome_filter",
            self.outcome_filter,
        )
        require(
            self.perturbation_band is None
            or self.perturbation_band in _PERTURBATION_BANDS,
            "unknown perturbation_band",
            self.perturbation_band,
        )
        require(
            self.perturbation_band is None or self.perturbation_source_key is not None,
            "perturbation_band requires a perturbation source",
        )

    def _validate_dispersion_state(self) -> None:
        """Require complete metric-specific quiet-zone state for geometric plots."""
        require(
            self.dispersion_metric in LOW_VARIABILITY_METRICS,
            "geometric plot requires a supported dispersion_metric",
            self.dispersion_metric,
        )
        expected_unit = "m^3" if self.dispersion_metric == ELLIPSOID_VOLUME else "m"
        require(self.dispersion_unit == expected_unit, "invalid dispersion_unit")
        threshold = self.quiet_threshold
        require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold))
            and threshold > 0,
            "quiet_threshold must be finite and greater than zero",
            threshold,
        )
        if self.dispersion_metric == ELLIPSOID_VOLUME:
            require(self.confidence_level is not None, "volume requires confidence")
            if self.confidence_level is not None:
                validated_confidence_level(self.confidence_level)
        else:
            require(
                self.confidence_level is None,
                "confidence applies only to confidence-ellipsoid volume",
            )
        duration = self.min_quiet_duration_s
        require(
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(float(duration))
            and duration >= 0,
            "min_quiet_duration_s must be finite and non-negative",
        )
        require(
            isinstance(self.min_quiet_samples, int)
            and not isinstance(self.min_quiet_samples, bool)
            and self.min_quiet_samples >= 1,
            "min_quiet_samples must be an integer >= 1",
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a versioned JSON-safe mapping, preserving explicit nulls."""
        self._validate_full_object()
        payload: dict[str, object] = asdict(self)
        if self.variable_keys is not None:
            payload["variable_keys"] = list(self.variable_keys)
        return {"schema_version": PLOT_DEFINITION_SCHEMA_VERSION, **payload}

    @classmethod
    def from_json_dict(cls, document: object) -> PlotDefinition:
        """Strictly parse v3 or migrate one exact v1/v2 document."""
        require(isinstance(document, dict), "plot definition must be an object")
        root = cast(dict[str, object], document)
        version = _strict_integer(root.get("schema_version"), "schema_version")
        if version == 1:
            root = migrate_v1(root, {item.name for item in fields(cls)})
        elif version == 2:
            root = migrate_v2(root, {item.name for item in fields(cls)})
        else:
            require(version == PLOT_DEFINITION_SCHEMA_VERSION, "unsupported schema")
        payload = _validated_v3_payload(root)
        return cls(**cast(dict[str, Any], payload))


def _validated_v3_payload(document: dict[str, object]) -> dict[str, object]:
    """Normalize one exact v3 payload without coercive conversions."""
    field_names = {item.name for item in fields(PlotDefinition)}
    _validate_exact_fields(document, field_names | {"schema_version"})
    require(
        _strict_integer(document["schema_version"], "schema_version") == 3,
        "unsupported schema",
    )
    result_id = _strict_nullable_string(document["result_id"], "result_id")
    require(result_id is not None, "result_id is required")
    plot_type = _strict_nullable_string(document["plot_type"], "plot_type")
    require(plot_type in _PLOT_TYPES, "unknown plot_type", plot_type)
    payload: dict[str, object] = {"result_id": result_id, "plot_type": plot_type}
    for name in _NULLABLE_STRING_FIELDS:
        payload[name] = _strict_nullable_string(document[name], name)
    for name in _NULLABLE_REAL_FIELDS:
        payload[name] = _strict_nullable_real(document[name], name)
    for name in _NULLABLE_INTEGER_FIELDS:
        payload[name] = _strict_nullable_integer(document[name], name)
    payload["variable_keys"] = _strict_variable_keys(document["variable_keys"])
    payload["show_confidence_ellipsoids"] = _strict_nullable_boolean(
        document["show_confidence_ellipsoids"], "show_confidence_ellipsoids"
    )
    return payload


def write_plot_definition(definition: PlotDefinition, path: str | Path) -> None:
    """Write a reproducible plot definition as UTF-8 JSON."""
    require(
        isinstance(definition, PlotDefinition),
        "definition must be a PlotDefinition",
    )
    definition._validate_full_object()
    Path(path).write_text(
        json.dumps(definition.to_json_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_plot_definition(path: str | Path) -> PlotDefinition:
    """Read and strictly parse one UTF-8 variation plot definition."""
    return PlotDefinition.from_json_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


__all__ = [
    "PLOT_DEFINITION_SCHEMA_VERSION",
    "PlotDefinition",
    "PlotType",
    "read_plot_definition",
    "write_plot_definition",
]
