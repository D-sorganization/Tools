"""Versioned, JSON-safe definitions for reproducible variation plots."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
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

PLOT_DEFINITION_SCHEMA_VERSION = 2
_LEGACY_RMS_THRESHOLD_M = 0.005
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

    def __post_init__(self) -> None:
        require(bool(self.result_id.strip()), "result_id must be non-empty")
        require(
            self.selected_trial_index is None or self.selected_trial_index >= 0,
            "selected_trial_index must be non-negative",
        )
        require(
            self.camera_zoom is None or self.camera_zoom > 0,
            "camera_zoom must be greater than zero",
        )
        require(
            self.phase_end_fraction is None or 0 < self.phase_end_fraction <= 1,
            "phase_end_fraction must be in (0, 1]",
        )
        if self.plot_type == "scalar_scatter":
            require(bool(self.x_variable_key), "scatter requires x_variable_key")
            require(bool(self.y_variable_key), "scatter requires y_variable_key")
        if self.plot_type in {"swing_arc_overlay", "geometric_variability"}:
            require(bool(self.point_id), "geometric plot requires point_id")
            require(bool(self.coordinate_frame), "geometric plot requires a frame")
            self._validate_dispersion_state()
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
        return {"schema_version": PLOT_DEFINITION_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_json_dict(cls, document: object) -> PlotDefinition:
        """Strictly parse v2 or migrate one exact v1 document."""
        require(isinstance(document, dict), "plot definition must be an object")
        root = cast(dict[str, object], document)
        version = _strict_integer(root.get("schema_version"), "schema_version")
        if version == 1:
            root = _migrate_v1(root)
        else:
            require(version == PLOT_DEFINITION_SCHEMA_VERSION, "unsupported schema")
        payload = _validated_v2_payload(root)
        return cls(**cast(dict[str, Any], payload))


def _strict_integer(value: object, name: str) -> int:
    """Return one genuine JSON integer, excluding booleans."""
    require(type(value) is int, f"{name} must be an integer", value)
    return cast(int, value)


def _strict_nullable_string(value: object, name: str) -> str | None:
    """Return null or one non-empty trimmed string."""
    require(
        value is None
        or (isinstance(value, str) and bool(value) and value == value.strip()),
        f"{name} must be null or a non-empty trimmed string",
        value,
    )
    return cast(str | None, value)


def _strict_nullable_real(value: object, name: str) -> float | None:
    """Return null or one finite JSON real, excluding booleans."""
    require(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ),
        f"{name} must be null or a finite real number",
        value,
    )
    return None if value is None else float(cast(float, value))


def _strict_nullable_integer(value: object, name: str) -> int | None:
    """Return null or one genuine JSON integer."""
    require(value is None or type(value) is int, f"{name} must be null or integer")
    return cast(int | None, value)


def _strict_variable_keys(value: object) -> tuple[str, ...] | None:
    """Return null or an exact string-array tuple."""
    require(value is None or isinstance(value, list), "variable_keys must be an array")
    if value is None:
        return None
    items = cast(list[object], value)
    require(
        all(
            isinstance(item, str) and bool(item) and item == item.strip()
            for item in items
        ),
        "variable_keys must contain non-empty trimmed strings",
    )
    return tuple(cast(list[str], items))


def _validate_exact_fields(document: Mapping[str, object], expected: set[str]) -> None:
    """Reject omitted and unknown wire fields symmetrically."""
    require(set(document) == expected, "invalid plot definition fields")


def _validated_v2_payload(document: dict[str, object]) -> dict[str, object]:
    """Normalize one exact v2 payload without coercive conversions."""
    field_names = {item.name for item in fields(PlotDefinition)}
    _validate_exact_fields(document, field_names | {"schema_version"})
    require(
        _strict_integer(document["schema_version"], "schema_version") == 2,
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
    return payload


def _migrate_v1(document: dict[str, object]) -> dict[str, object]:
    """Migrate exact v1 state with declared legacy RMS defaults."""
    v2_fields = {item.name for item in fields(PlotDefinition)}
    new_fields = {
        "dispersion_metric",
        "dispersion_unit",
        "quiet_threshold",
        "confidence_level",
        "min_quiet_duration_s",
        "min_quiet_samples",
    }
    expected = (v2_fields - new_fields) | {"schema_version", "quiet_threshold_m"}
    _validate_exact_fields(document, expected)
    legacy = dict(document)
    threshold = _strict_nullable_real(
        legacy.pop("quiet_threshold_m"), "quiet_threshold_m"
    )
    require(
        threshold is None or threshold > 0,
        "quiet_threshold_m must be greater than zero",
    )
    legacy["schema_version"] = 2
    geometric = legacy.get("plot_type") in {
        "swing_arc_overlay",
        "geometric_variability",
    }
    legacy.update(
        dispersion_metric="rms-radius" if geometric else None,
        dispersion_unit="m" if geometric else None,
        quiet_threshold=(
            threshold if threshold is not None else _LEGACY_RMS_THRESHOLD_M
        )
        if geometric
        else None,
        confidence_level=None,
        min_quiet_duration_s=0.0 if geometric else None,
        min_quiet_samples=1 if geometric else None,
    )
    return legacy


def write_plot_definition(definition: PlotDefinition, path: str | Path) -> None:
    """Write a reproducible plot definition as UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(definition.to_json_dict(), indent=2),
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
