"""Versioned, JSON-safe definitions for reproducible variation plots."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LOW_VARIABILITY_METRICS,
)
from shared.python.swing_sim.variation.dispersion_metric_types import (
    validated_confidence_level,
)

PLOT_DEFINITION_SCHEMA_VERSION = 2
PlotType = Literal[
    "scalar_scatter",
    "swing_arc_overlay",
    "geometric_variability",
    "distribution_matrix",
]


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


def write_plot_definition(definition: PlotDefinition, path: str | Path) -> None:
    """Write a reproducible plot definition as UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(definition.to_json_dict(), indent=2),
        encoding="utf-8",
    )


__all__ = [
    "PLOT_DEFINITION_SCHEMA_VERSION",
    "PlotDefinition",
    "PlotType",
    "write_plot_definition",
]
