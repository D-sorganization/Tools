"""Versioned, JSON-safe definitions for reproducible variation plots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from shared.python.contracts import require

PLOT_DEFINITION_SCHEMA_VERSION = 1
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
    quiet_threshold_m: float | None = None
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
            self.quiet_threshold_m is None or self.quiet_threshold_m > 0,
            "quiet_threshold_m must be greater than zero",
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
