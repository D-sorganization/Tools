"""Explicit schema-v1 migration for variation plot definitions."""

from __future__ import annotations

from shared.python.contracts import require

from ._plot_definition_contract import (
    _strict_nullable_real,
    _validate_exact_fields,
)
from .simulation_types import APP_FRAME_ID

_LEGACY_RMS_THRESHOLD_M = 0.005
_GEOMETRIC_PLOT_TYPES = {"swing_arc_overlay", "geometric_variability"}
_HISTORICAL_FRAME_PLOT_TYPES = {"scalar_scatter", "distribution_matrix"}


def migrate_v1(document: dict[str, object], v2_fields: set[str]) -> dict[str, object]:
    """Migrate one exact v1 document with declared legacy RMS defaults."""
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
    plot_type = legacy.get("plot_type")
    if plot_type in _HISTORICAL_FRAME_PLOT_TYPES:
        frame = legacy.get("coordinate_frame")
        require(
            frame is None or frame == APP_FRAME_ID,
            "legacy coordinate_frame is unsupported",
            frame,
        )
        legacy["coordinate_frame"] = None
    legacy["schema_version"] = 2
    geometric = plot_type in _GEOMETRIC_PLOT_TYPES
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


__all__ = ["migrate_v1"]
