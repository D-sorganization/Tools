"""Shared scientific and transport limits for complete Rate ensembles."""

from __future__ import annotations

from shared.python.contracts import require

MAX_ENSEMBLE_JSON_BYTES = 16_000_000
MAX_TRIALS = 100_000
MAX_SAMPLES = 100_000
MAX_POINTS = 256
MAX_POSITION_CELLS = 5_000_000


def require_ensemble_shape_limits(
    trial_count: int, sample_count: int, point_count: int
) -> None:
    """Require one ensemble shape to fit every scientific allocation limit."""
    require(trial_count <= MAX_TRIALS, "trial limit exceeded", trial_count)
    require(sample_count <= MAX_SAMPLES, "sample limit exceeded", sample_count)
    require(point_count <= MAX_POINTS, "point limit exceeded", point_count)
    position_cells = trial_count * sample_count * point_count * 3
    require(
        position_cells <= MAX_POSITION_CELLS,
        "position cell limit exceeded",
        position_cells,
    )


__all__ = [
    "MAX_ENSEMBLE_JSON_BYTES",
    "MAX_POINTS",
    "MAX_POSITION_CELLS",
    "MAX_SAMPLES",
    "MAX_TRIALS",
    "require_ensemble_shape_limits",
]
