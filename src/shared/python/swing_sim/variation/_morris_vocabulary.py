"""Canonical Morris wire vocabulary and status normalization."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

EVALUATED_HIT_VALUE = "evaluated_hit"
EVALUATED_NO_IMPACT_VALUE = "evaluated_no_impact"
NUMERICAL_FAILURE_VALUE = "numerical_failure"
OUTCOMES = (
    EVALUATED_HIT_VALUE,
    EVALUATED_NO_IMPACT_VALUE,
    NUMERICAL_FAILURE_VALUE,
)
OUTPUT_KINDS = ("scalar", "state-point", "impact", "shot-outcome")


def normalize_outcomes(value: Any) -> npt.NDArray[np.str_]:
    """Normalize canonical TrialEvaluationStatus values without reverse imports."""
    raw = np.asarray(value, dtype=object)
    normalized = [str(getattr(item, "value", item)) for item in raw.ravel()]
    return cast(
        npt.NDArray[np.str_],
        np.asarray(normalized, dtype=str).reshape(raw.shape),
    )
