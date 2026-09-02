"""Analysis utility functions shared by optimization and multi-parameter sweeps.

This module provides the ``evaluate_output`` helper that both
``optimization.py`` and ``multi_param_analysis.py`` depend on.
"""

from __future__ import annotations

import logging
import math
from typing import Any

_logger = logging.getLogger(__name__)


def evaluate_output(
    engine: Any,
    base_params: dict[str, float],
    manual_hhv: float,
    output_variable: str,
    overrides: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Run the engine with merged parameters and extract a named output.

    Parameters
    ----------
    engine:
        Calculation engine exposing a ``calculate(**params)`` method that
        returns a JSON-serialisable dictionary.
    base_params:
        Baseline parameter dictionary.
    manual_hhv:
        Higher-heating-value override supplied by the user.
    output_variable:
        Key to extract from the engine result dictionary.
    overrides:
        Parameter overrides applied on top of *base_params*.

    Returns
    -------
    tuple[float, dict, dict]
        ``(output_value, state_dict, composition_dict)`` where
        *state_dict* and *composition_dict* are sub-dicts from the engine
        result (empty dicts if not present).
    """
    if base_params is None:
        raise ValueError("base_params must be provided")
    params = {**base_params}
    if overrides:
        params.update(overrides)

    # Inject HHV if the engine expects it
    if manual_hhv > 0:
        params["manual_hhv"] = manual_hhv

    try:
        result = engine.calculate(**params)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        # NaN, not 0.0 (issue #3976): both callers (optimization.py's
        # gradient estimator and objective evaluator, multi_param_analysis's
        # grid sweep) already check `np.isfinite(...)` / rely on NaN to
        # detect a failed evaluation and apply their own fallback/penalty.
        # Returning 0.0 silently masqueraded a failure as "the answer is
        # exactly zero", bypassing that existing handling.
        _logger.warning("Engine calculation failed: %s", exc)
        return math.nan, {}, {}

    if not isinstance(result, dict):
        return math.nan, {}, {}

    output_value = float(result.get(output_variable, 0.0))

    state: dict[str, float] = result.get("state", {})
    composition: dict[str, float] = result.get("composition", {})

    return output_value, state, composition


__all__ = [
    "evaluate_output",
]
