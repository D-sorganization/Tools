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

        On **any** failure the first element is ``math.nan`` and both dicts are
        empty (issue #3976). A failure is: the engine raising, the engine
        returning a non-dict, ``output_variable`` being absent from the result,
        or its value not being coercible to ``float``. ``NaN`` -- never ``0.0``
        -- is the failure sentinel, because ``0.0`` is a legitimate objective
        value and both callers detect failure with ``np.isfinite(...)``.

    Notes
    -----
    This function does not raise on engine failure by design: the callers are a
    gradient estimator and a grid sweep that must continue past a bad point and
    apply their own penalty. The contract is therefore "explicit NaN sentinel",
    and callers **must** check finiteness before using the value.
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

    # A *missing* output key is an evaluation failure, not "the answer is zero"
    # (issue #3976). The original `result.get(output_variable, 0.0)` turned a
    # typo'd `output_variable` -- or an engine that simply does not publish that
    # key -- into a perfectly plausible objective of 0.0. In
    # `optimization._gradient_component` both perturbed evaluations then return
    # 0.0, the gradient is exactly zero, and the optimizer "converges" on
    # garbage; a multi-parameter sweep plots a flat zero surface that is
    # indistinguishable from real data. NaN is the failure sentinel both callers
    # already test for with `np.isfinite(...)`.
    if output_variable not in result:
        _logger.warning(
            "Engine result has no %r key (available: %s); returning NaN",
            output_variable,
            sorted(result),
        )
        return math.nan, {}, {}

    try:
        output_value = float(result[output_variable])
    except (TypeError, ValueError) as exc:
        # A non-numeric value under the requested key is equally a failure; it
        # must not escape as a raw TypeError from deep inside a sweep.
        _logger.warning(
            "Engine result key %r is not numeric (%s); returning NaN",
            output_variable,
            exc,
        )
        return math.nan, {}, {}

    state: dict[str, float] = result.get("state", {})
    composition: dict[str, float] = result.get("composition", {})

    return output_value, state, composition


__all__ = [
    "evaluate_output",
]
