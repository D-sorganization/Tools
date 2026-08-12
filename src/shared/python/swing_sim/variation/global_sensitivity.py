"""Deterministic Morris screening with explicit scientific availability.

The Morris elementary-effects method screens simultaneous bounded inputs for
nonlinear or interacting influence at substantially lower cost than a Sobol
study. It does not separate nonlinearity from interaction and it does not turn
association into causation. Every estimate retains the full trajectory
denominator, including failed and no-impact pairs whose outputs remain absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from shared.python.contracts import require

from .morris_design import (
    EVALUATED_NO_IMPACT_VALUE,
    NUMERICAL_FAILURE_VALUE,
    MorrisObservations,
)

_ADEQUATE_TRAJECTORIES = 10
MORRIS_REPORT_SCHEMA_ID = "swing-sim/morris-global-sensitivity-report"
MORRIS_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MorrisEstimate:
    """One source-to-output elementary-effects estimate and denominator."""

    spec_id: str
    variable_key: str
    source_unit: str
    source_lower: float
    source_upper: float
    source_time_window_s: tuple[float, float] | None
    source_point_ids: tuple[str, ...]
    output_name: str
    output_unit: str
    coordinate_frame: str | None
    target_kind: str
    target_time_s: float | None
    target_point_id: str | None
    mu: float
    mu_star: float
    mu_star_standard_error: float
    sigma: float
    availability: str
    sample_adequacy: str
    total_effect_pairs: int
    valid_effect_pairs: int
    no_impact_pairs: int
    no_impact_unavailable_pairs: int
    failed_pairs: int
    nonfinite_pairs: int


@dataclass(frozen=True)
class MorrisReport:
    """Complete Morris screening report with method assumptions."""

    estimates: tuple[MorrisEstimate, ...]
    trajectories: int
    levels: int
    seed: int
    total_design_samples: int
    normalized_step: float
    assumptions: tuple[str, ...]
    interaction_caveat: str
    method: str = "morris-elementary-effects"

    def estimate(self, spec_id: str, output_name: str) -> MorrisEstimate:
        """Return the unique estimate for one source and output."""
        matches = tuple(
            estimate
            for estimate in self.estimates
            if estimate.spec_id == spec_id and estimate.output_name == output_name
        )
        require(
            len(matches) == 1,
            "unknown Morris source/output pair",
            (spec_id, output_name),
        )
        return matches[0]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a deterministic, non-finite-safe cross-runtime document."""
        return {
            "schema_id": MORRIS_REPORT_SCHEMA_ID,
            "schema_version": MORRIS_REPORT_SCHEMA_VERSION,
            "method": self.method,
            "design": {
                "trajectories": self.trajectories,
                "levels": self.levels,
                "seed": self.seed,
                "total_samples": self.total_design_samples,
                "normalized_step": self.normalized_step,
            },
            "assumptions": list(self.assumptions),
            "interaction_caveat": self.interaction_caveat,
            "estimates": [_estimate_to_json(item) for item in self.estimates],
        }


def _finite_or_none(value: float) -> float | None:
    """Map unavailable non-finite estimates to JSON null."""
    return value if math.isfinite(value) else None


def _estimate_to_json(estimate: MorrisEstimate) -> dict[str, Any]:
    """Serialize one estimate with explicit source, target, and denominator."""
    return {
        "source": {
            "spec_id": estimate.spec_id,
            "variable_key": estimate.variable_key,
            "unit": estimate.source_unit,
            "bounds": [estimate.source_lower, estimate.source_upper],
            "time_window_s": (
                None
                if estimate.source_time_window_s is None
                else list(estimate.source_time_window_s)
            ),
            "point_ids": list(estimate.source_point_ids),
        },
        "target": {
            "name": estimate.output_name,
            "unit": estimate.output_unit,
            "kind": estimate.target_kind,
            "time_s": estimate.target_time_s,
            "point_id": estimate.target_point_id,
            "coordinate_frame": estimate.coordinate_frame,
        },
        "effects": {
            "mu": _finite_or_none(estimate.mu),
            "mu_star": _finite_or_none(estimate.mu_star),
            "mu_star_standard_error": _finite_or_none(estimate.mu_star_standard_error),
            "sigma": _finite_or_none(estimate.sigma),
        },
        "availability": estimate.availability,
        "sample_adequacy": estimate.sample_adequacy,
        "denominator": {
            "total_pairs": estimate.total_effect_pairs,
            "valid_pairs": estimate.valid_effect_pairs,
            "typed_no_impact_pairs": estimate.no_impact_pairs,
            "no_impact_unavailable_pairs": estimate.no_impact_unavailable_pairs,
            "failed_pairs": estimate.failed_pairs,
            "nonfinite_pairs": estimate.nonfinite_pairs,
        },
    }


def _effect_arrays(
    observations: MorrisObservations,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return factor-ordered effects and exclusion masks."""
    starts = observations.values[:, :-1, :]
    ends = observations.values[:, 1:, :]
    start_status = observations.outcomes[:, :-1]
    end_status = observations.outcomes[:, 1:]
    failed = (start_status == NUMERICAL_FAILURE_VALUE) | (
        end_status == NUMERICAL_FAILURE_VALUE
    )
    no_impact = ~failed & (
        (start_status == EVALUATED_NO_IMPACT_VALUE)
        | (end_status == EVALUATED_NO_IMPACT_VALUE)
    )
    finite = np.isfinite(starts) & np.isfinite(ends)
    raw = (ends - starts) / observations.design.signed_steps[:, :, np.newaxis]
    rows = np.arange(observations.design.trajectories)[:, np.newaxis]
    changed = observations.design.changed_factor_indices
    factor_count = len(observations.design.factors)
    effects = np.empty_like(raw)
    failed_by_factor: npt.NDArray[np.bool_] = np.empty(
        (observations.design.trajectories, factor_count), dtype=np.bool_
    )
    no_impact_by_factor = np.empty_like(failed_by_factor)
    finite_by_factor = np.empty_like(raw, dtype=bool)
    effects[rows, changed, :] = raw
    failed_by_factor[rows, changed] = failed
    no_impact_by_factor[rows, changed] = no_impact
    finite_by_factor[rows, changed, :] = finite
    return effects, failed_by_factor, no_impact_by_factor, finite_by_factor


def _estimate_values(
    effects: np.ndarray, valid_count: int, minimum_effects: int
) -> tuple[float, float, float, float, str, str]:
    """Summarize available elementary effects and scientific states."""
    if valid_count < minimum_effects:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "insufficient-data",
            "insufficient",
        )
    mu = float(np.mean(effects))
    absolute_effects = np.abs(effects)
    mu_star = float(np.mean(absolute_effects))
    mu_star_standard_error = float(
        np.std(absolute_effects, ddof=1) / math.sqrt(valid_count)
    )
    sigma = float(np.std(effects, ddof=1))
    numerical_tolerance = 64.0 * np.finfo(float).eps * max(1.0, mu_star)
    if mu_star_standard_error <= numerical_tolerance:
        mu_star_standard_error = 0.0
    if sigma <= numerical_tolerance:
        sigma = 0.0
    adequacy = "adequate" if valid_count >= _ADEQUATE_TRAJECTORIES else "limited"
    availability = "constant-output" if np.all(effects == 0.0) else "available"
    return mu, mu_star, mu_star_standard_error, sigma, availability, adequacy


def _build_estimate(
    observations: MorrisObservations,
    indices: tuple[int, int],
    minimum_effects: int,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> MorrisEstimate:
    factor_index, output_index = indices
    effects, failed, no_impact, finite = arrays
    factor = observations.design.factors[factor_index]
    output = observations.outputs[output_index]
    valid, counts = _availability_masks(
        factor_index, output_index, failed, no_impact, finite
    )
    failed_count, miss_count, miss_unavailable, nonfinite_count = counts
    valid_effects = effects[valid, factor_index, output_index]
    valid_count = int(valid_effects.size)
    require(
        valid_count + failed_count + miss_unavailable + nonfinite_count
        == observations.design.trajectories,
        "Morris effect denominator invariant failed",
    )
    summary = _estimate_values(valid_effects, valid_count, minimum_effects)
    mu, mu_star, standard_error, sigma, availability, adequacy = summary
    return MorrisEstimate(
        spec_id=factor.spec_id,
        variable_key=factor.variable_key,
        source_unit=factor.unit,
        source_lower=factor.lower,
        source_upper=factor.upper,
        source_time_window_s=factor.source_time_window_s,
        source_point_ids=factor.source_point_ids,
        output_name=output.name,
        output_unit=output.unit,
        coordinate_frame=output.coordinate_frame,
        target_kind=output.target_kind,
        target_time_s=output.target_time_s,
        target_point_id=output.target_point_id,
        mu=mu,
        mu_star=mu_star,
        mu_star_standard_error=standard_error,
        sigma=sigma,
        availability=availability,
        sample_adequacy=adequacy,
        total_effect_pairs=observations.design.trajectories,
        valid_effect_pairs=valid_count,
        no_impact_pairs=miss_count,
        no_impact_unavailable_pairs=miss_unavailable,
        failed_pairs=failed_count,
        nonfinite_pairs=nonfinite_count,
    )


def _availability_masks(
    factor_index: int,
    output_index: int,
    failed: np.ndarray,
    no_impact: np.ndarray,
    finite: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Classify valid and unavailable pairs for one source/output."""
    failed_for_factor = failed[:, factor_index]
    no_impact_for_factor = no_impact[:, factor_index]
    available = finite[:, factor_index, output_index]
    no_impact_unavailable = no_impact_for_factor & ~available
    eligible = ~failed_for_factor & ~no_impact_unavailable
    valid = eligible & available
    nonfinite = ~failed_for_factor & ~no_impact_for_factor & ~available
    counts = (
        int(np.count_nonzero(failed_for_factor)),
        int(np.count_nonzero(no_impact_for_factor)),
        int(np.count_nonzero(no_impact_unavailable)),
        int(np.count_nonzero(nonfinite)),
    )
    return valid, counts


def analyze_morris(
    observations: MorrisObservations, minimum_effects: int = 4
) -> MorrisReport:
    """Calculate Morris ``mu``, ``mu*``, and ``sigma`` without inventing data.

    A pair is usable when neither endpoint failed and both output values are
    finite. Evaluated no-impact endpoints therefore contribute available
    pre-impact/state metrics, while absent impact/shot outputs remain counted
    as no-impact-unavailable. Every excluded pair stays in the denominator.
    """
    minimum_effects = _require_effect_count(minimum_effects)
    arrays = _effect_arrays(observations)
    indices = (
        (factor_index, output_index)
        for factor_index in range(len(observations.design.factors))
        for output_index in range(len(observations.outputs))
    )
    estimates = tuple(
        _build_estimate(observations, pair, minimum_effects, arrays) for pair in indices
    )
    return MorrisReport(
        estimates=estimates,
        trajectories=observations.design.trajectories,
        levels=observations.design.levels,
        seed=observations.design.seed,
        total_design_samples=(
            observations.design.trajectories * (len(observations.design.factors) + 1)
        ),
        normalized_step=(
            observations.design.levels / (2.0 * (observations.design.levels - 1))
        ),
        assumptions=(
            "Factors are independently screened over declared finite bounds.",
            "The evaluator is deterministic or its stochastic stream is controlled.",
            "Elementary effects are scaled to a full normalized factor range.",
            "Only paired finite outputs from non-failed endpoints are estimable.",
        ),
        interaction_caveat=(
            "Morris sigma is a screening indicator that conflates nonlinearity and "
            "interaction; it does not isolate either effect or establish causality."
        ),
    )


def _require_effect_count(value: object) -> int:
    """Require a true integer elementary-effect adequacy threshold."""
    require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer)),
        "minimum_effects must be an integer >= 2",
        value,
    )
    result = int(cast(int | np.integer[Any], value))
    require(result >= 2, "minimum_effects must be an integer >= 2", result)
    return result


__all__ = [
    "MORRIS_REPORT_SCHEMA_ID",
    "MORRIS_REPORT_SCHEMA_VERSION",
    "MorrisEstimate",
    "MorrisReport",
    "analyze_morris",
]
