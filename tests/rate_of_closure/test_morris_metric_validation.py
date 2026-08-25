"""Tests for Morris metric invariant validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from rate_of_closure.application.morris._metric_validation import (
    validate_finite_metrics,
)
from rate_of_closure.application.morris._response_constants import (
    MAX_SAFELY_SQUARED_METRIC,
    PRODUCER_CLAMP_MULTIPLIER,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@dataclass
class _EffectValues:
    mu: float | None
    mu_star: float | None
    mu_star_standard_error: float | None
    sigma: float | None


def _make_valid_effects(
    valid_pairs: int, scale: float = 1.0
) -> tuple[_EffectValues, str, int]:
    mu = 0.5 * scale
    mu_star = 1.5 * scale
    sigma = scale * math.sqrt(valid_pairs * 2.25 / (valid_pairs - 1))
    standard_error = scale * 0.5 / math.sqrt(valid_pairs - 1)
    effects = _EffectValues(
        mu=mu,
        mu_star=mu_star,
        mu_star_standard_error=standard_error,
        sigma=sigma,
    )
    return effects, "available", valid_pairs


@pytest.mark.parametrize(
    ("valid_pairs", "scale"),
    [
        (4, 1.0),
        (12, 1.0),
        (12, 1e6),
        (100, 0.01),
    ],
)
def test_validate_finite_metrics_accepts_valid_metric_identities(
    valid_pairs: int, scale: float
) -> None:
    effects, availability, count = _make_valid_effects(valid_pairs, scale)
    validate_finite_metrics(effects, availability, count)


def test_validate_finite_metrics_ignores_partially_or_wholly_null_effects() -> None:
    effects = _EffectValues(None, None, None, None)
    validate_finite_metrics(effects, "insufficient-data", 0)

    partial = _EffectValues(1.0, None, 0.5, 0.2)
    validate_finite_metrics(partial, "available", 4)


def test_validate_finite_metrics_rejects_magnitude_exceeding_safe_square() -> None:
    effects = _EffectValues(
        mu=MAX_SAFELY_SQUARED_METRIC * 1.1,
        mu_star=MAX_SAFELY_SQUARED_METRIC * 1.1,
        mu_star_standard_error=0.0,
        sigma=0.0,
    )
    with pytest.raises(ValueError, match="safely squared"):
        validate_finite_metrics(effects, "available", 4)


@pytest.mark.parametrize(
    ("mu", "mu_star", "standard_error", "sigma"),
    [
        (5.0, 2.0, 0.1, 0.2),  # mu_star < abs(mu)
        (-5.0, 2.0, 0.1, 0.2),  # mu_star < abs(mu)
        (1.0, -0.5, 0.1, 0.2),  # negative mu_star
        (1.0, 1.5, -0.1, 0.2),  # negative standard error
        (1.0, 1.5, 0.1, -0.2),  # negative sigma
    ],
)
def test_validate_finite_metrics_rejects_negative_or_sub_mean_magnitudes(
    mu: float, mu_star: float, standard_error: float, sigma: float
) -> None:
    effects = _EffectValues(mu, mu_star, standard_error, sigma)
    pattern = "(non-negative|at least absolute mean|inconsistent)"
    with pytest.raises(ValueError, match=pattern):
        validate_finite_metrics(effects, "available", 4)


def test_validate_finite_metrics_constant_output_semantics() -> None:
    zero_effects = _EffectValues(0.0, 0.0, 0.0, 0.0)
    validate_finite_metrics(zero_effects, "constant-output", 4)

    with pytest.raises(ValueError, match="constant-output"):
        validate_finite_metrics(zero_effects, "available", 4)

    nonzero_effects = _EffectValues(1.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="constant-output"):
        validate_finite_metrics(nonzero_effects, "constant-output", 4)


def test_validate_finite_metrics_rejects_producer_clamp_violations() -> None:
    mu_star = 2.0
    delta = PRODUCER_CLAMP_MULTIPLIER * math.ulp(1.0) * max(1.0, mu_star)
    clamped_val = delta / 2.0

    effects_sigma_clamped = _EffectValues(
        mu=mu_star,
        mu_star=mu_star,
        mu_star_standard_error=0.0,
        sigma=clamped_val,
    )
    with pytest.raises(ValueError, match="producer zero clamp"):
        validate_finite_metrics(effects_sigma_clamped, "available", 4)

    effects_se_clamped = _EffectValues(
        mu=mu_star,
        mu_star=mu_star,
        mu_star_standard_error=clamped_val,
        sigma=0.0,
    )
    with pytest.raises(ValueError, match="producer zero clamp"):
        validate_finite_metrics(effects_se_clamped, "available", 4)


def test_validate_finite_metrics_zero_sigma_relationships() -> None:
    # Zero sigma with nonzero standard error is impossible
    effects = _EffectValues(
        mu=2.0,
        mu_star=2.0,
        mu_star_standard_error=0.5,
        sigma=0.0,
    )
    with pytest.raises(ValueError, match="zero-sigma"):
        validate_finite_metrics(effects, "available", 4)

    # Zero sigma with mu_star != abs(mu) is impossible
    effects_mismatch = _EffectValues(
        mu=1.0,
        mu_star=2.0,
        mu_star_standard_error=0.0,
        sigma=0.0,
    )
    with pytest.raises(ValueError, match="zero-sigma"):
        validate_finite_metrics(effects_mismatch, "available", 4)


def test_validate_finite_metrics_rejects_clamp_scale_degeneracy() -> None:
    # mu_star == abs(mu), standard_error == 0, but sigma is large (> sqrt(n)*delta)
    effects = _EffectValues(
        mu=2.0,
        mu_star=2.0,
        mu_star_standard_error=0.0,
        sigma=1e-8,
    )
    with pytest.raises(ValueError, match="clamp-scale degeneracy"):
        validate_finite_metrics(effects, "available", 4)


def test_validate_finite_metrics_rejects_sample_moment_identity_inconsistency() -> None:
    effects = _EffectValues(
        mu=0.5,
        mu_star=1.5,
        mu_star_standard_error=0.5 / math.sqrt(3),
        sigma=0.25,  # Inconsistent with mu, mu_star, standard_error
    )
    with pytest.raises(ValueError, match="identity is inconsistent"):
        validate_finite_metrics(effects, "available", 4)


def test_validate_finite_metrics_requires_at_least_two_samples() -> None:
    effects = _EffectValues(
        mu=1.0,
        mu_star=1.0,
        mu_star_standard_error=0.0,
        sigma=0.0,
    )
    with pytest.raises(ValueError, match="at least two pairs"):
        validate_finite_metrics(effects, "available", 1)
