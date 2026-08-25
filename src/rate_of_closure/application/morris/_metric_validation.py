"""Private numerical realizability checks for Morris wire metrics."""

from __future__ import annotations

import math
from typing import Protocol, cast

from ._response_constants import (
    IDENTITY_EPSILON_MULTIPLIER,
    MAX_SAFELY_SQUARED_METRIC,
    PRODUCER_CLAMP_MULTIPLIER,
)


class EffectValues(Protocol):
    """Structural finite-or-null effect values used by the validator."""

    @property
    def mu(self) -> float | None: ...

    @property
    def mu_star(self) -> float | None: ...

    @property
    def mu_star_standard_error(self) -> float | None: ...

    @property
    def sigma(self) -> float | None: ...


def validate_finite_metrics(
    effects: EffectValues, availability: str, sample_count: int
) -> None:
    """Enforce the producer clamp and exact sample-moment identity."""
    optional = (
        effects.mu,
        effects.mu_star,
        effects.mu_star_standard_error,
        effects.sigma,
    )
    if any(value is None for value in optional):
        return
    finite = cast(tuple[float, float, float, float], optional)
    mu, mu_star, standard_error, sigma = finite
    if any(abs(value) > MAX_SAFELY_SQUARED_METRIC for value in finite):
        raise ValueError("Morris metrics must be safely squared finite values")
    if mu_star < 0.0 or standard_error < 0.0 or sigma < 0.0:
        raise ValueError("Morris metric magnitudes must be non-negative")
    if mu_star < abs(mu):
        raise ValueError("Morris effect mu_star must be at least absolute mean effect")
    if mu_star == 0.0:
        if (mu, standard_error, sigma) != (
            0.0,
            0.0,
            0.0,
        ) or availability != "constant-output":
            raise ValueError("zero mu_star requires zero constant-output metrics")
        return
    if availability == "constant-output":
        raise ValueError("constant-output Morris effects must be zero")
    delta = PRODUCER_CLAMP_MULTIPLIER * math.ulp(1.0) * max(1.0, mu_star)
    if any(0.0 < value <= delta for value in (standard_error, sigma)):
        raise ValueError("serialized metric violates the producer zero clamp")
    difference = abs(mu_star - abs(mu))
    tolerance = IDENTITY_EPSILON_MULTIPLIER * math.ulp(1.0) * max(abs(mu), mu_star)
    if sigma == 0.0 and (standard_error != 0.0 or difference > tolerance):
        raise ValueError("zero-sigma Morris metric relationship failed")
    if (
        difference <= tolerance
        and standard_error == 0.0
        and sigma > math.sqrt(sample_count) * delta + tolerance
    ):
        raise ValueError("Morris metric clamp-scale degeneracy is inconsistent")
    if sample_count < 2:
        raise ValueError("available Morris metrics require at least two pairs")
    scale = max(abs(mu), mu_star, standard_error, sigma, delta)
    normalized_mu, normalized_star, normalized_error, normalized_sigma = (
        value / scale for value in (mu, mu_star, standard_error, sigma)
    )
    correction = sample_count / (sample_count - 1)
    residual = (
        normalized_sigma**2
        - sample_count * normalized_error**2
        - correction * normalized_star**2
        + correction * normalized_mu**2
    )
    term_scale = (
        normalized_sigma**2
        + sample_count * normalized_error**2
        + correction * normalized_star**2
        + correction * normalized_mu**2
    )
    rounding = IDENTITY_EPSILON_MULTIPLIER * math.ulp(1.0) * term_scale
    normalized_delta = delta / scale
    clamps = (normalized_delta**2 if sigma == 0.0 else 0.0) + (
        sample_count * normalized_delta**2 if standard_error == 0.0 else 0.0
    )
    if abs(residual) > rounding + clamps:
        raise ValueError("Morris metric identity is inconsistent with valid_pairs")


__all__ = ["validate_finite_metrics"]
