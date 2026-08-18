"""Deterministic low-discrepancy capability candidate and ensemble sampling."""

from __future__ import annotations

import math

import numpy as np

from .capability_profile import ClubCapability

_FACTOR_TOLERANCE = 1e-9


def _is_prime(candidate: int) -> bool:
    divisor = 2
    while divisor * divisor <= candidate:
        if candidate % divisor == 0:
            return False
        divisor += 1
    return candidate >= 2


def _primes(count: int) -> tuple[int, ...]:
    result: list[int] = []
    candidate = 2
    while len(result) < count:
        if _is_prime(candidate):
            result.append(candidate)
        candidate += 1
    return tuple(result)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        result += factor * (index % base)
        index //= base
        factor /= base
    return result


def sample_candidate_parameters(
    club: ClubCapability, index: int, seed: int
) -> dict[str, float]:
    """Return one continuous candidate inside the club's safe envelope."""
    if index == 0:
        return {item.parameter_id: item.baseline for item in club.parameters}
    sequence_index = index + seed
    return {
        item.parameter_id: item.lower_bound
        + _radical_inverse(sequence_index, base) * (item.upper_bound - item.lower_bound)
        for item, base in zip(
            club.parameters, _primes(len(club.parameters)), strict=True
        )
    }


def _covariance_factor(club: ClubCapability) -> np.ndarray:
    covariance: np.ndarray = np.asarray(club.covariance_matrix(), dtype=float)
    size = covariance.shape[0]
    factor: np.ndarray = np.zeros_like(covariance)
    for row in range(size):
        for column in range(row + 1):
            residual = covariance[row, column] - float(
                factor[row, :column] @ factor[column, :column]
            )
            if row == column:
                factor[row, column] = math.sqrt(max(0.0, residual))
            elif factor[column, column] > _FACTOR_TOLERANCE:
                factor[row, column] = residual / factor[column, column]
    return factor


def sample_perturbed_parameters(
    club: ClubCapability, nominal: dict[str, float], sample_index: int, seed: int
) -> dict[str, float]:
    """Return one biased, correlated, safety-clipped ensemble delivery."""
    bases = _primes(len(club.parameters))
    sequence_index = sample_index + seed + 1
    independent = np.asarray(
        [
            math.sqrt(3.0) * (2.0 * _radical_inverse(sequence_index, base) - 1.0)
            for base in bases
        ]
    )
    correlated = _covariance_factor(club) @ independent
    return {
        item.parameter_id: min(
            item.upper_bound,
            max(
                item.lower_bound,
                nominal[item.parameter_id] + item.bias + correlated[index],
            ),
        )
        for index, item in enumerate(club.parameters)
    }


__all__ = ["sample_candidate_parameters", "sample_perturbed_parameters"]
