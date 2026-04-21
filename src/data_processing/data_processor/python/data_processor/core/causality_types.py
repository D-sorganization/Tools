"""Causality analysis result types.

Extracted from cross_correlation.py to give causality-specific data classes
their own module, reducing the responsibility footprint of the parent module.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GrangerCausalityResult:
    """Results from Granger causality test."""

    # X causes Y
    x_causes_y: bool
    x_causes_y_pvalue: float
    x_causes_y_fstat: float

    # Y causes X
    y_causes_x: bool
    y_causes_x_pvalue: float
    y_causes_x_fstat: float

    # Optimal lags
    optimal_lag_xy: int
    optimal_lag_yx: int

    # Direction
    causal_direction: str  # 'X->Y', 'Y->X', 'bidirectional', 'none'

    # Model details
    aic_values: dict[str, float] = field(default_factory=dict)


@dataclass
class TransferEntropyResult:
    """Results from transfer entropy analysis."""

    # Transfer entropy values
    te_x_to_y: float
    te_y_to_x: float
    net_te: float  # X->Y minus Y->X

    # Significance
    te_x_to_y_pvalue: float
    te_y_to_x_pvalue: float

    # Direction
    dominant_direction: str  # 'X->Y', 'Y->X', 'none'


__all__ = [
    "GrangerCausalityResult",
    "TransferEntropyResult",
]
