"""Regression tests for signal generator preconditions."""

from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("numpy")
import numpy as np
from signal_toolkit.core import SignalGenerator


def test_chirp_requires_at_least_two_time_points() -> None:
    with pytest.raises(ValueError, match="at least two time points"):
        SignalGenerator.chirp(np.array([0.0]))


@pytest.mark.parametrize(
    "factory",
    [
        SignalGenerator.sawtooth,
        SignalGenerator.triangle,
        SignalGenerator.square,
    ],
)
def test_periodic_generators_reject_zero_frequency(
    factory: Callable[..., object],
) -> None:
    with pytest.raises(ValueError, match="frequency must be positive"):
        factory(np.linspace(0.0, 1.0, 5), frequency=0.0)
