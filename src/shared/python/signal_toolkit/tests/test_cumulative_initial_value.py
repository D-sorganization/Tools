"""Cumulative integral honours a nonzero ``initial_value`` offset.

Expected values come from the closed form: the running integral of y = 1 over
t in [0, 4] is t, so with an initial value C it is C + t.
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.calculus import Integrator
from signal_toolkit.core import Signal


def _unit_signal() -> Signal:
    t = np.linspace(0.0, 4.0, 5)
    return Signal(time=t, values=np.ones_like(t), name="one")


def test_cumulative_integral_offsets_every_sample_by_initial_value() -> None:
    result = Integrator().cumulative_integral(_unit_signal(), initial_value=10.0)
    np.testing.assert_allclose(result.values, [10.0, 11.0, 12.0, 13.0, 14.0])


def test_cumulative_integral_default_starts_at_zero() -> None:
    result = Integrator().cumulative_integral(_unit_signal())
    np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_integrate_cumulative_signal_offsets_by_initial_value() -> None:
    result = Integrator().integrate(_unit_signal(), initial_value=-2.5)
    np.testing.assert_allclose(
        result.cumulative_signal.values, [-2.5, -1.5, -0.5, 0.5, 1.5]
    )
    # The definite integral itself is independent of the offset.
    assert result.value == pytest.approx(4.0)


def test_nonfinite_initial_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="initial_value"):
        Integrator().cumulative_integral(_unit_signal(), initial_value=float("nan"))
