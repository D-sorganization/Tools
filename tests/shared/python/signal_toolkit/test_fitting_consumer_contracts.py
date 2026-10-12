"""Consumer contracts required to retire the downstream Signal shadow (#5571)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from signal_toolkit.core import Signal
from signal_toolkit.fitting import (
    CustomFunctionFitter,
    ExponentialFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)

from signal_toolkit import fitting as fitting_module


@pytest.mark.parametrize(
    "fit",
    [
        lambda signal: SinusoidFitter().fit(signal),
        lambda signal: ExponentialFitter().fit_decay(signal),
        lambda signal: ExponentialFitter().fit_growth(signal),
        lambda signal: LinearFitter().fit(signal),
        lambda signal: PolynomialFitter(order=1).fit(signal),
        lambda signal: CustomFunctionFitter(lambda t, a: a * t, ["a"]).fit(signal),
    ],
    ids=["sinusoid", "decay", "growth", "linear", "polynomial", "custom"],
)
def test_empty_signal_rejected_before_fit(fit: Callable[[Signal], object]) -> None:
    """Empty time/value arrays are a caller error, never an indexing failure."""
    empty = Signal(time=np.array([]), values=np.array([]))
    with pytest.raises(ValueError, match="Signal cannot be empty"):
        fit(empty)


@pytest.mark.parametrize(
    ("fit", "guess"),
    [
        (
            lambda signal, guess: SinusoidFitter().fit(signal, initial_guess=guess),
            [1.0, 1.0, 0.0, 0.0],
        ),
        (
            lambda signal, guess: ExponentialFitter().fit_decay(
                signal, initial_guess=guess
            ),
            [1.0, 0.5, 0.0],
        ),
        (
            lambda signal, guess: ExponentialFitter().fit_growth(
                signal, initial_guess=guess
            ),
            [1.0, 0.5, 0.0],
        ),
        (
            lambda signal, guess: CustomFunctionFitter(
                lambda t, a, b: a * t + b, ["a", "b"]
            ).fit(signal, initial_guess=guess),
            [1.0, 0.0],
        ),
    ],
    ids=["sinusoid", "decay", "growth", "custom"],
)
def test_failed_fit_reuses_ndarray_guess(
    monkeypatch: pytest.MonkeyPatch,
    fit: Callable[[Signal, np.ndarray], object],
    guess: list[float],
) -> None:
    """The failure fallback must not copy an already normalized guess (#5912)."""
    time = np.linspace(0.0, 1.0, 32)
    sample = Signal(time=time, values=np.sin(2 * np.pi * time))
    initial_guess = np.asarray(guess)
    original_array = fitting_module.np.array
    original_asarray = fitting_module.np.asarray
    asarray_inputs: list[object] = []

    def fail_curve_fit(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced fit failure")

    def guarded_array(*args: object, **kwargs: object) -> np.ndarray:
        if args and args[0] is initial_guess:
            raise AssertionError("fallback copied the ndarray guess")
        return original_array(*args, **kwargs)

    def recording_asarray(*args: object, **kwargs: object) -> np.ndarray:
        if args:
            asarray_inputs.append(args[0])
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(fitting_module.optimize, "curve_fit", fail_curve_fit)
    monkeypatch.setattr(fitting_module.np, "array", guarded_array)
    monkeypatch.setattr(fitting_module.np, "asarray", recording_asarray)

    result = fit(sample, initial_guess)

    assert result.success is False
    assert any(value is initial_guess for value in asarray_inputs)
