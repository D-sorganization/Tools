"""Regression tests for #3673.

``design_frequency_window`` previously guarded only ``filter_type is not None``
with an assert and then divided by ``transition_bw``. A zero ``transition_bw``
produced inf/NaN coefficients, and an unrecognized ``filter_type`` returned a
silent all-zero array. Both must now raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.fft_filter_ops import design_frequency_window


def test_zero_transition_bw_raises() -> None:
    with pytest.raises(ValueError, match="transition_bw"):
        design_frequency_window(
            filter_type="FFT Low-pass",
            freq_low=0.1,
            freq_high=0.4,
            window_shape="Rectangular",
            n_samples=64,
            transition_bw=0.0,
        )


def test_negative_transition_bw_raises() -> None:
    with pytest.raises(ValueError, match="transition_bw"):
        design_frequency_window(
            filter_type="FFT High-pass",
            freq_low=0.1,
            freq_high=0.4,
            window_shape="Rectangular",
            n_samples=64,
            transition_bw=-0.05,
        )


def test_nonpositive_n_samples_raises() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        design_frequency_window(
            filter_type="FFT Low-pass",
            freq_low=0.1,
            freq_high=0.4,
            window_shape="Rectangular",
            n_samples=0,
            transition_bw=0.05,
        )


def test_unknown_filter_type_raises() -> None:
    with pytest.raises(ValueError, match="filter_type"):
        design_frequency_window(
            filter_type="FFT Bogus",
            freq_low=0.1,
            freq_high=0.4,
            window_shape="Rectangular",
            n_samples=64,
            transition_bw=0.05,
        )


def test_valid_input_returns_finite_response() -> None:
    response = design_frequency_window(
        filter_type="FFT Low-pass",
        freq_low=0.1,
        freq_high=0.4,
        window_shape="Rectangular",
        n_samples=64,
        transition_bw=0.05,
    )
    assert response.shape == (64,)
    assert np.all(np.isfinite(response))
