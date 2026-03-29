"""Tests for WaveletDenoiser -- wavelet-based signal denoising.

Covers: config, enum types, denoise, NaN handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_processor.core.wavelet_denoising import (
    ThresholdMethod,
    ThresholdSelection,
    ThresholdingMethod,
    WaveletDenoiseConfig,
    WaveletDenoiser,
    WaveletFamily,
)


class TestWaveletEnums:
    """Test enum types."""

    def test_wavelet_families(self) -> None:
        assert WaveletFamily.HAAR.value == "haar"
        assert WaveletFamily.DAUBECHIES.value == "db"
        assert WaveletFamily.SYMLET.value == "sym"

    def test_threshold_methods(self) -> None:
        assert ThresholdMethod.SOFT.value == "soft"
        assert ThresholdMethod.HARD.value == "hard"
        assert ThresholdMethod.GARROTE.value == "garrote"

    def test_threshold_selection(self) -> None:
        assert ThresholdSelection.UNIVERSAL.value == "universal"
        assert ThresholdSelection.SURE.value == "sure"

    def test_backward_compat_alias(self) -> None:
        assert ThresholdingMethod is ThresholdMethod


class TestWaveletDenoiseConfig:
    """Test WaveletDenoiseConfig."""

    def test_default_config(self) -> None:
        cfg = WaveletDenoiseConfig()
        assert cfg.wavelet_family == WaveletFamily.DAUBECHIES
        assert cfg.wavelet_order == 4
        assert cfg.threshold_method == ThresholdMethod.SOFT

    def test_config_with_wavelet_string_haar(self) -> None:
        cfg = WaveletDenoiseConfig(wavelet="haar")
        assert cfg.wavelet_family == WaveletFamily.HAAR
        assert cfg.wavelet_order == 1

    def test_config_with_wavelet_string_db(self) -> None:
        cfg = WaveletDenoiseConfig(wavelet="db8")
        assert cfg.wavelet_family == WaveletFamily.DAUBECHIES
        assert cfg.wavelet_order == 8

    def test_config_with_wavelet_string_sym(self) -> None:
        cfg = WaveletDenoiseConfig(wavelet="sym6")
        assert cfg.wavelet_family == WaveletFamily.SYMLET
        assert cfg.wavelet_order == 6

    def test_config_kwargs(self) -> None:
        cfg = WaveletDenoiseConfig(stationary=True, noise_estimation="std")
        assert cfg.stationary is True
        assert cfg.noise_estimation == "std"


class TestWaveletDenoiser:
    """Test WaveletDenoiser.denoise."""

    @pytest.fixture()
    def noisy_signal(self) -> np.ndarray:
        """Create a sine wave with noise."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2 * np.pi, 200)
        return np.sin(t) + 0.3 * rng.standard_normal(len(t))

    def test_denoise_returns_result(self, noisy_signal: np.ndarray) -> None:
        denoiser = WaveletDenoiser()
        result = denoiser.denoise(noisy_signal)
        assert result.denoised is not None
        assert len(result.denoised) == len(noisy_signal)

    def test_denoised_has_lower_noise(self, noisy_signal: np.ndarray) -> None:
        denoiser = WaveletDenoiser()
        result = denoiser.denoise(noisy_signal)
        # The denoised signal should be smoother (lower diff variance)
        orig_var = np.var(np.diff(noisy_signal))
        den_var = np.var(np.diff(result.denoised))
        assert den_var < orig_var

    def test_denoise_preserves_length(self, noisy_signal: np.ndarray) -> None:
        denoiser = WaveletDenoiser()
        result = denoiser.denoise(noisy_signal)
        assert len(result.original) == len(noisy_signal)

    def test_denoise_none_raises(self) -> None:
        denoiser = WaveletDenoiser()
        with pytest.raises(ValueError, match="signal must be provided"):
            denoiser.denoise(None)  # type: ignore[arg-type]

    def test_denoise_with_nans_preserves_length(self) -> None:
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        denoiser = WaveletDenoiser()
        result = denoiser.denoise(signal)
        assert len(result.denoised) == len(signal)
        # Denoiser may or may not fill NaNs depending on window size
        # but it should not crash

    def test_denoise_short_signal(self) -> None:
        signal = np.array([1.0, 2.0, 3.0])
        denoiser = WaveletDenoiser()
        result = denoiser.denoise(signal)
        assert len(result.denoised) == 3

    def test_custom_config(self, noisy_signal: np.ndarray) -> None:
        cfg = WaveletDenoiseConfig(wavelet="haar", threshold_method=ThresholdMethod.HARD)
        denoiser = WaveletDenoiser(config=cfg)
        result = denoiser.denoise(noisy_signal)
        assert result.wavelet_name is not None
