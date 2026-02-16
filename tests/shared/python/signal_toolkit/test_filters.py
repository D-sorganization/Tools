"""Tests for the signal_toolkit.filters module.

Covers:
- FilterType and FilterDesign enums
- FilterSpec creation and response analysis
- FilterDesigner factory methods (Butterworth, Chebyshev, Elliptic, Bessel)
- apply_filter function
- Convenience wrapper functions
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from signal_toolkit.core import Signal
from signal_toolkit.filters import (
    FilterDesign,
    FilterDesigner,
    FilterSpec,
    FilterType,
    apply_filter,
    apply_moving_average,
    apply_savgol,
    create_butterworth_filter,
)

# ── Helper ───────────────────────────────────────────────────────────────

def _make_signal(fs: float, duration: float, values: np.ndarray) -> Signal:
    """Create a Signal from sample rate, duration, and value array."""
    t = np.arange(len(values)) / fs
    return Signal(time=t, values=values)


# ── FilterType / FilterDesign Enums ──────────────────────────────────────


class TestFilterType:
    """Verify FilterType enum values."""

    def test_lowpass_value(self) -> None:
        assert FilterType.LOWPASS.value == "lowpass"

    def test_highpass_value(self) -> None:
        assert FilterType.HIGHPASS.value == "highpass"

    def test_bandpass_value(self) -> None:
        assert FilterType.BANDPASS.value == "bandpass"

    def test_bandstop_value(self) -> None:
        assert FilterType.BANDSTOP.value == "bandstop"

    def test_notch_value(self) -> None:
        assert FilterType.NOTCH.value == "notch"


class TestFilterDesign:
    """Verify FilterDesign enum values."""

    def test_butterworth_value(self) -> None:
        assert FilterDesign.BUTTERWORTH.value == "butterworth"

    def test_chebyshev1_value(self) -> None:
        assert FilterDesign.CHEBYSHEV1.value == "chebyshev1"

    def test_elliptic_value(self) -> None:
        assert FilterDesign.ELLIPTIC.value == "elliptic"

    def test_bessel_value(self) -> None:
        assert FilterDesign.BESSEL.value == "bessel"


# ── FilterSpec ───────────────────────────────────────────────────────────


class TestFilterSpec:
    """Test FilterSpec dataclass and its methods."""

    @pytest.fixture()
    def simple_lowpass(self) -> FilterSpec:
        """Create a simple Butterworth lowpass filter for testing."""
        return FilterDesigner.butterworth(
            filter_type=FilterType.LOWPASS,
            cutoff=100.0,
            fs=1000.0,
            order=4,
        )

    def test_filter_spec_has_coefficients(self, simple_lowpass: FilterSpec) -> None:
        assert simple_lowpass.b is not None
        assert simple_lowpass.a is not None
        assert len(simple_lowpass.b) > 0
        assert len(simple_lowpass.a) > 0

    def test_filter_spec_has_metadata(self, simple_lowpass: FilterSpec) -> None:
        assert simple_lowpass.filter_type == FilterType.LOWPASS
        assert simple_lowpass.design == FilterDesign.BUTTERWORTH
        assert simple_lowpass.fs == 1000.0

    def test_frequency_response_shape(self, simple_lowpass: FilterSpec) -> None:
        freqs, mag, phase = simple_lowpass.get_frequency_response(num_points=256)
        assert len(freqs) == 256
        assert len(mag) == 256
        assert len(phase) == 256

    def test_frequency_response_passband(self, simple_lowpass: FilterSpec) -> None:
        """Passband gain should be near 0 dB (magnitude ~1.0)."""
        freqs, mag, _phase = simple_lowpass.get_frequency_response(num_points=512)
        # At DC (0 Hz), magnitude should be ~1.0
        assert_allclose(mag[0], 1.0, atol=0.01)

    def test_frequency_response_stopband(self, simple_lowpass: FilterSpec) -> None:
        """Stopband should have significant attenuation."""
        freqs, mag, _phase = simple_lowpass.get_frequency_response(num_points=512)
        high_freq_mag = mag[-1]
        assert high_freq_mag < 0.1, f"Stopband magnitude too high: {high_freq_mag}"

    def test_impulse_response_shape(self, simple_lowpass: FilterSpec) -> None:
        time, impulse = simple_lowpass.get_impulse_response(num_samples=100)
        assert len(time) == 100
        assert len(impulse) == 100

    def test_impulse_response_decays(self, simple_lowpass: FilterSpec) -> None:
        """Impulse response of a stable filter should decay."""
        _time, impulse = simple_lowpass.get_impulse_response(num_samples=200)
        assert abs(impulse[-1]) < abs(impulse[0]) or abs(impulse[-1]) < 0.01


# ── FilterDesigner Factory Methods ───────────────────────────────────────


class TestFilterDesigner:
    """Test filter design factory methods."""

    def test_butterworth_lowpass(self) -> None:
        spec = FilterDesigner.butterworth(FilterType.LOWPASS, 100.0, 1000.0, order=4)
        assert spec.filter_type == FilterType.LOWPASS
        assert spec.design == FilterDesign.BUTTERWORTH
        assert spec.order == 4

    def test_butterworth_highpass(self) -> None:
        spec = FilterDesigner.butterworth(FilterType.HIGHPASS, 200.0, 1000.0, order=3)
        assert spec.filter_type == FilterType.HIGHPASS
        freqs, mag, _ = spec.get_frequency_response(512)
        assert mag[0] < 0.01

    def test_butterworth_bandpass(self) -> None:
        spec = FilterDesigner.butterworth(
            FilterType.BANDPASS, (50.0, 150.0), 1000.0, order=3
        )
        assert spec.filter_type == FilterType.BANDPASS

    def test_butterworth_bandstop(self) -> None:
        spec = FilterDesigner.butterworth(
            FilterType.BANDSTOP, (50.0, 150.0), 1000.0, order=3
        )
        assert spec.filter_type == FilterType.BANDSTOP

    def test_chebyshev1_lowpass(self) -> None:
        spec = FilterDesigner.chebyshev1(
            FilterType.LOWPASS, 100.0, 1000.0, order=4, ripple_db=1.0
        )
        assert spec.design == FilterDesign.CHEBYSHEV1

    def test_chebyshev2_lowpass(self) -> None:
        spec = FilterDesigner.chebyshev2(
            FilterType.LOWPASS, 100.0, 1000.0, order=4, attenuation_db=40.0
        )
        assert spec.design == FilterDesign.CHEBYSHEV2

    def test_elliptic_lowpass(self) -> None:
        spec = FilterDesigner.elliptic(
            FilterType.LOWPASS, 100.0, 1000.0,
            order=4, ripple_db=1.0, attenuation_db=40.0,
        )
        assert spec.design == FilterDesign.ELLIPTIC

    def test_bessel_lowpass(self) -> None:
        spec = FilterDesigner.bessel(FilterType.LOWPASS, 100.0, 1000.0, order=4)
        assert spec.design == FilterDesign.BESSEL


# ── apply_filter ─────────────────────────────────────────────────────────


class TestApplyFilter:
    """Test the apply_filter function."""

    def test_lowpass_removes_high_frequency(self) -> None:
        """A lowpass filter should remove high-frequency components."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        # Signal: 10 Hz (passband) + 400 Hz (stopband)
        data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 400 * t)
        sig = Signal(time=t, values=data)

        filt = FilterDesigner.butterworth(FilterType.LOWPASS, 50.0, fs, order=4)
        filtered = apply_filter(sig, filt, zero_phase=True)

        rms_filtered = np.sqrt(np.mean(filtered.values**2))
        assert 0.4 < rms_filtered < 1.0

    def test_apply_filter_preserves_length(self) -> None:
        fs = 1000.0
        n = 1000
        t = np.arange(n) / fs
        sig = Signal(time=t, values=np.random.randn(n))
        filt = FilterDesigner.butterworth(FilterType.LOWPASS, 100.0, fs, order=3)
        filtered = apply_filter(sig, filt)
        assert len(filtered.values) == n

    def test_apply_filter_preserves_time(self) -> None:
        fs = 500.0
        n = 500
        t = np.arange(n) / fs
        sig = Signal(time=t, values=np.random.randn(n))
        filt = FilterDesigner.butterworth(FilterType.LOWPASS, 50.0, fs, order=2)
        filtered = apply_filter(sig, filt)
        assert_allclose(filtered.time, sig.time)


# ── Smoothing Filters ────────────────────────────────────────────────────


class TestSmoothingFilters:
    """Test smoothing filter functions."""

    def test_moving_average_reduces_noise(self) -> None:
        fs = 100.0
        t = np.arange(500) / fs
        clean = np.sin(2 * np.pi * 2.0 * t)
        noisy = clean + np.random.randn(500) * 0.3
        sig = Signal(time=t, values=noisy)

        smoothed = apply_moving_average(sig, window_size=11)
        # Smoothed should have lower noise variance
        noise_before = np.std(sig.values - clean)
        noise_after = np.std(smoothed.values - clean)
        assert noise_after < noise_before

    def test_savgol_preserves_shape(self) -> None:
        fs = 100.0
        t = np.arange(500) / fs
        sig = Signal(time=t, values=np.sin(2 * np.pi * 2.0 * t))
        filtered = apply_savgol(sig, window_length=11, polyorder=3)
        assert len(filtered.values) == 500


# ── Convenience Functions ────────────────────────────────────────────────


class TestConvenienceFilters:
    """Test convenience wrapper functions."""

    def test_create_butterworth_filter_lowpass(self) -> None:
        spec = create_butterworth_filter("lowpass", 100.0, 1000.0, order=4)
        assert spec.filter_type == FilterType.LOWPASS

    def test_create_butterworth_filter_highpass(self) -> None:
        spec = create_butterworth_filter("highpass", 200.0, 1000.0)
        assert spec.filter_type == FilterType.HIGHPASS

    def test_create_butterworth_filter_bandpass(self) -> None:
        spec = create_butterworth_filter("bandpass", (50.0, 150.0), 1000.0)
        assert spec.filter_type == FilterType.BANDPASS
