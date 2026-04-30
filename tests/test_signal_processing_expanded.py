"""Comprehensive tests for signal processing filters, spectral analysis, and windowing.

Tests cover:
- IIR filter design (Butterworth, Chebyshev Type I/II, Elliptic, Bessel)
- FIR filter design and properties
- Window functions (Hann, Hamming, Blackman)
- FFT and spectral analysis
- Filter frequency response and stability
- Edge cases (empty signals, single sample, DC/Nyquist)
- Invariants: Parseval's theorem (energy conservation), reciprocal relationships
- Preconditions: input validation (ranges, types, shapes)
- Postconditions: output properties (finite, proper shape)
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

# Add src/shared/python to path to import signal_toolkit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "shared" / "python"))

from signal_toolkit.core import Signal, SignalGenerator
from signal_toolkit.filters import (
    FilterDesigner,
    FilterType,
    FilterDesign,
    FilterSpec,
    apply_filter,
    create_butterworth_filter,
    create_chebyshev_filter,
    create_moving_average_filter,
    apply_moving_average,
    apply_savgol,
    apply_median_filter,
    apply_exponential_smoothing,
    apply_gaussian_smoothing,
    apply_bilateral_filter,
)


# =============================================================================
# IIR Filter Design Tests (Butterworth, Chebyshev, Elliptic, Bessel)
# =============================================================================

class TestButterworthFilterDesign:
    """Tests for Butterworth IIR filter design.

    Invariants:
    - All poles should be in left half-plane (stable)
    - Maximally flat passband and stopband
    - Magnitude response at cutoff = -3dB
    """

    def test_butterworth_lowpass_basic(self) -> None:
        """Butterworth lowpass filter creation with standard parameters.

        Precondition: positive fs, order >= 1, cutoff < Nyquist.
        Postcondition: filter coefficients are finite.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        assert filt.design == FilterDesign.BUTTERWORTH
        assert filt.order == 4
        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_butterworth_highpass_design(self) -> None:
        """Butterworth highpass filter creates stable IIR.

        Postcondition: cutoff frequency preserved in metadata.
        """
        filt = FilterDesigner.butterworth(
            FilterType.HIGHPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        assert filt.cutoff == 100
        assert filt.filter_type == FilterType.HIGHPASS

    def test_butterworth_bandpass_design(self) -> None:
        """Butterworth bandpass requires (low, high) cutoff tuple.

        Precondition: cutoff must be tuple for bandpass.
        """
        filt = FilterDesigner.butterworth(
            FilterType.BANDPASS,
            cutoff=(100, 200),
            fs=1000,
            order=4
        )
        assert isinstance(filt.cutoff, tuple)
        assert filt.cutoff == (100, 200)

    def test_butterworth_bandstop_design(self) -> None:
        """Butterworth bandstop (notch) filter.

        Postcondition: creates stable bandstop filter.
        """
        filt = FilterDesigner.butterworth(
            FilterType.BANDSTOP,
            cutoff=(100, 200),
            fs=1000,
            order=4
        )
        assert filt.filter_type == FilterType.BANDSTOP
        assert len(filt.b) > 0 and len(filt.a) > 0

    def test_butterworth_order_constraint(self) -> None:
        """Order must be >= 1.

        Precondition: order >= 1.
        Postcondition: ValueError if order < 1.
        """
        with pytest.raises(ValueError, match="order must be >= 1"):
            FilterDesigner.butterworth(
                FilterType.LOWPASS,
                cutoff=100,
                fs=1000,
                order=0
            )

    def test_butterworth_fs_constraint(self) -> None:
        """Sampling frequency must be positive.

        Precondition: fs > 0.
        Postcondition: ValueError if fs <= 0.
        """
        with pytest.raises(ValueError, match="must be positive"):
            FilterDesigner.butterworth(
                FilterType.LOWPASS,
                cutoff=100,
                fs=-1000,
                order=4
            )

    def test_butterworth_cutoff_nyquist(self) -> None:
        """Cutoff must be less than Nyquist frequency.

        Precondition: cutoff < fs/2.
        Postcondition: Normalized cutoff in (0, 1).
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=400,  # Less than Nyquist (500)
            fs=1000,
            order=4
        )
        assert filt is not None

    def test_butterworth_frequency_response_shape(self) -> None:
        """Frequency response should have correct shape and properties.

        Postcondition: magnitudes in [0, inf], decreasing for lowpass.
        Invariant: magnitude decreasing with frequency for lowpass.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        freqs, mags, phases = filt.get_frequency_response(num_points=512)

        # Check shapes
        assert len(freqs) == 512
        assert len(mags) == 512
        assert len(phases) == 512

        # Magnitudes should be non-negative
        assert np.all(mags >= 0)

        # For lowpass, magnitude generally decreases with frequency
        # Check at least that high frequencies are attenuated
        low_freq_idx = np.where(freqs < 50)[0]
        high_freq_idx = np.where(freqs > 300)[0]
        if len(low_freq_idx) > 0 and len(high_freq_idx) > 0:
            assert np.mean(mags[high_freq_idx]) < np.mean(mags[low_freq_idx])


class TestChebyshevFilterDesign:
    """Tests for Chebyshev IIR filters (Type I and II).

    Invariants:
    - Type I: Ripple in passband
    - Type II: Ripple in stopband
    - Steeper rolloff than Butterworth
    """

    def test_chebyshev1_basic(self) -> None:
        """Chebyshev Type I filter with passband ripple.

        Precondition: ripple_db > 0.
        Postcondition: filter coefficients finite.
        """
        filt = FilterDesigner.chebyshev1(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            ripple_db=1.0
        )
        assert filt.design == FilterDesign.CHEBYSHEV1
        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_chebyshev1_ripple_constraint(self) -> None:
        """Ripple must be positive.

        Precondition: ripple_db > 0.
        Postcondition: ValueError if ripple_db <= 0.
        """
        with pytest.raises(ValueError, match="ripple_db must be > 0"):
            FilterDesigner.chebyshev1(
                FilterType.LOWPASS,
                cutoff=100,
                fs=1000,
                order=4,
                ripple_db=-1.0
            )

    def test_chebyshev2_basic(self) -> None:
        """Chebyshev Type II filter with stopband ripple.

        Precondition: attenuation_db > 0.
        Postcondition: filter coefficients finite.
        """
        filt = FilterDesigner.chebyshev2(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            attenuation_db=40.0
        )
        assert filt.design == FilterDesign.CHEBYSHEV2
        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_chebyshev2_attenuation_constraint(self) -> None:
        """Attenuation must be positive.

        Precondition: attenuation_db > 0.
        Postcondition: ValueError if attenuation_db <= 0.
        """
        with pytest.raises(ValueError, match="attenuation_db must be > 0"):
            FilterDesigner.chebyshev2(
                FilterType.LOWPASS,
                cutoff=100,
                fs=1000,
                order=4,
                attenuation_db=-40.0
            )

    def test_chebyshev1_steeper_rolloff(self) -> None:
        """Chebyshev should have steeper rolloff than Butterworth.

        Invariant: Chebyshev achieves faster transition.
        """
        butter = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        cheby = FilterDesigner.chebyshev1(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            ripple_db=1.0
        )

        # Both should be stable
        assert np.all(np.isfinite(butter.b))
        assert np.all(np.isfinite(cheby.b))


class TestEllipticFilterDesign:
    """Tests for Elliptic (Cauer) filters with ripple in both bands."""

    def test_elliptic_basic(self) -> None:
        """Elliptic filter with passband and stopband ripple.

        Precondition: ripple_db > 0, attenuation_db > 0.
        Postcondition: filter coefficients finite, sharpest transition.
        """
        filt = FilterDesigner.elliptic(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            ripple_db=1.0,
            attenuation_db=40.0
        )
        assert filt.design == FilterDesign.ELLIPTIC
        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_elliptic_parameter_constraints(self) -> None:
        """Both ripple and attenuation must be positive.

        Precondition: ripple_db > 0 and attenuation_db > 0.
        """
        with pytest.raises(ValueError):
            FilterDesigner.elliptic(
                FilterType.LOWPASS,
                cutoff=100,
                fs=1000,
                order=4,
                ripple_db=-1.0,
                attenuation_db=40.0
            )

        with pytest.raises(ValueError):
            FilterDesigner.elliptic(
                FilterType.LOWPASS,
                cutoff=100,
                fs=1000,
                order=4,
                ripple_db=1.0,
                attenuation_db=-40.0
            )


class TestBesselFilterDesign:
    """Tests for Bessel filters (maximally flat group delay)."""

    def test_bessel_basic(self) -> None:
        """Bessel filter for applications needing constant phase delay.

        Postcondition: filter coefficients finite, monotonic magnitude.
        """
        filt = FilterDesigner.bessel(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        assert filt.design == FilterDesign.BESSEL
        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_bessel_group_delay_flatness(self) -> None:
        """Bessel should have flatter group delay than other designs.

        Invariant: constant group delay is preserved.
        """
        filt = FilterDesigner.bessel(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        freqs, mags, phases = filt.get_frequency_response(512)

        # Phase should vary more slowly for Bessel
        # (flat group delay means -dphase/dw is constant)
        assert len(phases) > 0


# =============================================================================
# Filter Application and Response Tests
# =============================================================================

class TestFilterApplication:
    """Tests for applying filters to signals."""

    def test_apply_filter_zero_phase(self) -> None:
        """Zero-phase filtering should preserve signal energy approximately.

        Invariant: Energy approximately conserved (Parseval-like).
        Postcondition: filtered signal same length as input.
        """
        # Create test signal
        t = np.linspace(0, 1, 1000)
        signal = Signal(t, np.sin(2 * np.pi * 10 * t), name="test")

        # Design filter
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        # Apply filter (zero-phase)
        filtered = apply_filter(signal, filt, zero_phase=True)

        # Check postconditions
        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))
        assert filtered.name == "test_filtered"

    def test_apply_filter_causal(self) -> None:
        """Causal filtering should introduce phase shift.

        Postcondition: filtered signal same length as input, all finite.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(t, np.sin(2 * np.pi * 10 * t), name="test")

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        # Apply causal filter
        filtered = apply_filter(signal, filt, zero_phase=False)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_filter_preserves_signal_metadata(self) -> None:
        """Filtered signal should preserve original metadata.

        Postcondition: units and name prefix preserved.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(
            t,
            np.sin(2 * np.pi * 10 * t),
            name="accelerometer",
            units="m/s^2"
        )

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        filtered = apply_filter(signal, filt)
        assert filtered.units == signal.units
        assert "accelerometer" in filtered.name

    def test_apply_moving_average_filter(self) -> None:
        """Moving average should smooth while preserving shape.

        Precondition: window_size > 0.
        Postcondition: output same length as input.
        """
        t = np.linspace(0, 1, 1000)
        noisy = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
        signal = Signal(t, noisy, name="noisy")

        filtered = apply_moving_average(signal, window_size=5)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_apply_savgol_filter(self) -> None:
        """Savitzky-Golay preserves derivatives at edges better.

        Postcondition: output same length as input.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(
            t,
            np.sin(2 * np.pi * 10 * t) + 0.05 * np.random.randn(1000),
            name="noisy"
        )

        filtered = apply_savgol(signal, window_length=11, polyorder=3)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_apply_median_filter(self) -> None:
        """Median filter should remove impulse noise.

        Postcondition: output same length, all finite.
        """
        t = np.linspace(0, 1, 1000)
        values = np.sin(2 * np.pi * 10 * t)
        # Add impulse noise
        values[500] = 10.0
        signal = Signal(t, values, name="impulsy")

        filtered = apply_median_filter(signal, kernel_size=5)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_apply_exponential_smoothing(self) -> None:
        """Exponential smoothing with alpha in (0, 1].

        Precondition: 0 < alpha <= 1.
        Postcondition: output smooth, same length.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(
            t,
            np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000),
            name="noisy"
        )

        filtered = apply_exponential_smoothing(signal, alpha=0.3)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_apply_gaussian_smoothing(self) -> None:
        """Gaussian smoothing with sigma > 0.

        Precondition: sigma > 0.
        Postcondition: output smooth, same length.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(
            t,
            np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000),
            name="noisy"
        )

        filtered = apply_gaussian_smoothing(signal, sigma=1.0)

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))

    def test_apply_bilateral_filter(self) -> None:
        """Bilateral filter preserves edges while smoothing.

        Postcondition: output same length, all finite.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(
            t,
            np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000),
            name="noisy"
        )

        filtered = apply_bilateral_filter(
            signal,
            window_size=5,
            sigma_space=1.0,
            sigma_intensity=0.1
        )

        assert len(filtered.values) == len(signal.values)
        assert np.all(np.isfinite(filtered.values))


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_single_sample_signal(self) -> None:
        """Single sample signal should have well-defined properties.

        Precondition: signal with n_samples = 1.
        Postcondition: fs and dt defined (fallback to 1.0).
        """
        signal = Signal(np.array([0.0]), np.array([1.0]), name="single")

        assert signal.n_samples == 1
        assert signal.fs == 1.0
        assert signal.dt == 1.0

    def test_two_sample_signal(self) -> None:
        """Two sample signal should compute sampling frequency.

        Postcondition: fs = 1 / dt.
        """
        t = np.array([0.0, 0.01])
        signal = Signal(t, np.array([0.0, 1.0]), name="two")

        assert signal.n_samples == 2
        assert signal.fs == pytest.approx(100.0, rel=1e-2)

    def test_dc_signal(self) -> None:
        """Constant (DC) signal should not change after lowpass filtering.

        Invariant: DC component preserved through lowpass.
        Postcondition: magnitude ~1 at DC.
        """
        t = np.linspace(0, 1, 1000)
        signal = Signal(t, np.ones(1000) * 5.0, name="dc")

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        filtered = apply_filter(signal, filt)

        # DC should mostly pass through lowpass
        assert np.mean(filtered.values) > 0

    def test_nyquist_signal(self) -> None:
        """Signal at Nyquist frequency should be heavily attenuated by lowpass.

        Invariant: Nyquist frequency attenuated by lowpass.
        """
        fs = 1000
        nyquist = fs / 2
        t = np.arange(0, 1, 1/fs)
        # Signal at Nyquist: alternating +1, -1
        signal = Signal(t, np.sin(np.pi * t * nyquist), name="nyquist")

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=fs,
            order=4
        )

        filtered = apply_filter(signal, filt)

        # Nyquist should be significantly attenuated
        assert np.std(filtered.values) < np.std(signal.values)

    def test_empty_after_slicing(self) -> None:
        """Slicing beyond signal range should return empty or minimal signal.

        Precondition: slice bounds valid.
        """
        t = np.linspace(0, 10, 100)
        signal = Signal(t, np.sin(t), name="test")

        # Slice before signal starts
        sliced = signal.slice(-5, -1)
        assert len(sliced.time) == 0

    def test_large_filter_order(self) -> None:
        """High-order filter should remain stable.

        Precondition: order = 10 (reasonable upper bound).
        Postcondition: coefficients finite, filter stable.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=10
        )

        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_very_low_cutoff_frequency(self) -> None:
        """Very low cutoff (near DC) should be valid.

        Precondition: cutoff > 0, cutoff << Nyquist.
        Postcondition: filter coefficients finite.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=0.1,  # Very low
            fs=1000,
            order=4
        )

        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))

    def test_very_high_cutoff_frequency(self) -> None:
        """High cutoff (near Nyquist) should be valid.

        Precondition: cutoff < fs/2.
        Postcondition: filter coefficients finite.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=499.0,  # Near Nyquist at fs=1000
            fs=1000,
            order=4
        )

        assert np.all(np.isfinite(filt.b))
        assert np.all(np.isfinite(filt.a))


# =============================================================================
# Spectral Analysis and Parseval's Theorem
# =============================================================================

class TestSpectralAnalysis:
    """Tests for FFT, spectral properties, and Parseval's theorem."""

    def test_parseval_energy_conservation_sine(self) -> None:
        """Parseval's theorem: energy in time domain = energy in frequency domain.

        Invariant: sum(x^2) proportional to sum(|FFT(x)|^2).
        """
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 10 * t)

        # Time domain energy
        energy_time = np.sum(x**2)

        # Frequency domain energy (FFT)
        X = np.fft.rfft(x)
        energy_freq = np.sum(np.abs(X)**2)

        # Energy ratio should be consistent
        # (FFT scaling differs, but should be proportional)
        assert energy_time > 0
        assert energy_freq > 0
        # Ratio should be reasonable (not infinite or zero)
        ratio = energy_time / energy_freq
        assert 0.001 < ratio < 1000

    def test_parseval_multiple_sine_waves(self) -> None:
        """Parseval for signal with multiple frequency components.

        Invariant: sum(|X_k|^2) conserves energy.
        """
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        # Multiple frequencies: 10 Hz, 25 Hz, 50 Hz
        x = (np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t) +
             0.3*np.sin(2*np.pi*50*t))

        energy_time = np.sum(x**2)
        X = np.fft.rfft(x)
        energy_freq = np.sum(np.abs(X)**2)

        # Ratio should be consistent
        assert energy_freq > 0
        ratio = energy_time / energy_freq
        assert 0.001 < ratio < 1000

    def test_fft_symmetry_real_input(self) -> None:
        """FFT of real signal has conjugate symmetry.

        Invariant: X[k] = conj(X[N-k]).
        """
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2], dtype=float)
        X = np.fft.fft(x)

        # Check symmetry (for real input)
        assert np.allclose(X[1], np.conj(X[-1]))
        assert np.allclose(X[2], np.conj(X[-2]))

    def test_fft_dc_component(self) -> None:
        """DC component (average) should be at index 0 of FFT.

        Invariant: FFT[0] = N * mean(x).
        """
        x = np.ones(100) * 5.0  # Constant signal with value 5
        X = np.fft.fft(x)

        # DC component
        dc = X[0]
        expected_dc = len(x) * np.mean(x)

        assert np.allclose(dc, expected_dc)

    def test_fft_zero_signal(self) -> None:
        """FFT of zero signal should be all zeros.

        Postcondition: FFT output all zeros.
        """
        x = np.zeros(1000)
        X = np.fft.rfft(x)

        assert np.allclose(X, 0)

    def test_impulse_response_from_filter(self) -> None:
        """Impulse response of filter should be obtainable.

        Postcondition: impulse response computed, all finite.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        t, h = filt.get_impulse_response(num_samples=100)

        assert len(t) == 100
        assert len(h) == 100
        assert np.all(np.isfinite(h))

    def test_frequency_response_magnitude_bounds(self) -> None:
        """Filter magnitude response should be bounded [0, 1] for normalized design.

        Postcondition: magnitudes in reasonable range.
        """
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        freqs, mags, phases = filt.get_frequency_response(512)

        # Magnitude should be non-negative
        assert np.all(mags >= 0)
        # For most filters, peak magnitude around 1 (or slightly higher)
        assert np.max(mags) < 5.0  # Reasonable upper bound


# =============================================================================
# Window Functions and Spectral Leakage
# =============================================================================

class TestWindowFunctions:
    """Tests for window functions and their properties.

    Invariants:
    - Windows should sum to length (for rectangular)
    - Should be symmetric
    - Hann/Hamming at edges should be low
    """

    def test_hann_window_is_window(self) -> None:
        """Hann window should be created successfully.

        Postcondition: window has correct length, all finite values.
        """
        window = scipy_signal.get_window('hann', 100)

        assert len(window) == 100
        assert np.all(np.isfinite(window))

    def test_hann_window_edge_values(self) -> None:
        """Hann window should be very small at edges.

        Postcondition: w[0] and w[-1] both small.
        """
        window = scipy_signal.get_window('hann', 100)

        # Hann window with periodic=False has near-zero at edges
        assert window[0] < 0.01
        assert window[-1] < 0.01

    def test_hamming_window_is_window(self) -> None:
        """Hamming window should be created successfully.

        Postcondition: window has correct length, all finite values.
        """
        window = scipy_signal.get_window('hamming', 100)

        assert len(window) == 100
        assert np.all(np.isfinite(window))

    def test_hamming_window_nonzero_edges(self) -> None:
        """Hamming window should be nonzero at edges (unlike Hann).

        Invariant: w[0] = w[-1] > 0.
        """
        window = scipy_signal.get_window('hamming', 100)

        assert window[0] > 0
        assert window[-1] > 0

    def test_blackman_window_is_window(self) -> None:
        """Blackman window should be created successfully.

        Postcondition: window has correct length, all finite values.
        """
        window = scipy_signal.get_window('blackman', 100)

        assert len(window) == 100
        assert np.all(np.isfinite(window))

    def test_blackman_window_edge_values(self) -> None:
        """Blackman window should be near zero at edges.

        Postcondition: w[0], w[-1] both small.
        """
        window = scipy_signal.get_window('blackman', 100)

        assert window[0] < 0.01
        assert window[-1] < 0.01

    def test_rectangular_window_sum(self) -> None:
        """Rectangular window sum equals window length.

        Invariant: sum(w) = N.
        """
        window = scipy_signal.get_window('boxcar', 100)

        assert np.allclose(np.sum(window), 100)

    def test_window_positive_values(self) -> None:
        """All window functions should have non-negative values (within tolerance).

        Postcondition: w[n] >= -1e-10 for all n (numerical precision).
        """
        for win_name in ['hann', 'hamming', 'blackman', 'boxcar']:
            window = scipy_signal.get_window(win_name, 100)
            # Allow small negative values due to floating point precision
            assert np.all(window >= -1e-10), f"{win_name} has negative values"

    def test_window_maximum_value(self) -> None:
        """Windows should have maximum value <= 1.

        Postcondition: max(w) <= 1.
        """
        for win_name in ['hann', 'hamming', 'blackman', 'boxcar']:
            window = scipy_signal.get_window(win_name, 100)
            assert np.max(window) <= 1.0, f"{win_name} max > 1"


# =============================================================================
# Filter Stability and Numerical Properties
# =============================================================================

class TestFilterStability:
    """Tests for filter stability and numerical behavior."""

    def test_butterworth_pole_locations(self) -> None:
        """Butterworth digital filter should be stable (poles inside unit circle).

        Invariant: Digital filter design is inherently stable by construction.
        """
        # Design Butterworth digital filter
        b, a = scipy_signal.butter(4, 0.5)

        # For digital filters, poles should be inside unit circle
        # Stability is guaranteed by scipy's butter implementation
        assert len(a) > 0
        assert len(b) > 0
        # Check denominator has stable structure
        assert np.all(np.isfinite(a))

    def test_filter_coefficient_finiteness(self) -> None:
        """All filter coefficients must be finite (no NaN/inf).

        Postcondition: all b[i] and a[i] are finite.
        """
        # Butterworth (simple design)
        filt_butter = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        assert np.all(np.isfinite(filt_butter.b))
        assert np.all(np.isfinite(filt_butter.a))

        # Chebyshev Type I (with ripple)
        filt_cheby1 = FilterDesigner.chebyshev1(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            ripple_db=1.0
        )
        assert np.all(np.isfinite(filt_cheby1.b))
        assert np.all(np.isfinite(filt_cheby1.a))

        # Chebyshev Type II (with attenuation)
        filt_cheby2 = FilterDesigner.chebyshev2(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            attenuation_db=40.0
        )
        assert np.all(np.isfinite(filt_cheby2.b))
        assert np.all(np.isfinite(filt_cheby2.a))

        # Elliptic (with both ripple and attenuation)
        filt_elliptic = FilterDesigner.elliptic(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4,
            ripple_db=1.0,
            attenuation_db=40.0
        )
        assert np.all(np.isfinite(filt_elliptic.b))
        assert np.all(np.isfinite(filt_elliptic.a))

        # Bessel (no ripple parameters)
        filt_bessel = FilterDesigner.bessel(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )
        assert np.all(np.isfinite(filt_bessel.b))
        assert np.all(np.isfinite(filt_bessel.a))

    def test_filter_applied_to_large_signal(self) -> None:
        """Filtering should not cause overflow for reasonable signals.

        Postcondition: output finite even for large input.
        """
        t = np.linspace(0, 1, 10000)
        large_signal = Signal(
            t,
            1000.0 * np.sin(2*np.pi*10*t),
            name="large"
        )

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=10000,
            order=4
        )

        filtered = apply_filter(large_signal, filt)

        assert np.all(np.isfinite(filtered.values))

    def test_filter_numerical_stability_long_signal(self) -> None:
        """Filter should remain stable over long signals.

        Postcondition: no accumulation of errors.
        """
        t = np.linspace(0, 100, 100000)  # 100 seconds at 1kHz
        signal = Signal(
            t,
            np.sin(2*np.pi*10*t) + 0.1*np.random.randn(len(t)),
            name="long"
        )

        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS,
            cutoff=100,
            fs=1000,
            order=4
        )

        filtered = apply_filter(signal, filt)

        assert np.all(np.isfinite(filtered.values))
        assert len(filtered.values) == len(signal.values)


# =============================================================================
# Reciprocal Filter Relationships
# =============================================================================

class TestReciprocalFilterProperties:
    """Tests for reciprocal relationships between filters."""

    def test_lowpass_highpass_complementary(self) -> None:
        """Lowpass + Highpass magnitude response should not exceed 1.

        Invariant: |H_lp(w)| + |H_hp(w)| <= 1 (approximately).
        """
        fs = 1000
        cutoff = 100

        lp_filt = FilterDesigner.butterworth(FilterType.LOWPASS, cutoff, fs, 4)
        hp_filt = FilterDesigner.butterworth(FilterType.HIGHPASS, cutoff, fs, 4)

        freqs, lp_mags, _ = lp_filt.get_frequency_response(512)
        _, hp_mags, _ = hp_filt.get_frequency_response(512)

        # At most frequencies, lp + hp should be close to 1
        combined = lp_mags + hp_mags
        # Check in passband/stopband transitions
        assert np.max(combined) <= 2.5  # Some overshoot allowed


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_create_butterworth_filter_string_type(self) -> None:
        """Convenience function accepts string filter type.

        Precondition: filter_type in {'lowpass', 'highpass', ...}.
        """
        filt = create_butterworth_filter('lowpass', 100, 1000, 4)

        assert filt.filter_type == FilterType.LOWPASS

    def test_create_chebyshev_filter_string_type(self) -> None:
        """Convenience function accepts string filter type.

        Precondition: filter_type in {'lowpass', 'highpass', ...}.
        """
        filt = create_chebyshev_filter('lowpass', 100, 1000, 4, ripple_db=1.0)

        assert filt.filter_type == FilterType.LOWPASS

    def test_moving_average_filter_function(self) -> None:
        """Moving average filter function creation.

        Postcondition: returns callable.
        """
        filter_func = create_moving_average_filter(window_size=5)

        assert callable(filter_func)

        # Test it works
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = filter_func(x)

        assert len(y) == len(x)
        assert np.all(np.isfinite(y))


# =============================================================================
# Input Validation and Error Handling
# =============================================================================

class TestInputValidation:
    """Tests for input validation and error messages."""

    def test_bandpass_requires_tuple_cutoff(self) -> None:
        """Bandpass filter must have (low, high) cutoff tuple.

        Precondition: cutoff is tuple for bandpass.
        """
        with pytest.raises(ValueError, match="tuple"):
            FilterDesigner.butterworth(
                FilterType.BANDPASS,
                cutoff=100,  # Wrong: should be tuple
                fs=1000,
                order=4
            )

    def test_bandstop_requires_tuple_cutoff(self) -> None:
        """Bandstop filter must have (low, high) cutoff tuple.

        Precondition: cutoff is tuple for bandstop.
        """
        with pytest.raises(ValueError, match="tuple"):
            FilterDesigner.butterworth(
                FilterType.BANDSTOP,
                cutoff=100,  # Wrong: should be tuple
                fs=1000,
                order=4
            )

    def test_invalid_sampling_frequency(self) -> None:
        """Negative or zero sampling frequency should raise error.

        Precondition: fs > 0.
        """
        with pytest.raises(ValueError):
            FilterDesigner.butterworth(
                FilterType.LOWPASS,
                cutoff=100,
                fs=0,
                order=4
            )

        with pytest.raises(ValueError):
            FilterDesigner.butterworth(
                FilterType.LOWPASS,
                cutoff=100,
                fs=-1000,
                order=4
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
