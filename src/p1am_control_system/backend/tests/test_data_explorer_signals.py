"""Tests for the pure-numpy signal kernels (data_explorer_signals)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_explorer_signals import (  # noqa: E402
    apply_filter,
    differentiate,
    exponential_smoothing,
    fft_filter,
    gaussian_filter,
    hampel_filter,
    integrate,
    median_filter,
    moving_average,
    resample_series,
    savgol_filter,
    zscore_filter,
)


# --------------------------------------------------------------------------- #
# moving_average                                                              #
# --------------------------------------------------------------------------- #
def test_moving_average_window1_identity() -> None:
    y = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(moving_average(y, 1), y)


def test_moving_average_hand_computed_window3() -> None:
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = moving_average(y, 3)
    # edges shrink: [mean(1,2), mean(1,2,3), mean(2,3,4), mean(3,4,5), mean(4,5)]
    np.testing.assert_allclose(out, [1.5, 2.0, 3.0, 4.0, 4.5])


def test_moving_average_constant_preserved() -> None:
    y = np.full(10, 7.0)
    np.testing.assert_allclose(moving_average(y, 5), y)


# --------------------------------------------------------------------------- #
# exponential_smoothing                                                       #
# --------------------------------------------------------------------------- #
def test_exponential_recursion_matches_manual() -> None:
    y = [1.0, 2.0, 3.0]
    a = 0.5
    out = exponential_smoothing(y, a)
    s0 = 1.0
    s1 = a * 2.0 + (1 - a) * s0
    s2 = a * 3.0 + (1 - a) * s1
    np.testing.assert_allclose(out, [s0, s1, s2])


def test_exponential_alpha_one_is_identity() -> None:
    y = [3.0, 1.0, 4.0]
    np.testing.assert_allclose(exponential_smoothing(y, 1.0), y)


# --------------------------------------------------------------------------- #
# median_filter                                                               #
# --------------------------------------------------------------------------- #
def test_median_removes_single_spike() -> None:
    y = [1.0, 1.0, 100.0, 1.0, 1.0]
    out = median_filter(y, 3)
    assert out[2] == 1.0
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0, 1.0])


# --------------------------------------------------------------------------- #
# gaussian_filter                                                             #
# --------------------------------------------------------------------------- #
def test_gaussian_preserves_constant() -> None:
    y = np.full(20, 5.0)
    np.testing.assert_allclose(gaussian_filter(y, 2.0), y, atol=1e-12)


def test_gaussian_reduces_variance() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 500)
    out = gaussian_filter(y, 3.0)
    assert out.var() < y.var()


# --------------------------------------------------------------------------- #
# savgol_filter                                                               #
# --------------------------------------------------------------------------- #
def test_savgol_preserves_quadratic_exactly() -> None:
    x = np.arange(21, dtype=float)
    y = 2.0 * x**2 - 3.0 * x + 1.0
    out = savgol_filter(y, 7, 2)
    np.testing.assert_allclose(out, y, atol=1e-8)


def test_savgol_smooths_noise() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 200)
    y = x + rng.normal(0, 0.5, x.size)
    out = savgol_filter(y, 11, 2)
    assert np.std(out - x) < np.std(y - x)


# --------------------------------------------------------------------------- #
# hampel_filter                                                               #
# --------------------------------------------------------------------------- #
def test_hampel_replaces_injected_outlier() -> None:
    y = np.ones(11)
    y[5] = 50.0
    out = hampel_filter(y, 5, 3.0)
    assert out[5] == pytest.approx(1.0)
    np.testing.assert_allclose(out, np.ones(11))


# --------------------------------------------------------------------------- #
# zscore_filter                                                               #
# --------------------------------------------------------------------------- #
def test_zscore_interpolates_spike() -> None:
    y = np.ones(11)
    y[5] = 100.0
    out = zscore_filter(y, 2.0)
    assert out[5] == pytest.approx(1.0)


def test_zscore_constant_unchanged() -> None:
    y = np.full(8, 3.0)
    np.testing.assert_allclose(zscore_filter(y, 2.0), y)


# --------------------------------------------------------------------------- #
# fft_filter                                                                  #
# --------------------------------------------------------------------------- #
def test_fft_lowpass_removes_high_keeps_low() -> None:
    sr = 100.0
    t = np.arange(0, 4, 1.0 / sr)
    low = np.sin(2 * np.pi * 2.0 * t)
    high = np.sin(2 * np.pi * 30.0 * t)
    out = fft_filter(low + high, sr, None, 10.0)
    # The high-frequency component is removed; result tracks the low sinusoid.
    np.testing.assert_allclose(out, low, atol=0.05)


def test_fft_highpass_removes_low() -> None:
    sr = 100.0
    t = np.arange(0, 4, 1.0 / sr)
    low = np.sin(2 * np.pi * 2.0 * t)
    high = np.sin(2 * np.pi * 30.0 * t)
    out = fft_filter(low + high, sr, 10.0, None)
    np.testing.assert_allclose(out, high, atol=0.05)


def test_fft_bandpass_keeps_middle() -> None:
    sr = 200.0
    t = np.arange(0, 4, 1.0 / sr)
    mid = np.sin(2 * np.pi * 20.0 * t)
    sig = np.sin(2 * np.pi * 2.0 * t) + mid + np.sin(2 * np.pi * 80.0 * t)
    out = fft_filter(sig, sr, 10.0, 40.0)
    np.testing.assert_allclose(out, mid, atol=0.05)


def test_fft_low_ge_high_rejected() -> None:
    with pytest.raises(ValueError):
        fft_filter([1.0, 2.0, 3.0, 4.0], 10.0, 5.0, 5.0)


# --------------------------------------------------------------------------- #
# integrate / differentiate                                                   #
# --------------------------------------------------------------------------- #
def test_integrate_constant_slope_is_linear() -> None:
    # y = 2 (constant), integral over unit spacing -> 0,2,4,6,...
    y = np.full(5, 2.0)
    out = integrate(y)
    np.testing.assert_allclose(out, [0.0, 2.0, 4.0, 6.0, 8.0])


def test_integrate_with_initial_and_x() -> None:
    y = np.array([1.0, 1.0, 1.0])
    x = np.array([0.0, 2.0, 4.0])
    out = integrate(y, x, initial=10.0)
    np.testing.assert_allclose(out, [10.0, 12.0, 14.0])


def test_differentiate_of_linear_is_constant() -> None:
    x = np.arange(10, dtype=float)
    y = 3.0 * x + 1.0
    out = differentiate(y)
    np.testing.assert_allclose(out, np.full(10, 3.0))


def test_differentiate_with_x() -> None:
    x = np.array([0.0, 2.0, 4.0, 6.0])
    y = 5.0 * x
    out = differentiate(y, x)
    np.testing.assert_allclose(out, np.full(4, 5.0))


# --------------------------------------------------------------------------- #
# resample_series                                                             #
# --------------------------------------------------------------------------- #
def test_resample_mean_known_grid() -> None:
    t = np.array([0.0, 0.4, 1.0, 1.5, 2.2])
    y = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    centers, vals = resample_series(t, y, 1.0, "mean", interpolate=False)
    # bins: [0,1)->{1,3}=2 ; [1,2)->{10,20}=15 ; [2,3)->{100}=100
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(vals, [2.0, 15.0, 100.0])


def test_resample_last_and_sum() -> None:
    t = np.array([0.0, 0.4, 1.0, 1.5])
    y = np.array([1.0, 3.0, 10.0, 20.0])
    _, last = resample_series(t, y, 1.0, "last", interpolate=False)
    np.testing.assert_allclose(last, [3.0, 20.0])
    _, total = resample_series(t, y, 1.0, "sum", interpolate=False)
    np.testing.assert_allclose(total, [4.0, 30.0])


def test_resample_interpolates_empty_bin() -> None:
    # Bin edges start at t.min()=0.5: bins [0.5,1.5),[1.5,2.5),[2.5,3.5).
    # Samples land in bin0 and bin2; the middle bin is empty -> interpolated.
    t = np.array([0.5, 2.5])
    y = np.array([0.0, 20.0])
    centers, vals = resample_series(t, y, 1.0, "mean", interpolate=True)
    np.testing.assert_allclose(centers, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vals, [0.0, 10.0, 20.0])


def test_resample_drops_empty_bin_when_no_interp() -> None:
    t = np.array([0.5, 2.5])
    y = np.array([0.0, 20.0])
    centers, vals = resample_series(t, y, 1.0, "mean", interpolate=False)
    np.testing.assert_allclose(centers, [1.0, 3.0])
    np.testing.assert_allclose(vals, [0.0, 20.0])


# --------------------------------------------------------------------------- #
# apply_filter dispatch                                                       #
# --------------------------------------------------------------------------- #
def test_apply_filter_moving_average() -> None:
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = apply_filter(y, "moving_average", {"window": 3})
    np.testing.assert_allclose(out, moving_average(y, 3))


def test_apply_filter_integrate_uses_t() -> None:
    y = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 2.0, 4.0])
    out = apply_filter(y, "integrate", {"initial": 0.0}, t=t)
    np.testing.assert_allclose(out, [0.0, 2.0, 4.0])


def test_apply_filter_fft_lowpass_sample_rate_from_t() -> None:
    sr = 100.0
    t = np.arange(0, 2, 1.0 / sr)
    low = np.sin(2 * np.pi * 2.0 * t)
    high = np.sin(2 * np.pi * 30.0 * t)
    out = apply_filter(low + high, "fft_lowpass", {"high": 10.0}, t=t)
    np.testing.assert_allclose(out, low, atol=0.05)


def test_apply_filter_fft_uses_explicit_sample_rate() -> None:
    sr = 100.0
    t = np.arange(0, 2, 1.0 / sr)
    low = np.sin(2 * np.pi * 2.0 * t)
    out = apply_filter(low, "fft_lowpass", {"high": 10.0, "sample_rate_hz": sr})
    np.testing.assert_allclose(out, low, atol=0.05)


def test_apply_filter_differentiate() -> None:
    x = np.arange(6, dtype=float)
    y = 3.0 * x
    out = apply_filter(y, "differentiate", {}, t=x)
    np.testing.assert_allclose(out, np.full(6, 3.0))


def test_apply_filter_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        apply_filter([1.0, 2.0], "nope", {})


def test_apply_filter_missing_param_raises() -> None:
    with pytest.raises(ValueError):
        apply_filter([1.0, 2.0, 3.0], "moving_average", {})


# --------------------------------------------------------------------------- #
# DbC: type / value error paths                                               #
# --------------------------------------------------------------------------- #
def test_empty_input_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        moving_average([], 3)


def test_string_input_raises_typeerror() -> None:
    with pytest.raises(TypeError):
        moving_average("abc", 3)  # type: ignore[arg-type]


def test_nonfinite_input_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        gaussian_filter([1.0, np.nan, 3.0], 1.0)


def test_moving_average_window_type_and_range() -> None:
    with pytest.raises(TypeError):
        moving_average([1.0, 2.0], 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        moving_average([1.0, 2.0], 0)


def test_exponential_alpha_out_of_range() -> None:
    with pytest.raises(ValueError):
        exponential_smoothing([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        exponential_smoothing([1.0, 2.0], 1.5)


def test_median_even_window_rejected() -> None:
    with pytest.raises(ValueError):
        median_filter([1.0, 2.0, 3.0], 4)


def test_gaussian_sigma_nonpositive_rejected() -> None:
    with pytest.raises(ValueError):
        gaussian_filter([1.0, 2.0], 0.0)


def test_savgol_window_le_polyorder_rejected() -> None:
    with pytest.raises(ValueError):
        savgol_filter([1.0, 2.0, 3.0, 4.0, 5.0], 3, 3)


def test_savgol_even_window_rejected() -> None:
    with pytest.raises(ValueError):
        savgol_filter(np.arange(10.0), 4, 2)


def test_hampel_nsigma_nonpositive_rejected() -> None:
    with pytest.raises(ValueError):
        hampel_filter([1.0, 2.0, 3.0], 3, 0.0)


def test_zscore_threshold_nonpositive_rejected() -> None:
    with pytest.raises(ValueError):
        zscore_filter([1.0, 2.0, 3.0], 0.0)


def test_fft_sample_rate_nonpositive_rejected() -> None:
    with pytest.raises(ValueError):
        fft_filter([1.0, 2.0, 3.0], 0.0, None, 1.0)


def test_fft_no_cutoff_rejected() -> None:
    with pytest.raises(ValueError):
        fft_filter([1.0, 2.0, 3.0], 10.0, None, None)


def test_integrate_x_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        integrate([1.0, 2.0, 3.0], [0.0, 1.0])


def test_resample_bad_agg_rejected() -> None:
    with pytest.raises(ValueError):
        resample_series([0.0, 1.0], [1.0, 2.0], 1.0, "bogus", interpolate=False)


def test_resample_nonpositive_interval_rejected() -> None:
    with pytest.raises(ValueError):
        resample_series([0.0, 1.0], [1.0, 2.0], 0.0, "mean", interpolate=False)


def test_resample_non_ascending_t_rejected() -> None:
    with pytest.raises(ValueError):
        resample_series([1.0, 0.0], [1.0, 2.0], 1.0, "mean", interpolate=False)


def test_resample_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        resample_series([0.0, 1.0, 2.0], [1.0, 2.0], 1.0, "mean", False)


def test_apply_filter_ftype_type_rejected() -> None:
    with pytest.raises(TypeError):
        apply_filter([1.0, 2.0], 123, {})  # type: ignore[arg-type]


def test_apply_filter_params_type_rejected() -> None:
    with pytest.raises(TypeError):
        apply_filter([1.0, 2.0], "median", [1, 2])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Regression: resample grid DoS guard + present-bin correctness               #
# --------------------------------------------------------------------------- #
def test_resample_rejects_oversized_grid() -> None:
    # A tiny interval over a wide span would allocate billions of bins.
    with pytest.raises(ValueError, match="resample grid too large"):
        resample_series([0.0, 1.0e6], [1.0, 2.0], 1.0e-3, "mean", False)


def test_resample_drops_empty_bins_when_not_interpolating() -> None:
    # Samples at t=0 and t=10 with 1s bins -> only bins 0 and 10 are present.
    centers, vals = resample_series([0.0, 10.0], [5.0, 7.0], 1.0, "mean", False)
    np.testing.assert_allclose(centers, [0.5, 10.5])
    np.testing.assert_allclose(vals, [5.0, 7.0])


def test_resample_interpolates_empty_bins() -> None:
    centers, vals = resample_series([0.0, 2.0], [0.0, 2.0], 1.0, "mean", True)
    # 3 bins (centers 0.5,1.5,2.5); middle bin linearly interpolated.
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(vals, [0.0, 1.0, 2.0])


def test_resample_sum_and_last_aggregations() -> None:
    t = [0.0, 0.4, 0.8, 1.2]
    y = [1.0, 2.0, 3.0, 10.0]
    _, s = resample_series(t, y, 1.0, "sum", False)
    np.testing.assert_allclose(s, [6.0, 10.0])  # bin0 = 1+2+3, bin1 = 10
    _, last = resample_series(t, y, 1.0, "last", False)
    np.testing.assert_allclose(last, [3.0, 10.0])
