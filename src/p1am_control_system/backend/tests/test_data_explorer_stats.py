"""Tests for the pure-numpy Data Explorer statistical kernels.

Numeric correctness is asserted against hand-computed/analytic values, and
every Design-by-Contract error path (``TypeError`` for wrong types,
``ValueError`` for empty/NaN/out-of-range/length-mismatch) is exercised.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_explorer_stats import (  # noqa: E402
    correlation_matrix,
    cross_correlation,
    describe,
    fit_trendline,
    histogram,
    pca,
    spectrum,
)

# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def test_describe_known_array() -> None:
    result = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["count"] == 5.0
    assert result["mean"] == pytest.approx(3.0)
    # sample std (ddof=1) of 1..5 = sqrt(2.5)
    assert result["std"] == pytest.approx(np.sqrt(2.5))
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["p25"] == pytest.approx(2.0)
    assert result["p75"] == pytest.approx(4.0)
    # rms = sqrt(mean(1,4,9,16,25)) = sqrt(11)
    assert result["rms"] == pytest.approx(np.sqrt(11.0))


def test_describe_single_value_std_zero() -> None:
    result = describe([7.0])
    assert result["count"] == 1.0
    assert result["std"] == 0.0
    assert result["mean"] == pytest.approx(7.0)
    assert result["rms"] == pytest.approx(7.0)


def test_describe_rejects_empty() -> None:
    with pytest.raises(ValueError):
        describe([])


def test_describe_rejects_nan() -> None:
    with pytest.raises(ValueError):
        describe([1.0, np.nan, 3.0])


def test_describe_rejects_str() -> None:
    with pytest.raises(TypeError):
        describe("not an array")


def test_describe_rejects_2d() -> None:
    with pytest.raises(ValueError):
        describe([[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------- #
# correlation_matrix
# --------------------------------------------------------------------------- #


def test_correlation_pearson_perfect_positive() -> None:
    x = np.arange(10, dtype=float)
    labels, matrix = correlation_matrix({"x": x, "y": 2.0 * x + 1.0}, "pearson")
    assert labels == ["x", "y"]
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diag(matrix), [1.0, 1.0])


def test_correlation_pearson_perfect_negative() -> None:
    x = np.arange(10, dtype=float)
    _, matrix = correlation_matrix({"x": x, "y": -3.0 * x}, "pearson")
    assert matrix[0, 1] == pytest.approx(-1.0)


def test_correlation_spearman_monotonic_nonlinear() -> None:
    x = np.linspace(1.0, 5.0, 12)
    # strictly increasing nonlinear transform -> spearman == 1
    y = np.exp(x)
    _, matrix = correlation_matrix({"x": x, "y": y}, "spearman")
    assert matrix[0, 1] == pytest.approx(1.0)


def test_correlation_spearman_differs_from_pearson() -> None:
    x = np.linspace(1.0, 5.0, 20)
    y = x**3
    _, pear = correlation_matrix({"x": x, "y": y}, "pearson")
    _, spear = correlation_matrix({"x": x, "y": y}, "spearman")
    assert spear[0, 1] == pytest.approx(1.0)
    assert pear[0, 1] < spear[0, 1]


def test_correlation_rejects_unknown_method() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        correlation_matrix({"a": x, "b": x}, "kendall")


def test_correlation_rejects_single_column() -> None:
    with pytest.raises(ValueError):
        correlation_matrix({"a": np.arange(5, dtype=float)}, "pearson")


def test_correlation_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        correlation_matrix(
            {"a": np.arange(5, dtype=float), "b": np.arange(6, dtype=float)},
            "pearson",
        )


def test_correlation_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        correlation_matrix([1, 2, 3], "pearson")  # type: ignore[arg-type]


def test_correlation_rejects_method_type() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(TypeError):
        correlation_matrix({"a": x, "b": x}, 3)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# cross_correlation
# --------------------------------------------------------------------------- #


def test_cross_correlation_detects_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(200)
    shift = 7
    a = base.copy()
    b = np.roll(base, shift)  # b is a delayed by `shift`
    lags, values, best_lag = cross_correlation(a, b, max_lag=20)
    assert best_lag == shift
    assert lags[int(np.argmax(values))] == shift
    assert values.max() == pytest.approx(np.max(values))
    assert values[int(np.argmax(values))] <= 1.0 + 1e-9


def test_cross_correlation_zero_lag_autocorr() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(100)
    lags, values, best_lag = cross_correlation(a, a, max_lag=10)
    assert best_lag == 0
    assert values[int(np.where(lags == 0)[0][0])] == pytest.approx(1.0)


def test_cross_correlation_rejects_constant() -> None:
    with pytest.raises(ValueError):
        cross_correlation(np.ones(10), np.arange(10, dtype=float), max_lag=2)


def test_cross_correlation_rejects_bad_max_lag() -> None:
    a = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        cross_correlation(a, a, max_lag=0)
    with pytest.raises(ValueError):
        cross_correlation(a, a, max_lag=10)


def test_cross_correlation_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        cross_correlation(np.arange(10, dtype=float), np.arange(9, dtype=float), 2)


def test_cross_correlation_rejects_max_lag_type() -> None:
    a = np.arange(10, dtype=float)
    with pytest.raises(TypeError):
        cross_correlation(a, a, max_lag=2.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #


def test_spectrum_fft_peak_at_sine_frequency() -> None:
    fs = 100.0
    n = 1000  # 100 integer cycles of a 10 Hz tone -> no spectral leakage
    t = np.arange(n) / fs
    f0 = 10.0
    y = np.sin(2.0 * np.pi * f0 * t)
    freqs, mag = spectrum(y, fs, "fft", "none", None, detrend=False)
    peak_freq = freqs[int(np.argmax(mag))]
    assert peak_freq == pytest.approx(f0, abs=fs / n)
    # amplitude ~ 1.0 for a unit sine (no leakage at an exact bin frequency)
    assert mag.max() == pytest.approx(1.0, abs=0.05)


def test_spectrum_welch_peak_location() -> None:
    fs = 200.0
    n = 4096
    t = np.arange(n) / fs
    f0 = 25.0
    y = np.sin(2.0 * np.pi * f0 * t)
    freqs, psd = spectrum(y, fs, "welch", "hanning", 256, detrend=True)
    peak_freq = freqs[int(np.argmax(psd))]
    assert peak_freq == pytest.approx(f0, abs=fs / 256 * 1.5)


def test_spectrum_detrend_removes_dc() -> None:
    fs = 50.0
    y = np.ones(256) * 5.0 + np.sin(2.0 * np.pi * 5.0 * np.arange(256) / fs)
    freqs, mag = spectrum(y, fs, "fft", "none", None, detrend=True)
    # DC bin (index 0) should be near zero after detrending.
    assert mag[0] == pytest.approx(0.0, abs=1e-6)


def test_spectrum_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        spectrum(np.arange(10, dtype=float), 0.0, "fft", "none", None, False)


def test_spectrum_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        spectrum(np.arange(10, dtype=float), 1.0, "bogus", "none", None, False)


def test_spectrum_rejects_unknown_window() -> None:
    with pytest.raises(ValueError):
        spectrum(np.arange(10, dtype=float), 1.0, "fft", "kaiser", None, False)


def test_spectrum_rejects_detrend_type() -> None:
    with pytest.raises(TypeError):
        spectrum(np.arange(10, dtype=float), 1.0, "fft", "none", None, "yes")  # type: ignore[arg-type]


def test_spectrum_rejects_bad_segment_size() -> None:
    with pytest.raises(ValueError):
        spectrum(np.arange(10, dtype=float), 1.0, "welch", "none", 0, False)


# --------------------------------------------------------------------------- #
# fit_trendline
# --------------------------------------------------------------------------- #


def test_fit_trendline_linear_exact() -> None:
    x = np.arange(10, dtype=float)
    y = 3.0 * x + 2.0
    result = fit_trendline(x, y, "linear", 1)
    assert result["coefficients"][0] == pytest.approx(3.0)
    assert result["coefficients"][1] == pytest.approx(2.0)
    assert result["r_squared"] == pytest.approx(1.0)
    assert len(result["x_fit"]) == 200
    assert len(result["y_fit"]) == 200
    assert "y =" in result["equation"]


def test_fit_trendline_polynomial_exact() -> None:
    x = np.linspace(-3.0, 3.0, 25)
    y = 2.0 * x**2 - x + 4.0
    result = fit_trendline(x, y, "polynomial", 2)
    coeffs = result["coefficients"]
    assert coeffs[0] == pytest.approx(2.0, abs=1e-6)
    assert coeffs[1] == pytest.approx(-1.0, abs=1e-6)
    assert coeffs[2] == pytest.approx(4.0, abs=1e-6)
    assert result["r_squared"] == pytest.approx(1.0)


def test_fit_trendline_exponential_recovers_b() -> None:
    x = np.linspace(0.0, 2.0, 40)
    a_true, b_true = 1.5, 0.7
    y = a_true * np.exp(b_true * x)
    result = fit_trendline(x, y, "exponential", 1)
    a, b = result["coefficients"]
    assert a == pytest.approx(a_true, rel=1e-6)
    assert b == pytest.approx(b_true, rel=1e-6)
    assert 0.0 <= result["r_squared"] <= 1.0
    assert result["r_squared"] == pytest.approx(1.0)


def test_fit_trendline_power_recovers_b() -> None:
    x = np.linspace(1.0, 5.0, 40)
    a_true, b_true = 2.0, 1.3
    y = a_true * np.power(x, b_true)
    result = fit_trendline(x, y, "power", 1)
    a, b = result["coefficients"]
    assert a == pytest.approx(a_true, rel=1e-6)
    assert b == pytest.approx(b_true, rel=1e-6)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_fit_trendline_r_squared_in_unit_range_with_noise() -> None:
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0 + rng.standard_normal(50)
    result = fit_trendline(x, y, "linear", 1)
    assert 0.0 <= result["r_squared"] <= 1.0
    assert result["r_squared"] > 0.9


def test_fit_trendline_exponential_rejects_nonpositive_y() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        fit_trendline(x, np.array([1.0, -2.0, 3.0, 4.0, 5.0]), "exponential", 1)


def test_fit_trendline_power_rejects_nonpositive_x() -> None:
    with pytest.raises(ValueError):
        fit_trendline(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "power", 1)


def test_fit_trendline_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        fit_trendline(np.arange(5, dtype=float), np.arange(6, dtype=float), "linear", 1)


def test_fit_trendline_rejects_unknown_kind() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        fit_trendline(x, x, "spline", 1)


def test_fit_trendline_rejects_bad_num_points() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        fit_trendline(x, x, "linear", 1, num_points=1)


def test_fit_trendline_polynomial_needs_enough_points() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        fit_trendline(x, y, "polynomial", 5)


def test_fit_trendline_rejects_degree_type() -> None:
    x = np.arange(5, dtype=float)
    with pytest.raises(TypeError):
        fit_trendline(x, x, "linear", 1.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# pca
# --------------------------------------------------------------------------- #


def test_pca_two_correlated_vars_pc1_explains_almost_all() -> None:
    x = np.linspace(0.0, 10.0, 100)
    y = 2.0 * x  # perfectly correlated
    result = pca({"x": x, "y": y}, standardize=False, n_components=0)
    ratio = result["explained_variance_ratio"]
    assert ratio[0] == pytest.approx(1.0, abs=1e-9)
    assert result["cumulative_variance"][-1] == pytest.approx(1.0, abs=1e-9)
    assert result["component_labels"][0] == "PC1"
    assert result["loadings"].shape == (2, 2)
    assert result["scores_pc1"].shape == (100,)
    assert result["scores_pc2"].shape == (100,)


def test_pca_standardize_correlation_based() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(80)
    b = a * 100.0 + 0.01 * rng.standard_normal(80)  # tightly correlated
    result = pca({"a": a, "b": b}, standardize=True, n_components=2)
    ratio = result["explained_variance_ratio"]
    assert ratio[0] > 0.99
    assert result["singular_values"].shape == (2,)


def test_pca_n_components_limits_output() -> None:
    rng = np.random.default_rng(4)
    cols = {name: rng.standard_normal(50) for name in ("a", "b", "c", "d")}
    result = pca(cols, standardize=False, n_components=2)
    assert result["loadings"].shape == (2, 4)
    assert len(result["component_labels"]) == 2
    assert result["explained_variance_ratio"].shape == (2,)


def test_pca_rejects_single_variable() -> None:
    with pytest.raises(ValueError):
        pca({"a": np.arange(10, dtype=float)}, standardize=False, n_components=0)


def test_pca_rejects_zero_variance_when_standardize() -> None:
    with pytest.raises(ValueError):
        pca(
            {"a": np.ones(10), "b": np.arange(10, dtype=float)},
            standardize=True,
            n_components=0,
        )


def test_pca_rejects_too_many_components() -> None:
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        pca({"a": x, "b": 2.0 * x}, standardize=False, n_components=5)


def test_pca_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        pca(
            {"a": np.arange(10, dtype=float), "b": np.arange(9, dtype=float)},
            standardize=False,
            n_components=0,
        )


def test_pca_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        pca([1, 2, 3], standardize=False, n_components=0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# histogram
# --------------------------------------------------------------------------- #


def test_histogram_counts_sum_to_n() -> None:
    rng = np.random.default_rng(5)
    y = rng.standard_normal(500)
    edges, counts = histogram(y, bins=20, density=False)
    assert edges.shape == (21,)
    assert counts.shape == (20,)
    assert counts.sum() == pytest.approx(500.0)


def test_histogram_known_bins() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0])
    edges, counts = histogram(y, bins=2, density=False)
    # range [0, 2], two bins: [0,1) and [1,2]; numpy puts 1.0 in second bin.
    assert counts.sum() == 6.0
    assert len(counts) == 2


def test_histogram_density_integrates_to_one() -> None:
    rng = np.random.default_rng(6)
    y = rng.standard_normal(1000)
    edges, counts = histogram(y, bins=30, density=True)
    widths = np.diff(edges)
    assert np.sum(counts * widths) == pytest.approx(1.0)


def test_histogram_rejects_bad_bins() -> None:
    with pytest.raises(ValueError):
        histogram(np.arange(10, dtype=float), bins=0, density=False)


def test_histogram_rejects_empty() -> None:
    with pytest.raises(ValueError):
        histogram([], bins=5, density=False)


def test_histogram_rejects_nan() -> None:
    with pytest.raises(ValueError):
        histogram([1.0, np.nan], bins=5, density=False)


def test_histogram_rejects_bins_type() -> None:
    with pytest.raises(TypeError):
        histogram(np.arange(10, dtype=float), bins=2.0, density=False)  # type: ignore[arg-type]
