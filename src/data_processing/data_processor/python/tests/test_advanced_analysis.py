from numba import jit

"""Tests for advanced statistical analysis modules.

Tests for:
- Kalman Filtering
- Wavelet Denoising
- Spectral Analysis
- Outlier Detection
- Time Series Decomposition
- Cross-Correlation Analysis
- State Space Modeling
- Uncertainty Quantification
- Data Augmentation
- Feature Engineering
"""

import numpy as np
import pytest


class TestKalmanFilter:
    """Tests for Kalman filter module."""

    def test_standard_kalman_filter(self) -> None:
        """Test standard Kalman filter on noisy random walk."""
        from data_processor.core.kalman_filter import KalmanFilter, KalmanFilterConfig

        # Generate noisy random walk
        np.random.seed(42)
        n = 100
        true_state = np.cumsum(np.random.randn(n) * 0.1)
        observations = true_state + np.random.randn(n) * 0.5

        # Create filter
        config = KalmanFilterConfig(state_dim=1, obs_dim=1)
        kf = KalmanFilter(config)

        # Configure as random walk model
        kf.set_transition_matrix(np.array([[1.0]]))
        kf.set_observation_matrix(np.array([[1.0]]))
        kf.set_process_noise(np.array([[0.01]]))
        kf.set_observation_noise(np.array([[0.25]]))

        # Filter
        result = kf.filter(observations)

        assert result.filtered_states.shape == (n, 1)
        assert result.log_likelihood < 0  # Should be negative

        # Filtered should be smoother than observations
        obs_var = np.var(np.diff(observations))
        filtered_var = np.var(np.diff(result.filtered_states.flatten()))
        assert filtered_var < obs_var

    def test_kalman_smooth_function(self) -> None:
        """Test convenience function for Kalman smoothing."""
        from data_processor.core.kalman_filter import kalman_smooth

        np.random.seed(42)
        noisy_signal = np.cumsum(np.random.randn(50)) + np.random.randn(50) * 0.5

        smoothed = kalman_smooth(noisy_signal)

        assert len(smoothed) == len(noisy_signal)
        # Smoothed should have lower variance in differences
        assert np.var(np.diff(smoothed)) < np.var(np.diff(noisy_signal))

    def test_extended_kalman_filter(self) -> None:
        """Test Extended Kalman Filter."""
        from data_processor.core.kalman_filter import ExtendedKalmanFilter

        # Simple nonlinear model: x_t = x_{t-1}^2 / 10 + noise
        def transition(x: np.ndarray) -> np.ndarray:
            return x * 0.9  # Simplified linear for test

        def observation(x: np.ndarray) -> np.ndarray:
            return x

        ekf = ExtendedKalmanFilter(state_dim=1, obs_dim=1)

        # Generate test data
        np.random.seed(42)
        observations = np.random.randn(30)

        result = ekf.filter(
            observations,
            transition_func=transition,
            observation_func=observation,
        )

        assert result.filtered_states.shape[0] == 30


class TestWaveletDenoising:
    """Tests for wavelet denoising module."""

    def test_basic_denoising(self) -> None:
        """Test basic wavelet denoising."""
        from data_processor.core.wavelet_denoising import (
            WaveletDenoiseConfig,
            WaveletDenoiser,
        )

        np.random.seed(42)
        # Create noisy sinusoid
        t = np.linspace(0, 4 * np.pi, 200)
        clean = np.sin(t)
        noisy = clean + np.random.randn(200) * 0.3

        config = WaveletDenoiseConfig(wavelet="db4", level=4)
        denoiser = WaveletDenoiser(config)
        result = denoiser.denoise(noisy)

        assert len(result.denoised) == len(noisy)
        # Denoised should be closer to clean than noisy
        noisy_error = np.mean((noisy - clean) ** 2)
        denoised_error = np.mean((result.denoised - clean) ** 2)
        assert denoised_error < noisy_error

    def test_denoise_signal_function(self) -> None:
        """Test convenience denoising function."""
        from data_processor.core.wavelet_denoising import denoise_signal

        np.random.seed(42)
        noisy = np.random.randn(100)

        denoised = denoise_signal(noisy)

        assert len(denoised) == len(noisy)
        # Denoised should have smaller high-frequency content
        noisy_hf = np.var(np.diff(np.diff(noisy)))
        denoised_hf = np.var(np.diff(np.diff(denoised)))
        assert denoised_hf < noisy_hf


class TestSpectralAnalysis:
    """Tests for spectral analysis module."""

    def test_fft_spectrum(self) -> None:
        """Test FFT-based spectrum computation."""
        from data_processor.core.spectral_analysis import (
            SpectralAnalyzer,
            SpectralConfig,
        )

        # Create signal with known frequency
        fs = 100  # Sample rate
        t = np.arange(0, 2, 1 / fs)
        freq = 10  # Hz
        signal = np.sin(2 * np.pi * freq * t)

        config = SpectralConfig(sample_rate=fs)
        analyzer = SpectralAnalyzer(config)
        result = analyzer.compute_fft(signal)

        assert len(result.frequencies) > 0
        assert len(result.power) == len(result.frequencies)

        # Peak should be near 10 Hz
        peak_freq = result.frequencies[np.argmax(result.power)]
        assert abs(peak_freq - freq) < 2  # Within 2 Hz

    def test_welch_periodogram(self) -> None:
        """Test Welch's method."""
        from data_processor.core.spectral_analysis import (
            SpectralAnalyzer,
            SpectralConfig,
        )

        np.random.seed(42)
        signal = np.random.randn(1000)

        config = SpectralConfig(sample_rate=100)
        analyzer = SpectralAnalyzer(config)
        result = analyzer.compute_welch(signal)

        assert len(result.frequencies) > 0
        assert result.dominant_frequency >= 0

    def test_spectrogram(self) -> None:
        """Test spectrogram computation."""
        from data_processor.core.spectral_analysis import (
            SpectralAnalyzer,
            SpectralConfig,
        )

        # Chirp signal
        fs = 100
        t = np.arange(0, 5, 1 / fs)
        signal = np.sin(2 * np.pi * (5 + 5 * t) * t)

        config = SpectralConfig(sample_rate=fs)
        analyzer = SpectralAnalyzer(config)
        result = analyzer.compute_spectrogram(signal, window_size=64)

        assert result.spectrogram.shape[0] > 0
        assert result.spectrogram.shape[1] > 0

    def test_compute_spectrum_function(self) -> None:
        """Test convenience spectrum function."""
        from data_processor.core.spectral_analysis import compute_spectrum

        signal = np.random.randn(256)
        result = compute_spectrum(signal, sample_rate=100)

        assert len(result.frequencies) > 0
        assert result.total_power > 0


class TestOutlierDetection:
    """Tests for outlier detection module."""

    def test_zscore_detection(self) -> None:
        """Test Z-score outlier detection."""
        from data_processor.core.outlier_detection import OutlierConfig, OutlierDetector

        np.random.seed(42)
        # Normal data with outliers
        data = np.random.randn(100)
        data[10] = 10  # Outlier
        data[50] = -8  # Outlier

        config = OutlierConfig(methods=["zscore"], zscore_threshold=3.0)
        detector = OutlierDetector(config)
        result = detector.detect(data)

        assert 10 in result.outlier_indices
        assert 50 in result.outlier_indices
        assert result.n_outliers >= 2

    def test_iqr_detection(self) -> None:
        """Test IQR outlier detection."""
        from data_processor.core.outlier_detection import OutlierConfig, OutlierDetector

        np.random.seed(42)
        data = np.random.randn(100)
        data[25] = 15  # Outlier

        config = OutlierConfig(methods=["iqr"])
        detector = OutlierDetector(config)
        result = detector.detect(data)

        assert 25 in result.outlier_indices

    def test_ensemble_detection(self) -> None:
        """Test ensemble outlier detection."""
        from data_processor.core.outlier_detection import OutlierConfig, OutlierDetector

        np.random.seed(42)
        data = np.random.randn(100)
        data[30] = 20  # Clear outlier

        config = OutlierConfig(
            methods=["zscore", "iqr", "isolation_forest"],
            ensemble_threshold=0.5,
        )
        detector = OutlierDetector(config)
        result = detector.detect(data)

        assert 30 in result.outlier_indices
        assert result.method_agreement is not None

    def test_detect_outliers_function(self) -> None:
        """Test convenience outlier detection function."""
        from data_processor.core.outlier_detection import detect_outliers

        data = np.array([1, 2, 3, 100, 4, 5])
        result = detect_outliers(data)

        assert 3 in result.outlier_indices  # Index of 100


class TestTimeSeriesDecomposition:
    """Tests for time series decomposition module."""

    def test_stl_decomposition(self) -> None:
        """Test STL decomposition."""
        from data_processor.core.time_series_decomposition import (
            DecompositionConfig,
            TimeSeriesDecomposer,
        )

        np.random.seed(42)
        # Create signal with trend, seasonal, and residual
        t = np.arange(200)
        trend = 0.01 * t
        seasonal = np.sin(2 * np.pi * t / 20)  # Period of 20
        residual = np.random.randn(200) * 0.1
        signal = trend + seasonal + residual

        config = DecompositionConfig(period=20)
        decomposer = TimeSeriesDecomposer(config)
        result = decomposer.decompose(signal, period=20)

        assert len(result.trend) == len(signal)
        assert len(result.seasonal) == len(signal)
        assert len(result.residual) == len(signal)
        assert result.period == 20

        # Reconstruction should be close to original
        reconstructed = result.reconstruct()
        assert np.allclose(reconstructed, signal, atol=0.5)

    def test_seasonality_detection(self) -> None:
        """Test seasonality detection."""
        from data_processor.core.time_series_decomposition import TimeSeriesDecomposer

        np.random.seed(42)
        # Strong seasonal signal
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / 10) + np.random.randn(100) * 0.1

        decomposer = TimeSeriesDecomposer()
        detection = decomposer.detect_seasonality(signal)

        assert detection.is_seasonal
        assert detection.dominant_period is not None
        # Should detect period near 10
        assert abs(detection.dominant_period - 10) < 3

    def test_decompose_time_series_function(self) -> None:
        """Test convenience decomposition function."""
        from data_processor.core.time_series_decomposition import decompose_time_series

        signal = np.sin(np.linspace(0, 8 * np.pi, 100)) + np.random.randn(100) * 0.1
        result = decompose_time_series(signal, period=25)

        assert result.trend_strength >= 0
        assert result.seasonal_strength >= 0


class TestCrossCorrelation:
    """Tests for cross-correlation module."""

    def test_basic_cross_correlation(self) -> None:
        """Test basic cross-correlation."""
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        np.random.seed(42)
        # Create two related signals
        x = np.random.randn(100)
        y = np.roll(x, 5) + np.random.randn(100) * 0.3  # y lags x by 5

        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.cross_correlate(x, y)

        assert len(result.ccf_values) > 0
        # Optimal lag should be around 5
        assert abs(result.optimal_lag - 5) < 3

    @jit(nopython=True, fastmath=True)
    def test_granger_causality(self) -> None:
        """Test Granger causality test."""
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        np.random.seed(42)
        # x causes y with lag
        x = np.cumsum(np.random.randn(100))
        y = np.zeros(100)
        for i in range(2, 100):
            y[i] = 0.7 * y[i - 1] + 0.5 * x[i - 2] + np.random.randn() * 0.1

        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.granger_causality_test(x, y, max_lag=5)

        assert result.causal_direction in ["X->Y", "Y->X", "bidirectional", "none"]

    def test_rolling_correlation(self) -> None:
        """Test rolling cross-correlation."""
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5

        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.rolling_cross_correlation(x, y, window=20)

        assert len(result.correlations) > 0
        assert result.window_size == 20

    def test_cross_correlate_function(self) -> None:
        """Test convenience cross-correlation function."""
        from data_processor.core.cross_correlation import cross_correlate

        x = np.random.randn(50)
        y = np.random.randn(50)

        result = cross_correlate(x, y)

        assert result.correlation_at_zero is not None
        assert len(result.lags) > 0


class TestStateSpace:
    """Tests for state space modeling module."""

    def test_local_level_model(self) -> None:
        """Test local level model."""
        from data_processor.core.state_space import LocalLevelModel

        np.random.seed(42)
        # Random walk plus noise
        true_state = np.cumsum(np.random.randn(100) * 0.1)
        observations = true_state + np.random.randn(100) * 0.5

        model = LocalLevelModel()
        result = model.fit(observations)

        assert len(result.fitted_values) == len(observations)
        assert result.converged
        assert result.aic > 0

    def test_local_linear_trend_model(self) -> None:
        """Test local linear trend model."""
        from data_processor.core.state_space import LocalLinearTrendModel

        np.random.seed(42)
        # Linear trend with noise
        t = np.arange(50)
        observations = 0.5 * t + np.random.randn(50) * 2

        model = LocalLinearTrendModel()
        result = model.fit(observations)

        assert len(result.smoothed_states) == len(observations)
        assert result.n_states == 2

    def test_state_space_forecasting(self) -> None:
        """Test state space model forecasting."""
        from data_processor.core.state_space import LocalLevelModel

        np.random.seed(42)
        observations = np.cumsum(np.random.randn(50))

        model = LocalLevelModel()
        model.fit(observations)
        forecast = model.forecast(steps=10)

        assert len(forecast.forecast) == 10
        assert len(forecast.lower_ci) == 10
        assert len(forecast.upper_ci) == 10
        assert all(forecast.lower_ci <= forecast.forecast)
        assert all(forecast.forecast <= forecast.upper_ci)

    def test_fit_state_space_function(self) -> None:
        """Test convenience state space function."""
        from data_processor.core.state_space import fit_state_space

        observations = np.random.randn(30)
        result = fit_state_space(observations, model_type="local_level")

        assert result.model_type.value == "local_level"
        assert len(result.fitted_values) == 30


class TestUncertaintyQuantification:
    """Tests for uncertainty quantification module."""

    def test_bootstrap_ci(self) -> None:
        """Test bootstrap confidence interval."""
        from data_processor.core.uncertainty_quantification import (
            UncertaintyConfig,
            UncertaintyQuantifier,
        )

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        config = UncertaintyConfig(n_bootstrap=500)
        uq = UncertaintyQuantifier(config)
        result = uq.bootstrap_ci(data, np.mean)

        # CI should contain true mean (10)
        assert result.ci_lower < 10 < result.ci_upper
        assert abs(result.point_estimate - 10) < 1

    def test_monte_carlo_propagation(self) -> None:
        """Test Monte Carlo uncertainty propagation."""
        from data_processor.core.uncertainty_quantification import UncertaintyQuantifier

        def area(radius: float) -> float:
            return np.pi * radius**2

        uq = UncertaintyQuantifier()
        result = uq.monte_carlo_propagation(
            area,
            {"radius": ("normal", {"loc": 5, "scale": 0.1})},
        )

        # Mean should be close to pi * 5^2
        expected = np.pi * 25
        assert abs(result.mean - expected) < 1

    def test_error_propagation(self) -> None:
        """Test linear error propagation."""
        from data_processor.core.uncertainty_quantification import UncertaintyQuantifier

        def sum_func(a: float, b: float) -> float:
            return a + b

        uq = UncertaintyQuantifier()
        result, unc = uq.error_propagation(
            sum_func,
            {"a": 10, "b": 20},
            {"a": 1, "b": 2},
        )

        assert result == 30
        # For sum, errors add in quadrature: sqrt(1^2 + 2^2) = sqrt(5)
        assert abs(unc - np.sqrt(5)) < 0.1

    def test_sensitivity_analysis(self) -> None:
        """Test sensitivity analysis."""
        from data_processor.core.uncertainty_quantification import UncertaintyQuantifier

        def model(x: float, y: float) -> float:
            return 2 * x + y  # x has more influence

        uq = UncertaintyQuantifier()
        result = uq.sensitivity_analysis(
            model,
            {"x": (0, 10), "y": (0, 10)},
            n_samples=100,
        )

        # x should have higher sensitivity
        assert result.ranking[0] == "x"

    def test_bootstrap_confidence_interval_function(self) -> None:
        """Test convenience bootstrap function."""
        from data_processor.core.uncertainty_quantification import (
            bootstrap_confidence_interval,
        )

        data = np.random.randn(50)
        result = bootstrap_confidence_interval(data, np.mean)

        assert result.ci_lower < result.ci_upper


class TestDataAugmentation:
    """Tests for data augmentation module."""

    def test_gaussian_noise(self) -> None:
        """Test Gaussian noise augmentation."""
        from data_processor.core.data_augmentation import DataAugmenter

        np.random.seed(42)
        data = np.ones((10, 50))

        augmenter = DataAugmenter()
        noisy = augmenter.add_gaussian_noise(data, std=0.1)

        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

    def test_time_warp(self) -> None:
        """Test time warping augmentation."""
        from data_processor.core.data_augmentation import DataAugmenter

        np.random.seed(42)
        data = np.sin(np.linspace(0, 4 * np.pi, 100))

        augmenter = DataAugmenter()
        warped = augmenter.time_warp(data)

        assert warped.shape == data.shape
        # Should be different but similar pattern
        corr = np.corrcoef(data, warped)[0, 1]
        assert corr > 0.5

    def test_scaling(self) -> None:
        """Test scaling augmentation."""
        from data_processor.core.data_augmentation import DataAugmenter

        data = np.random.randn(10, 50)

        augmenter = DataAugmenter()
        scaled = augmenter.scale(data, range_=(0.5, 1.5))

        assert scaled.shape == data.shape
        # Scaled values should be different
        assert not np.allclose(scaled, data)

    def test_mixup(self) -> None:
        """Test mixup augmentation."""
        from data_processor.core.data_augmentation import DataAugmenter

        np.random.seed(42)
        data = np.random.randn(20, 50)
        labels = np.array([0, 1] * 10)

        augmenter = DataAugmenter()
        mixed_data, mixed_labels = augmenter.mixup(data, labels)

        assert mixed_data.shape == data.shape
        assert mixed_labels is not None

    def test_augment_data_function(self) -> None:
        """Test convenience augmentation function."""
        from data_processor.core.data_augmentation import augment_data

        data = np.random.randn(10, 50)
        result = augment_data(data, methods=["gaussian_noise", "scaling"])

        assert result.augmentation_factor > 1
        assert result.n_samples_augmented > result.n_samples_original


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_statistical_features(self) -> None:
        """Test statistical feature extraction."""
        from data_processor.core.feature_engineering import FeatureExtractor

        data = np.random.randn(100)

        extractor = FeatureExtractor()
        features, names = extractor.extract_statistical(data, "test")

        assert len(features) > 0
        assert len(names) == len(features)
        assert "test_mean" in names
        assert "test_std" in names

    def test_rolling_features(self) -> None:
        """Test rolling window features."""
        from data_processor.core.feature_engineering import FeatureExtractor

        data = np.random.randn(100)

        extractor = FeatureExtractor()
        features, names = extractor.extract_rolling(data, windows=[5, 10])

        assert len(features) > 0
        assert any("rolling_mean" in name for name in names)

    def test_lag_features(self) -> None:
        """Test lag feature extraction."""
        from data_processor.core.feature_engineering import FeatureExtractor

        data = np.random.randn(50)

        extractor = FeatureExtractor()
        features, names = extractor.extract_lag(data, lags=[1, 5])

        assert len(features) > 0
        assert any("lag_1" in name for name in names)

    def test_polynomial_features(self) -> None:
        """Test polynomial feature creation."""
        from data_processor.core.feature_engineering import FeatureExtractor

        data = np.random.randn(20, 3)

        extractor = FeatureExtractor()
        poly_features, names = extractor.create_polynomial_features(data, degree=2)

        assert poly_features.shape[0] == 20
        assert poly_features.shape[1] > 3  # More features than original

    def test_feature_selection_correlation(self) -> None:
        """Test correlation-based feature selection."""
        from data_processor.core.feature_engineering import FeatureSelector

        # Create features with high correlation
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = x1 + np.random.randn(100) * 0.01  # Highly correlated with x1
        x3 = np.random.randn(100)  # Independent
        features = np.column_stack([x1, x2, x3])

        selector = FeatureSelector()
        result = selector.select_by_correlation(features, ["x1", "x2", "x3"], threshold=0.9)

        # Should remove one of x1 or x2
        assert result.n_selected < 3
        assert len(result.removed_names) >= 1

    def test_feature_selection_variance(self) -> None:
        """Test variance-based feature selection."""
        from data_processor.core.feature_engineering import FeatureSelector

        # Create features with different variances
        np.random.seed(42)
        x1 = np.random.randn(100) * 10  # High variance
        x2 = np.zeros(100) + 0.001 * np.random.randn(100)  # Near-zero variance
        x3 = np.random.randn(100)  # Normal variance
        features = np.column_stack([x1, x2, x3])

        selector = FeatureSelector()
        result = selector.select_by_variance(features, ["x1", "x2", "x3"], threshold=0.1)

        # x2 should be removed due to low variance
        assert "x2" in result.removed_names
        assert result.n_selected == 2

    def test_feature_transformer(self) -> None:
        """Test feature transformation."""
        from data_processor.core.feature_engineering import (
            FeatureTransformer,
            TransformationType,
        )

        data = np.random.randn(50, 3) + 5  # Positive values

        transformer = FeatureTransformer()

        # Test standardization
        standardized = transformer.fit_transform(data, TransformationType.STANDARDIZE)
        assert np.abs(np.mean(standardized)) < 0.1
        assert np.abs(np.std(standardized) - 1) < 0.1

        # Test normalization
        transformer2 = FeatureTransformer()
        normalized = transformer2.fit_transform(data, TransformationType.NORMALIZE)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_extract_features_function(self) -> None:
        """Test convenience feature extraction function."""
        from data_processor.core.feature_engineering import extract_features

        # Time series data
        data = np.random.randn(10, 50, 3)  # 10 samples, 50 timesteps, 3 channels

        result = extract_features(data, ["x", "y", "z"])

        assert result.n_samples == 10
        assert result.n_features > 0
        assert len(result.feature_names) == result.n_features


class TestIntegration:
    """Integration tests for multiple modules."""

    def test_denoising_then_decomposition(self) -> None:
        """Test denoising followed by decomposition."""
        from data_processor.core.time_series_decomposition import TimeSeriesDecomposer
        from data_processor.core.wavelet_denoising import WaveletDenoiser

        np.random.seed(42)
        # Noisy seasonal signal
        t = np.arange(200)
        signal = np.sin(2 * np.pi * t / 20) + 0.01 * t + np.random.randn(200) * 0.5

        # Denoise first
        denoiser = WaveletDenoiser()
        denoised = denoiser.denoise(signal).denoised

        # Then decompose
        decomposer = TimeSeriesDecomposer()
        result = decomposer.decompose(denoised, period=20)

        assert result.seasonal_strength > 0.3

    @jit(nopython=True, fastmath=True)
    def test_outlier_detection_then_smoothing(self) -> None:
        """Test outlier detection then Kalman smoothing."""
        from data_processor.core.kalman_filter import kalman_smooth
        from data_processor.core.outlier_detection import detect_outliers

        np.random.seed(42)
        # Signal with outliers
        signal = np.cumsum(np.random.randn(100) * 0.1)
        signal[25] = 50  # Outlier
        signal[75] = -50  # Outlier

        # Detect outliers
        result = detect_outliers(signal)

        # Replace outliers with interpolated values
        clean_signal = signal.copy()
        for idx in result.outlier_indices:
            if 0 < idx < len(signal) - 1:
                clean_signal[idx] = (signal[idx - 1] + signal[idx + 1]) / 2

        # Smooth
        smoothed = kalman_smooth(clean_signal)

        assert np.var(np.diff(smoothed)) < np.var(np.diff(signal))

    def test_feature_extraction_pipeline(self) -> None:
        """Test full feature extraction pipeline."""
        from data_processor.core.feature_engineering import (
            FeatureExtractor,
            FeatureSelector,
            FeatureTransformer,
            TransformationType,
        )

        np.random.seed(42)
        # Multi-channel time series
        data = np.random.randn(50, 100, 4)

        # Extract features
        extractor = FeatureExtractor()
        result = extractor.extract_all(data, ["ch1", "ch2", "ch3", "ch4"])

        # Select features
        selector = FeatureSelector()
        selection = selector.select_by_variance(
            result.features, result.feature_names, threshold=0.01
        )

        # Transform selected features
        selected_features = result.features[:, selection.selected_indices]
        transformer = FeatureTransformer()
        final_features = transformer.fit_transform(
            selected_features, TransformationType.STANDARDIZE
        )

        assert final_features.shape[0] == 50
        assert final_features.shape[1] <= result.n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
