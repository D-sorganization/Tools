"""Tests for signal processing core functionality.

Tests follow TDD principles - written before implementation.
Covers: integration, differentiation, resampling, custom variables, trendlines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSignalIntegration:
    """Tests for signal integration methods."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time series data for testing."""
        n_points = 100
        time = pd.date_range("2024-01-01", periods=n_points, freq="1s")
        # Constant signal - integral should be linear
        constant = np.ones(n_points) * 5.0
        # Linear signal - integral should be quadratic
        linear = np.arange(n_points, dtype=float)
        # Sine wave for testing
        sine = np.sin(np.linspace(0, 4 * np.pi, n_points))
        return pd.DataFrame(
            {
                "time": time,
                "constant": constant,
                "linear": linear,
                "sine": sine,
            }
        )

    def test_trapezoidal_integration_constant_signal(self, sample_data: pd.DataFrame):
        """Integral of constant 5 over 1 second intervals should grow linearly."""
        from data_processor.core.signal_processing import integrate_signals

        result = integrate_signals(sample_data, "time", ["constant"], method="trapezoidal")

        assert "cumulative_constant" in result.columns
        # After 10 seconds (10 points), integral should be ~50
        # (5 * 10 = 50, but trapezoidal starts at 0)
        cumulative = result["cumulative_constant"].values
        assert cumulative[0] == 0  # Starts at zero
        assert np.isclose(cumulative[10], 50, rtol=0.01)

    def test_rectangular_integration(self, sample_data: pd.DataFrame):
        """Test rectangular (left-endpoint) integration."""
        from data_processor.core.signal_processing import integrate_signals

        result = integrate_signals(sample_data, "time", ["constant"], method="rectangular")

        assert "cumulative_constant" in result.columns
        cumulative = result["cumulative_constant"].values
        assert cumulative[0] == 0

    def test_simpson_integration(self, sample_data: pd.DataFrame):
        """Test Simpson's rule integration."""
        from data_processor.core.signal_processing import integrate_signals

        result = integrate_signals(sample_data, "time", ["constant"], method="simpson")

        assert "cumulative_constant" in result.columns

    def test_integration_preserves_nan(self, sample_data: pd.DataFrame):
        """NaN values should be handled gracefully."""
        from data_processor.core.signal_processing import integrate_signals

        sample_data.loc[5, "constant"] = np.nan
        result = integrate_signals(sample_data, "time", ["constant"], method="trapezoidal")

        # Integration should continue past NaN
        assert not np.isnan(result["cumulative_constant"].iloc[-1])

    def test_integration_multiple_signals(self, sample_data: pd.DataFrame):
        """Test integrating multiple signals at once."""
        from data_processor.core.signal_processing import integrate_signals

        result = integrate_signals(
            sample_data, "time", ["constant", "linear"], method="trapezoidal"
        )

        assert "cumulative_constant" in result.columns
        assert "cumulative_linear" in result.columns


class TestSignalDifferentiation:
    """Tests for signal differentiation methods."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time series data for testing."""
        n_points = 100
        time = pd.date_range("2024-01-01", periods=n_points, freq="1s")
        # Quadratic: y = t^2, derivative = 2t
        t = np.arange(n_points, dtype=float)
        quadratic = t**2
        # Cubic: y = t^3, derivative = 3t^2
        cubic = t**3
        # Sine: derivative = cos
        sine = np.sin(t * 0.1)
        return pd.DataFrame(
            {
                "time": time,
                "quadratic": quadratic,
                "cubic": cubic,
                "sine": sine,
            }
        )

    def test_spline_first_derivative(self, sample_data: pd.DataFrame):
        """Test spline-based first derivative."""
        from data_processor.core.signal_processing import differentiate_signals

        result = differentiate_signals(
            sample_data, "time", ["quadratic"], method="spline", orders=[1]
        )

        assert "quadratic_d1" in result.columns
        # Derivative of t^2 is 2t
        # At t=50, derivative should be ~100
        d1 = result["quadratic_d1"].values
        assert np.isclose(d1[50], 100, rtol=0.1)

    def test_rolling_polynomial_first_derivative(self, sample_data: pd.DataFrame):
        """Test rolling polynomial (causal) first derivative."""
        from data_processor.core.signal_processing import differentiate_signals

        result = differentiate_signals(
            sample_data,
            "time",
            ["quadratic"],
            method="rolling_polynomial",
            orders=[1],
            window_size=11,
            poly_order=3,
        )

        assert "quadratic_d1" in result.columns

    def test_second_derivative(self, sample_data: pd.DataFrame):
        """Test second derivative calculation."""
        from data_processor.core.signal_processing import differentiate_signals

        result = differentiate_signals(
            sample_data, "time", ["quadratic"], method="spline", orders=[2]
        )

        assert "quadratic_d2" in result.columns
        # Second derivative of t^2 is 2 (constant)
        d2 = result["quadratic_d2"].values
        # Should be approximately 2 in the middle of the data
        assert np.isclose(np.nanmean(d2[20:80]), 2, rtol=0.2)

    def test_multiple_derivative_orders(self, sample_data: pd.DataFrame):
        """Test computing multiple derivative orders at once."""
        from data_processor.core.signal_processing import differentiate_signals

        result = differentiate_signals(
            sample_data, "time", ["cubic"], method="spline", orders=[1, 2, 3]
        )

        assert "cubic_d1" in result.columns
        assert "cubic_d2" in result.columns
        assert "cubic_d3" in result.columns

    def test_differentiation_handles_nan(self, sample_data: pd.DataFrame):
        """NaN values should be handled gracefully."""
        from data_processor.core.signal_processing import differentiate_signals

        sample_data.loc[50, "quadratic"] = np.nan
        result = differentiate_signals(
            sample_data, "time", ["quadratic"], method="spline", orders=[1]
        )

        # Should still produce output
        assert "quadratic_d1" in result.columns


class TestTimeResampling:
    """Tests for time resampling functionality."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time series at 1 second intervals."""
        n_points = 100
        time = pd.date_range("2024-01-01", periods=n_points, freq="1s")
        values = np.sin(np.linspace(0, 4 * np.pi, n_points))
        return pd.DataFrame({"time": time, "signal": values})

    def test_resample_to_lower_frequency(self, sample_data: pd.DataFrame):
        """Test downsampling from 1s to 5s intervals."""
        from data_processor.core.signal_processing import resample_data

        result = resample_data(sample_data, "time", "5s")

        # 100 points at 1s -> ~20 points at 5s
        assert len(result) < len(sample_data)
        assert len(result) == 20

    def test_resample_to_higher_frequency(self, sample_data: pd.DataFrame):
        """Test upsampling from 1s to 500ms intervals."""
        from data_processor.core.signal_processing import resample_data

        result = resample_data(sample_data, "time", "500ms", interpolate=True)

        # Should have more points after upsampling
        assert len(result) > len(sample_data)

    def test_resample_preserves_time_range(self, sample_data: pd.DataFrame):
        """Resampling should preserve the overall time range."""
        from data_processor.core.signal_processing import resample_data

        result = resample_data(sample_data, "time", "2s")

        # Time range should be similar
        original_duration = (sample_data["time"].max() - sample_data["time"].min()).total_seconds()
        result_duration = (result["time"].max() - result["time"].min()).total_seconds()
        assert np.isclose(original_duration, result_duration, rtol=0.1)

    def test_resample_with_aggregation(self, sample_data: pd.DataFrame):
        """Test resampling with mean aggregation."""
        from data_processor.core.signal_processing import resample_data

        result = resample_data(sample_data, "time", "10s", method="mean")

        assert len(result) == 10


class TestCustomVariables:
    """Tests for custom calculated variables."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data with multiple signals."""
        n_points = 50
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=n_points, freq="1s"),
                "temperature_c": np.linspace(0, 100, n_points),
                "pressure": np.linspace(100, 200, n_points),
                "flow_rate": np.ones(n_points) * 10,
            }
        )

    def test_simple_arithmetic_formula(self, sample_data: pd.DataFrame):
        """Test simple arithmetic: Celsius to Fahrenheit."""
        from data_processor.core.signal_processing import apply_custom_variable

        result = apply_custom_variable(
            sample_data,
            name="temperature_f",
            formula="([temperature_c] * 1.8) + 32",
        )

        assert "temperature_f" in result.columns
        # 0C = 32F, 100C = 212F
        assert np.isclose(result["temperature_f"].iloc[0], 32)
        assert np.isclose(result["temperature_f"].iloc[-1], 212)

    def test_multi_signal_formula(self, sample_data: pd.DataFrame):
        """Test formula using multiple signals."""
        from data_processor.core.signal_processing import apply_custom_variable

        result = apply_custom_variable(
            sample_data,
            name="power",
            formula="[pressure] * [flow_rate]",
        )

        assert "power" in result.columns
        # pressure * flow_rate
        expected = sample_data["pressure"] * sample_data["flow_rate"]
        np.testing.assert_array_almost_equal(result["power"], expected)

    def test_math_functions_in_formula(self, sample_data: pd.DataFrame):
        """Test using math functions like sqrt, log, sin."""
        from data_processor.core.signal_processing import apply_custom_variable

        result = apply_custom_variable(
            sample_data,
            name="sqrt_pressure",
            formula="sqrt([pressure])",
        )

        assert "sqrt_pressure" in result.columns
        expected = np.sqrt(sample_data["pressure"])
        np.testing.assert_array_almost_equal(result["sqrt_pressure"], expected)

    def test_formula_validation_rejects_unsafe_code(self, sample_data: pd.DataFrame):
        """Unsafe operations like imports should be rejected."""
        from data_processor.core.signal_processing import apply_custom_variable

        with pytest.raises(ValueError, match="[Uu]nsafe|not allowed"):
            apply_custom_variable(
                sample_data,
                name="unsafe",
                formula="__import__('os').system('rm -rf /')",
            )

    def test_formula_with_unknown_signal_raises_error(self, sample_data: pd.DataFrame):
        """Using a non-existent signal should raise an error."""
        from data_processor.core.signal_processing import apply_custom_variable

        with pytest.raises(ValueError, match="[Uu]nknown|not found"):
            apply_custom_variable(
                sample_data,
                name="bad",
                formula="[nonexistent_signal] * 2",
            )


class TestTrendlineAnalysis:
    """Tests for trendline/regression analysis."""

    @pytest.fixture
    def linear_data(self) -> pd.DataFrame:
        """Create data that follows a linear trend: y = 2x + 5."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 5 + np.random.normal(0, 0.1, 50)  # Small noise
        return pd.DataFrame({"x": x, "y": y})

    @pytest.fixture
    def exponential_data(self) -> pd.DataFrame:
        """Create data that follows exponential trend: y = 2 * e^(0.3x)."""
        x = np.linspace(0, 5, 50)
        y = 2 * np.exp(0.3 * x) + np.random.normal(0, 0.1, 50)
        return pd.DataFrame({"x": x, "y": y})

    def test_linear_trendline(self, linear_data: pd.DataFrame):
        """Test linear regression trendline."""
        from data_processor.core.signal_processing import calculate_trendline

        result = calculate_trendline(linear_data, x_col="x", y_col="y", trend_type="linear")

        assert "slope" in result
        assert "intercept" in result
        assert "r_squared" in result
        assert np.isclose(result["slope"], 2, rtol=0.1)
        assert np.isclose(result["intercept"], 5, rtol=0.2)
        assert result["r_squared"] > 0.99

    def test_polynomial_trendline(self, linear_data: pd.DataFrame):
        """Test polynomial trendline."""
        from data_processor.core.signal_processing import calculate_trendline

        result = calculate_trendline(
            linear_data, x_col="x", y_col="y", trend_type="polynomial", degree=2
        )

        assert "coefficients" in result
        assert "r_squared" in result
        assert len(result["coefficients"]) == 3  # degree 2 has 3 coeffs

    def test_exponential_trendline(self, exponential_data: pd.DataFrame):
        """Test exponential trendline."""
        from data_processor.core.signal_processing import calculate_trendline

        result = calculate_trendline(
            exponential_data, x_col="x", y_col="y", trend_type="exponential"
        )

        assert "a" in result  # y = a * e^(b*x)
        assert "b" in result
        assert "r_squared" in result
        assert np.isclose(result["a"], 2, rtol=0.2)
        assert np.isclose(result["b"], 0.3, rtol=0.2)

    def test_power_trendline(self):
        """Test power trendline: y = a * x^b."""
        from data_processor.core.signal_processing import calculate_trendline

        x = np.linspace(1, 10, 50)  # Avoid x=0 for power
        y = 3 * (x**2) + np.random.normal(0, 0.5, 50)
        data = pd.DataFrame({"x": x, "y": y})

        result = calculate_trendline(data, x_col="x", y_col="y", trend_type="power")

        assert "a" in result
        assert "b" in result
        assert np.isclose(result["b"], 2, rtol=0.2)

    def test_trendline_with_time_window(self, linear_data: pd.DataFrame):
        """Test calculating trendline over a specific portion of data."""
        from data_processor.core.signal_processing import calculate_trendline

        result = calculate_trendline(
            linear_data, x_col="x", y_col="y", trend_type="linear", x_min=2, x_max=8
        )

        assert "slope" in result
        # Should still be approximately 2


class TestTimeRangeUtilities:
    """Tests for time range manipulation."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample time series data."""
        time = pd.date_range("2024-01-01 10:00:00", periods=100, freq="1min")
        values = np.arange(100)
        return pd.DataFrame({"time": time, "value": values})

    def test_trim_by_time(self, sample_data: pd.DataFrame):
        """Test trimming data to a specific time range."""
        from data_processor.core.signal_processing import trim_time_range

        result = trim_time_range(sample_data, "time", start_time="10:30:00", end_time="11:00:00")

        assert len(result) < len(sample_data)
        assert result["time"].min().hour == 10
        assert result["time"].min().minute >= 30
        assert result["time"].max().hour == 11

    def test_trim_by_date_and_time(self, sample_data: pd.DataFrame):
        """Test trimming with date and time specification."""
        from data_processor.core.signal_processing import trim_time_range

        result = trim_time_range(
            sample_data,
            "time",
            date="2024-01-01",
            start_time="10:30:00",
            end_time="11:00:00",
        )

        assert len(result) < len(sample_data)


class TestConfigurationManagement:
    """Tests for configuration save/load functionality."""

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        from data_processor.core.config_manager import ConfigManager

        config = ConfigManager(config_dir=tmp_path)

        settings = {
            "filter_type": "moving_average",
            "window_size": 10,
            "output_format": "csv",
            "signals": ["temp", "pressure"],
        }

        config.save_config("test_config", settings)
        loaded = config.load_config("test_config")

        assert loaded == settings

    def test_list_configurations(self, tmp_path):
        """Test listing all saved configurations."""
        from data_processor.core.config_manager import ConfigManager

        config = ConfigManager(config_dir=tmp_path)
        config.save_config("config1", {"a": 1})
        config.save_config("config2", {"b": 2})

        configs = config.list_configs()

        assert "config1" in configs
        assert "config2" in configs

    def test_delete_configuration(self, tmp_path):
        """Test deleting a configuration."""
        from data_processor.core.config_manager import ConfigManager

        config = ConfigManager(config_dir=tmp_path)
        config.save_config("to_delete", {"x": 1})
        config.delete_config("to_delete")

        configs = config.list_configs()
        assert "to_delete" not in configs


class TestSignalListManagement:
    """Tests for signal list save/load functionality."""

    def test_save_and_load_signal_list(self, tmp_path):
        """Test saving and loading signal selections."""
        from data_processor.core.signal_list_manager import SignalListManager

        manager = SignalListManager(config_dir=tmp_path)

        signals = ["temperature", "pressure", "flow_rate"]
        manager.save_signal_list("my_signals", signals)
        loaded = manager.load_signal_list("my_signals")

        assert loaded == signals

    def test_list_signal_sets(self, tmp_path):
        """Test listing all saved signal sets."""
        from data_processor.core.signal_list_manager import SignalListManager

        manager = SignalListManager(config_dir=tmp_path)
        manager.save_signal_list("set1", ["a", "b"])
        manager.save_signal_list("set2", ["c", "d"])

        sets = manager.list_signal_sets()

        assert "set1" in sets
        assert "set2" in sets


class TestPlotConfigurationManagement:
    """Tests for plot configuration management."""

    def test_save_and_load_plot_config(self, tmp_path):
        """Test saving and loading plot configurations."""
        from data_processor.core.plot_config_manager import PlotConfigManager

        manager = PlotConfigManager(config_dir=tmp_path)

        plot_config = {
            "name": "Temperature Plot",
            "signals": ["temp1", "temp2"],
            "x_axis": "time",
            "chart_type": "line",
            "color_scheme": "default",
            "trendline": {"type": "linear", "enabled": True},
            "time_range": {"start": "10:00", "end": "12:00"},
        }

        manager.save_plot_config("temp_plot", plot_config)
        loaded = manager.load_plot_config("temp_plot")

        assert loaded == plot_config

    def test_list_plot_configs(self, tmp_path):
        """Test listing all saved plot configurations."""
        from data_processor.core.plot_config_manager import PlotConfigManager

        manager = PlotConfigManager(config_dir=tmp_path)
        manager.save_plot_config("plot1", {"name": "Plot 1"})
        manager.save_plot_config("plot2", {"name": "Plot 2"})

        plots = manager.list_plot_configs()

        assert "plot1" in plots
        assert "plot2" in plots


class TestDatasetNaming:
    """Tests for dataset naming utilities."""

    def test_auto_generate_name(self):
        """Test automatic dataset name generation."""
        from data_processor.core.dataset_naming import generate_dataset_name

        name = generate_dataset_name(
            base_name="data",
            include_timestamp=True,
            include_filter=True,
            filter_type="moving_average",
        )

        assert "data" in name
        assert "moving_average" in name

    def test_custom_name_validation(self):
        """Test custom name validation."""
        from data_processor.core.dataset_naming import validate_dataset_name

        assert validate_dataset_name("valid_name") is True
        assert validate_dataset_name("valid-name-123") is True
        assert validate_dataset_name("") is False
        assert validate_dataset_name("invalid/name") is False

    def test_unique_name_generation(self, tmp_path):
        """Test generating unique names to avoid overwrites."""
        from data_processor.core.dataset_naming import generate_unique_name

        # Create existing file
        (tmp_path / "data.csv").touch()

        unique = generate_unique_name(tmp_path, "data", ".csv")

        assert unique != "data.csv"
        assert "data" in unique
