from numba import jit

"""Comprehensive tests for statistical analysis modules.

Tests cover:
- Dataset management with undo/redo
- Surface plot generation
- PCA analysis
- ANOVA statistical tests
- Multivariable regression
- Neural network interface
- Script generation

Following TDD principles with comprehensive edge case coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="s"),
            "signal_a": np.sin(np.linspace(0, 4 * np.pi, n))
            + np.random.normal(0, 0.1, n),
            "signal_b": np.cos(np.linspace(0, 4 * np.pi, n))
            + np.random.normal(0, 0.1, n),
            "signal_c": np.linspace(0, 10, n) + np.random.normal(0, 0.5, n),
        }
    )


@pytest.fixture
def multivariate_df() -> pd.DataFrame:
    """Create multivariate data for regression/PCA testing."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = x1 * 0.5 + np.random.normal(0, 0.3, n)  # Correlated with x1
    noise = np.random.normal(0, 0.5, n)
    y = 2 * x1 + 3 * x2 - 1.5 * x3 + 5 + noise

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y": y,
        }
    )


@pytest.fixture
def anova_df() -> pd.DataFrame:
    """Create data for ANOVA testing."""
    np.random.seed(42)
    groups = []
    values = []

    # Group A: mean 10
    groups.extend(["A"] * 30)
    values.extend(np.random.normal(10, 2, 30))

    # Group B: mean 12
    groups.extend(["B"] * 30)
    values.extend(np.random.normal(12, 2, 30))

    # Group C: mean 15
    groups.extend(["C"] * 30)
    values.extend(np.random.normal(15, 2, 30))

    return pd.DataFrame(
        {
            "group": groups,
            "value": values,
        }
    )


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for file operations."""
    return tmp_path


# =============================================================================
# DATASET MANAGER TESTS
# =============================================================================


class TestDatasetManager:
    """Tests for DatasetManager and undo/redo functionality."""

    def test_load_from_dataframe(self, sample_df: pd.DataFrame) -> None:
        """Test loading a dataset from DataFrame."""
        from data_processor.core.dataset_manager import DatasetManager

        manager = DatasetManager()
        dataset_id = manager.load_from_dataframe(sample_df, name="Test Dataset")

        assert dataset_id is not None
        assert manager.active_data is not None
        assert len(manager.active_data) == len(sample_df)

    def test_save_version_creates_history(self, sample_df: pd.DataFrame) -> None:
        """Test that saving versions creates history entries."""
        from data_processor.core.dataset_manager import DatasetManager

        manager = DatasetManager()
        manager.load_from_dataframe(sample_df, name="Test")

        # Apply some transformation
        modified_df = sample_df.copy()
        modified_df["signal_a"] = modified_df["signal_a"] * 2

        manager.save_version(modified_df, "multiply", "Multiplied signal_a by 2")

        history = manager.get_history()
        assert len(history) == 2
        assert history[1].operation == "multiply"

    def test_undo_reverts_to_previous_version(self, sample_df: pd.DataFrame) -> None:
        """Test that undo reverts to previous version."""
        from data_processor.core.dataset_manager import DatasetManager

        manager = DatasetManager()
        manager.load_from_dataframe(sample_df, name="Test")

        # Modify and save
        modified_df = sample_df.copy()
        modified_df["signal_a"] = modified_df["signal_a"] * 2
        manager.save_version(modified_df, "modify", "Modified")

        assert manager.can_undo is True

        # Undo
        result = manager.undo()
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    def test_redo_restores_undone_version(self, sample_df: pd.DataFrame) -> None:
        """Test that redo restores undone version."""
        from data_processor.core.dataset_manager import DatasetManager

        manager = DatasetManager()
        manager.load_from_dataframe(sample_df, name="Test")

        modified_df = sample_df.copy()
        modified_df["signal_a"] = 0
        manager.save_version(modified_df, "zero", "Zeroed signal")

        manager.undo()
        assert manager.can_redo is True

        result = manager.redo()
        assert result is not None
        assert all(result["signal_a"] == 0)

    def test_workspace_save_and_load(
        self, sample_df: pd.DataFrame, temp_dir: Path
    ) -> None:
        """Test saving and loading workspace."""
        from data_processor.core.dataset_manager import DatasetManager

        # Create and save
        manager1 = DatasetManager()
        manager1.load_from_dataframe(sample_df, name="Test")
        manager1.save_workspace(temp_dir)

        # Load in new manager
        manager2 = DatasetManager()
        manager2.load_workspace(temp_dir)

        assert manager2.active_data is not None
        assert len(manager2.active_data) == len(sample_df)


class TestUndoRedoManager:
    """Tests for the UndoRedoManager command pattern."""

    def test_execute_adds_to_history(self) -> None:
        """Test that executing commands adds them to history."""
        from data_processor.core.undo_redo import LambdaCommand, UndoRedoManager

        manager: UndoRedoManager[int] = UndoRedoManager()
        state = [0]

        cmd = LambdaCommand(
            execute_fn=lambda: state.__setitem__(0, state[0] + 1) or state[0],
            undo_fn=lambda: state.__setitem__(0, state[0] - 1) or state[0],
            name="Increment",
        )

        manager.execute(cmd)
        assert state[0] == 1
        assert manager.can_undo is True
        assert len(manager.history) == 1

    def test_undo_reverses_command(self) -> None:
        """Test that undo reverses the command."""
        from data_processor.core.undo_redo import LambdaCommand, UndoRedoManager

        manager: UndoRedoManager[int] = UndoRedoManager()
        state = [0]

        cmd = LambdaCommand(
            execute_fn=lambda: state.__setitem__(0, 10) or 10,
            undo_fn=lambda: state.__setitem__(0, 0) or 0,
            name="Set to 10",
        )

        manager.execute(cmd)
        manager.undo()

        assert state[0] == 0
        assert manager.can_redo is True


# =============================================================================
# SURFACE PLOT TESTS
# =============================================================================


class TestSurfacePlot:
    """Tests for surface plot generation."""

    def test_create_surface_basic(self) -> None:
        """Test basic surface creation."""
        from data_processor.core.surface_plot import (
            SurfacePlotConfig,
            SurfacePlotEngine,
        )

        # Create test data
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = np.random.uniform(0, 10, n)
        z = np.sin(x) * np.cos(y) + np.random.normal(0, 0.1, n)

        df = pd.DataFrame({"x": x, "y": y, "z": z})

        config = SurfacePlotConfig(
            x_column="x",
            y_column="y",
            z_column="z",
            grid_resolution=20,
        )

        engine = SurfacePlotEngine()
        result = engine.create_surface(df, config)

        assert result.x_grid.shape == (20, 20)
        assert result.y_grid.shape == (20, 20)
        assert result.z_grid.shape == (20, 20)

    def test_surface_with_smoothing(self) -> None:
        """Test surface creation with smoothing."""
        from data_processor.core.surface_plot import (
            SmoothingMethod,
            SurfacePlotConfig,
            SurfacePlotEngine,
        )

        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = np.random.uniform(0, 10, n)
        z = x + y + np.random.normal(0, 2, n)

        df = pd.DataFrame({"x": x, "y": y, "z": z})

        config = SurfacePlotConfig(
            x_column="x",
            y_column="y",
            z_column="z",
            smoothing_method=SmoothingMethod.GAUSSIAN,
            smoothing_sigma=1.5,
        )

        engine = SurfacePlotEngine()
        result = engine.create_surface(df, config)

        assert result.z_grid is not None
        assert result.statistics["n_points"] == n

    def test_surface_with_outlier_removal(self) -> None:
        """Test surface with outlier removal."""
        from data_processor.core.surface_plot import (
            SurfacePlotConfig,
            SurfacePlotEngine,
        )

        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = np.random.uniform(0, 10, n)
        z = x + y

        # Add outliers
        z[0] = 1000
        z[1] = -1000

        df = pd.DataFrame({"x": x, "y": y, "z": z})

        config = SurfacePlotConfig(
            x_column="x",
            y_column="y",
            z_column="z",
            remove_outliers=True,
            outlier_threshold=3.0,
        )

        engine = SurfacePlotEngine()
        result = engine.create_surface(df, config)

        # Should have fewer points after outlier removal
        assert result.statistics["n_points"] < n


# =============================================================================
# PCA TESTS
# =============================================================================


class TestPCAAnalysis:
    """Tests for PCA analysis."""

    def test_pca_basic_analysis(self, multivariate_df: pd.DataFrame) -> None:
        """Test basic PCA analysis."""
        from data_processor.core.pca_analysis import PCAAnalyzer

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(multivariate_df, columns=["x1", "x2", "x3"])

        assert result.n_components == 3
        assert result.n_features == 3
        assert len(result.components) == 3
        assert result.total_variance_explained > 0.99

    def test_pca_variance_explained(self, multivariate_df: pd.DataFrame) -> None:
        """Test that variance explained sums correctly."""
        from data_processor.core.pca_analysis import PCAAnalyzer

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(multivariate_df, columns=["x1", "x2", "x3"])

        total_var = sum(c.explained_variance_ratio for c in result.components)
        assert abs(total_var - 1.0) < 0.01

    def test_pca_feature_importance(self, multivariate_df: pd.DataFrame) -> None:
        """Test feature importance calculation."""
        from data_processor.core.pca_analysis import PCAAnalyzer

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(multivariate_df, columns=["x1", "x2", "x3"])

        assert len(result.feature_importance) == 3
        total_importance = sum(result.feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01

    def test_pca_transformed_data(self, multivariate_df: pd.DataFrame) -> None:
        """Test that transformed data has correct dimensions."""
        from data_processor.core.pca_analysis import PCAAnalyzer

        analyzer = PCAAnalyzer()
        result = analyzer.analyze(multivariate_df, columns=["x1", "x2", "x3"])

        assert result.transformed_data.shape[0] == len(multivariate_df)
        assert result.transformed_data.shape[1] == 3


# =============================================================================
# ANOVA TESTS
# =============================================================================


class TestANOVA:
    """Tests for ANOVA statistical analysis."""

    def test_one_way_anova_basic(self, anova_df: pd.DataFrame) -> None:
        """Test basic one-way ANOVA."""
        from data_processor.core.anova import ANOVAAnalyzer

        analyzer = ANOVAAnalyzer()
        result = analyzer.one_way_anova(
            anova_df, dependent_var="value", group_var="group"
        )

        assert result.f_statistic > 0
        assert result.p_value < 0.05  # Groups have different means
        assert result.df_between == 2  # 3 groups - 1
        assert len(result.group_means) == 3

    def test_anova_effect_sizes(self, anova_df: pd.DataFrame) -> None:
        """Test effect size calculations."""
        from data_processor.core.anova import ANOVAAnalyzer

        analyzer = ANOVAAnalyzer()
        result = analyzer.one_way_anova(
            anova_df, dependent_var="value", group_var="group"
        )

        assert 0 <= result.eta_squared <= 1
        assert result.omega_squared <= result.eta_squared
        assert result.cohens_f >= 0

    def test_anova_post_hoc_tests(self, anova_df: pd.DataFrame) -> None:
        """Test post-hoc comparisons."""
        from data_processor.core.anova import ANOVAAnalyzer, PostHocMethod

        analyzer = ANOVAAnalyzer()
        result = analyzer.one_way_anova(
            anova_df,
            dependent_var="value",
            group_var="group",
            post_hoc=PostHocMethod.BONFERRONI,
        )

        # Should have 3 pairwise comparisons (3 choose 2)
        assert len(result.post_hoc_results) == 3

    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    def test_two_way_anova(self) -> None:
        """Test two-way ANOVA."""
        from data_processor.core.anova import ANOVAAnalyzer

        np.random.seed(42)
        # Create 2x2 factorial design
        data = []
        for factor_a in ["low", "high"]:
            for factor_b in ["control", "treatment"]:
                effect_a = 5 if factor_a == "high" else 0
                effect_b = 3 if factor_b == "treatment" else 0
                for _ in range(20):
                    value = 10 + effect_a + effect_b + np.random.normal(0, 2)
                    data.append(
                        {"factor_a": factor_a, "factor_b": factor_b, "value": value}
                    )

        df = pd.DataFrame(data)

        analyzer = ANOVAAnalyzer()
        result = analyzer.two_way_anova(
            df,
            dependent_var="value",
            factor_a="factor_a",
            factor_b="factor_b",
        )

        assert result.factor_a_p < 0.05  # Significant main effect
        assert result.factor_b_p < 0.05  # Significant main effect

    def test_anova_assumption_tests(self, anova_df: pd.DataFrame) -> None:
        """Test assumption testing."""
        from data_processor.core.anova import ANOVAAnalyzer

        analyzer = ANOVAAnalyzer()
        result = analyzer.one_way_anova(
            anova_df,
            dependent_var="value",
            group_var="group",
            test_assumptions=True,
        )

        assert len(result.assumption_tests) > 0
        # Should include normality and homogeneity tests
        test_names = [t.test_name for t in result.assumption_tests]
        assert any("Normality" in name for name in test_names)


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestRegression:
    """Tests for multivariable regression."""

    def test_linear_regression_basic(self, multivariate_df: pd.DataFrame) -> None:
        """Test basic linear regression."""
        from data_processor.core.regression import MultivariateRegressor

        regressor = MultivariateRegressor()
        result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        assert result.r_squared > 0.8  # Should explain most variance
        assert len(result.coefficients) == 3
        assert result.f_p_value < 0.05

    def test_regression_coefficients(self, multivariate_df: pd.DataFrame) -> None:
        """Test coefficient estimation."""
        from data_processor.core.regression import MultivariateRegressor

        regressor = MultivariateRegressor()
        result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        # Check coefficients are close to true values (2, 3, -1.5)
        coef_dict = {c.name: c.estimate for c in result.coefficients}
        assert abs(coef_dict["x1"] - 2.0) < 0.5
        assert abs(coef_dict["x2"] - 3.0) < 0.5

    def test_regression_diagnostics(self, multivariate_df: pd.DataFrame) -> None:
        """Test diagnostic calculations."""
        from data_processor.core.regression import (
            MultivariateRegressor,
            RegressionConfig,
        )

        config = RegressionConfig(compute_diagnostics=True)
        regressor = MultivariateRegressor(config)
        result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        assert result.diagnostics is not None
        assert len(result.diagnostics.residuals) == len(multivariate_df)
        assert result.diagnostics.durbin_watson > 0

    def test_ridge_regression(self, multivariate_df: pd.DataFrame) -> None:
        """Test ridge regression."""
        from data_processor.core.regression import (
            MultivariateRegressor,
            RegressionConfig,
            RegularizationType,
        )

        config = RegressionConfig(regularization=RegularizationType.RIDGE, alpha=0.1)
        regressor = MultivariateRegressor(config)
        result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        assert result.r_squared > 0.5

    def test_surface_prediction(self, multivariate_df: pd.DataFrame) -> None:
        """Test surface prediction for plotting."""
        from data_processor.core.regression import MultivariateRegressor

        regressor = MultivariateRegressor()
        result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        x_grid, y_grid, z_grid = regressor.predict_surface(
            result,
            x_var="x1",
            y_var="x2",
            x_range=(-2, 2),
            y_range=(-2, 2),
            grid_size=10,
            fixed_values={"x3": 0},
        )

        assert x_grid.shape == (10, 10)
        assert z_grid.shape == (10, 10)


# =============================================================================
# NEURAL NETWORK TESTS
# =============================================================================


class TestNeuralNetwork:
    """Tests for neural network interface."""

    def test_create_config(self) -> None:
        """Test network configuration creation."""
        from data_processor.core.neural_network import (
            NetworkType,
            NeuralNetworkInterface,
        )

        nn = NeuralNetworkInterface()
        config = nn.create_config(
            input_features=10,
            output_features=1,
            network_type=NetworkType.MLP,
            hidden_layers=[64, 32],
        )

        assert config.input_features == 10
        assert config.output_features == 1
        assert len(config.layers) > 0

    def test_prepare_data(self, multivariate_df: pd.DataFrame) -> None:
        """Test data preparation."""
        from data_processor.core.neural_network import NeuralNetworkInterface

        nn = NeuralNetworkInterface()
        nn.create_config(input_features=3, output_features=1)

        data = nn.prepare_data(
            multivariate_df,
            target_columns=["y"],
            feature_columns=["x1", "x2", "x3"],
        )

        assert "X_train" in data
        assert "y_train" in data
        assert "X_val" in data
        assert "X_test" in data

    def test_simple_training(self, multivariate_df: pd.DataFrame) -> None:
        """Test simple NumPy-based training."""
        from data_processor.core.neural_network import NeuralNetworkInterface

        nn = NeuralNetworkInterface()
        config = nn.create_config(
            input_features=3,
            output_features=1,
            hidden_layers=[16, 8],
            epochs=10,
        )

        data = nn.prepare_data(
            multivariate_df,
            target_columns=["y"],
            feature_columns=["x1", "x2", "x3"],
        )

        result = nn.train_simple(data, config)

        assert len(result.train_loss_history) > 0
        assert result.final_train_loss < result.train_loss_history[0]

    def test_export_pytorch_script(self, temp_dir: Path) -> None:
        """Test PyTorch script export."""
        from data_processor.core.neural_network import (
            Framework,
            NeuralNetworkInterface,
        )

        nn = NeuralNetworkInterface()
        config = nn.create_config(input_features=5, output_features=1)

        output_path = temp_dir / "model.py"
        nn.export_script(config, output_path, framework=Framework.PYTORCH)

        assert output_path.exists()
        content = output_path.read_text()
        assert "import torch" in content
        assert "class NeuralNetwork" in content

    def test_export_tensorflow_script(self, temp_dir: Path) -> None:
        """Test TensorFlow script export."""
        from data_processor.core.neural_network import (
            Framework,
            NeuralNetworkInterface,
        )

        nn = NeuralNetworkInterface()
        config = nn.create_config(input_features=5, output_features=1)

        output_path = temp_dir / "model_tf.py"
        nn.export_script(config, output_path, framework=Framework.TENSORFLOW)

        assert output_path.exists()
        content = output_path.read_text()
        assert "tensorflow" in content
        assert "keras" in content


# =============================================================================
# SCRIPT GENERATOR TESTS
# =============================================================================


class TestScriptGenerator:
    """Tests for script generation."""

    def test_pipeline_recording(self) -> None:
        """Test pipeline operation recording."""
        from data_processor.core.script_generator import (
            OperationType,
            PipelineRecorder,
        )

        recorder = PipelineRecorder("Test Pipeline")

        recorder.record_load("input.csv")
        recorder.record_filter("Moving Average", {"ma_window": 5})
        recorder.record_export("output.csv")

        pipeline = recorder.pipeline
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].operation == OperationType.LOAD
        assert pipeline.steps[1].operation == OperationType.FILTER

    def test_python_script_generation(self, temp_dir: Path) -> None:
        """Test Python script generation."""
        from data_processor.core.script_generator import (
            OperationType,
            ProcessingPipeline,
            ProcessingStep,
            ScriptGenerator,
        )

        pipeline = ProcessingPipeline(name="Test")
        pipeline.steps = [
            ProcessingStep(
                operation=OperationType.LOAD,
                parameters={"file_path": "input.csv"},
            ),
            ProcessingStep(
                operation=OperationType.FILTER,
                parameters={
                    "filter_type": "Moving Average",
                    "filter_params": {"ma_window": 5},
                },
            ),
        ]

        generator = ScriptGenerator()
        script = generator.generate_python_script(pipeline)

        assert "import pandas" in script
        assert "def process_data" in script

    def test_pipeline_save_and_load(self, temp_dir: Path) -> None:
        """Test pipeline configuration save/load."""
        from data_processor.core.script_generator import (
            OperationType,
            ProcessingPipeline,
            ProcessingStep,
            ScriptGenerator,
        )

        pipeline = ProcessingPipeline(
            name="Test Pipeline",
            description="A test pipeline",
        )
        pipeline.steps = [
            ProcessingStep(
                operation=OperationType.FILTER,
                parameters={"filter_type": "Gaussian", "sigma": 2.0},
            ),
        ]

        generator = ScriptGenerator()
        config_path = temp_dir / "pipeline.json"
        generator.export_pipeline_config(pipeline, config_path)

        loaded = generator.import_pipeline_config(config_path)

        assert loaded.name == "Test Pipeline"
        assert len(loaded.steps) == 1

    def test_batch_script_generation(self) -> None:
        """Test batch processing script generation."""
        from data_processor.core.script_generator import (
            ProcessingPipeline,
            ScriptGenerator,
        )

        pipeline = ProcessingPipeline(name="Batch Test")
        generator = ScriptGenerator()

        script = generator.generate_batch_script(
            pipeline,
            input_patterns=["data/*.csv"],
            output_dir="output",
            parallel=True,
        )

        assert "ProcessPoolExecutor" in script
        assert "glob.glob" in script


# =============================================================================
# PLOT ZOOM TESTS
# =============================================================================


class TestPlotZoom:
    """Tests for mouse wheel zoom functionality."""

    def test_zoom_config_defaults(self) -> None:
        """Test zoom configuration defaults."""
        from data_processor.core.plot_zoom import ZoomConfig

        config = ZoomConfig()

        assert config.zoom_in_factor > 1
        assert config.zoom_out_factor < 1
        assert config.center_on_cursor is True

    def test_zoom_handler_creation(self) -> None:
        """Test zoom handler creation."""
        from data_processor.core.plot_zoom import MouseWheelZoom, ZoomConfig

        config = ZoomConfig(zoom_in_factor=1.5, zoom_out_factor=0.67)
        zoom = MouseWheelZoom(config)

        assert zoom.config.zoom_in_factor == 1.5

    def test_interactive_plot_manager(self) -> None:
        """Test interactive plot manager."""
        from data_processor.core.plot_zoom import InteractivePlotManager

        manager = InteractivePlotManager()
        assert manager.zoom_handler is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_analysis_pipeline(self, multivariate_df: pd.DataFrame) -> None:
        """Test a complete analysis workflow."""
        from data_processor.core.pca_analysis import PCAAnalyzer
        from data_processor.core.regression import MultivariateRegressor

        # 1. PCA to understand data structure
        pca = PCAAnalyzer()
        pca_result = pca.analyze(multivariate_df, columns=["x1", "x2", "x3"])

        # 2. Check which components are important
        important_components = pca.select_components_by_variance(
            pca_result, variance_threshold=0.95
        )
        assert len(important_components) >= 2

        # 3. Regression with original features
        regressor = MultivariateRegressor()
        reg_result = regressor.fit(
            multivariate_df, target="y", predictors=["x1", "x2", "x3"]
        )

        assert reg_result.r_squared > 0.8

    def test_dataset_manager_with_filter_workflow(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test dataset manager with filtering workflow."""
        from data_processor.core.dataset_manager import DatasetManager

        manager = DatasetManager()
        manager.load_from_dataframe(sample_df, name="Test")

        # Simulate filter application
        filtered_df = sample_df.copy()
        filtered_df["signal_a"] = filtered_df["signal_a"].rolling(5).mean()

        manager.save_version(
            filtered_df.dropna(),
            "moving_average",
            "Applied 5-point moving average",
            {"window": 5},
        )

        # Verify we can undo
        assert manager.can_undo
        original = manager.undo()
        assert len(original) == len(sample_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
