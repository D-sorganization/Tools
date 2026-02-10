"""Core data processor modules.

This package provides the shared processing logic for all GUI implementations.
All modules here are UI-agnostic and can be used by TKinter, PyQt6, React, or CLI.

Fixed in issue #530: converted to lazy imports via ``__getattr__`` to break
the fragile import chain that required ``src/python/src`` on ``sys.path``
at package-load time.

Modules:
    signal_processing: Integration, differentiation, resampling, formulas, trendlines
    config_manager: Application settings persistence
    signal_list_manager: Signal selection persistence
    plot_config_manager: Plot configuration persistence
    dataset_naming: File naming utilities
    dat_importer: DAT/DBF file import
    data_loader: CSV/data file loading
    signal_processor: Filter orchestration
    dataset_manager: Dataset state management with undo/redo
    undo_redo: Command pattern undo/redo system
    surface_plot: 3D surface plotting with smoothing
    pca_analysis: Principal Component Analysis
    anova: Comprehensive ANOVA statistical suite
    regression: Multivariable regression analysis
    neural_network: Neural network training interface
    script_generator: Automated processing script generation
    plot_zoom: Mouse wheel zoom for plots
    kalman_filter: Kalman filtering (Standard, Extended, Unscented)
    wavelet_denoising: Wavelet-based signal denoising
    spectral_analysis: Spectral and frequency domain analysis
    outlier_detection: Ensemble outlier detection methods
    time_series_decomposition: STL and classical decomposition
    cross_correlation: Cross-correlation and causality analysis
    state_space: State space modeling and estimation
    uncertainty_quantification: Bootstrap and Monte Carlo uncertainty
    data_augmentation: Data augmentation techniques
    feature_engineering: Automated feature extraction and selection
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Signal processing
    "integrate_signals",
    "differentiate_signals",
    "resample_data",
    "apply_custom_variable",
    "calculate_trendline",
    "trim_time_range",
    "IntegrationMethod",
    "DifferentiationMethod",
    "TrendlineType",
    # Configuration
    "ConfigManager",
    "SignalListManager",
    "PlotConfigManager",
    # Dataset naming
    "generate_dataset_name",
    "validate_dataset_name",
    "sanitize_dataset_name",
    "generate_unique_name",
    # DAT import
    "read_dat_file",
    "read_dbf_tags",
    "get_dat_columns",
    "import_dat_with_tags",
    "export_dat_to_csv",
    "preview_dat_file",
    "get_dat_file_info",
    "DBF_AVAILABLE",
    # Dataset management
    "DatasetManager",
    "DatasetVersion",
    "DatasetHistory",
    "DatasetMetadata",
    # Undo/Redo
    "UndoRedoManager",
    "Command",
    "FilterCommand",
    "ColumnOperationCommand",
    "RowFilterCommand",
    "CompositeCommand",
    "LambdaCommand",
    # Surface plots
    "SurfacePlotEngine",
    "SurfacePlotConfig",
    "SurfacePlotResult",
    "InterpolationMethod",
    "SmoothingMethod",
    "plot_surface_matplotlib",
    # PCA
    "PCAAnalyzer",
    "PCAConfig",
    "PCAComponent",
    "PCAResult",
    "create_scree_plot",
    "create_loading_plot",
    # ANOVA
    "ANOVAAnalyzer",
    "ANOVATable",
    "OneWayANOVAResult",
    "TwoWayANOVAResult",
    "RepeatedMeasuresResult",
    "PostHocMethod",
    "PostHocComparison",
    "format_anova_report",
    # Regression
    "MultivariateRegressor",
    "RegressionConfig",
    "RegressionResult",
    "RegressionDiagnostics",
    "CoefficientInfo",
    "RegularizationType",
    "SelectionMethod",
    "format_regression_report",
    # Neural Networks
    "NeuralNetworkInterface",
    "NetworkConfig",
    "NetworkType",
    "LayerConfig",
    "ActivationFunction",
    "LossFunction",
    "Optimizer",
    "Framework",
    "TrainingResult",
    "DataSplitConfig",
    # Script generation
    "ScriptGenerator",
    "PipelineRecorder",
    "PipelineExecutor",
    "ProcessingPipeline",
    "ProcessingStep",
    "OperationType",
    # Plot zoom
    "MouseWheelZoom",
    "ZoomConfig",
    "InteractivePlotManager",
    "enable_wheel_zoom",
    "enable_wheel_zoom_all_figures",
    # Kalman Filter
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "KalmanFilterConfig",
    "KalmanFilterResult",
    "kalman_smooth",
    # Wavelet Denoising
    "WaveletDenoiser",
    "WaveletDenoiseConfig",
    "WaveletDenoiseResult",
    "WaveletFamily",
    "ThresholdingMethod",
    "denoise_signal",
    # Spectral Analysis
    "SpectralAnalyzer",
    "SpectralConfig",
    "SpectralResult",
    "SpectrogramResult",
    "CoherenceResult",
    "WindowFunction",
    "compute_spectrum",
    # Outlier Detection
    "OutlierDetector",
    "OutlierConfig",
    "OutlierResult",
    "OutlierDetectionMethod",
    "detect_outliers",
    # Time Series Decomposition
    "TimeSeriesDecomposer",
    "DecompositionConfig",
    "DecompositionResult",
    "DecompositionMethod",
    "SeasonalModel",
    "TrendModel",
    "SeasonalityDetectionResult",
    "decompose_time_series",
    # Cross-Correlation
    "CrossCorrelationAnalyzer",
    "CrossCorrelationConfig",
    "CrossCorrelationResult",
    "GrangerCausalityResult",
    "TransferEntropyResult",
    "RollingCorrelationResult",
    "NormalizationMethod",
    "CausalityMethod",
    "cross_correlate",
    "granger_causality",
    # State Space
    "BaseStateSpaceModel",
    "LocalLevelModel",
    "LocalLinearTrendModel",
    "SeasonalStateSpaceModel",
    "ARIMAStateSpace",
    "StateSpaceModelFactory",
    "StateSpaceConfig",
    "StateSpaceResult",
    "ForecastResult",
    "StateSpaceModelType",
    "OptimizationMethod",
    "fit_state_space",
    # Uncertainty Quantification
    "UncertaintyQuantifier",
    "UncertaintyConfig",
    "ConfidenceInterval",
    "BootstrapResult",
    "MonteCarloResult",
    "SensitivityResult",
    "PredictionInterval",
    "BootstrapMethod",
    "UncertaintyMethod",
    "bootstrap_confidence_interval",
    "propagate_uncertainty",
    # Data Augmentation
    "DataAugmenter",
    "AugmentationConfig",
    "AugmentationResult",
    "AugmentationMethod",
    "augment_data",
    # Feature Engineering
    "FeatureExtractor",
    "FeatureSelector",
    "FeatureTransformer",
    "FeatureConfig",
    "FeatureResult",
    "SelectionResult",
    "FeatureCategory",
    "TransformationType",
    "extract_features",
    "select_features",
]

# ---------------------------------------------------------------------------
# Lazy import mapping: name -> (module_path, attribute_name)
# See issue #530 -- deferred to avoid loading all 20+ analysis modules
# (and their heavy deps like scipy, sklearn, etc.) at package import time.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # signal_processing
    "integrate_signals": ("data_processor.core.signal_processing", "integrate_signals"),
    "differentiate_signals": (
        "data_processor.core.signal_processing",
        "differentiate_signals",
    ),
    "resample_data": ("data_processor.core.signal_processing", "resample_data"),
    "apply_custom_variable": (
        "data_processor.core.signal_processing",
        "apply_custom_variable",
    ),
    "calculate_trendline": (
        "data_processor.core.signal_processing",
        "calculate_trendline",
    ),
    "trim_time_range": ("data_processor.core.signal_processing", "trim_time_range"),
    "IntegrationMethod": ("data_processor.core.signal_processing", "IntegrationMethod"),
    "DifferentiationMethod": (
        "data_processor.core.signal_processing",
        "DifferentiationMethod",
    ),
    "TrendlineType": ("data_processor.core.signal_processing", "TrendlineType"),
    # config_manager
    "ConfigManager": ("data_processor.core.config_manager", "ConfigManager"),
    "SignalListManager": (
        "data_processor.core.signal_list_manager",
        "SignalListManager",
    ),
    "PlotConfigManager": (
        "data_processor.core.plot_config_manager",
        "PlotConfigManager",
    ),
    # dataset_naming
    "generate_dataset_name": (
        "data_processor.core.dataset_naming",
        "generate_dataset_name",
    ),
    "validate_dataset_name": (
        "data_processor.core.dataset_naming",
        "validate_dataset_name",
    ),
    "sanitize_dataset_name": (
        "data_processor.core.dataset_naming",
        "sanitize_dataset_name",
    ),
    "generate_unique_name": (
        "data_processor.core.dataset_naming",
        "generate_unique_name",
    ),
    # dat_importer
    "read_dat_file": ("data_processor.core.dat_importer", "read_dat_file"),
    "read_dbf_tags": ("data_processor.core.dat_importer", "read_dbf_tags"),
    "get_dat_columns": ("data_processor.core.dat_importer", "get_dat_columns"),
    "import_dat_with_tags": (
        "data_processor.core.dat_importer",
        "import_dat_with_tags",
    ),
    "export_dat_to_csv": ("data_processor.core.dat_importer", "export_dat_to_csv"),
    "preview_dat_file": ("data_processor.core.dat_importer", "preview_dat_file"),
    "get_dat_file_info": ("data_processor.core.dat_importer", "get_dat_file_info"),
    "DBF_AVAILABLE": ("data_processor.core.dat_importer", "DBF_AVAILABLE"),
    # dataset_manager
    "DatasetManager": ("data_processor.core.dataset_manager", "DatasetManager"),
    "DatasetVersion": ("data_processor.core.dataset_manager", "DatasetVersion"),
    "DatasetHistory": ("data_processor.core.dataset_manager", "DatasetHistory"),
    "DatasetMetadata": ("data_processor.core.dataset_manager", "DatasetMetadata"),
    # undo_redo
    "UndoRedoManager": ("data_processor.core.undo_redo", "UndoRedoManager"),
    "Command": ("data_processor.core.undo_redo", "Command"),
    "FilterCommand": ("data_processor.core.undo_redo", "FilterCommand"),
    "ColumnOperationCommand": (
        "data_processor.core.undo_redo",
        "ColumnOperationCommand",
    ),
    "RowFilterCommand": ("data_processor.core.undo_redo", "RowFilterCommand"),
    "CompositeCommand": ("data_processor.core.undo_redo", "CompositeCommand"),
    "LambdaCommand": ("data_processor.core.undo_redo", "LambdaCommand"),
    # surface_plot
    "SurfacePlotEngine": ("data_processor.core.surface_plot", "SurfacePlotEngine"),
    "SurfacePlotConfig": ("data_processor.core.surface_plot", "SurfacePlotConfig"),
    "SurfacePlotResult": ("data_processor.core.surface_plot", "SurfacePlotResult"),
    "InterpolationMethod": ("data_processor.core.surface_plot", "InterpolationMethod"),
    "SmoothingMethod": ("data_processor.core.surface_plot", "SmoothingMethod"),
    "plot_surface_matplotlib": (
        "data_processor.core.surface_plot",
        "plot_surface_matplotlib",
    ),
    # pca_analysis
    "PCAAnalyzer": ("data_processor.core.pca_analysis", "PCAAnalyzer"),
    "PCAConfig": ("data_processor.core.pca_analysis", "PCAConfig"),
    "PCAComponent": ("data_processor.core.pca_analysis", "PCAComponent"),
    "PCAResult": ("data_processor.core.pca_analysis", "PCAResult"),
    "create_scree_plot": ("data_processor.core.pca_analysis", "create_scree_plot"),
    "create_loading_plot": ("data_processor.core.pca_analysis", "create_loading_plot"),
    # anova
    "ANOVAAnalyzer": ("data_processor.core.anova", "ANOVAAnalyzer"),
    "ANOVATable": ("data_processor.core.anova", "ANOVATable"),
    "OneWayANOVAResult": ("data_processor.core.anova", "OneWayANOVAResult"),
    "TwoWayANOVAResult": ("data_processor.core.anova", "TwoWayANOVAResult"),
    "RepeatedMeasuresResult": ("data_processor.core.anova", "RepeatedMeasuresResult"),
    "PostHocMethod": ("data_processor.core.anova", "PostHocMethod"),
    "PostHocComparison": ("data_processor.core.anova", "PostHocComparison"),
    "format_anova_report": ("data_processor.core.anova", "format_anova_report"),
    # regression
    "MultivariateRegressor": (
        "data_processor.core.regression",
        "MultivariateRegressor",
    ),
    "RegressionConfig": ("data_processor.core.regression", "RegressionConfig"),
    "RegressionResult": ("data_processor.core.regression", "RegressionResult"),
    "RegressionDiagnostics": (
        "data_processor.core.regression",
        "RegressionDiagnostics",
    ),
    "CoefficientInfo": ("data_processor.core.regression", "CoefficientInfo"),
    "RegularizationType": ("data_processor.core.regression", "RegularizationType"),
    "SelectionMethod": ("data_processor.core.regression", "SelectionMethod"),
    "format_regression_report": (
        "data_processor.core.regression",
        "format_regression_report",
    ),
    # neural_network
    "NeuralNetworkInterface": (
        "data_processor.core.neural_network",
        "NeuralNetworkInterface",
    ),
    "NetworkConfig": ("data_processor.core.neural_network", "NetworkConfig"),
    "NetworkType": ("data_processor.core.neural_network", "NetworkType"),
    "LayerConfig": ("data_processor.core.neural_network", "LayerConfig"),
    "ActivationFunction": ("data_processor.core.neural_network", "ActivationFunction"),
    "LossFunction": ("data_processor.core.neural_network", "LossFunction"),
    "Optimizer": ("data_processor.core.neural_network", "Optimizer"),
    "Framework": ("data_processor.core.neural_network", "Framework"),
    "TrainingResult": ("data_processor.core.neural_network", "TrainingResult"),
    "DataSplitConfig": ("data_processor.core.neural_network", "DataSplitConfig"),
    # script_generator
    "ScriptGenerator": ("data_processor.core.script_generator", "ScriptGenerator"),
    "PipelineRecorder": ("data_processor.core.script_generator", "PipelineRecorder"),
    "PipelineExecutor": ("data_processor.core.script_generator", "PipelineExecutor"),
    "ProcessingPipeline": (
        "data_processor.core.script_generator",
        "ProcessingPipeline",
    ),
    "ProcessingStep": ("data_processor.core.script_generator", "ProcessingStep"),
    "OperationType": ("data_processor.core.script_generator", "OperationType"),
    # plot_zoom
    "MouseWheelZoom": ("data_processor.core.plot_zoom", "MouseWheelZoom"),
    "ZoomConfig": ("data_processor.core.plot_zoom", "ZoomConfig"),
    "InteractivePlotManager": (
        "data_processor.core.plot_zoom",
        "InteractivePlotManager",
    ),
    "enable_wheel_zoom": ("data_processor.core.plot_zoom", "enable_wheel_zoom"),
    "enable_wheel_zoom_all_figures": (
        "data_processor.core.plot_zoom",
        "enable_wheel_zoom_all_figures",
    ),
    # kalman_filter
    "KalmanFilter": ("data_processor.core.kalman_filter", "KalmanFilter"),
    "ExtendedKalmanFilter": (
        "data_processor.core.kalman_filter",
        "ExtendedKalmanFilter",
    ),
    "UnscentedKalmanFilter": (
        "data_processor.core.kalman_filter",
        "UnscentedKalmanFilter",
    ),
    "KalmanFilterConfig": ("data_processor.core.kalman_filter", "KalmanFilterConfig"),
    "KalmanFilterResult": ("data_processor.core.kalman_filter", "KalmanFilterResult"),
    "kalman_smooth": ("data_processor.core.kalman_filter", "kalman_smooth"),
    # wavelet_denoising
    "WaveletDenoiser": ("data_processor.core.wavelet_denoising", "WaveletDenoiser"),
    "WaveletDenoiseConfig": (
        "data_processor.core.wavelet_denoising",
        "WaveletDenoiseConfig",
    ),
    "WaveletDenoiseResult": (
        "data_processor.core.wavelet_denoising",
        "WaveletDenoiseResult",
    ),
    "WaveletFamily": ("data_processor.core.wavelet_denoising", "WaveletFamily"),
    "ThresholdingMethod": (
        "data_processor.core.wavelet_denoising",
        "ThresholdingMethod",
    ),
    "denoise_signal": ("data_processor.core.wavelet_denoising", "denoise_signal"),
    # spectral_analysis
    "SpectralAnalyzer": ("data_processor.core.spectral_analysis", "SpectralAnalyzer"),
    "SpectralConfig": ("data_processor.core.spectral_analysis", "SpectralConfig"),
    "SpectralResult": ("data_processor.core.spectral_analysis", "SpectralResult"),
    "SpectrogramResult": ("data_processor.core.spectral_analysis", "SpectrogramResult"),
    "CoherenceResult": ("data_processor.core.spectral_analysis", "CoherenceResult"),
    "WindowFunction": ("data_processor.core.spectral_analysis", "WindowFunction"),
    "compute_spectrum": ("data_processor.core.spectral_analysis", "compute_spectrum"),
    # outlier_detection
    "OutlierDetector": ("data_processor.core.outlier_detection", "OutlierDetector"),
    "OutlierConfig": ("data_processor.core.outlier_detection", "OutlierConfig"),
    "OutlierResult": ("data_processor.core.outlier_detection", "OutlierResult"),
    "OutlierDetectionMethod": (
        "data_processor.core.outlier_detection",
        "OutlierDetectionMethod",
    ),
    "detect_outliers": ("data_processor.core.outlier_detection", "detect_outliers"),
    # time_series_decomposition
    "TimeSeriesDecomposer": (
        "data_processor.core.time_series_decomposition",
        "TimeSeriesDecomposer",
    ),
    "DecompositionConfig": (
        "data_processor.core.time_series_decomposition",
        "DecompositionConfig",
    ),
    "DecompositionResult": (
        "data_processor.core.time_series_decomposition",
        "DecompositionResult",
    ),
    "DecompositionMethod": (
        "data_processor.core.time_series_decomposition",
        "DecompositionMethod",
    ),
    "SeasonalModel": ("data_processor.core.time_series_decomposition", "SeasonalModel"),
    "TrendModel": ("data_processor.core.time_series_decomposition", "TrendModel"),
    "SeasonalityDetectionResult": (
        "data_processor.core.time_series_decomposition",
        "SeasonalityDetectionResult",
    ),
    "decompose_time_series": (
        "data_processor.core.time_series_decomposition",
        "decompose_time_series",
    ),
    # cross_correlation
    "CrossCorrelationAnalyzer": (
        "data_processor.core.cross_correlation",
        "CrossCorrelationAnalyzer",
    ),
    "CrossCorrelationConfig": (
        "data_processor.core.cross_correlation",
        "CrossCorrelationConfig",
    ),
    "CrossCorrelationResult": (
        "data_processor.core.cross_correlation",
        "CrossCorrelationResult",
    ),
    "GrangerCausalityResult": (
        "data_processor.core.cross_correlation",
        "GrangerCausalityResult",
    ),
    "TransferEntropyResult": (
        "data_processor.core.cross_correlation",
        "TransferEntropyResult",
    ),
    "RollingCorrelationResult": (
        "data_processor.core.cross_correlation",
        "RollingCorrelationResult",
    ),
    "NormalizationMethod": (
        "data_processor.core.cross_correlation",
        "NormalizationMethod",
    ),
    "CausalityMethod": ("data_processor.core.cross_correlation", "CausalityMethod"),
    "cross_correlate": ("data_processor.core.cross_correlation", "cross_correlate"),
    "granger_causality": ("data_processor.core.cross_correlation", "granger_causality"),
    # state_space
    "BaseStateSpaceModel": ("data_processor.core.state_space", "BaseStateSpaceModel"),
    "LocalLevelModel": ("data_processor.core.state_space", "LocalLevelModel"),
    "LocalLinearTrendModel": (
        "data_processor.core.state_space",
        "LocalLinearTrendModel",
    ),
    "SeasonalStateSpaceModel": ("data_processor.core.state_space", "SeasonalModel"),
    "ARIMAStateSpace": ("data_processor.core.state_space", "ARIMAStateSpace"),
    "StateSpaceModelFactory": (
        "data_processor.core.state_space",
        "StateSpaceModelFactory",
    ),
    "StateSpaceConfig": ("data_processor.core.state_space", "StateSpaceConfig"),
    "StateSpaceResult": ("data_processor.core.state_space", "StateSpaceResult"),
    "ForecastResult": ("data_processor.core.state_space", "ForecastResult"),
    "StateSpaceModelType": ("data_processor.core.state_space", "StateSpaceModelType"),
    "OptimizationMethod": ("data_processor.core.state_space", "OptimizationMethod"),
    "fit_state_space": ("data_processor.core.state_space", "fit_state_space"),
    # uncertainty_quantification
    "UncertaintyQuantifier": (
        "data_processor.core.uncertainty_quantification",
        "UncertaintyQuantifier",
    ),
    "UncertaintyConfig": (
        "data_processor.core.uncertainty_quantification",
        "UncertaintyConfig",
    ),
    "ConfidenceInterval": (
        "data_processor.core.uncertainty_quantification",
        "ConfidenceInterval",
    ),
    "BootstrapResult": (
        "data_processor.core.uncertainty_quantification",
        "BootstrapResult",
    ),
    "MonteCarloResult": (
        "data_processor.core.uncertainty_quantification",
        "MonteCarloResult",
    ),
    "SensitivityResult": (
        "data_processor.core.uncertainty_quantification",
        "SensitivityResult",
    ),
    "PredictionInterval": (
        "data_processor.core.uncertainty_quantification",
        "PredictionInterval",
    ),
    "BootstrapMethod": (
        "data_processor.core.uncertainty_quantification",
        "BootstrapMethod",
    ),
    "UncertaintyMethod": (
        "data_processor.core.uncertainty_quantification",
        "UncertaintyMethod",
    ),
    "bootstrap_confidence_interval": (
        "data_processor.core.uncertainty_quantification",
        "bootstrap_confidence_interval",
    ),
    "propagate_uncertainty": (
        "data_processor.core.uncertainty_quantification",
        "propagate_uncertainty",
    ),
    # data_augmentation
    "DataAugmenter": ("data_processor.core.data_augmentation", "DataAugmenter"),
    "AugmentationConfig": (
        "data_processor.core.data_augmentation",
        "AugmentationConfig",
    ),
    "AugmentationResult": (
        "data_processor.core.data_augmentation",
        "AugmentationResult",
    ),
    "AugmentationMethod": (
        "data_processor.core.data_augmentation",
        "AugmentationMethod",
    ),
    "augment_data": ("data_processor.core.data_augmentation", "augment_data"),
    # feature_engineering
    "FeatureExtractor": ("data_processor.core.feature_engineering", "FeatureExtractor"),
    "FeatureSelector": ("data_processor.core.feature_engineering", "FeatureSelector"),
    "FeatureTransformer": (
        "data_processor.core.feature_engineering",
        "FeatureTransformer",
    ),
    "FeatureConfig": ("data_processor.core.feature_engineering", "FeatureConfig"),
    "FeatureResult": ("data_processor.core.feature_engineering", "FeatureResult"),
    "SelectionResult": ("data_processor.core.feature_engineering", "SelectionResult"),
    "FeatureCategory": ("data_processor.core.feature_engineering", "FeatureCategory"),
    "TransformationType": (
        "data_processor.core.feature_engineering",
        "TransformationType",
    ),
    "extract_features": ("data_processor.core.feature_engineering", "extract_features"),
    "select_features": ("data_processor.core.feature_engineering", "select_features"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load attributes on first access (see issue #530)."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'data_processor.core' has no attribute {name!r}")
