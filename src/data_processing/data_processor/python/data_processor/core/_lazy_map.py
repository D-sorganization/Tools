"""Lazy import dispatch table for ``data_processor.core``.

Extracted from ``__init__.py`` (issue #1696) to keep the package entry-point
below 120 lines.  Each entry maps a public name to the (module_path,
attribute_name) pair that provides it.
"""

from __future__ import annotations

LAZY_IMPORTS: dict[str, tuple[str, str]] = {
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
