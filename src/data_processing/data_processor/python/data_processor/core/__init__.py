"""Core data processor modules.

This package provides the shared processing logic for all GUI implementations.
All modules here are UI-agnostic and can be used by TKinter, PyQt6, React, or CLI.

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

from data_processor.core.anova import (
    ANOVAAnalyzer,
    ANOVATable,
    OneWayANOVAResult,
    PostHocComparison,
    PostHocMethod,
    RepeatedMeasuresResult,
    TwoWayANOVAResult,
    format_anova_report,
)
from data_processor.core.config_manager import ConfigManager
from data_processor.core.cross_correlation import (
    CausalityMethod,
    CrossCorrelationAnalyzer,
    CrossCorrelationConfig,
    CrossCorrelationResult,
    GrangerCausalityResult,
    NormalizationMethod,
    RollingCorrelationResult,
    TransferEntropyResult,
    cross_correlate,
    granger_causality,
)
from data_processor.core.dat_importer import (
    DBF_AVAILABLE,
    export_dat_to_csv,
    get_dat_columns,
    get_dat_file_info,
    import_dat_with_tags,
    preview_dat_file,
    read_dat_file,
    read_dbf_tags,
)
from data_processor.core.data_augmentation import (
    AugmentationConfig,
    AugmentationMethod,
    AugmentationResult,
    DataAugmenter,
    augment_data,
)

# New analysis modules
from data_processor.core.dataset_manager import (
    DatasetHistory,
    DatasetManager,
    DatasetMetadata,
    DatasetVersion,
)
from data_processor.core.dataset_naming import (
    generate_dataset_name,
    generate_unique_name,
    sanitize_dataset_name,
    validate_dataset_name,
)
from data_processor.core.feature_engineering import (
    FeatureCategory,
    FeatureConfig,
    FeatureExtractor,
    FeatureResult,
    FeatureSelector,
    FeatureTransformer,
    SelectionResult,
    TransformationType,
    extract_features,
    select_features,
)

# Advanced analysis modules
from data_processor.core.kalman_filter import (
    ExtendedKalmanFilter,
    KalmanFilter,
    KalmanFilterConfig,
    KalmanFilterResult,
    UnscentedKalmanFilter,
    kalman_smooth,
)
from data_processor.core.neural_network import (
    ActivationFunction,
    DataSplitConfig,
    Framework,
    LayerConfig,
    LossFunction,
    NetworkConfig,
    NetworkType,
    NeuralNetworkInterface,
    Optimizer,
    TrainingResult,
)
from data_processor.core.outlier_detection import (
    OutlierConfig,
    OutlierDetectionMethod,
    OutlierDetector,
    OutlierResult,
    detect_outliers,
)
from data_processor.core.pca_analysis import (
    PCAAnalyzer,
    PCAComponent,
    PCAConfig,
    PCAResult,
    create_loading_plot,
    create_scree_plot,
)
from data_processor.core.plot_config_manager import PlotConfigManager
from data_processor.core.plot_zoom import (
    InteractivePlotManager,
    MouseWheelZoom,
    ZoomConfig,
    enable_wheel_zoom,
    enable_wheel_zoom_all_figures,
)
from data_processor.core.regression import (
    CoefficientInfo,
    MultivariateRegressor,
    RegressionConfig,
    RegressionDiagnostics,
    RegressionResult,
    RegularizationType,
    SelectionMethod,
    format_regression_report,
)
from data_processor.core.script_generator import (
    OperationType,
    PipelineExecutor,
    PipelineRecorder,
    ProcessingPipeline,
    ProcessingStep,
    ScriptGenerator,
)
from data_processor.core.signal_list_manager import SignalListManager
from data_processor.core.signal_processing import (
    DifferentiationMethod,
    IntegrationMethod,
    TrendlineType,
    apply_custom_variable,
    calculate_trendline,
    differentiate_signals,
    integrate_signals,
    resample_data,
    trim_time_range,
)
from data_processor.core.spectral_analysis import (
    CoherenceResult,
    SpectralAnalyzer,
    SpectralConfig,
    SpectralResult,
    SpectrogramResult,
    WindowFunction,
    compute_spectrum,
)
from data_processor.core.state_space import (
    ARIMAStateSpace,
    BaseStateSpaceModel,
    ForecastResult,
    LocalLevelModel,
    LocalLinearTrendModel,
    OptimizationMethod,
)
from data_processor.core.state_space import SeasonalModel as SeasonalStateSpaceModel
from data_processor.core.state_space import (
    StateSpaceConfig,
    StateSpaceModelFactory,
    StateSpaceModelType,
    StateSpaceResult,
    fit_state_space,
)
from data_processor.core.surface_plot import (
    InterpolationMethod,
    SmoothingMethod,
    SurfacePlotConfig,
    SurfacePlotEngine,
    SurfacePlotResult,
    plot_surface_matplotlib,
)
from data_processor.core.time_series_decomposition import (
    DecompositionConfig,
    DecompositionMethod,
    DecompositionResult,
    SeasonalityDetectionResult,
    SeasonalModel,
    TimeSeriesDecomposer,
    TrendModel,
    decompose_time_series,
)
from data_processor.core.uncertainty_quantification import (
    BootstrapMethod,
    BootstrapResult,
    ConfidenceInterval,
    MonteCarloResult,
    PredictionInterval,
    SensitivityResult,
    UncertaintyConfig,
    UncertaintyMethod,
    UncertaintyQuantifier,
    bootstrap_confidence_interval,
    propagate_uncertainty,
)
from data_processor.core.undo_redo import (
    ColumnOperationCommand,
    Command,
    CompositeCommand,
    FilterCommand,
    LambdaCommand,
    RowFilterCommand,
    UndoRedoManager,
)
from data_processor.core.wavelet_denoising import (
    ThresholdingMethod,
    WaveletDenoiseConfig,
    WaveletDenoiser,
    WaveletDenoiseResult,
    WaveletFamily,
    denoise_signal,
)

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
