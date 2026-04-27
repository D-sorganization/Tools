"""Core data processor modules.

This package provides the shared processing logic for all GUI implementations.
All modules here are UI-agnostic and can be used by TKinter, PyQt6, React, or CLI.

Fixed in issue #530: converted to lazy imports via ``__getattr__`` to break
the fragile import chain that required ``src/python/src`` on ``sys.path``
at package-load time.

Refactored in issue #1696: the lazy-import dispatch table has been moved to
``_lazy_map.py`` to keep this file below 120 lines.

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

from data_processor.core._lazy_map import LAZY_IMPORTS

__all__ = list(LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load attributes on first access (see issue #530)."""
    if name in LAZY_IMPORTS:
        module_path, attr_name = LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'data_processor.core' has no attribute {name!r}")
