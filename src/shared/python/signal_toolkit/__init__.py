"""Signal Processing Toolkit - Comprehensive Signal Analysis Library.

A production-ready signal processing library for generating, fitting,
filtering, and analyzing signals. Designed for use in control systems,
simulation, robotics, and data analysis applications.

This package is part of the shared Tools repository and can be used
across multiple projects including UpstreamDrift and Gasification_Model.

Features:
    - Signal Generation: 13 signal types (sine, cosine, chirp, etc.)
    - Function Fitting: Sinusoid, exponential, linear, polynomial, custom
    - Digital Filters: Butterworth, Chebyshev, Bessel, adaptive (LMS/RLS)
    - Calculus: Differentiation, integration, tangent lines, curvature
    - Series: Taylor/Maclaurin series expansions with convergence analysis
    - Noise Generation: White, pink, brown, blue, violet, impulse
    - Limits: Saturation, rate limiting, deadband, hysteresis, backlash
    - I/O: CSV, JSON, NPZ, MAT, numpy array support
    - Visualization: PyQt6 widget for interactive signal analysis

Usage:
    from signal_toolkit import Signal, SignalGenerator, FunctionFitter

    # Create a signal
    import numpy as np
    t = np.linspace(0, 10, 1000)
    signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=2.0)

    # Fit a function
    fitter = FunctionFitter()
    result = fitter.fit_sinusoid(signal)

    # Apply a filter
    from signal_toolkit import create_butterworth_filter, apply_filter
    filter_spec = create_butterworth_filter('lowpass', cutoff=5, fs=100, order=4)
    filtered = apply_filter(signal, filter_spec)

Dependencies:
    Required: numpy, scipy
    Optional: matplotlib (visualization), PyQt6 (GUI widget)

Version: 2.1.0

Lazy-loading strategy (issue #1696 — god module refactor):
    All submodule imports are deferred to first attribute access via
    ``__getattr__``.  This avoids importing scipy, numpy, and all
    submodules on every ``import signal_toolkit``.
    Only the logger and optional-widget guards are eagerly evaluated.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__ = "2.1.0"

# ---------------------------------------------------------------------------
# Lazy import dispatch table
# Maps public name -> (relative_module, attribute_in_that_module)
# ---------------------------------------------------------------------------
_LAZY: dict[str, tuple[str, str]] = {
    # Core
    "Signal": (".core", "Signal"),
    "SignalGenerator": (".core", "SignalGenerator"),
    # Fitting
    "FitResult": (".fitting", "FitResult"),
    "FunctionFitter": (".fitting", "FunctionFitter"),
    "SinusoidFitter": (".fitting", "SinusoidFitter"),
    "CosineFitter": (".fitting", "CosineFitter"),
    "ExponentialFitter": (".fitting", "ExponentialFitter"),
    "LinearFitter": (".fitting", "LinearFitter"),
    "PolynomialFitter": (".fitting", "PolynomialFitter"),
    "CustomFunctionFitter": (".fitting", "CustomFunctionFitter"),
    # Filters
    "FilterType": (".filters", "FilterType"),
    "FilterDesign": (".filters", "FilterDesign"),
    "FilterSpec": (".filters", "FilterSpec"),
    "FilterDesigner": (".filters", "FilterDesigner"),
    "AdaptiveFilter": (".filters", "AdaptiveFilter"),
    "apply_filter": (".filters", "apply_filter"),
    "apply_moving_average": (".filters", "apply_moving_average"),
    "apply_savgol": (".filters", "apply_savgol"),
    "apply_median_filter": (".filters", "apply_median_filter"),
    "apply_exponential_smoothing": (".filters", "apply_exponential_smoothing"),
    "apply_gaussian_smoothing": (".filters", "apply_gaussian_smoothing"),
    "apply_bilateral_filter": (".filters", "apply_bilateral_filter"),
    "create_butterworth_filter": (".filters", "create_butterworth_filter"),
    "create_chebyshev_filter": (".filters", "create_chebyshev_filter"),
    "create_moving_average_filter": (".filters", "create_moving_average_filter"),
    "create_savgol_filter": (".filters", "create_savgol_filter"),
    # Calculus
    "DifferentiationMethod": (".calculus", "DifferentiationMethod"),
    "IntegrationMethod": (".calculus", "IntegrationMethod"),
    "TangentLine": (".calculus", "TangentLine"),
    "IntegralResult": (".calculus", "IntegralResult"),
    "Differentiator": (".calculus", "Differentiator"),
    "Integrator": (".calculus", "Integrator"),
    "compute_derivative": (".calculus", "compute_derivative"),
    "compute_integral": (".calculus", "compute_integral"),
    "compute_tangent_line": (".calculus", "compute_tangent_line"),
    "compute_curvature": (".calculus", "compute_curvature"),
    "compute_arc_length": (".calculus", "compute_arc_length"),
    "find_extrema": (".calculus", "find_extrema"),
    "find_inflection_points": (".calculus", "find_inflection_points"),
    # Limits
    "SaturationMode": (".limits", "SaturationMode"),
    "apply_saturation": (".limits", "apply_saturation"),
    "apply_rate_limiter": (".limits", "apply_rate_limiter"),
    "apply_deadband": (".limits", "apply_deadband"),
    "apply_hysteresis": (".limits", "apply_hysteresis"),
    "apply_backlash": (".limits", "apply_backlash"),
    "create_saturation_function": (".limits", "create_saturation_function"),
    "visualize_saturation_curves": (".limits", "visualize_saturation_curves"),
    # Noise
    "NoiseType": (".noise", "NoiseType"),
    "NoiseGenerator": (".noise", "NoiseGenerator"),
    "DisturbanceSimulator": (".noise", "DisturbanceSimulator"),
    "add_noise_to_signal": (".noise", "add_noise_to_signal"),
    "generate_disturbance_profile": (".noise", "generate_disturbance_profile"),
    # I/O
    "SignalImporter": (".io", "SignalImporter"),
    "SignalExporter": (".io", "SignalExporter"),
    "SignalLoader": (".io", "SignalLoader"),
    "BatchProcessor": (".io", "BatchProcessor"),
    "import_from_csv": (".io", "import_from_csv"),
    "export_to_csv": (".io", "export_to_csv"),
    # Series (Taylor/Maclaurin)
    "SeriesExpansion": (".series", "SeriesExpansion"),
    "SeriesResult": (".series", "SeriesResult"),
    "exp_series": (".series", "exp_series"),
    "sin_series": (".series", "sin_series"),
    "cos_series": (".series", "cos_series"),
    "ln_series": (".series", "ln_series"),
    "geometric_series": (".series", "geometric_series"),
    "arctan_series": (".series", "arctan_series"),
    "sinh_series": (".series", "sinh_series"),
    "cosh_series": (".series", "cosh_series"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute loader — defers submodule imports to first access."""
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        # Cache in module namespace to avoid repeated __getattr__ calls
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Optional imports (PyQt6 / matplotlib) — eagerly resolved at import time
# so that HAS_* flags are available without triggering __getattr__.
# ---------------------------------------------------------------------------

# Optional: Polynomial generator (PyQt6 required)
try:
    from .polynomial_generator import PolynomialGeneratorWidget

    HAS_POLYNOMIAL_GENERATOR = True
except ImportError:
    PolynomialGeneratorWidget = None  # type: ignore[misc, assignment]
    HAS_POLYNOMIAL_GENERATOR = False
    logger.debug("PolynomialGeneratorWidget not available (requires PyQt6)")

# Optional: Interactive widget (PyQt6 + matplotlib required)
try:
    from .widget import SignalToolkitWidget

    HAS_WIDGET = True
except ImportError:
    SignalToolkitWidget = None  # type: ignore[misc, assignment]
    HAS_WIDGET = False
    logger.debug("SignalToolkitWidget not available (requires PyQt6 + matplotlib)")


__all__ = [
    # Core
    "Signal",
    "SignalGenerator",
    # Fitting
    "FitResult",
    "FunctionFitter",
    "SinusoidFitter",
    "CosineFitter",
    "ExponentialFitter",
    "LinearFitter",
    "PolynomialFitter",
    "CustomFunctionFitter",
    # Filters
    "FilterType",
    "FilterDesign",
    "FilterSpec",
    "FilterDesigner",
    "AdaptiveFilter",
    "apply_filter",
    "apply_moving_average",
    "apply_savgol",
    "apply_median_filter",
    "apply_exponential_smoothing",
    "apply_gaussian_smoothing",
    "apply_bilateral_filter",
    "create_butterworth_filter",
    "create_chebyshev_filter",
    "create_moving_average_filter",
    "create_savgol_filter",
    # Calculus
    "DifferentiationMethod",
    "IntegrationMethod",
    "TangentLine",
    "IntegralResult",
    "Differentiator",
    "Integrator",
    "compute_derivative",
    "compute_integral",
    "compute_tangent_line",
    "compute_curvature",
    "compute_arc_length",
    "find_extrema",
    "find_inflection_points",
    # Limits
    "SaturationMode",
    "apply_saturation",
    "apply_rate_limiter",
    "apply_deadband",
    "apply_hysteresis",
    "apply_backlash",
    "create_saturation_function",
    "visualize_saturation_curves",
    # Noise
    "NoiseType",
    "NoiseGenerator",
    "DisturbanceSimulator",
    "add_noise_to_signal",
    "generate_disturbance_profile",
    # I/O
    "SignalImporter",
    "SignalExporter",
    "SignalLoader",
    "BatchProcessor",
    "import_from_csv",
    "export_to_csv",
    # Series (Taylor/Maclaurin)
    "SeriesExpansion",
    "SeriesResult",
    "exp_series",
    "sin_series",
    "cos_series",
    "ln_series",
    "geometric_series",
    "arctan_series",
    "sinh_series",
    "cosh_series",
    # Optional (GUI)
    "PolynomialGeneratorWidget",
    "SignalToolkitWidget",
    "HAS_POLYNOMIAL_GENERATOR",
    "HAS_WIDGET",
]
