"""Signal Processing Toolkit - Comprehensive Signal Analysis Library.

A production-ready signal processing library for generating, fitting,
filtering, and analyzing signals.

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

Version: 2.1.0

Lazy-loading strategy (issue #1696 - god module refactor):
    All submodule imports are deferred to first attribute access via
    ``__getattr__``.  The dispatch table has been moved to ``_lazy_map.py``
    to keep this file below 120 lines.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

<<<<<<< HEAD
from signal_toolkit._lazy_map import LAZY
=======
from .adaptive_filter import AdaptiveFilter
from .calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegralResult,
    IntegrationMethod,
    Integrator,
    TangentLine,
    compute_arc_length,
    compute_curvature,
    compute_derivative,
    compute_integral,
    compute_tangent_line,
    find_extrema,
    find_inflection_points,
)
from .core import Signal, SignalGenerator
from .filters import (
    FilterDesign,
    FilterDesigner,
    FilterSpec,
    FilterType,
    apply_bilateral_filter,
    apply_exponential_smoothing,
    apply_filter,
    apply_gaussian_smoothing,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
    create_butterworth_filter,
    create_chebyshev_filter,
    create_moving_average_filter,
    create_savgol_filter,
)
from .fitting import (
    CosineFitter,
    CustomFunctionFitter,
    ExponentialFitter,
    FitResult,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)
from .io import (
    BatchProcessor,
    SignalExporter,
    SignalImporter,
    SignalLoader,
    export_to_csv,
    import_from_csv,
)
from .limits import (
    SaturationMode,
    apply_backlash,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
    create_saturation_function,
    visualize_saturation_curves,
)
from .noise import (
    DisturbanceSimulator,
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
    generate_disturbance_profile,
)
from .series import (
    SeriesExpansion,
    SeriesResult,
    arctan_series,
    cos_series,
    cosh_series,
    exp_series,
    geometric_series,
    ln_series,
    sin_series,
    sinh_series,
)
>>>>>>> origin/main

logger = logging.getLogger(__name__)

__version__ = "2.1.0"

# ---------------------------------------------------------------------------
# Optional imports (PyQt6 / matplotlib)
# ---------------------------------------------------------------------------

try:
    from .polynomial_generator import PolynomialGeneratorWidget

    HAS_POLYNOMIAL_GENERATOR = True
except ImportError:
    PolynomialGeneratorWidget = None  # type: ignore[misc, assignment]
    HAS_POLYNOMIAL_GENERATOR = False
    logger.debug("PolynomialGeneratorWidget not available (requires PyQt6)")

try:
    from .widget import SignalToolkitWidget

    HAS_WIDGET = True
except ImportError:
    SignalToolkitWidget = None  # type: ignore[misc, assignment]
    HAS_WIDGET = False
    logger.debug("SignalToolkitWidget not available (requires PyQt6 + matplotlib)")


def __getattr__(name: str) -> Any:
    """Lazy attribute loader - defers submodule imports to first access."""
    if name in LAZY:
        module_path, attr = LAZY[name]
        mod = importlib.import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *LAZY.keys(),
    "PolynomialGeneratorWidget",
    "SignalToolkitWidget",
    "HAS_POLYNOMIAL_GENERATOR",
    "HAS_WIDGET",
]
