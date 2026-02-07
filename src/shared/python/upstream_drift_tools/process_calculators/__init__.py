"""Process Engineering Calculators - Standalone Utility Tools.

This package contains standalone process engineering calculators that can be used
independently of any specific application. These tools are designed to be reusable
across multiple projects.

Categories:
    - Equipment Sizing: Pressure drop, scrubbers, baghouse, flare
    - Thermodynamic: Acid gas dewpoint, syngas water, WGS reactor
    - Process Analysis: PSA, syngas compression, optimization
    - Financial: NPV/IRR analysis, project economics
    - Mathematical: ODE solver, thermal profiles, multi-parameter analysis

Usage:
    from upstream_drift_tools.process_calculators import (
        AcidGasDewpointCalculator,
        ScrubberCalculator,
        FlareCalculator,
        FinancialCalculator,
    )

Note:
    Some calculators have optional dependencies (PyQt6, scipy, etc.).
    Core calculation functionality works without UI dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .acid_gas_dewpoint_calculator import AcidGasDewpointCalculator
from .baghouse_calculator import BaghouseCalculator, BaghouseResult
from .constants import (
    ATOL_ZERO,
    R_GAS_J_MOL_K,
    R_UNIVERSAL,
    STANDARD_GRAVITY,
    celsius_to_kelvin,
    kelvin_to_celsius,
)
from .electrode_advancement_calculator import ElectrodeAdvancementCalculator
from .financial_calculator import FinancialModelCalculator as FinancialCalculator
from .flare_calculator import FlareCalculator, FlareDesign
from .ode_solver import ODESolver
from .thermal_profile_predictor import (
    fit_heating_parameters,
    predict_temperature_profile,
)

if TYPE_CHECKING:
    from .multi_param_analysis import (
        MultiParameterAnalysis as MultiParameterAnalysisType,
    )
    from .optimization import AdamOptimizer as AdamOptimizerType
    from .optimization import Optimizer as OptimizerType
    from .pressure_drop_calculator import (
        PressureDropCalculator as PressureDropCalculatorType,
    )
    from .scrubber_calculator import ScrubberCalculator as ScrubberCalculatorType
    from .syngas_compression_calculator import (
        SyngasCompressionCalculator as SyngasCompressionCalculatorType,
    )
    from .syngas_water_calculator import (
        SyngasWaterCalculator as SyngasWaterCalculatorType,
    )
    from .water_vapor_pressure_calculator import (
        WaterVaporPressureCalculator as WaterVaporPressureCalculatorType,
    )
    from .wgs_reactor_calculator import WGSReactorEngine as WGSReactorEngineType

logger = logging.getLogger(__name__)

# Track import errors for optional modules
_import_errors: list[str] = []

# Calculators with numpy/scipy dependencies
try:
    from .scrubber_calculator import ScrubberCalculator
except ImportError as e:
    _import_errors.append(f"ScrubberCalculator not available: {e}")
    ScrubberCalculator: type[ScrubberCalculatorType] | None = None  # type: ignore[no-redef]

try:
    from .syngas_water_calculator import SyngasWaterCalculator
    from .water_vapor_pressure_calculator import WaterVaporPressureCalculator
except ImportError as e:
    _import_errors.append(f"Water calculators not available: {e}")
    SyngasWaterCalculator: type[SyngasWaterCalculatorType] | None = None  # type: ignore[no-redef]
    WaterVaporPressureCalculator: type[WaterVaporPressureCalculatorType] | None = None  # type: ignore[no-redef]

try:
    from .wgs_reactor_calculator import WGSReactorEngine

    WGSReactorCalculator = WGSReactorEngine  # Alias
except ImportError as e:
    _import_errors.append(f"WGSReactorCalculator not available: {e}")
    WGSReactorCalculator: type[WGSReactorEngineType] | None = None  # type: ignore[no-redef]
    WGSReactorEngine: type[WGSReactorEngineType] | None = None  # type: ignore[no-redef]

try:
    from .optimization import AdamOptimizer, Optimizer
except ImportError as e:
    _import_errors.append(f"Optimization not available: {e}")
    Optimizer: type[OptimizerType] | None = None  # type: ignore[no-redef]
    AdamOptimizer: type[AdamOptimizerType] | None = None  # type: ignore[no-redef]

try:
    from .multi_param_analysis import MultiParameterAnalysis
except ImportError as e:
    _import_errors.append(f"MultiParameterAnalysis not available: {e}")
    MultiParameterAnalysis: type[MultiParameterAnalysisType] | None = None  # type: ignore[no-redef]

# UI-dependent calculators (require PyQt6)
try:
    from .syngas_compression_calculator import SyngasCompressionCalculator
except ImportError as e:
    _import_errors.append(f"SyngasCompressionCalculator not available: {e}")
    SyngasCompressionCalculator: type[SyngasCompressionCalculatorType] | None = None  # type: ignore[no-redef]

# Modular packages
try:
    from .pressure_drop_calculator import PressureDropCalculator
except ImportError as e:
    _import_errors.append(f"PressureDropCalculator not available: {e}")
    PressureDropCalculator: type[PressureDropCalculatorType] | None = None  # type: ignore[no-redef]

# Log any import errors that occurred
for error in _import_errors:
    logger.debug(error)

__all__ = [
    # Always available
    "AcidGasDewpointCalculator",
    "FlareCalculator",
    "FlareDesign",
    "BaghouseCalculator",
    "BaghouseResult",
    "ElectrodeAdvancementCalculator",
    "FinancialCalculator",
    "ODESolver",
    "fit_heating_parameters",
    "predict_temperature_profile",
    # Constants
    "R_UNIVERSAL",
    "R_GAS_J_MOL_K",
    "STANDARD_GRAVITY",
    "ATOL_ZERO",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    # Conditionally available
    "ScrubberCalculator",
    "SyngasWaterCalculator",
    "WaterVaporPressureCalculator",
    "WGSReactorCalculator",
    "WGSReactorEngine",
    "Optimizer",
    "AdamOptimizer",
    "MultiParameterAnalysis",
    "SyngasCompressionCalculator",
    "PressureDropCalculator",
]
