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

import logging

logger = logging.getLogger(__name__)

# Core calculators that work standalone
from .acid_gas_dewpoint_calculator import AcidGasDewpointCalculator
from .flare_calculator import FlareCalculator, FlareDesign
from .baghouse_calculator import BaghouseCalculator, BaghouseResult
from .electrode_advancement_calculator import ElectrodeAdvancementCalculator
from .financial_calculator import FinancialCalculator
from .ode_solver import ODESolver
from .thermal_profile_predictor import ThermalProfilePredictor

# Constants module
from .constants import (
    R_UNIVERSAL,
    R_GAS_J_MOL_K,
    STANDARD_GRAVITY,
    ATOL_ZERO,
    celsius_to_kelvin,
    kelvin_to_celsius,
)

# Calculators with numpy/scipy dependencies
try:
    from .scrubber_calculator import ScrubberCalculator
except ImportError as e:
    logger.debug(f"ScrubberCalculator not available: {e}")
    ScrubberCalculator = None  # type: ignore

try:
    from .syngas_water_calculator import SyngasWaterCalculator
    from .water_vapor_pressure_calculator import WaterVaporPressureCalculator
except ImportError as e:
    logger.debug(f"Water calculators not available: {e}")
    SyngasWaterCalculator = None  # type: ignore
    WaterVaporPressureCalculator = None  # type: ignore

try:
    from .wgs_reactor_calculator import WGSReactorEngine
    WGSReactorCalculator = WGSReactorEngine  # Alias
except ImportError as e:
    logger.debug(f"WGSReactorCalculator not available: {e}")
    WGSReactorCalculator = None  # type: ignore
    WGSReactorEngine = None  # type: ignore

try:
    from .optimization import Optimizer, AdamOptimizer
except ImportError as e:
    logger.debug(f"Optimization not available: {e}")
    Optimizer = None  # type: ignore
    AdamOptimizer = None  # type: ignore

try:
    from .multi_param_analysis import MultiParameterAnalysis
except ImportError as e:
    logger.debug(f"MultiParameterAnalysis not available: {e}")
    MultiParameterAnalysis = None  # type: ignore

# UI-dependent calculators (require PyQt6)
try:
    from .syngas_compression_calculator import SyngasCompressionCalculator
except ImportError as e:
    logger.debug(f"SyngasCompressionCalculator not available: {e}")
    SyngasCompressionCalculator = None  # type: ignore

# Modular packages
try:
    from .pressure_drop_calculator import PressureDropCalculator
except ImportError as e:
    logger.debug(f"PressureDropCalculator not available: {e}")
    PressureDropCalculator = None  # type: ignore

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
    "ThermalProfilePredictor",
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
