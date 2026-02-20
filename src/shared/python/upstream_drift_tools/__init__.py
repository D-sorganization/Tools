"""
UpstreamDrift Shared Tools Library
==================================

Shared components for Gasification Model and UpstreamDrift (Golf Suite).

Subpackages (import by domain):
    calculators           - Calculation engines (conversion, electrical, mechanical, thermo)
    data_processing       - DataProcessorEngine, readers/writers, typed exceptions
    lab                   - Laboratory tools (bio/C3D reader)
    process_calculators   - Standalone process engineering calculators
    ui                    - PyQt6 widgets, themes, managers, mixins
    utils                 - Logging, paths, state management, physical constants
"""

from .protocols import (
    DataTransformer,
    ProcessCalculator,
    StateSerializable,
    UnitConverter,
)

__version__ = "0.1.0"

__all__ = [
    # Protocols
    "DataTransformer",
    "ProcessCalculator",
    "StateSerializable",
    "UnitConverter",
    # Subpackages (explicit for discovery)
    "calculators",
    "data_processing",
    "lab",
    "process_calculators",
    "ui",
    "utils",
]
