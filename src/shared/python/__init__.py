"""
Python-specific shared utilities and libraries.
This package contains reusable Python logic for tools.

Available packages:
    - upstream_drift_tools: Process engineering calculators
    - signal_toolkit: Signal processing and analysis
    - humanoid_character_builder: URDF humanoid model generation
    - model_generation: URDF/MJCF model building and conversion

Usage:
    from humanoid_character_builder import CharacterBuilder, BodyParameters
    from model_generation import quick_urdf, ManualBuilder, FrankensteinEditor
    from signal_toolkit import Signal, SignalGenerator, FunctionFitter
    from upstream_drift_tools.process_calculators import FlareCalculator
"""

__all__ = [
    "humanoid_character_builder",
    "model_generation",
    "signal_toolkit",
    "upstream_drift_tools",
]
