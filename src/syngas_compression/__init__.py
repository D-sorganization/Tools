"""Syngas Compression Calculator Module.

Provides standalone GUI interfaces for syngas compression analysis.
Uses the shared engine from upstream_drift_tools.
"""

from shared.python.upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)

__all__ = [
    "CompressionStage",
    "SyngasCompressionEngine",
]
