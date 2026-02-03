"""Syngas Compression Calculator Python Package.

Provides PyQt6 GUI for syngas compression analysis.
"""

from shared.python.upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
    create_syngas_compression_calculator,
)

__all__ = [
    "CompressionStage",
    "SyngasCompressionEngine",
    "create_syngas_compression_calculator",
]
