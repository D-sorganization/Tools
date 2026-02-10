"""Interfaces to external systems and adapters."""

from .electrode_adapter import ElectrodeAdapter
from .geometry_sync import GeometrySynchronizer, GeometryValidationResult

__all__ = ["ElectrodeAdapter", "GeometrySynchronizer", "GeometryValidationResult"]
