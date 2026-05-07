"""Visualization module for glass models."""

from .colormaps import ColormapManager
from .contours import ContourExtractor, ContourResult, extract_contours, label_contours
from .derived_fields import DerivedFieldCalculator, DerivedFieldMetadata
from .glyphs import GlyphDensityController, GlyphStyle
from .isosurface import IsoSurfaceExtractor, IsoSurfaceResult
from .lighting import Light, LightingManager, MaterialProperties
from .transparency import (
    TransparencyRenderer,
    disable_transparent_background,
    enable_transparent_background,
    export_with_transparency,
)
from .viewpoints import Viewpoint, ViewpointManager

__all__ = [
    "ColormapManager",
    "ContourExtractor",
    "ContourResult",
    "extract_contours",
    "label_contours",
    "GlyphDensityController",
    "GlyphStyle",
    "IsoSurfaceExtractor",
    "IsoSurfaceResult",
    "Light",
    "LightingManager",
    "MaterialProperties",
    "DerivedFieldCalculator",
    "DerivedFieldMetadata",
    "TransparencyRenderer",
    "enable_transparent_background",
    "disable_transparent_background",
    "export_with_transparency",
    "Viewpoint",
    "ViewpointManager",
]
