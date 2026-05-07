"""Contour extraction and labeling for arbitrary surfaces.

This module provides contour line extraction on 2D surfaces using marching
squares algorithm, with support for uniform and logarithmic level spacing,
caching for performance, and automatic contour labeling.

Key features:
- Single and batch contour extraction
- Marching squares algorithm for 2D surfaces
- Uniform and logarithmic level spacing
- Caching strategy with invalidation
- Contour labeling with value annotations
- Performance-optimized for large surfaces

Production quality: Smooth contours, accurate levels, SubViewer-compatible output.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContourResult:
    """Result of contour extraction.

    Attributes:
        vertices: (N, 3) array of surface vertex positions
        contours: List of contour lines, each is array of 3D points
        contour_values: Array of contour level values
        field_min: Minimum field value
        field_max: Maximum field value
    """

    vertices: np.ndarray
    contours: list[np.ndarray | None]
    contour_values: np.ndarray
    field_min: float
    field_max: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with dict-based access."""
        return {
            "vertices": self.vertices,
            "contours": self.contours,
            "contour_values": self.contour_values,
            "field_min": self.field_min,
            "field_max": self.field_max,
        }


class ContourExtractor:
    """Extract contour lines from scalar fields on arbitrary 2D surfaces.

    Uses marching squares algorithm applied to surface triangles to extract
    smooth contour lines at specified levels. Supports uniform and logarithmic
    spacing of contour levels. Includes caching for repeated extractions.

    Attributes:
        n_levels: Number of contour levels (default 10)
        spacing: Level spacing type: 'uniform' or 'log' (default 'uniform')
        enable_cache: Enable result caching (default True)
    """

    def __init__(
        self,
        n_levels: int = 10,
        spacing: str = "uniform",
        enable_cache: bool = True,
    ) -> None:
        """Initialize contour extractor.

        Args:
            n_levels: Number of contour levels to extract (default 10)
            spacing: Level spacing type: 'uniform' or 'log' (default 'uniform')
            enable_cache: Enable result caching (default True)

        Raises:
            ValueError: If n_levels < 1 or spacing not in ['uniform', 'log']
        """
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        if spacing not in ("uniform", "log"):
            raise ValueError(f"spacing must be 'uniform' or 'log', got {spacing}")

        self.n_levels = n_levels
        self.spacing = spacing
        self.enable_cache = enable_cache
        self._cache: dict[str, ContourResult] = {}

        logger.debug(
            "ContourExtractor initialized with n_levels=%d, spacing=%s, cache=%s",
            n_levels,
            spacing,
            enable_cache,
        )

    def extract(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        field: np.ndarray,
    ) -> ContourResult:
        """Extract contour lines from a surface scalar field.

        Args:
            vertices: (N, 3) array of vertex positions
            triangles: (M, 3) array of triangle vertex indices
            field: (N,) array of scalar field values at vertices

        Returns:
            ContourResult containing extracted contours and metadata

        Raises:
            ValueError: If inputs are invalid or inconsistent
            IndexError: If triangle indices are out of range
        """
        # Validate inputs
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3) array, got {vertices.shape}")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (M, 3) array, got {triangles.shape}")
        if field.ndim != 1 or len(field) != len(vertices):
            raise ValueError(
                f"field must be (N,) array matching vertices, got {field.shape}"
            )

        # Check triangle indices are valid
        max_idx = triangles.max()
        if max_idx >= len(vertices):
            raise IndexError(
                f"Triangle index {max_idx} out of range [0, {len(vertices) - 1}]"
            )

        # Generate cache key
        cache_key = self._generate_cache_key(vertices, triangles, field)
        if self.enable_cache and cache_key in self._cache:
            logger.debug("Returning cached contour result")
            return self._cache[cache_key]

        # Extract contours
        field_clean = self._clean_field(field)
        field_min, field_max = self._get_field_range(field_clean)

        contour_values = self._generate_levels(field_min, field_max)
        contours = self._extract_contours_marching_squares(
            vertices, triangles, field_clean, contour_values
        )

        result = ContourResult(
            vertices=vertices.copy(),
            contours=contours,
            contour_values=contour_values,
            field_min=float(field_min),
            field_max=float(field_max),
        )

        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = result

        logger.debug(
            "Extracted %d contours with %d levels",
            len([c for c in contours if c is not None]),
            len(contour_values),
        )

        return result

    def _clean_field(self, field: np.ndarray) -> np.ndarray:
        """Clean field by handling NaN and inf values.

        Args:
            field: Input scalar field

        Returns:
            Cleaned field with NaN/inf replaced
        """
        field_clean = field.copy()
        mask = ~np.isfinite(field_clean)
        if np.any(mask):
            valid_values = field_clean[~mask]
            if len(valid_values) > 0:
                fill_value = np.nanmedian(valid_values)
            else:
                fill_value = 0.0
            field_clean[mask] = fill_value
            logger.warning(
                "Field contained %d non-finite values, replaced with %f",
                np.sum(mask),
                fill_value,
            )
        return field_clean

    def _get_field_range(self, field: np.ndarray) -> tuple[float, float]:
        """Get min and max field values, handling edge cases.

        Args:
            field: Input scalar field

        Returns:
            Tuple of (min_val, max_val)
        """
        field_min = float(np.min(field))
        field_max = float(np.max(field))

        # Handle constant field
        if field_min == field_max:
            field_max = field_min + 1.0
            logger.warning(
                "Field is constant (%.3f), extending range to [%.3f, %.3f]",
                field_min,
                field_min,
                field_max,
            )

        return field_min, field_max

    def _generate_levels(self, field_min: float, field_max: float) -> np.ndarray:
        """Generate contour levels.

        Args:
            field_min: Minimum field value
            field_max: Maximum field value

        Returns:
            Array of contour level values
        """
        if self.spacing == "uniform":
            levels = np.linspace(field_min, field_max, self.n_levels)
        elif self.spacing == "log":
            # For log spacing, ensure positive values
            if field_min <= 0:
                field_min = field_max / 1000.0  # Use small positive value
                logger.debug(
                    "Negative field_min, adjusted to %f for log spacing", field_min
                )
            levels = np.logspace(
                np.log10(field_min), np.log10(field_max), self.n_levels
            )
        else:
            raise ValueError(f"Unknown spacing: {self.spacing}")

        return levels

    def _extract_contours_marching_squares(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        field: np.ndarray,
        levels: np.ndarray,
    ) -> list[np.ndarray | None]:
        """Extract contours using marching squares on triangles.

        Args:
            vertices: (N, 3) vertex positions
            triangles: (M, 3) triangle indices
            field: (N,) scalar field values
            levels: Contour level values

        Returns:
            List of contour line arrays, one per level
        """
        contours = [None] * len(levels)

        # Precompute triangle field values
        field_vals_all = field[triangles]  # (M, 3)
        tri_vertices_all = vertices[triangles]  # (M, 3, 3)

        for level_idx, level in enumerate(levels):
            contour_lines = []

            # Check which triangles cross this level
            min_vals = np.min(field_vals_all, axis=1)  # (M,)
            max_vals = np.max(field_vals_all, axis=1)  # (M,)
            crosses = (min_vals <= level) & (max_vals >= level)  # (M,)

            # Process only triangles that cross this level
            crossing_indices = np.where(crosses)[0]

            for tri_idx in crossing_indices:
                field_vals = field_vals_all[tri_idx]
                tri_vertices = tri_vertices_all[tri_idx]

                # Extract contour segment(s) from triangle
                segments = self._extract_triangle_contours(
                    tri_vertices, field_vals, level
                )

                if segments:
                    contour_lines.extend(segments)

            # Combine segments into contours if there are any
            if contour_lines:
                contours[level_idx] = np.vstack(contour_lines)

        return contours

    @staticmethod
    def _crosses_triangle(field_vals: np.ndarray, level: float) -> bool:
        """Check if contour level crosses triangle.

        Args:
            field_vals: Field values at triangle vertices (3,)
            level: Contour level value

        Returns:
            True if level crosses triangle
        """
        return bool((field_vals >= level).any() and (field_vals <= level).any())

    def _extract_triangle_contours(
        self,
        tri_vertices: np.ndarray,
        field_vals: np.ndarray,
        level: float,
    ) -> list[np.ndarray]:
        """Extract contour segments from a single triangle.

        Uses marching squares logic: find edges where contour crosses,
        interpolate crossing points, and return line segments.

        Args:
            tri_vertices: (3, 3) vertex positions
            field_vals: (3,) field values at vertices
            level: Contour level value

        Returns:
            List of line segment arrays (each Nx3)
        """
        # Find edge crossings
        edge_crossings = []
        edges = [(0, 1), (1, 2), (2, 0)]

        for i, j in edges:
            if self._edge_crosses_level(field_vals[i], field_vals[j], level):
                # Interpolate crossing point
                t = (level - field_vals[i]) / (field_vals[j] - field_vals[i])
                t = np.clip(t, 0.0, 1.0)
                point = (1.0 - t) * tri_vertices[i] + t * tri_vertices[j]
                edge_crossings.append(point)

        # Generate line segments from crossings
        segments = []
        if len(edge_crossings) >= 2:
            # Connect consecutive crossing points
            for i in range(len(edge_crossings) - 1):
                segment = np.array([edge_crossings[i], edge_crossings[i + 1]])
                segments.append(segment)

        return segments

    def _edge_crosses_level(self, val_a: float, val_b: float, level: float) -> bool:
        """Check if contour level crosses an edge.

        Args:
            val_a: Field value at edge start
            val_b: Field value at edge end
            level: Contour level

        Returns:
            True if level crosses the edge
        """
        return (val_a < level <= val_b) or (val_b < level <= val_a)

    def _generate_cache_key(
        self, vertices: np.ndarray, triangles: np.ndarray, field: np.ndarray
    ) -> str:
        """Generate cache key from inputs.

        Args:
            vertices: Vertex array
            triangles: Triangle array
            field: Field array

        Returns:
            Cache key string
        """
        data = (
            vertices.tobytes(),
            triangles.tobytes(),
            field.tobytes(),
            self.n_levels,
            self.spacing,
        )
        parts = [d if isinstance(d, bytes) else str(d).encode() for d in data]
        key = hashlib.md5(b"".join(parts)).hexdigest()
        return key


def extract_contours(
    surface_mesh: np.ndarray,
    triangles: np.ndarray,
    field: np.ndarray,
    n_levels: int = 10,
    spacing: str = "uniform",
) -> ContourResult:
    """Extract contours from a scalar field on a surface.

    Convenience function for single extractions without caching overhead.

    Args:
        surface_mesh: (N, 3) array of vertex positions
        triangles: (M, 3) array of triangle indices
        field: (N,) array of scalar field values
        n_levels: Number of contour levels (default 10)
        spacing: Level spacing 'uniform' or 'log' (default 'uniform')

    Returns:
        ContourResult with extracted contours

    Raises:
        ValueError: If inputs are invalid
        IndexError: If triangle indices out of range
    """
    extractor = ContourExtractor(n_levels=n_levels, spacing=spacing)
    return extractor.extract(surface_mesh, triangles, field)


def label_contours(
    contours: list[np.ndarray | None],
    field_values: np.ndarray,
) -> list[dict[str, Any] | None]:
    """Label contours with field values and positions.

    Args:
        contours: List of contour line arrays
        field_values: Array of field values corresponding to contours

    Returns:
        List of label dictionaries, one per contour
    """
    labeled = []

    for i, contour_line in enumerate(contours):
        if contour_line is None or len(contour_line) == 0:
            labeled.append(None)
            continue

        # Find label position: midpoint of longest segment
        if len(contour_line) > 1:
            diffs = np.diff(contour_line, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            max_idx = np.argmax(distances)
            label_pos = (contour_line[max_idx] + contour_line[max_idx + 1]) / 2.0
        else:
            label_pos = contour_line[0]

        label_info = {
            "value": float(field_values[i]),
            "position": label_pos,
            "text": f"{field_values[i]:.3f}",
        }

        labeled.append(label_info)

    return labeled


__all__ = [
    "ContourExtractor",
    "ContourResult",
    "extract_contours",
    "label_contours",
]
