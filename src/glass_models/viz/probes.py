"""
Point/Line/Plane probe tool for FEA mesh field value queries.

Provides classes for extracting field values from FEA meshes:
- PointProbe: Single-point interpolation
- LineProbe: 1D profile extraction (100 samples)
- PlaneProbe: 2D cross-section extraction (50x50 grid)
- ProbeManager: Coordination and caching

Design by Contract:
- Validate interpolation stability, bounds checking
- All results must be finite (no NaN/Inf)
- Interpolation error <0.1% on linear fields

Performance targets:
- Point probe: <50ms
- Line probe: <200ms
- Plane probe: <500ms
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import griddata

# ============================================================================
# ProbeResult: Data Container
# ============================================================================


@dataclass
class ProbeResult:
    """Result container for probe evaluation.

    Attributes:
        location: Central location of probe (e.g., point or plane center)
        values: Extracted field values (1D array for any probe type)
        coordinates: Mesh coordinates where values were sampled
        probe_type: Type of probe ('point', 'line', or 'plane')
    """

    location: np.ndarray
    values: np.ndarray
    coordinates: np.ndarray
    probe_type: str

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if len(self.values) != len(self.coordinates):
            msg = (
                f"Values and coordinates length mismatch: "
                f"{len(self.values)} vs {len(self.coordinates)}"
            )
            raise ValueError(msg)

        if np.any(np.isnan(self.values)) or np.any(np.isinf(self.values)):
            msg = "Result contains NaN or Inf values"
            raise ValueError(msg)

        if self.probe_type not in ("point", "line", "plane"):
            msg = f"Unknown probe type: {self.probe_type}"
            raise ValueError(msg)


# ============================================================================
# Probe Base Class
# ============================================================================


class Probe(ABC):
    """Abstract base class for probe types.

    Subclasses must implement evaluate() to extract field values.
    """

    def __init__(self, mesh: Any) -> None:
        """Initialize probe with mesh.

        Args:
            mesh: Mesh object with attributes:
                - nodes: N x 3 array of node coordinates
                - field_values: N array of scalar field values
        """
        if mesh is None:
            msg = "Mesh cannot be None"
            raise TypeError(msg)
        self.mesh = mesh

    @abstractmethod
    def evaluate(self) -> ProbeResult:
        """Extract field values and return probe result.

        Returns:
            ProbeResult with location, values, coordinates, and probe_type
        """


# ============================================================================
# Point Probe: Single-Point Interpolation
# ============================================================================


class PointProbe(Probe):
    """Single-point field value interpolation.

    Evaluates field at a specified point using linear interpolation
    on the mesh. Returns exact values at nodes, interpolated values elsewhere.
    """

    def __init__(self, mesh: Any, location: np.ndarray) -> None:
        """Initialize point probe.

        Args:
            mesh: Mesh object
            location: 3D point coordinates [x, y, z]
        """
        super().__init__(mesh)

        location = np.asarray(location, dtype=np.float64)
        if location.shape != (3,):
            msg = f"Location must be shape (3,), got {location.shape}"
            raise ValueError(msg)

        self.location = location

    def evaluate(self) -> ProbeResult:
        """Interpolate field value at point.

        Returns:
            ProbeResult with single interpolated value
        """
        # Use scipy's griddata for robust interpolation
        # Method='linear' for fast, stable linear interpolation
        try:
            values = griddata(
                self.mesh.nodes,
                self.mesh.field_values,
                self.location.reshape(1, -1),
                method="linear",
                fill_value=np.nan,
            )
        except Exception as err:
            msg = f"Interpolation failed: {err}"
            raise RuntimeError(msg) from err

        # Check for NaN (out of bounds)
        if np.isnan(values[0]):
            # Try nearest neighbor for out-of-bounds
            values = griddata(
                self.mesh.nodes,
                self.mesh.field_values,
                self.location.reshape(1, -1),
                method="nearest",
            )

        return ProbeResult(
            location=self.location.copy(),
            values=values,
            coordinates=self.location.reshape(1, -1).copy(),
            probe_type="point",
        )


# ============================================================================
# Line Probe: 1D Profile Extraction
# ============================================================================


class LineProbe(Probe):
    """1D profile extraction along a line.

    Samples exactly 100 points along a path from start to end,
    interpolating field values at each sample point.
    """

    def __init__(self, mesh: Any, start: np.ndarray, end: np.ndarray) -> None:
        """Initialize line probe.

        Args:
            mesh: Mesh object
            start: Starting point [x, y, z]
            end: Ending point [x, y, z]
        """
        super().__init__(mesh)

        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)

        if start.shape != (3,):
            msg = f"Start must be shape (3,), got {start.shape}"
            raise ValueError(msg)
        if end.shape != (3,):
            msg = f"End must be shape (3,), got {end.shape}"
            raise ValueError(msg)

        self.start = start
        self.end = end

    def evaluate(self) -> ProbeResult:
        """Sample field along line at 100 points.

        Returns:
            ProbeResult with 100 interpolated values
        """
        # Create 100 sample points along line
        t = np.linspace(0, 1, 100)
        coordinates = self.start[np.newaxis, :] + t[:, np.newaxis] * (
            self.end - self.start
        )

        # Interpolate at all points
        values = griddata(
            self.mesh.nodes,
            self.mesh.field_values,
            coordinates,
            method="linear",
            fill_value=np.nan,
        )

        # Handle any NaN from out-of-bounds with nearest neighbor
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            nn_values = griddata(
                self.mesh.nodes,
                self.mesh.field_values,
                coordinates[nan_mask],
                method="nearest",
            )
            values[nan_mask] = nn_values

        # Center point
        center = self.start + 0.5 * (self.end - self.start)

        return ProbeResult(
            location=center.copy(),
            values=values,
            coordinates=coordinates.copy(),
            probe_type="line",
        )


# ============================================================================
# Plane Probe: 2D Cross-Section Extraction
# ============================================================================


class PlaneProbe(Probe):
    """2D cross-section extraction on a plane.

    Defines a plane by position and normal vector, then creates
    a 50x50 grid of sample points in that plane and interpolates
    field values at each point.
    """

    def __init__(self, mesh: Any, position: np.ndarray, normal: np.ndarray) -> None:
        """Initialize plane probe.

        Args:
            mesh: Mesh object
            position: Point on plane [x, y, z]
            normal: Normal vector to plane (will be normalized)
        """
        super().__init__(mesh)

        position = np.asarray(position, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)

        if position.shape != (3,):
            msg = f"Position must be shape (3,), got {position.shape}"
            raise ValueError(msg)
        if normal.shape != (3,):
            msg = f"Normal must be shape (3,), got {normal.shape}"
            raise ValueError(msg)

        # Normalize normal
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-10:
            msg = "Normal vector cannot be zero"
            raise ValueError(msg)
        normal = normal / normal_length

        self.position = position
        self.normal = normal

    def evaluate(self) -> ProbeResult:
        """Sample field on 50x50 grid in plane.

        Returns:
            ProbeResult with 2500 interpolated values
        """
        # Create orthonormal basis for plane
        # Find two vectors perpendicular to normal
        if abs(self.normal[0]) < 0.9:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.array([0.0, 1.0, 0.0])

        # First tangent vector
        tangent1 = np.cross(self.normal, u)
        tangent1 = tangent1 / np.linalg.norm(tangent1)

        # Second tangent vector
        tangent2 = np.cross(self.normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)

        # Create 50x50 grid in plane
        s = np.linspace(-1, 1, 50)
        t = np.linspace(-1, 1, 50)
        S, T = np.meshgrid(s, t)

        # Convert to 3D coordinates
        grid_points = (
            self.position[np.newaxis, np.newaxis, :]
            + S[:, :, np.newaxis] * tangent1[np.newaxis, np.newaxis, :]
            + T[:, :, np.newaxis] * tangent2[np.newaxis, np.newaxis, :]
        )

        # Flatten for interpolation
        coordinates = grid_points.reshape(-1, 3)

        # Interpolate
        values = griddata(
            self.mesh.nodes,
            self.mesh.field_values,
            coordinates,
            method="linear",
            fill_value=np.nan,
        )

        # Handle NaN with nearest neighbor
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            nn_values = griddata(
                self.mesh.nodes,
                self.mesh.field_values,
                coordinates[nan_mask],
                method="nearest",
            )
            values[nan_mask] = nn_values

        return ProbeResult(
            location=self.position.copy(),
            values=values,
            coordinates=coordinates.copy(),
            probe_type="plane",
        )


# ============================================================================
# Probe Manager: Coordination and Caching
# ============================================================================


class ProbeManager:
    """Manager for coordinating multiple probes with caching.

    Maintains a list of probes, provides methods to add/remove probes,
    and evaluates all probes returning results. Results are cached
    to avoid recomputation.
    """

    def __init__(self, mesh: Any) -> None:
        """Initialize manager.

        Args:
            mesh: Mesh object shared by all probes
        """
        if mesh is None:
            msg = "Mesh cannot be None"
            raise TypeError(msg)
        self.mesh = mesh
        self.probes: list[Probe] = []
        self._result_cache: dict[int, ProbeResult | None] = {}

    def add_point_probe(self, location: np.ndarray) -> PointProbe:
        """Add a point probe.

        Args:
            location: Point coordinates [x, y, z]

        Returns:
            The created PointProbe
        """
        probe = PointProbe(self.mesh, location)
        self.probes.append(probe)
        self._result_cache[id(probe)] = None
        return probe

    def add_line_probe(self, start: np.ndarray, end: np.ndarray) -> LineProbe:
        """Add a line probe.

        Args:
            start: Line start point [x, y, z]
            end: Line end point [x, y, z]

        Returns:
            The created LineProbe
        """
        probe = LineProbe(self.mesh, start, end)
        self.probes.append(probe)
        self._result_cache[id(probe)] = None
        return probe

    def add_plane_probe(self, position: np.ndarray, normal: np.ndarray) -> PlaneProbe:
        """Add a plane probe.

        Args:
            position: Point on plane [x, y, z]
            normal: Normal vector to plane

        Returns:
            The created PlaneProbe
        """
        probe = PlaneProbe(self.mesh, position, normal)
        self.probes.append(probe)
        self._result_cache[id(probe)] = None
        return probe

    def remove_probe(self, index: int) -> None:
        """Remove probe by index.

        Args:
            index: Index in probes list
        """
        if not 0 <= index < len(self.probes):
            msg = f"Index {index} out of range [0, {len(self.probes)})"
            raise IndexError(msg)

        probe = self.probes.pop(index)
        del self._result_cache[id(probe)]

    def evaluate_all(self) -> list[ProbeResult]:
        """Evaluate all probes and return results.

        Returns:
            List of ProbeResult objects, in same order as probes
        """
        results = []
        for probe in self.probes:
            probe_id = id(probe)
            cached = self._result_cache.get(probe_id)
            if cached is not None:
                results.append(cached)
            else:
                result = probe.evaluate()
                self._result_cache[probe_id] = result
                results.append(result)
        return results

    def get_probe_result(self, index: int) -> ProbeResult | None:
        """Get result for specific probe.

        Args:
            index: Index in probes list

        Returns:
            ProbeResult if evaluated, None otherwise
        """
        if not 0 <= index < len(self.probes):
            msg = f"Index {index} out of range [0, {len(self.probes)})"
            raise IndexError(msg)

        probe = self.probes[index]
        probe_id = id(probe)
        cached = self._result_cache.get(probe_id)

        if cached is None:
            cached = probe.evaluate()
            self._result_cache[probe_id] = cached

        return cached
