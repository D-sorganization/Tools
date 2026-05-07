"""Derived field calculations and caching for visualization.

This module provides calculations for derived fields computed from primary
field data, including gradient magnitude, vector magnitude, and divergence.

Key features:
- Gradient magnitude computation |∇f| from scalar fields
- Vector magnitude computation |v| from vector fields
- Divergence computation ∇·v from vector fields
- Caching with parent field tracking
- Numerical stability (no NaN/Inf artifacts)
- Performance optimized for production use

Production quality: All operations are stable, efficient, and thoroughly tested.
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cache invalidation message (DRY)
_CACHE_INVALIDATED_MSG = "Cache invalidated due to field change"


@dataclass
class DerivedFieldMetadata:
    """Metadata about a derived field.

    Attributes:
        name: Display name for the derived field
        type: Type of derivation ('gradient_magnitude', 'magnitude', 'divergence')
        parent_field_hash: Hash of the parent field
        computation_time_ms: Time taken to compute in milliseconds
    """

    name: str
    type: str
    parent_field_hash: str
    computation_time_ms: float


class DerivedFieldCalculator:
    """Calculate derived fields from primary field data.

    This class provides methods to compute derived quantities like gradient
    magnitude, vector magnitude, and divergence from scalar and vector fields.
    Includes caching with parent field tracking for efficiency.

    Attributes:
        cache_size_limit: Maximum number of cached computations (default: 20)
    """

    def __init__(self, cache_size_limit: int = 20) -> None:
        """Initialize the derived field calculator.

        Args:
            cache_size_limit: Maximum entries in LRU cache

        Raises:
            ValueError: If cache_size_limit is not positive
        """
        if cache_size_limit <= 0:
            raise ValueError(
                f"cache_size_limit must be positive, got {cache_size_limit}"
            )

        self.cache_size_limit = cache_size_limit
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parent_field_hash: str | None = None

        logger.debug(
            "DerivedFieldCalculator initialized with cache_limit=%d",
            cache_size_limit,
        )

    def compute_gradient_magnitude(self, scalar_field: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude |∇f| from scalar field.

        Uses central difference approximation for numerical gradients.
        Output shape matches input shape.

        Args:
            scalar_field: 3D scalar field array (shape: (nx, ny, nz))

        Returns:
            3D gradient magnitude array (shape: (nx, ny, nz))

        Raises:
            TypeError: If scalar_field is not array-like
            ValueError: If scalar_field is not 3D
        """
        # Input validation
        if not isinstance(scalar_field, np.ndarray):
            try:
                scalar_field = np.asarray(scalar_field)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"scalar_field must be array-like, got {type(scalar_field)}"
                ) from e

        if not isinstance(scalar_field, np.ndarray):
            raise TypeError("scalar_field must be ndarray")

        if scalar_field.ndim != 3:
            raise ValueError(f"scalar_field must be 3D, got shape {scalar_field.shape}")

        # Check cache
        field_hash = self._compute_field_hash(scalar_field)
        cache_key = self._cache_key(field_hash, "gradient_magnitude")

        if self._parent_field_hash == field_hash and cache_key in self._cache:
            logger.debug("Cache hit for gradient_magnitude")
            return self._cache[cache_key]["result"].copy()

        # Invalidate cache if field changed
        if self._parent_field_hash != field_hash:
            self._cache.clear()
            self._parent_field_hash = field_hash
            logger.debug(_CACHE_INVALIDATED_MSG)

        # Compute gradient using numpy's gradient
        try:
            # gradient returns list of gradients along each axis
            grads = np.gradient(scalar_field, axis=(0, 1, 2))
            grad_x, grad_y, grad_z = grads[0], grads[1], grads[2]

            # Compute magnitude: sqrt(gx^2 + gy^2 + gz^2)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

            # Ensure no NaN or Inf
            grad_magnitude = np.nan_to_num(
                grad_magnitude, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Store in cache
            self._store_in_cache(
                cache_key,
                {"result": grad_magnitude.copy()},
                len(scalar_field.tobytes()),
            )

            return grad_magnitude

        except Exception as e:
            logger.error("Gradient magnitude computation failed: %s", str(e))
            raise ValueError(f"Gradient magnitude computation failed: {str(e)}") from e

    def compute_vector_magnitude(self, vector_field: np.ndarray) -> np.ndarray:
        """Compute magnitude |v| from vector field.

        Computes sqrt(vx^2 + vy^2 + vz^2) for each point in the field.

        Args:
            vector_field: 4D vector field array (shape: (3, nx, ny, nz))
                         where first dimension is [vx, vy, vz]

        Returns:
            3D magnitude array (shape: (nx, ny, nz))

        Raises:
            TypeError: If vector_field is not array-like
            ValueError: If vector_field is not shape (3, nx, ny, nz)
        """
        # Input validation
        if not isinstance(vector_field, np.ndarray):
            try:
                vector_field = np.asarray(vector_field)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"vector_field must be array-like, got {type(vector_field)}"
                ) from e

        if not isinstance(vector_field, np.ndarray):
            raise TypeError("vector_field must be ndarray")

        if vector_field.ndim != 4 or vector_field.shape[0] != 3:
            raise ValueError(
                f"vector_field must have shape (3, nx, ny, nz), "
                f"got shape {vector_field.shape}"
            )

        # Check cache
        field_hash = self._compute_field_hash(vector_field)
        cache_key = self._cache_key(field_hash, "magnitude")

        if self._parent_field_hash == field_hash and cache_key in self._cache:
            logger.debug("Cache hit for vector_magnitude")
            return self._cache[cache_key]["result"].copy()

        # Invalidate cache if field changed
        if self._parent_field_hash != field_hash:
            self._cache.clear()
            self._parent_field_hash = field_hash
            logger.debug(_CACHE_INVALIDATED_MSG)

        # Compute magnitude
        try:
            vx = vector_field[0, :, :, :]
            vy = vector_field[1, :, :, :]
            vz = vector_field[2, :, :, :]

            magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

            # Ensure no NaN or Inf
            magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

            # Store in cache
            self._store_in_cache(
                cache_key, {"result": magnitude.copy()}, len(vector_field.tobytes())
            )

            return magnitude

        except Exception as e:
            logger.error("Vector magnitude computation failed: %s", str(e))
            raise ValueError(f"Vector magnitude computation failed: {str(e)}") from e

    def compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """Compute divergence ∇·v from vector field.

        Computes ∂vx/∂x + ∂vy/∂y + ∂vz/∂z using central differences.

        Args:
            vector_field: 4D vector field array (shape: (3, nx, ny, nz))
                         where first dimension is [vx, vy, vz]

        Returns:
            3D divergence array (shape: (nx, ny, nz))

        Raises:
            TypeError: If vector_field is not array-like
            ValueError: If vector_field is not shape (3, nx, ny, nz)
        """
        # Input validation
        if not isinstance(vector_field, np.ndarray):
            try:
                vector_field = np.asarray(vector_field)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"vector_field must be array-like, got {type(vector_field)}"
                ) from e

        if not isinstance(vector_field, np.ndarray):
            raise TypeError("vector_field must be ndarray")

        if vector_field.ndim != 4 or vector_field.shape[0] != 3:
            raise ValueError(
                f"vector_field must have shape (3, nx, ny, nz), "
                f"got shape {vector_field.shape}"
            )

        # Check cache
        field_hash = self._compute_field_hash(vector_field)
        cache_key = self._cache_key(field_hash, "divergence")

        if self._parent_field_hash == field_hash and cache_key in self._cache:
            logger.debug("Cache hit for divergence")
            return self._cache[cache_key]["result"].copy()

        # Invalidate cache if field changed
        if self._parent_field_hash != field_hash:
            self._cache.clear()
            self._parent_field_hash = field_hash
            logger.debug(_CACHE_INVALIDATED_MSG)

        # Compute divergence
        try:
            vx = vector_field[0, :, :, :]
            vy = vector_field[1, :, :, :]
            vz = vector_field[2, :, :, :]

            # Compute partial derivatives
            dvx_dx = np.gradient(vx, axis=0)
            dvy_dy = np.gradient(vy, axis=1)
            dvz_dz = np.gradient(vz, axis=2)

            # Sum to get divergence
            divergence = dvx_dx + dvy_dy + dvz_dz

            # Ensure no NaN or Inf
            divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

            # Store in cache
            self._store_in_cache(
                cache_key, {"result": divergence.copy()}, len(vector_field.tobytes())
            )

            return divergence

        except Exception as e:
            logger.error("Divergence computation failed: %s", str(e))
            raise ValueError(f"Divergence computation failed: {str(e)}") from e

    @staticmethod
    def get_derived_field_name(parent_name: str, operation_type: str) -> str:
        """Generate display name for a derived field.

        Args:
            parent_name: Name of the parent field (e.g., "Temperature")
            operation_type: Type of operation ('gradient_magnitude', 'magnitude',
                          'divergence')

        Returns:
            Formatted field name suitable for UI display
        """
        if operation_type == "gradient_magnitude":
            return f"{parent_name} Gradient Magnitude"
        elif operation_type == "magnitude":
            return f"|{parent_name}|"
        elif operation_type == "divergence":
            return f"{parent_name} Divergence"
        else:
            return f"{parent_name} [{operation_type}]"

    def _compute_field_hash(self, field: np.ndarray) -> str:
        """Compute hash of field for cache keying.

        Args:
            field: Field array (any shape)

        Returns:
            SHA256 hash of field data
        """
        field_bytes = field.astype(np.float32).tobytes()
        return hashlib.sha256(field_bytes).hexdigest()

    def _cache_key(self, field_hash: str, operation_type: str) -> str:
        """Generate cache key.

        Args:
            field_hash: Hash of the field
            operation_type: Type of operation

        Returns:
            Cache key string
        """
        return f"{field_hash}:{operation_type}"

    def _store_in_cache(
        self, cache_key: str, result: dict[str, Any], field_size_bytes: int
    ) -> None:
        """Store computation result in LRU cache.

        Args:
            cache_key: Cache key
            result: Computation result to cache
            field_size_bytes: Size of field data in bytes (reserved for future use)
        """
        # Move to end (most recently used)
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)

        # Evict oldest if over limit
        while len(self._cache) > self.cache_size_limit:
            self._cache.popitem(last=False)  # Remove least recently used

        logger.debug(
            "Cache store: %d entries, limit=%d",
            len(self._cache),
            self.cache_size_limit,
        )

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with 'size' and 'limit' keys
        """
        return {"size": len(self._cache), "limit": self.cache_size_limit}

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()
        self._parent_field_hash = None
        logger.debug("Cache cleared")


__all__ = ["DerivedFieldCalculator", "DerivedFieldMetadata"]
