"""Validated correlation/covariance group schema for perturbation plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from shared.python.contracts import require

from .identity_contracts import stable_id, stable_id_array, strict_string

MATRIX_KINDS: tuple[str, ...] = ("correlation", "covariance")
_MATRIX_TOLERANCE = 1e-12


def _as_matrix(value: object) -> np.ndarray:
    """Convert matrix-like input or raise a domain-level contract error."""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        require(False, "matrix must be a numeric square matrix", value)
        raise ValueError("matrix must be a numeric square matrix") from exc
    return cast(np.ndarray, result)


def _validate_group_ids(group_id: str, spec_ids: tuple[str, ...]) -> None:
    """Validate stable group/member identifiers."""
    stable_id(group_id, "group_id")
    require(
        len(spec_ids) >= 2, "correlation group needs at least two spec_ids", spec_ids
    )
    for spec_id in spec_ids:
        stable_id(spec_id, "spec_ids")
    require(len(set(spec_ids)) == len(spec_ids), "spec_ids must be unique", spec_ids)


def _validate_matrix(matrix: np.ndarray, size: int, matrix_kind: str) -> None:
    """Validate shape, symmetry, positive semidefiniteness, and diagonal."""
    require(
        matrix.shape == (size, size), "matrix shape must match spec_ids", matrix.shape
    )
    require(np.all(np.isfinite(matrix)), "matrix entries must be finite", matrix)
    require(
        np.allclose(matrix, matrix.T, rtol=0.0, atol=_MATRIX_TOLERANCE),
        "matrix must be symmetric",
        matrix,
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    require(
        float(np.min(eigenvalues)) >= -_MATRIX_TOLERANCE,
        "matrix must be positive semidefinite",
        eigenvalues,
    )
    diagonal = np.diag(matrix)
    if matrix_kind == "correlation":
        require(
            np.allclose(diagonal, 1.0, rtol=0.0, atol=_MATRIX_TOLERANCE),
            "correlation matrix must have a unit diagonal",
            diagonal,
        )
    else:
        require(
            np.all(diagonal > 0.0), "covariance diagonal must be positive", diagonal
        )


@dataclass(frozen=True)
class PerturbationGroup:
    """One disjoint group of jointly normal perturbation specifications.

    A correlation matrix is dimensionless and combines with member
    :class:`NoiseSpec` scales. A covariance matrix carries squared mixed units;
    its diagonal must equal the corresponding member scale squared. Optional
    per-spec truncation is applied after joint sampling and can therefore
    alter the covariance realized by a finite exported dataset.
    """

    group_id: str
    spec_ids: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    matrix_kind: str = "correlation"

    def __post_init__(self) -> None:
        spec_ids = tuple(self.spec_ids)
        require(
            self.matrix_kind in MATRIX_KINDS,
            f"matrix_kind must be one of {MATRIX_KINDS}",
            self.matrix_kind,
        )
        _validate_group_ids(self.group_id, spec_ids)
        matrix_array = _as_matrix(self.matrix)
        _validate_matrix(matrix_array, len(spec_ids), self.matrix_kind)
        immutable_matrix = tuple(
            tuple(float(value) for value in row) for row in matrix_array
        )
        object.__setattr__(self, "spec_ids", spec_ids)
        object.__setattr__(self, "matrix", immutable_matrix)

    def covariance_matrix(self, scales: Sequence[float]) -> np.ndarray:
        """Return the dimensional covariance represented by this group."""
        scale_array = np.asarray(scales, dtype=float)
        require(
            scale_array.shape == (len(self.spec_ids),),
            "scales must match spec_ids",
            scale_array,
        )
        matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix_kind == "covariance":
            return cast(np.ndarray, matrix.copy())
        covariance: np.ndarray = (
            scale_array[:, np.newaxis] * matrix * scale_array[np.newaxis, :]
        )
        return covariance

    def to_json_dict(self) -> dict[str, Any]:
        """Return the version-independent JSON representation."""
        return {
            "group_id": self.group_id,
            "spec_ids": list(self.spec_ids),
            "matrix_kind": self.matrix_kind,
            "matrix": [list(row) for row in self.matrix],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> PerturbationGroup:
        """Build a validated group from JSON-compatible data."""
        return cls(
            group_id=stable_id(data["group_id"], "group_id"),
            spec_ids=stable_id_array(data["spec_ids"], "spec_ids"),
            matrix=tuple(
                tuple(float(value) for value in row) for row in data["matrix"]
            ),
            matrix_kind=strict_string(
                data.get("matrix_kind", "correlation"), "matrix_kind"
            ),
        )


__all__ = ["MATRIX_KINDS", "PerturbationGroup"]
