"""Canonical geometry references and exact-CAD artifact validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

Vector3: TypeAlias = tuple[float, float, float]

MM_PER_M = 1_000.0
M3_PER_MM3 = 1.0e-9
_MAX_EXACT_VOLUME_RELATIVE_ERROR = 1.0e-8
_MAX_EXACT_BOUNDS_ERROR_M = 1.0e-9


@dataclass(frozen=True)
class CadGeometryReference:
    """Canonical B-Rep volume and axis-aligned bounds in SI units."""

    volume_m3: float
    bounds_min_m: Vector3
    bounds_max_m: Vector3

    def __post_init__(self) -> None:
        volume = finite_float(self.volume_m3, "volume_m3", positive=True)
        minimum = vector3(self.bounds_min_m, "bounds_min_m")
        maximum = vector3(self.bounds_max_m, "bounds_max_m")
        if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("bounds_max_m must exceed bounds_min_m on every axis")
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "bounds_min_m", minimum)
        object.__setattr__(self, "bounds_max_m", maximum)

    @property
    def minimum_span_m(self) -> float:
        """Return the shortest positive reference bounding-box span."""
        return min(
            high - low
            for low, high in zip(
                self.bounds_min_m,
                self.bounds_max_m,
                strict=True,
            )
        )


@dataclass(frozen=True)
class ExactCadValidation:
    """Post-export exact-shape validation from an independent read operation."""

    passed: bool
    reader: str
    is_valid: bool
    solid_count: int
    volume_m3: float
    volume_relative_error: float
    max_bounds_error_m: float


def validate_exact_cad(
    path: Path | str,
    reference: CadGeometryReference,
    *,
    format_name: str,
) -> ExactCadValidation:
    """Reopen STEP or BREP and fail unless exact geometry matches its source."""
    artifact_path = artifact_path_from(path)
    if not isinstance(reference, CadGeometryReference):
        raise TypeError("reference must be CadGeometryReference")
    if format_name not in {"step", "brep"}:
        raise ValueError("format_name must be 'step' or 'brep'")
    restored, reader = _read_exact_shape(artifact_path, format_name)
    is_valid = bool(restored.is_valid)
    solid_count = len(restored.solids())
    measured = reference_from_build123d_shape(restored)
    volume_error = relative_error(measured.volume_m3, reference.volume_m3)
    bounds_error = max_bounds_error(measured, reference)
    _require_exact_match(is_valid, solid_count, volume_error, bounds_error)
    return ExactCadValidation(
        passed=True,
        reader=reader,
        is_valid=is_valid,
        solid_count=solid_count,
        volume_m3=measured.volume_m3,
        volume_relative_error=volume_error,
        max_bounds_error_m=bounds_error,
    )


def reference_from_build123d_shape(shape: object) -> CadGeometryReference:
    """Convert one build123d shape into the generic SI validation reference."""
    if shape is None:
        raise TypeError("shape must be a build123d shape")
    typed_shape: Any = shape
    try:
        volume_mm3 = float(typed_shape.volume)
        bounds = typed_shape.bounding_box()
        minimum = tuple(float(value) / MM_PER_M for value in bounds.min)
        maximum = tuple(float(value) / MM_PER_M for value in bounds.max)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("shape must expose finite volume and bounding_box") from exc
    return CadGeometryReference(
        volume_m3=volume_mm3 * M3_PER_MM3,
        bounds_min_m=vector3(minimum, "bounds_min_m"),
        bounds_max_m=vector3(maximum, "bounds_max_m"),
    )


def max_bounds_error(
    measured: CadGeometryReference,
    reference: CadGeometryReference,
) -> float:
    """Return the maximum absolute error across both bounding-box corners."""
    differences = (
        abs(measured_value - reference_value)
        for measured_values, reference_values in (
            (measured.bounds_min_m, reference.bounds_min_m),
            (measured.bounds_max_m, reference.bounds_max_m),
        )
        for measured_value, reference_value in zip(
            measured_values,
            reference_values,
            strict=True,
        )
    )
    return max(differences)


def relative_error(measured: float, reference: float) -> float:
    """Return an absolute relative error against a positive reference."""
    return abs(measured - reference) / reference


def artifact_path_from(
    path: Path | str,
    *,
    maximum_bytes: int | None = None,
) -> Path:
    """Validate a readable artifact path and optional resource bound."""
    if not isinstance(path, (Path, str)):
        raise TypeError("path must be a filesystem path")
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ValueError("path must identify an existing artifact file")
    if maximum_bytes is not None and artifact_path.stat().st_size > maximum_bytes:
        raise ValueError(f"artifact exceeds the {maximum_bytes}-byte validation limit")
    return artifact_path


def vector3(value: object, name: str) -> Vector3:
    """Validate one finite three-value tuple."""
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-value tuple")
    return tuple(finite_float(item, name) for item in value)  # type: ignore[return-value]


def finite_float(value: object, name: str, *, positive: bool = False) -> float:
    """Validate a finite scalar, optionally requiring strict positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _read_exact_shape(path: Path, format_name: str) -> tuple[Any, str]:
    from build123d import import_brep, import_step

    if format_name == "step":
        return import_step(path), "build123d.import_step"
    return import_brep(path), "build123d.import_brep"


def _require_exact_match(
    is_valid: bool,
    solid_count: int,
    volume_error: float,
    bounds_error: float,
) -> None:
    if not is_valid or solid_count != 1:
        raise RuntimeError("exact CAD artifact must reopen as one valid solid")
    if volume_error > _MAX_EXACT_VOLUME_RELATIVE_ERROR:
        raise RuntimeError("exact CAD artifact volume exceeds validation tolerance")
    if bounds_error > _MAX_EXACT_BOUNDS_ERROR_M:
        raise RuntimeError("exact CAD artifact bounds exceed validation tolerance")


__all__ = [
    "CadGeometryReference",
    "ExactCadValidation",
    "reference_from_build123d_shape",
    "validate_exact_cad",
]
