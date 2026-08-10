"""Strict frame and interpretation contracts for the neutral terrain adapter."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .contract_types import GroundFrame, Vector3
from .profile_validation import parameter_value, strict_text
from .terrain_adapter_math import (
    UNIT_TOLERANCE,
    determinant,
    dot,
    matrix_vector,
    unit_vector,
    vector,
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 identity for one adapter record mapping."""
    text = str(canonical_numeric_json(payload))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrameTransform:
    """Explicit proper rigid transform into the canonical target frame."""

    source_frame_id: str
    rotation_rows: tuple[Vector3, Vector3, Vector3]
    translation_m: Vector3
    target_frame: GroundFrame = GroundFrame.TARGET

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_frame_id",
            strict_text(self.source_frame_id, "source_frame_id"),
        )
        if len(self.rotation_rows) != 3:
            raise ValueError("rotation_rows must contain three rows")
        rows = (
            unit_vector(self.rotation_rows[0], "rotation row"),
            unit_vector(self.rotation_rows[1], "rotation row"),
            unit_vector(self.rotation_rows[2], "rotation row"),
        )
        if any(
            abs(dot(rows[i], rows[j])) > UNIT_TOLERANCE
            for i in range(3)
            for j in range(i)
        ):
            raise ValueError("rotation_rows must be orthonormal")
        if abs(determinant(rows) - 1.0) > UNIT_TOLERANCE:
            raise ValueError("rotation_rows must be a proper rotation")
        object.__setattr__(self, "rotation_rows", rows)
        object.__setattr__(
            self, "translation_m", vector(self.translation_m, "translation_m")
        )
        object.__setattr__(self, "target_frame", GroundFrame(self.target_frame))

    def vector(self, value: Vector3) -> Vector3:
        """Rotate a source vector into the target frame."""
        return matrix_vector(self.rotation_rows, value)

    def point(self, value: Vector3) -> Vector3:
        """Rigidly transform a source point into the target frame."""
        rotated = self.vector(value)
        return (
            rotated[0] + self.translation_m[0],
            rotated[1] + self.translation_m[1],
            rotated[2] + self.translation_m[2],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact transform identity mapping."""
        if type(self) is not FrameTransform:
            raise TypeError("transform must use the exact contract type")
        return {
            "rotation_rows": self.rotation_rows,
            "source_frame_id": self.source_frame_id,
            "target_frame": str(self.target_frame),
            "translation_m": self.translation_m,
        }

    def canonical_sha256(self) -> str:
        """Return the exact canonical transform identity."""
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TerrainAdapterInterpretation:
    """Caller-owned values for source fields that cannot map uniquely."""

    source_friction_coefficient: float
    static_friction: float
    kinetic_friction: float
    firmness_pa: float
    friction_method: str
    firmness_method: str

    def __post_init__(self) -> None:
        source = parameter_value("static_friction", self.source_friction_coefficient)
        static = parameter_value("static_friction", self.static_friction)
        kinetic = parameter_value("kinetic_friction", self.kinetic_friction)
        firmness = parameter_value("firmness_pa", self.firmness_pa)
        if kinetic > static:
            raise ValueError("kinetic_friction must not exceed static_friction")
        object.__setattr__(self, "source_friction_coefficient", source)
        object.__setattr__(self, "static_friction", static)
        object.__setattr__(self, "kinetic_friction", kinetic)
        object.__setattr__(self, "firmness_pa", firmness)
        for name in ("friction_method", "firmness_method"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return the exact ambiguity-resolution identity mapping."""
        if type(self) is not TerrainAdapterInterpretation:
            raise TypeError("interpretation must use the exact contract type")
        return asdict(self)

    def canonical_sha256(self) -> str:
        """Return the exact canonical interpretation identity."""
        return canonical_sha256(self.to_dict())


__all__ = ["FrameTransform", "TerrainAdapterInterpretation", "canonical_sha256"]
