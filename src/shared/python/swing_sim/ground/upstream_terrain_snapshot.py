"""Strict neutral wire contract for one Upstream terrain point snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .contract_types import Vector3
from .profile_validation import (
    exact_fields,
    object_mapping,
    parameter_value,
    sha256_digest,
    strict_text,
)
from .terrain_adapter_math import UNIT_TOLERANCE, dot, unit_vector, vector

UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION = "upstream-terrain-snapshot/v1"


@dataclass(frozen=True)
class UpstreamTerrainSnapshot:
    """One immutable point/material snapshot with no Upstream class import."""

    terrain_id: str
    terrain_revision: str
    source_frame_id: str
    point_m: Vector3
    normal_unit: Vector3
    surface_velocity_m_s: Vector3
    material_id: str
    material_revision: str
    material_name: str
    friction_coefficient: float
    rolling_resistance: float
    restitution: float
    hardness_fraction: float
    grass_height_m: float
    compressibility_fraction: float
    compression_damping_fraction: float
    turf_density_kg_m3: float
    moisture_fraction: float
    source_sha256: str
    schema_version: str = UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "terrain_id",
            "terrain_revision",
            "source_frame_id",
            "material_id",
            "material_revision",
            "material_name",
        ):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        point = vector(self.point_m, "point_m")
        normal = unit_vector(self.normal_unit, "normal_unit")
        velocity = vector(self.surface_velocity_m_s, "surface_velocity_m_s")
        if abs(dot(normal, velocity)) > UNIT_TOLERANCE:
            raise ValueError("surface_velocity_m_s must be tangent to normal_unit")
        object.__setattr__(self, "point_m", point)
        object.__setattr__(self, "normal_unit", normal)
        object.__setattr__(self, "surface_velocity_m_s", velocity)
        self._normalize_material()
        object.__setattr__(
            self, "source_sha256", sha256_digest(self.source_sha256, "source_sha256")
        )

    def _normalize_material(self) -> None:
        mapping = {
            "friction_coefficient": "static_friction",
            "rolling_resistance": "rolling_resistance",
            "restitution": "normal_restitution",
            "hardness_fraction": "hardness_fraction",
            "grass_height_m": "grass_height_m",
            "compressibility_fraction": "compressibility_fraction",
            "compression_damping_fraction": "compression_damping_fraction",
            "turf_density_kg_m3": "turf_density_kg_m3",
            "moisture_fraction": "moisture_fraction",
        }
        for field_name, parameter_id in mapping.items():
            object.__setattr__(
                self,
                field_name,
                parameter_value(parameter_id, getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible v1 mapping."""
        if type(self) is not UpstreamTerrainSnapshot:
            raise TypeError("snapshot must use the exact v1 document type")
        return asdict(self)

    def to_json(self) -> str:
        """Return deterministic compact canonical JSON."""
        return str(canonical_numeric_json(self.to_dict()))

    def canonical_sha256(self) -> str:
        """Return the canonical snapshot document identity."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UpstreamTerrainSnapshot:
        """Parse an exact-field v1 snapshot mapping."""
        if cls is not UpstreamTerrainSnapshot:
            raise TypeError("snapshot must use the exact v1 document type")
        data = object_mapping(payload, "upstream terrain snapshot")
        exact_fields(data, _SNAPSHOT_FIELDS, "upstream terrain snapshot")
        return cls(**data)


_SNAPSHOT_FIELDS = {
    "compressibility_fraction",
    "compression_damping_fraction",
    "friction_coefficient",
    "grass_height_m",
    "hardness_fraction",
    "material_id",
    "material_name",
    "material_revision",
    "moisture_fraction",
    "normal_unit",
    "point_m",
    "restitution",
    "rolling_resistance",
    "schema_version",
    "source_frame_id",
    "source_sha256",
    "surface_velocity_m_s",
    "terrain_id",
    "terrain_revision",
    "turf_density_kg_m3",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def upstream_snapshot_from_json(text: str) -> UpstreamTerrainSnapshot:
    """Parse only exact canonical v1 neutral snapshot JSON."""
    if not isinstance(text, str):
        raise TypeError("upstream terrain snapshot JSON must be text")
    try:
        payload = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError("upstream terrain snapshot JSON is invalid") from exc
    snapshot = UpstreamTerrainSnapshot.from_dict(payload)
    if snapshot.to_json() != text:
        raise ValueError("upstream terrain snapshot JSON must be canonical")
    return snapshot


__all__ = [
    "UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION",
    "UpstreamTerrainSnapshot",
    "upstream_snapshot_from_json",
]
