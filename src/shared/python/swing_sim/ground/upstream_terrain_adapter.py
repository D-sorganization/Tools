"""Neutral, explicit-frame adapter for Upstream terrain point snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .contract_types import GroundSurfaceProfile
from .profile_validation import (
    exact_record,
    exact_records,
    sha256_digest,
    strict_text,
)
from .terrain_adapter_contracts import (
    FrameTransform,
    TerrainAdapterInterpretation,
    canonical_sha256,
)
from .terrain_adapter_math import UNIT_TOLERANCE, dot
from .upstream_terrain_snapshot import (
    UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION,
    UpstreamTerrainSnapshot,
    upstream_snapshot_from_json,
)

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

UPSTREAM_TERRAIN_ADAPTER_VERSION = "upstream-terrain-adapter/v1"


class TerrainDispositionKind(StrEnum):
    """Stable field-loss dispositions."""

    EXACT = "exact"
    EXPLICIT_INTERPRETATION = "explicit_interpretation"
    LOCAL_LINEARIZATION = "local_linearization"
    NOT_REPRESENTED = "not_represented"


@dataclass(frozen=True)
class TerrainFieldDisposition:
    """How one source concept is retained or lost at the target boundary."""

    source_field: str
    kind: TerrainDispositionKind
    target_fields: tuple[str, ...]
    note: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_field", strict_text(self.source_field, "source_field")
        )
        object.__setattr__(self, "kind", TerrainDispositionKind(self.kind))
        if type(self.target_fields) is not tuple:
            raise TypeError("target_fields must be an exact tuple")
        object.__setattr__(
            self,
            "target_fields",
            tuple(strict_text(item, "target field") for item in self.target_fields),
        )
        object.__setattr__(self, "note", strict_text(self.note, "disposition note"))
        if self.kind is TerrainDispositionKind.NOT_REPRESENTED:
            if self.target_fields:
                raise ValueError("not-represented fields cannot declare targets")
        elif not self.target_fields:
            raise ValueError("represented fields must declare target_fields")


@dataclass(frozen=True)
class AdaptedTerrainSurface:
    """One target surface and its complete explicit loss report."""

    surface: GroundSurfaceProfile
    snapshot: UpstreamTerrainSnapshot
    transform: FrameTransform
    interpretation: TerrainAdapterInterpretation
    adapter_version: str
    source_sha256: str
    snapshot_sha256: str
    transform_sha256: str
    interpretation_sha256: str
    adapter_input_sha256: str
    dispositions: tuple[TerrainFieldDisposition, ...]

    def _validated_hashes(self) -> tuple[str, str, str, str]:
        for name in (
            "source_sha256",
            "snapshot_sha256",
            "transform_sha256",
            "interpretation_sha256",
            "adapter_input_sha256",
        ):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        if self.source_sha256 != self.snapshot.source_sha256:
            raise ValueError("source_sha256 does not match snapshot")
        hashes = _adapter_hashes(self.snapshot, self.transform, self.interpretation)
        actual_hashes = (
            self.snapshot_sha256,
            self.transform_sha256,
            self.interpretation_sha256,
            self.adapter_input_sha256,
        )
        if actual_hashes != hashes:
            mismatched = next(
                name
                for name, actual, expected in zip(
                    (
                        "snapshot_sha256",
                        "transform_sha256",
                        "interpretation_sha256",
                        "adapter_input_sha256",
                    ),
                    actual_hashes,
                    hashes,
                    strict=True,
                )
                if actual != expected
            )
            raise ValueError(f"{mismatched} does not match adapter inputs")
        return hashes

    def __post_init__(self) -> None:
        if type(self.surface) is not GroundSurfaceProfile:
            raise TypeError("surface must use the exact solver contract type")
        exact_record(self.snapshot, UpstreamTerrainSnapshot, "snapshot")
        exact_record(self.transform, FrameTransform, "transform")
        exact_record(
            self.interpretation,
            TerrainAdapterInterpretation,
            "interpretation",
        )
        if self.adapter_version != UPSTREAM_TERRAIN_ADAPTER_VERSION:
            raise ValueError("adapter_version does not match adapter contract")
        self._validated_hashes()
        expected_surface = _surface_from_snapshot(
            self.snapshot, self.transform, self.interpretation
        )
        if self.surface != expected_surface:
            raise ValueError("surface does not match adapter inputs")
        dispositions = exact_records(
            self.dispositions, TerrainFieldDisposition, "terrain disposition"
        )
        if dispositions != _dispositions(self.interpretation):
            raise ValueError("dispositions do not match adapter contract")
        object.__setattr__(self, "dispositions", dispositions)

    @property
    def is_lossy(self) -> bool:
        """Return whether any concept required interpretation or was lost."""
        return any(
            item.kind != TerrainDispositionKind.EXACT for item in self.dispositions
        )


def _identity_dispositions() -> tuple[TerrainFieldDisposition, ...]:
    return (
        TerrainFieldDisposition(
            "terrain_identity",
            TerrainDispositionKind.EXACT,
            ("surface_id", "provider_version", "snapshot"),
            "Terrain and material identities and revisions are retained.",
        ),
        TerrainFieldDisposition(
            "snapshot_provenance",
            TerrainDispositionKind.EXACT,
            ("source_sha256", "snapshot_sha256"),
            "Source digest and strict snapshot document identity are retained.",
        ),
        TerrainFieldDisposition(
            "source_frame_id",
            TerrainDispositionKind.EXPLICIT_INTERPRETATION,
            ("frame", "transform_sha256"),
            "The complete caller-supplied proper transform is retained.",
        ),
        TerrainFieldDisposition(
            "geometry",
            TerrainDispositionKind.LOCAL_LINEARIZATION,
            ("height_m", "normal_unit"),
            "One tangent plane; changing normals are not preserved.",
        ),
        TerrainFieldDisposition(
            "surface_velocity_m_s",
            TerrainDispositionKind.EXACT,
            ("surface_velocity_m_s",),
            "Velocity is rotated by the same proper frame transform.",
        ),
        TerrainFieldDisposition(
            "frame_transform",
            TerrainDispositionKind.EXPLICIT_INTERPRETATION,
            ("transform", "transform_sha256"),
            "No axis convention is inferred; the exact transform is retained.",
        ),
    )


def _material_dispositions(
    interpretation: TerrainAdapterInterpretation,
) -> tuple[TerrainFieldDisposition, ...]:
    return (
        TerrainFieldDisposition(
            "friction_coefficient",
            TerrainDispositionKind.EXPLICIT_INTERPRETATION,
            ("static_friction", "kinetic_friction"),
            interpretation.friction_method,
        ),
        TerrainFieldDisposition(
            "firmness_pa",
            TerrainDispositionKind.EXPLICIT_INTERPRETATION,
            ("firmness_pa",),
            interpretation.firmness_method,
        ),
        TerrainFieldDisposition(
            "material_scalars",
            TerrainDispositionKind.EXACT,
            (
                "rolling_resistance",
                "normal_restitution",
                "hardness_fraction",
                "grass_height_m",
                "compressibility_fraction",
                "compression_damping_fraction",
                "turf_density_kg_m3",
                "moisture_fraction",
            ),
            "Values are retained in SI without inferred conversion.",
        ),
        TerrainFieldDisposition(
            "regional_topology",
            TerrainDispositionKind.NOT_REPRESENTED,
            (),
            "Region boundaries and changing terrain are outside this point snapshot.",
        ),
    )


def _dispositions(
    interpretation: TerrainAdapterInterpretation,
) -> tuple[TerrainFieldDisposition, ...]:
    return _identity_dispositions() + _material_dispositions(interpretation)


def _validate_adapter_inputs(
    snapshot: UpstreamTerrainSnapshot,
    transform: FrameTransform,
    interpretation: TerrainAdapterInterpretation,
) -> None:
    if type(snapshot) is not UpstreamTerrainSnapshot:
        raise TypeError("snapshot must use the exact v1 document type")
    if type(transform) is not FrameTransform:
        raise TypeError("transform must use the exact contract type")
    if type(interpretation) is not TerrainAdapterInterpretation:
        raise TypeError("interpretation must use the exact contract type")
    if snapshot.source_frame_id != transform.source_frame_id:
        raise ValueError("snapshot and transform source frames must match")
    if snapshot.friction_coefficient != interpretation.source_friction_coefficient:
        raise ValueError("source friction coefficient does not match interpretation")


def _surface_from_snapshot(
    snapshot: UpstreamTerrainSnapshot,
    transform: FrameTransform,
    interpretation: TerrainAdapterInterpretation,
) -> GroundSurfaceProfile:
    normal = transform.vector(snapshot.normal_unit)
    point = transform.point(snapshot.point_m)
    if normal[1] <= UNIT_TOLERANCE:
        raise ValueError("transformed normal must have a positive target up component")
    identity_sha = canonical_sha256(
        {
            "material_id": snapshot.material_id,
            "material_revision": snapshot.material_revision,
            "terrain_id": snapshot.terrain_id,
            "terrain_revision": snapshot.terrain_revision,
        }
    )
    return GroundSurfaceProfile(
        f"upstream-terrain:{identity_sha}",
        "upstream-terrain-snapshot",
        f"{UPSTREAM_TERRAIN_ADAPTER_VERSION}:{identity_sha}",
        transform.target_frame,
        dot(normal, point) / normal[1],
        normal,
        transform.vector(snapshot.surface_velocity_m_s),
        snapshot.restitution,
        interpretation.static_friction,
        interpretation.kinetic_friction,
        snapshot.rolling_resistance,
        interpretation.firmness_pa,
        snapshot.hardness_fraction,
        snapshot.grass_height_m,
        snapshot.compressibility_fraction,
        snapshot.compression_damping_fraction,
        snapshot.turf_density_kg_m3,
        snapshot.moisture_fraction,
    )


def _adapter_hashes(
    snapshot: UpstreamTerrainSnapshot,
    transform: FrameTransform,
    interpretation: TerrainAdapterInterpretation,
) -> tuple[str, str, str, str]:
    snapshot_sha = snapshot.canonical_sha256()
    transform_sha = transform.canonical_sha256()
    interpretation_sha = interpretation.canonical_sha256()
    adapter_input_sha = canonical_sha256(
        {
            "adapter_version": UPSTREAM_TERRAIN_ADAPTER_VERSION,
            "interpretation_sha256": interpretation_sha,
            "snapshot_sha256": snapshot_sha,
            "transform_sha256": transform_sha,
        }
    )
    return snapshot_sha, transform_sha, interpretation_sha, adapter_input_sha


def adapt_upstream_terrain_snapshot(
    snapshot: UpstreamTerrainSnapshot,
    transform: FrameTransform,
    interpretation: TerrainAdapterInterpretation,
) -> AdaptedTerrainSurface:
    """Create one local target-frame surface without importing Upstream types."""
    _validate_adapter_inputs(snapshot, transform, interpretation)
    surface = _surface_from_snapshot(snapshot, transform, interpretation)
    snapshot_sha, transform_sha, interpretation_sha, adapter_input_sha = (
        _adapter_hashes(snapshot, transform, interpretation)
    )
    return AdaptedTerrainSurface(
        surface,
        snapshot,
        transform,
        interpretation,
        UPSTREAM_TERRAIN_ADAPTER_VERSION,
        snapshot.source_sha256,
        snapshot_sha,
        transform_sha,
        interpretation_sha,
        adapter_input_sha,
        _dispositions(interpretation),
    )


__all__ = [
    "AdaptedTerrainSurface",
    "FrameTransform",
    "TerrainAdapterInterpretation",
    "TerrainDispositionKind",
    "TerrainFieldDisposition",
    "UPSTREAM_TERRAIN_ADAPTER_VERSION",
    "UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION",
    "UpstreamTerrainSnapshot",
    "adapt_upstream_terrain_snapshot",
    "upstream_snapshot_from_json",
]
