"""Strict identity and provenance binding to a shared golf-club assembly."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

import numpy as np

from shared.python.golf_club import (
    ClubAssembly,
    ClubComponent,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
)

from .types import ClubSpec

CLUB_ASSEMBLY_BINDING_FORMAT = "rate_of_closure.club_assembly_binding/1"
CLUB_SPEC_IDENTITY_FORMAT = "rate_of_closure.club_spec_identity/1"
SELECTED_HEAD_FRAME_ID = "rate_of_closure.head"
MAX_BINDING_BYTES = 4 * 1024 * 1024

_HEAD_MASS_TOLERANCE_KG = 1e-12


class MassPropertyAuthorityKind(str, Enum):  # noqa: UP042 - Python 3.10
    """Qualified origin categories accepted by the binding boundary."""

    MEASURED = "measured"
    MANUFACTURER = "manufacturer"
    CAD_INTEGRATED = "cad_integrated"
    QUALIFIED_ANALYSIS = "qualified_analysis"


@dataclass(frozen=True)
class ClubAssemblySourceAuthority:
    """Explicit source authority for imported assembly mass properties."""

    kind: MassPropertyAuthorityKind
    authority_id: str
    document_id: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MassPropertyAuthorityKind):
            raise TypeError("authority kind must be a MassPropertyAuthorityKind")
        for field_name in ("authority_id", "document_id", "revision"):
            object.__setattr__(
                self,
                field_name,
                _require_identifier(getattr(self, field_name), field_name),
            )

    def to_json_dict(self) -> dict[str, str]:
        """Return the version-one JSON record."""
        return {
            "kind": self.kind.value,
            "authority_id": self.authority_id,
            "document_id": self.document_id,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class ClubAssemblyBinding:
    """Validated exact association between one ClubSpec and ClubAssembly."""

    selected_spec: ClubSpec
    assembly: ClubAssembly
    authority: ClubAssemblySourceAuthority
    head_component_id: str
    head_component_from_selected_head: RigidTransform

    def __post_init__(self) -> None:
        if not isinstance(self.selected_spec, ClubSpec):
            raise TypeError("selected_spec must be a ClubSpec")
        if not isinstance(self.assembly, ClubAssembly):
            raise TypeError("assembly must be a ClubAssembly")
        if not isinstance(self.authority, ClubAssemblySourceAuthority):
            raise TypeError("authority must be a ClubAssemblySourceAuthority")
        if not isinstance(self.head_component_from_selected_head, RigidTransform):
            raise TypeError(
                "head_component_from_selected_head must be a RigidTransform"
            )
        object.__setattr__(
            self,
            "head_component_id",
            _require_identifier(self.head_component_id, "head_component_id"),
        )
        _validate_binding_relationships(self)

    @property
    def selected_spec_sha256(self) -> str:
        """Return the content identity of the exact selected specification."""
        return club_spec_identity(self.selected_spec)

    @property
    def assembly_sha256(self) -> str:
        """Return the content identity of the exact embedded assembly."""
        return club_assembly_identity(self.assembly)

    @property
    def head_component(self) -> ClubComponent:
        """Return the explicitly selected head component."""
        for component in self.assembly.components:
            if component.component_id == self.head_component_id:
                return component
        raise AssertionError("validated head component must remain present")

    def assert_matches(self, spec: ClubSpec) -> None:
        """Fail closed unless ``spec`` is the exact bound selection."""
        if not isinstance(spec, ClubSpec):
            raise TypeError("spec must be a ClubSpec")
        if club_spec_identity(spec) != self.selected_spec_sha256:
            raise ValueError("binding selected ClubSpec identity does not match")

    def head_properties_in_selected_frame(self) -> ComponentMassProperties:
        """Transform authoritative head properties into the selected-head frame."""
        source = self.head_component.mass_properties
        transform = self.head_component_from_selected_head
        rotation = np.asarray(transform.rotation)
        translation = np.asarray(transform.translation_m)
        center = rotation.T @ (np.asarray(source.center_of_mass_m) - translation)
        inertia = rotation.T @ np.asarray(source.inertia_at_com_kg_m2) @ rotation
        return ComponentMassProperties(
            component_id=source.component_id,
            role=ComponentRole.HEAD,
            frame_id=SELECTED_HEAD_FRAME_ID,
            mass_kg=source.mass_kg,
            center_of_mass_m=tuple(float(value) for value in center),
            inertia_at_com_kg_m2=tuple(
                tuple(float(value) for value in row) for row in inertia
            ),
        )


def club_spec_identity(spec: ClubSpec) -> str:
    """Return a stable SHA-256 over every normalized selected-spec field."""
    return hashlib.sha256(club_spec_identity_payload(spec)).hexdigest()


def club_spec_identity_payload(spec: ClubSpec) -> bytes:
    """Return canonical UTF-8 bytes used for selected-spec identity hashing."""
    if not isinstance(spec, ClubSpec):
        raise TypeError("spec must be a ClubSpec")
    return _identity_bytes(_spec_snapshot(spec))


def club_assembly_identity(assembly: ClubAssembly) -> str:
    """Return a stable SHA-256 over the complete current assembly record."""
    return hashlib.sha256(club_assembly_identity_payload(assembly)).hexdigest()


def club_assembly_identity_payload(assembly: ClubAssembly) -> bytes:
    """Return canonical UTF-8 bytes used for assembly identity hashing."""
    if not isinstance(assembly, ClubAssembly):
        raise TypeError("assembly must be a ClubAssembly")
    return _identity_bytes(assembly.to_json_dict())


def build_club_assembly_binding(
    *,
    spec: ClubSpec,
    assembly: ClubAssembly,
    authority: ClubAssemblySourceAuthority,
    head_component_id: str,
    head_component_from_selected_head: RigidTransform,
) -> ClubAssemblyBinding:
    """Build a validated binding without constructing any component defaults."""
    return ClubAssemblyBinding(
        selected_spec=spec,
        assembly=assembly,
        authority=authority,
        head_component_id=head_component_id,
        head_component_from_selected_head=head_component_from_selected_head,
    )


def serialize_club_assembly_binding(binding: ClubAssemblyBinding) -> bytes:
    """Serialize deterministic, versioned binding JSON as UTF-8."""
    from ._assembly_binding_io import serialize_binding

    return serialize_binding(binding)


def parse_club_assembly_binding(
    spec: ClubSpec, payload: str | bytes
) -> ClubAssemblyBinding:
    """Parse and validate one imported binding against the current selection."""
    from ._assembly_binding_io import parse_binding

    return parse_binding(spec, payload)


def _spec_snapshot(spec: ClubSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "club_type": spec.club_type.value,
        "length_m": spec.length_m,
        "head_mass_kg": spec.head_mass_kg,
        "loft_deg": spec.loft_deg,
        "lie_deg": spec.lie_deg,
        "moi_about_shaft_kg_m2": spec.moi_about_shaft_kg_m2,
        "cg_depth_m": spec.cg_depth_m,
        "cg_height_m": spec.cg_height_m,
        "face_bulge_radius_m": spec.face_bulge_radius_m,
        "face_roll_radius_m": spec.face_roll_radius_m,
        "head_style": spec.head_style.value,
    }


def _identity_bytes(value: object) -> bytes:
    normalized = _normalize_identity_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalize_identity_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("identity input numbers must be finite")
        return {"$float64_be": struct.pack(">d", number).hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("identity input keys must be strings")
        return {key: _normalize_identity_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    raise TypeError(f"unsupported identity input type {type(value).__name__}")


def _validate_binding_relationships(binding: ClubAssemblyBinding) -> None:
    head_components = tuple(
        component
        for component in binding.assembly.components
        if component.mass_properties.role is ComponentRole.HEAD
    )
    if len(head_components) != 1:
        raise ValueError("assembly must contain exactly one head component")
    head = head_components[0]
    if head.component_id != binding.head_component_id:
        raise ValueError("head component identifier does not select the unique head")
    transform = binding.head_component_from_selected_head
    if transform.from_frame_id != SELECTED_HEAD_FRAME_ID:
        raise ValueError("head transform must start in the selected head frame")
    if transform.to_frame_id != head.mass_properties.frame_id:
        raise ValueError("head component frame does not match the binding transform")
    if not math.isclose(
        head.mass_properties.mass_kg,
        binding.selected_spec.head_mass_kg,
        abs_tol=_HEAD_MASS_TOLERANCE_KG,
        rel_tol=0.0,
    ):
        raise ValueError("bound head mass does not match selected ClubSpec head mass")
    _require_positive_definite(
        head.mass_properties.inertia_at_com_kg_m2, "head inertia tensor"
    )
    _require_positive_definite(
        binding.assembly.mass_properties.inertia_at_com_kg_m2,
        "assembly inertia tensor",
    )


def _require_positive_definite(tensor: object, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(np.asarray(tensor, dtype=float))
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"{name} must be positive definite")


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


__all__ = [
    "CLUB_ASSEMBLY_BINDING_FORMAT",
    "CLUB_SPEC_IDENTITY_FORMAT",
    "MAX_BINDING_BYTES",
    "SELECTED_HEAD_FRAME_ID",
    "ClubAssemblyBinding",
    "ClubAssemblySourceAuthority",
    "MassPropertyAuthorityKind",
    "build_club_assembly_binding",
    "club_assembly_identity",
    "club_assembly_identity_payload",
    "club_spec_identity",
    "club_spec_identity_payload",
    "parse_club_assembly_binding",
    "serialize_club_assembly_binding",
]
