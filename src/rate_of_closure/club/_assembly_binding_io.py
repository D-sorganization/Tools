"""Version-one JSON I/O for selected-spec ClubAssembly bindings."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from shared.python.golf_club import (
    CURRENT_FORMAT as CLUB_ASSEMBLY_FORMAT,
)
from shared.python.golf_club import (
    ClubAssembly,
    RigidTransform,
)

from .assembly_binding import (
    CLUB_ASSEMBLY_BINDING_FORMAT,
    CLUB_SPEC_IDENTITY_FORMAT,
    MAX_BINDING_BYTES,
    ClubAssemblyBinding,
    ClubAssemblySourceAuthority,
    MassPropertyAuthorityKind,
    _spec_snapshot,
    build_club_assembly_binding,
    club_assembly_identity,
    club_spec_identity,
)
from .types import ClubSpec

_BINDING_FIELDS = frozenset(
    {
        "format",
        "selected_spec_identity",
        "assembly_identity",
        "source_authority",
        "units",
        "head_binding",
        "assembly",
    }
)
_SPEC_IDENTITY_FIELDS = frozenset({"format", "sha256", "snapshot"})
_ASSEMBLY_IDENTITY_FIELDS = frozenset({"format", "assembly_id", "sha256"})
_AUTHORITY_FIELDS = frozenset({"kind", "authority_id", "document_id", "revision"})
_HEAD_BINDING_FIELDS = frozenset(
    {"head_component_id", "head_component_from_selected_head"}
)
_TRANSFORM_FIELDS = frozenset(
    {"from_frame_id", "to_frame_id", "rotation", "translation_m"}
)
_SPEC_FIELDS = frozenset(
    {
        "name",
        "club_type",
        "length_m",
        "head_mass_kg",
        "loft_deg",
        "lie_deg",
        "moi_about_shaft_kg_m2",
        "cg_depth_m",
        "cg_height_m",
        "face_bulge_radius_m",
        "face_roll_radius_m",
        "head_style",
    }
)
_UNITS = {
    "angle": "degree",
    "inertia": "kg_m2",
    "length": "m",
    "mass": "kg",
}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def serialize_binding(binding: ClubAssemblyBinding) -> bytes:
    """Serialize deterministic, versioned binding JSON as UTF-8."""
    if not isinstance(binding, ClubAssemblyBinding):
        raise TypeError("binding must be a ClubAssemblyBinding")
    payload = (
        json.dumps(
            _binding_to_json_dict(binding),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_BINDING_BYTES:
        raise ValueError("club assembly binding exceeds the 4 MiB limit")
    return payload


def parse_binding(spec: ClubSpec, payload: str | bytes) -> ClubAssemblyBinding:
    """Parse and validate one imported binding against the current selection."""
    if not isinstance(spec, ClubSpec):
        raise TypeError("spec must be a ClubSpec")
    data = _parse_root(payload)
    _validate_units(data["units"])
    _validate_selected_identity(spec, data["selected_spec_identity"])
    assembly = _parse_assembly(data["assembly"])
    _validate_assembly_identity(assembly, data["assembly_identity"])
    authority = _parse_authority(data["source_authority"])
    head_component_id, transform = _parse_head_binding(data["head_binding"])
    return build_club_assembly_binding(
        spec=spec,
        assembly=assembly,
        authority=authority,
        head_component_id=head_component_id,
        head_component_from_selected_head=transform,
    )


def _binding_to_json_dict(binding: ClubAssemblyBinding) -> dict[str, Any]:
    transform = binding.head_component_from_selected_head
    return {
        "format": CLUB_ASSEMBLY_BINDING_FORMAT,
        "selected_spec_identity": {
            "format": CLUB_SPEC_IDENTITY_FORMAT,
            "sha256": binding.selected_spec_sha256,
            "snapshot": _spec_snapshot(binding.selected_spec),
        },
        "assembly_identity": {
            "format": CLUB_ASSEMBLY_FORMAT,
            "assembly_id": binding.assembly.assembly_id,
            "sha256": binding.assembly_sha256,
        },
        "source_authority": binding.authority.to_json_dict(),
        "units": dict(_UNITS),
        "head_binding": {
            "head_component_id": binding.head_component_id,
            "head_component_from_selected_head": {
                "from_frame_id": transform.from_frame_id,
                "to_frame_id": transform.to_frame_id,
                "rotation": [list(row) for row in transform.rotation],
                "translation_m": list(transform.translation_m),
            },
        },
        "assembly": binding.assembly.to_json_dict(),
    }


def _parse_root(payload: str | bytes) -> Mapping[str, Any]:
    text = _decode_payload(payload)
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("binding must contain valid JSON") from error
    data = _require_mapping(raw, "binding")
    _require_exact_fields(data, _BINDING_FIELDS, "binding")
    if data["format"] != CLUB_ASSEMBLY_BINDING_FORMAT:
        raise ValueError("unsupported club assembly binding format")
    return data


def _validate_selected_identity(spec: ClubSpec, value: object) -> None:
    data = _require_mapping(value, "selected_spec_identity")
    _require_exact_fields(data, _SPEC_IDENTITY_FIELDS, "selected_spec_identity")
    snapshot = _require_mapping(data["snapshot"], "selected spec snapshot")
    _require_exact_fields(snapshot, _SPEC_FIELDS, "selected spec snapshot")
    if data["format"] != CLUB_SPEC_IDENTITY_FORMAT:
        raise ValueError("unsupported selected ClubSpec identity format")
    digest = _require_sha256(data["sha256"], "selected ClubSpec identity")
    if snapshot != _spec_snapshot(spec) or digest != club_spec_identity(spec):
        raise ValueError("binding selected ClubSpec identity does not match")


def _parse_assembly(value: object) -> ClubAssembly:
    data = _require_mapping(value, "assembly")
    if data.get("format") != CLUB_ASSEMBLY_FORMAT:
        raise ValueError("binding requires the current golf-club assembly format")
    return ClubAssembly.from_json_dict(data)


def _validate_assembly_identity(assembly: ClubAssembly, value: object) -> None:
    data = _require_mapping(value, "assembly_identity")
    _require_exact_fields(data, _ASSEMBLY_IDENTITY_FIELDS, "assembly_identity")
    if data["format"] != CLUB_ASSEMBLY_FORMAT:
        raise ValueError("assembly identity format does not match")
    if data["assembly_id"] != assembly.assembly_id:
        raise ValueError("assembly identity does not match embedded assembly")
    digest = _require_sha256(data["sha256"], "assembly identity")
    if digest != club_assembly_identity(assembly):
        raise ValueError("assembly identity does not match embedded assembly")


def _parse_authority(value: object) -> ClubAssemblySourceAuthority:
    data = _require_mapping(value, "source_authority")
    _require_exact_fields(data, _AUTHORITY_FIELDS, "source_authority")
    kind_value = data["kind"]
    if not isinstance(kind_value, str):
        raise TypeError("authority kind must be a string")
    try:
        kind = MassPropertyAuthorityKind(kind_value)
    except ValueError as error:
        raise ValueError(f"unsupported authority kind {kind_value!r}") from error
    return ClubAssemblySourceAuthority(
        kind=kind,
        authority_id=data["authority_id"],
        document_id=data["document_id"],
        revision=data["revision"],
    )


def _parse_head_binding(value: object) -> tuple[str, RigidTransform]:
    data = _require_mapping(value, "head_binding")
    _require_exact_fields(data, _HEAD_BINDING_FIELDS, "head_binding")
    transform_data = _require_mapping(
        data["head_component_from_selected_head"], "head transform"
    )
    _require_exact_fields(transform_data, _TRANSFORM_FIELDS, "head transform")
    transform = RigidTransform(
        from_frame_id=transform_data["from_frame_id"],
        to_frame_id=transform_data["to_frame_id"],
        rotation=transform_data["rotation"],
        translation_m=transform_data["translation_m"],
    )
    component_id = _require_identifier(data["head_component_id"], "head component")
    return component_id, transform


def _validate_units(value: object) -> None:
    data = _require_mapping(value, "units")
    if data != _UNITS:
        raise ValueError("binding units must use the declared SI contract")


def _decode_payload(payload: str | bytes) -> str:
    if not isinstance(payload, (str, bytes)):
        raise TypeError("binding payload must be text or UTF-8 bytes")
    size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
    if size > MAX_BINDING_BYTES:
        raise ValueError("club assembly binding exceeds the 4 MiB limit")
    try:
        return payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as error:
        raise ValueError("binding payload must be valid UTF-8") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"binding JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object")
    return value


def _require_exact_fields(
    data: Mapping[str, Any], fields: frozenset[str], name: str
) -> None:
    missing = fields - set(data)
    unknown = set(data) - fields
    if missing or unknown:
        raise ValueError(
            f"{name} fields do not match schema; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


__all__: list[str] = []
