"""Strict engineering sidecar for the selected representative clubhead."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rate_of_closure._contracts import ensure, require

from ._atomic_file import write_bytes_atomic
from .assembly_binding import ClubAssemblyBinding
from .stl_export import default_clubhead_stl_filename, serialize_clubhead_stl
from .types import ClubSpec

CLUBHEAD_ENGINEERING_FORMAT = "rate_of_closure.clubhead_engineering/1"
CLUBHEAD_ENGINEERING_MEDIA_TYPE = "application/json"
_HEAD_FRAME_ID = "rate_of_closure.head"
_ASSEMBLY_UNAVAILABLE = (
    "No validated golf_club.ClubAssembly is bound to the selected Rate of "
    "Closure ClubSpec."
)
_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_ZERO = [0.0, 0.0, 0.0]

__all__ = [
    "CLUBHEAD_ENGINEERING_FORMAT",
    "CLUBHEAD_ENGINEERING_MEDIA_TYPE",
    "build_clubhead_engineering_sidecar",
    "default_clubhead_engineering_filename",
    "serialize_clubhead_engineering_sidecar",
    "write_clubhead_engineering_sidecar_atomic",
]


def default_clubhead_engineering_filename(spec: ClubSpec) -> str:
    """Return a portable filename paired with the selected-head STL name."""
    stl_filename = default_clubhead_stl_filename(spec)
    ensure(stl_filename.endswith(".stl"), "STL filename must have .stl suffix")
    return f"{stl_filename[:-4]}.engineering.json"


def _capabilities() -> dict[str, Any]:
    """Return the explicit present/absent mass-property capability ledger."""
    cg_missing = ["heel_toe_coordinate_m", "reconciled_head_frame_origin_transform"]
    return {
        "assembly_mass_properties": {
            "reason": _ASSEMBLY_UNAVAILABLE,
            "status": "unavailable",
        },
        "head_center_of_mass": {
            "missing": cg_missing,
            "status": "unavailable",
        },
        "head_full_inertia_tensor": {
            "missing": [
                "all six independent tensor components about the complete head CG",
                "complete CG and shaft-axis reference transform",
            ],
            "status": "unavailable",
        },
        "head_mass": {"status": "available"},
        "mesh_identity": {"status": "available"},
        "world_from_head_attitude": {
            "missing": ["complete world-from-head rotation"],
            "status": "unavailable",
        },
    }


def _bound_capabilities() -> dict[str, Any]:
    """Return capabilities unlocked by a validated assembly binding."""
    capabilities = _capabilities()
    for name in (
        "assembly_mass_properties",
        "head_center_of_mass",
        "head_full_inertia_tensor",
    ):
        capabilities[name] = {"status": "available"}
    return capabilities


def _frames() -> dict[str, Any]:
    """Declare the mesh/head identity transform and missing world attitude."""
    return {
        "head": {
            "axes": {
                "x_positive": "toward target",
                "y_positive": "up",
                "z_positive": "toward toe",
            },
            "frame_id": _HEAD_FRAME_ID,
            "handedness": "right",
            "length_unit": "m",
            "origin": (
                "parametric mesh head-frame origin; not reconciled to the "
                "physical center of mass"
            ),
        },
        "stl_from_head": {
            "coordinate_scale_mm_per_m": 1000.0,
            "rotation": _IDENTITY,
            "status": "available",
            "translation_m": _ZERO,
        },
        "world_from_head": {
            "reason": (
                "The selected static ClubSpec and STL do not carry a complete "
                "world-from-head attitude."
            ),
            "status": "unavailable",
        },
    }


def _bound_frames(binding: ClubAssemblyBinding) -> dict[str, Any]:
    """Add explicit head-component and assembly frames to the base ledger."""
    frames = _frames()
    transform = binding.head_component_from_selected_head
    frames["head_component_from_head"] = {
        "from_frame_id": transform.from_frame_id,
        "rotation": [list(row) for row in transform.rotation],
        "status": "available",
        "to_frame_id": transform.to_frame_id,
        "translation_m": list(transform.translation_m),
    }
    frames["assembly"] = {
        "frame_id": binding.assembly.frame_id,
        "length_unit": "m",
        "status": "available",
    }
    return frames


def _head_mass_properties(spec: ClubSpec) -> dict[str, Any]:
    """Expose authoritative inputs without promoting partial evidence."""
    return {
        "center_of_mass_m": {
            "evidence_only": {
                "cg_depth_m": {
                    "datum": "back from the face",
                    "value": spec.cg_depth_m,
                },
                "cg_height_m": {
                    "datum": "above the sole plane",
                    "value": spec.cg_height_m,
                },
            },
            "missing": [
                "heel_toe_coordinate_m",
                "reconciled_head_frame_origin_transform",
            ],
            "reason": (
                f"The available offsets are not a complete vector in {_HEAD_FRAME_ID}."
            ),
            "status": "unavailable",
        },
        "inertia_tensor_at_com_kg_m2": {
            "evidence_only": {
                "moi_about_shaft_kg_m2": {
                    "reference": (
                        "shaft axis; not a full tensor about the complete head CG"
                    ),
                    "value": spec.moi_about_shaft_kg_m2,
                }
            },
            "reason": (
                "A single shaft-axis scalar does not determine the six "
                "independent components of a symmetric tensor."
            ),
            "status": "unavailable",
        },
        "mass_kg": {
            "provenance": (
                "selected ClubSpec representative input; not a measurement certificate"
            ),
            "status": "available",
            "value": spec.head_mass_kg,
        },
    }


def _bound_mass_properties(binding: ClubAssemblyBinding) -> dict[str, Any]:
    """Expose complete properties only from the validated shared assembly."""
    head = binding.head_properties_in_selected_frame()
    assembly = binding.assembly.mass_properties
    provenance = "validated_club_assembly_binding"
    return {
        "assembly": {
            "center_of_mass_m": list(assembly.center_of_mass_m),
            "component_ids": list(assembly.component_ids),
            "frame_id": assembly.frame_id,
            "inertia_tensor_at_com_kg_m2": [
                list(row) for row in assembly.inertia_at_com_kg_m2
            ],
            "provenance": provenance,
            "status": "available",
            "total_mass_kg": assembly.total_mass_kg,
        },
        "head": {
            "center_of_mass_m": {
                "frame_id": head.frame_id,
                "provenance": provenance,
                "status": "available",
                "value": list(head.center_of_mass_m),
            },
            "inertia_tensor_at_com_kg_m2": {
                "about": "head_center_of_mass",
                "frame_id": head.frame_id,
                "provenance": provenance,
                "status": "available",
                "value": [list(row) for row in head.inertia_at_com_kg_m2],
            },
            "mass_kg": {
                "provenance": provenance,
                "status": "available",
                "value": head.mass_kg,
            },
        },
    }


def _binding_provenance(binding: ClubAssemblyBinding) -> dict[str, Any]:
    """Return identities and source authority without copying the assembly."""
    authority = binding.authority
    return {
        "assembly_id": binding.assembly.assembly_id,
        "assembly_sha256": binding.assembly_sha256,
        "binding_format": "rate_of_closure.club_assembly_binding/1",
        "head_component_id": binding.head_component_id,
        "selected_spec_sha256": binding.selected_spec_sha256,
        "source_authority": authority.to_json_dict(),
    }


def _mesh_record(spec: ClubSpec, stl_payload: bytes) -> dict[str, Any]:
    """Identify the exact companion STL and its shape-defining inputs."""
    return {
        "byte_length": len(stl_payload),
        "companion_filename": default_clubhead_stl_filename(spec),
        "format": "binary_stl",
        "generator": "rate_of_closure.parametric_head/1",
        "mesh_defining_inputs": {
            "club_type": spec.club_type.value,
            "face_bulge_radius_m": spec.face_bulge_radius_m,
            "face_roll_radius_m": spec.face_roll_radius_m,
            "head_mass_kg": spec.head_mass_kg,
            "head_style": spec.head_style.value,
            "loft_deg": spec.loft_deg,
        },
        "sha256": hashlib.sha256(stl_payload).hexdigest(),
    }


def build_clubhead_engineering_sidecar(
    spec: ClubSpec, binding: ClubAssemblyBinding | None = None
) -> dict[str, Any]:
    """Build a strict sidecar without inferring unavailable mass properties.

    The selected :class:`ClubSpec` authoritatively supplies only its declared
    head mass and partial/scalar evidence. A complete three-coordinate CG,
    full symmetric tensor, assembly record, and world attitude remain absent.
    Their capability records deliberately omit a substitutable ``value``.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    if binding is not None:
        require(
            isinstance(binding, ClubAssemblyBinding),
            "binding must be a ClubAssemblyBinding or None",
        )
        binding.assert_matches(spec)
    stl_payload = serialize_clubhead_stl(spec)
    document: dict[str, Any] = {
        "capabilities": _bound_capabilities() if binding else _capabilities(),
        "format": CLUBHEAD_ENGINEERING_FORMAT,
        "frames": _bound_frames(binding) if binding else _frames(),
        "limitations": [
            (
                "The representative render mesh is not a measured or "
                "density-integrated inertia CAD model."
            ),
            (
                "Two datum-relative CG offsets do not define a three-coordinate "
                "CG in the declared head frame."
            ),
            (
                "One shaft-axis scalar moment cannot determine a symmetric "
                "tensor about the head CG."
            ),
            (
                "A face normal or static loft does not define the complete "
                "world-from-head attitude."
            ),
            *(
                []
                if binding
                else [
                    (
                        "No validated shared golf-club assembly record is connected "
                        "to this selected club specification."
                    )
                ]
            ),
            *(
                [
                    (
                        "The imported source-authority declaration is preserved but "
                        "is not independently certified by this application."
                    )
                ]
                if binding
                else []
            ),
        ],
        "mass_properties": (
            _bound_mass_properties(binding)
            if binding
            else {
                "assembly": {
                    "reason": _ASSEMBLY_UNAVAILABLE,
                    "status": "unavailable",
                },
                "head": _head_mass_properties(spec),
            }
        ),
        "mesh": _mesh_record(spec, stl_payload),
        "provenance": {
            "application": "Rate of Closure Impact Explorer",
            "mass_property_authority": (
                "validated ClubAssembly binding"
                if binding
                else (
                    "selected ClubSpec fields only; no measured or CAD-integrated "
                    "tensor source"
                )
            ),
            "selected_spec": {
                "kind": "rate_of_closure.club.ClubSpec",
                "name": spec.name,
            },
        },
        "subject": {
            "kind": "selected_representative_clubhead",
            "name": spec.name,
        },
    }
    if binding:
        document["provenance"]["assembly_binding"] = _binding_provenance(binding)
    head = document["mass_properties"]["head"]
    if binding is None:
        ensure("value" not in head["center_of_mass_m"], "unavailable CG has no value")
        ensure(
            "value" not in head["inertia_tensor_at_com_kg_m2"],
            "unavailable tensor has no value",
        )
    return document


def serialize_clubhead_engineering_sidecar(
    spec: ClubSpec, binding: ClubAssemblyBinding | None = None
) -> bytes:
    """Serialize the deterministic versioned sidecar as UTF-8 JSON."""
    document = build_clubhead_engineering_sidecar(spec, binding)
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    ensure(bool(payload), "serialized engineering sidecar must not be empty")
    return payload


def write_clubhead_engineering_sidecar_atomic(
    spec: ClubSpec,
    path: str | Path,
    binding: ClubAssemblyBinding | None = None,
) -> Path:
    """Atomically replace ``path`` with the selected head's JSON sidecar."""
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    return Path(
        write_bytes_atomic(serialize_clubhead_engineering_sidecar(spec, binding), path)
    )
