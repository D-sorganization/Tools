"""Deterministic JSON serialization and deserialization for putter heads
(epic #4800, P3).

One versioned wire, ``golf_club.putter_head/1``: head mass, CG, full
inertia tensor, face loft/COR, provenance. Follows the package idiom
(``sort_keys``, ``allow_nan=False``, unknown fields rejected — fail closed).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ._validation import (
    reject_unknown_fields,
    require_finite_float,
    require_identifier,
    require_inertia,
    require_mapping,
    require_vector3,
)

if TYPE_CHECKING:
    from .putter_head import PutterHeadDocument

PUTTER_HEAD_FORMAT = "golf_club.putter_head/1"

_DOCUMENT_FIELDS = frozenset(
    {"format", "name", "head_mass_kg", "loft_deg", "cor", "cg_m"}
    | {"inertia_at_cg_kg_m2", "provenance"}
)
_PROVENANCE_FIELDS = frozenset(
    {"source_kind", "mesh_sha256", "density_kg_m3", "target_mass_kg", "library_name"}
)

__all__ = [
    "PUTTER_HEAD_FORMAT",
    "putter_head_from_json",
    "putter_head_to_json",
]


def putter_head_to_json(document: PutterHeadDocument) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
    from .putter_head import PutterHeadDocument

    if not isinstance(document, PutterHeadDocument):
        raise TypeError("document must be PutterHeadDocument")
    provenance: dict[str, Any] = {"source_kind": document.provenance.source_kind}
    for field in ("mesh_sha256", "density_kg_m3", "target_mass_kg", "library_name"):
        value = getattr(document.provenance, field)
        if value is not None:
            provenance[field] = value
    payload: dict[str, Any] = {
        "format": PUTTER_HEAD_FORMAT,
        "name": document.name,
        "head_mass_kg": document.head_mass_kg,
        "loft_deg": document.loft_deg,
        "cor": document.cor,
        "provenance": provenance,
    }
    if document.cg_m is not None:
        payload["cg_m"] = list(document.cg_m)
    if document.inertia_at_cg_kg_m2 is not None:
        payload["inertia_at_cg_kg_m2"] = [
            list(row) for row in document.inertia_at_cg_kg_m2
        ]
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def putter_head_from_json(text: str) -> PutterHeadDocument:
    """Parse and validate; unknown fields and wrong formats are refused."""
    from .putter_head import PutterHeadDocument, PutterHeadProvenance

    if not isinstance(text, str):
        raise TypeError("text must be str")
    data = require_mapping(json.loads(text), "putter head document")
    reject_unknown_fields(data, _DOCUMENT_FIELDS, "putter head document")
    if data.get("format") != PUTTER_HEAD_FORMAT:
        raise ValueError(f"format must be {PUTTER_HEAD_FORMAT!r}")
    provenance_data = require_mapping(data.get("provenance"), "provenance")
    reject_unknown_fields(provenance_data, _PROVENANCE_FIELDS, "provenance")
    provenance = PutterHeadProvenance(
        source_kind=require_identifier(
            provenance_data.get("source_kind"), "source_kind"
        ),
        mesh_sha256=provenance_data.get("mesh_sha256"),
        density_kg_m3=_optional_number(provenance_data, "density_kg_m3"),
        target_mass_kg=_optional_number(provenance_data, "target_mass_kg"),
        library_name=provenance_data.get("library_name"),
    )
    cg_m, tensor = data.get("cg_m"), data.get("inertia_at_cg_kg_m2")
    return PutterHeadDocument(
        name=require_identifier(data.get("name"), "name"),
        head_mass_kg=require_finite_float(data.get("head_mass_kg"), "head_mass_kg"),
        loft_deg=require_finite_float(data.get("loft_deg"), "loft_deg"),
        cor=require_finite_float(data.get("cor"), "cor"),
        provenance=provenance,
        cg_m=None if cg_m is None else require_vector3(cg_m, "cg_m"),
        inertia_at_cg_kg_m2=None if tensor is None else require_inertia(tensor),
    )


def _optional_number(data: Any, name: str) -> float | None:
    value = data.get(name)
    return None if value is None else require_finite_float(value, name)
