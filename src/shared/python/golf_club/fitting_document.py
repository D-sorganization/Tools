"""OEM club-fitting interchange document (club-tester C3, #4552).

One versioned wire, ``golf_club.fitting_document/1``, bundling everything a
fitting run needs to be reproducible from a single artifact:

- the rigid **assembly** (``golf_club.assembly/1`` sub-document),
- the measured **shaft profile** (``golf_club.shaft_profile/1``),
- **face geometry** (loft/lie/bulge/roll),
- the **tip mass** record the shaft-delivery model consumes
  (:class:`.shaft_delivery.ShaftTipMass` fields),
- an optional **mesh reference** — name + SHA-256 of the STL plus exactly
  one of declared density / target mass, the same selector
  :func:`.mesh_mass_properties.mesh_inertia` enforces,
- **provenance** (source kind, tool, ISO-8601 export date).

Serialization follows this package's established idiom (deterministic
``sort_keys`` JSON, ``allow_nan=False``, unknown fields rejected — fail
closed), so an OEM export either parses exactly or is refused with a
named reason; there is no silent field-dropping path. The human-readable
schema reference for OEM integrators is
``docs/specs/CLUB_FITTING_DOCUMENT.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._validation import (
    reject_unknown_fields,
    require_finite_float,
    require_identifier,
    require_mapping,
)
from .assembly import ClubAssembly
from .serialization import assembly_from_json_dict, assembly_to_json_dict
from .shaft_delivery import ShaftTipMass
from .shaft_profile import ShaftProfile
from .shaft_serialization import (
    shaft_profile_from_json_dict,
    shaft_profile_to_json_dict,
)

FITTING_DOCUMENT_FORMAT = "golf_club.fitting_document/1"

_SOURCE_KINDS = frozenset({"oem_export", "measured", "parametric", "cad_derived"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2}))?$")

_DOCUMENT_FIELDS = frozenset(
    {
        "format",
        "document_id",
        "face",
        "assembly",
        "shaft_profile",
        "tip_mass",
        "mesh_reference",
        "provenance",
    }
)
_FACE_FIELDS = frozenset({"loft_deg", "lie_deg", "bulge_m", "roll_m"})
_TIP_MASS_FIELDS = frozenset({"mass_kg", "cg_back_m", "cg_toe_m", "cg_drop_m"})
_MESH_FIELDS = frozenset({"name", "sha256", "density_kg_m3", "target_mass_kg"})
_PROVENANCE_FIELDS = frozenset({"source_kind", "tool_name", "exported_at"})

__all__ = [
    "FITTING_DOCUMENT_FORMAT",
    "ClubFittingDocument",
    "FaceGeometry",
    "FittingProvenance",
    "MeshReference",
    "fitting_document_from_json",
    "fitting_document_to_json",
]


@dataclass(frozen=True)
class FaceGeometry:
    """Static face geometry, degrees and meters.

    ``bulge_m`` / ``roll_m`` are face curvature radii; zero means flat
    (irons/wedges), positive values are the wood convention.
    """

    loft_deg: float
    lie_deg: float
    bulge_m: float = 0.0
    roll_m: float = 0.0

    def __post_init__(self) -> None:
        for name in ("loft_deg", "lie_deg", "bulge_m", "roll_m"):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if not 0.0 < self.loft_deg < 90.0:
            raise ValueError("loft_deg must lie in (0, 90)")
        if not 30.0 < self.lie_deg < 90.0:
            raise ValueError("lie_deg must lie in (30, 90)")
        if self.bulge_m < 0.0 or self.roll_m < 0.0:
            raise ValueError("bulge_m and roll_m must be >= 0 (0 = flat)")


@dataclass(frozen=True)
class MeshReference:
    """Reference to an external head mesh with an integrity pin.

    Exactly one of ``density_kg_m3`` / ``target_mass_kg`` must be given —
    the same scale selector the mesh inertia authority enforces, recorded
    here so the document alone reproduces the derived tensor.
    """

    name: str
    sha256: str
    density_kg_m3: float | None = None
    target_mass_kg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_identifier(self.name, "name"))
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if (self.density_kg_m3 is None) == (self.target_mass_kg is None):
            raise ValueError(
                "exactly one of density_kg_m3 or target_mass_kg must be given"
            )
        for name in ("density_kg_m3", "target_mass_kg"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, require_finite_float(value, name, positive=True)
                )


@dataclass(frozen=True)
class FittingProvenance:
    """Where the document came from, for traceable OEM workflows."""

    source_kind: str
    tool_name: str
    exported_at: str

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"source_kind must be one of {sorted(_SOURCE_KINDS)}",
            )
        object.__setattr__(
            self, "tool_name", require_identifier(self.tool_name, "tool_name")
        )
        if not isinstance(self.exported_at, str) or not _ISO_DATE.fullmatch(
            self.exported_at
        ):
            raise ValueError("exported_at must be an ISO-8601 date or datetime")


@dataclass(frozen=True)
class ClubFittingDocument:
    """The complete, self-describing fitting input for one club build."""

    document_id: str
    face: FaceGeometry
    assembly: ClubAssembly
    shaft_profile: ShaftProfile
    tip_mass: ShaftTipMass
    provenance: FittingProvenance
    mesh_reference: MeshReference | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "document_id", require_identifier(self.document_id, "document_id")
        )
        checks = (
            (self.face, FaceGeometry, "face"),
            (self.assembly, ClubAssembly, "assembly"),
            (self.shaft_profile, ShaftProfile, "shaft_profile"),
            (self.tip_mass, ShaftTipMass, "tip_mass"),
            (self.provenance, FittingProvenance, "provenance"),
        )
        for value, kind, name in checks:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}")
        if self.mesh_reference is not None and not isinstance(
            self.mesh_reference, MeshReference
        ):
            raise TypeError("mesh_reference must be MeshReference or None")


def fitting_document_to_json(document: ClubFittingDocument) -> str:
    """Serialize with deterministic key ordering and no non-finite extensions."""
    if not isinstance(document, ClubFittingDocument):
        raise TypeError("document must be ClubFittingDocument")
    payload: dict[str, Any] = {
        "format": FITTING_DOCUMENT_FORMAT,
        "document_id": document.document_id,
        "face": {
            "loft_deg": document.face.loft_deg,
            "lie_deg": document.face.lie_deg,
            "bulge_m": document.face.bulge_m,
            "roll_m": document.face.roll_m,
        },
        "assembly": assembly_to_json_dict(document.assembly),
        "shaft_profile": shaft_profile_to_json_dict(document.shaft_profile),
        "tip_mass": {
            "mass_kg": document.tip_mass.mass_kg,
            "cg_back_m": document.tip_mass.cg_back_m,
            "cg_toe_m": document.tip_mass.cg_toe_m,
            "cg_drop_m": document.tip_mass.cg_drop_m,
        },
        "provenance": {
            "source_kind": document.provenance.source_kind,
            "tool_name": document.provenance.tool_name,
            "exported_at": document.provenance.exported_at,
        },
    }
    if document.mesh_reference is not None:
        mesh: dict[str, Any] = {
            "name": document.mesh_reference.name,
            "sha256": document.mesh_reference.sha256,
        }
        if document.mesh_reference.density_kg_m3 is not None:
            mesh["density_kg_m3"] = document.mesh_reference.density_kg_m3
        else:
            mesh["target_mass_kg"] = document.mesh_reference.target_mass_kg
        payload["mesh_reference"] = mesh
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fitting_document_from_json(text: str) -> ClubFittingDocument:
    """Parse and validate; unknown fields and wrong formats are refused."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    data = require_mapping(json.loads(text), "fitting document")
    reject_unknown_fields(data, _DOCUMENT_FIELDS, "fitting document")
    if data.get("format") != FITTING_DOCUMENT_FORMAT:
        raise ValueError(
            f"format must be {FITTING_DOCUMENT_FORMAT!r}",
        )
    face_data = require_mapping(data.get("face"), "face")
    reject_unknown_fields(face_data, _FACE_FIELDS, "face")
    tip_data = require_mapping(data.get("tip_mass"), "tip_mass")
    reject_unknown_fields(tip_data, _TIP_MASS_FIELDS, "tip_mass")
    provenance_data = require_mapping(data.get("provenance"), "provenance")
    reject_unknown_fields(provenance_data, _PROVENANCE_FIELDS, "provenance")

    mesh_reference: MeshReference | None = None
    if "mesh_reference" in data:
        mesh_data = require_mapping(data.get("mesh_reference"), "mesh_reference")
        reject_unknown_fields(mesh_data, _MESH_FIELDS, "mesh_reference")
        mesh_reference = MeshReference(
            name=mesh_data.get("name"),
            sha256=mesh_data.get("sha256"),
            density_kg_m3=_optional_number(mesh_data, "density_kg_m3"),
            target_mass_kg=_optional_number(mesh_data, "target_mass_kg"),
        )

    return ClubFittingDocument(
        document_id=data.get("document_id"),
        face=FaceGeometry(
            loft_deg=face_data.get("loft_deg"),
            lie_deg=face_data.get("lie_deg"),
            bulge_m=face_data.get("bulge_m", 0.0),
            roll_m=face_data.get("roll_m", 0.0),
        ),
        assembly=assembly_from_json_dict(
            require_mapping(data.get("assembly"), "assembly")
        ),
        shaft_profile=shaft_profile_from_json_dict(
            require_mapping(data.get("shaft_profile"), "shaft_profile")
        ),
        tip_mass=ShaftTipMass(
            mass_kg=tip_data.get("mass_kg"),
            cg_back_m=tip_data.get("cg_back_m"),
            cg_toe_m=tip_data.get("cg_toe_m"),
            cg_drop_m=tip_data.get("cg_drop_m"),
        ),
        provenance=FittingProvenance(
            source_kind=provenance_data.get("source_kind"),
            tool_name=provenance_data.get("tool_name"),
            exported_at=provenance_data.get("exported_at"),
        ),
        mesh_reference=mesh_reference,
    )


def _optional_number(data: Mapping[str, Any], name: str) -> float | None:
    value = data.get(name)
    if value is None:
        return None
    validated: float = require_finite_float(value, name)
    return validated
