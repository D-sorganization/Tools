"""Putter-head import: mesh mass properties -> PutterSpec v2 (epic #4800, P3).

One versioned wire, ``golf_club.putter_head/1``: head mass, CG, full
inertia tensor, face loft/COR, provenance. Two construction paths:

* **Mesh** — a watertight STL through the C1 authority
  :func:`.mesh_mass_properties.mesh_inertia` (never a second mesh
  pipeline), with exactly one of ``density_kg_m3`` / ``target_mass_kg``
  (the same scale selector C1 enforces) and the STL SHA-256 pinned.
* **Club library** — the H1 library putters as the no-mesh fallback,
  resolving the reconciliation documented on
  :class:`shared.python.swing_sim.putting.impact.PutterSpec`: a v2
  document is built *on top of* the v1 spec (:func:`putter_spec`
  recovers it), never replacing it. A library head carries no tensor,
  so it strikes bit-identically to P1's default path (a test gate).

**Head frame** (matching ``swing_sim.impact``): ``x`` = target line
(face normal +x), ``y`` = up, ``z`` = toe; meshes must be authored in
these axes, and CG/tensor are reported in the mesh frame.

**Quasi-static twist.** Same lumped rigid-body posture as
:mod:`.impact_coupling`, one-way diagnostic coupling (no re-feed into
the launch solve): the contact normal impulse ``J = (1 + e) mu v_n``
(``mu`` the ball/effective-head reduced mass) at in-face offset ``r``
torques the head about its CG — toe offsets about the vertical axis
(``I_yy``), high offsets about the heel-toe axis (``I_zz``) — giving
``omega = J r / I`` and the mean contact-window rotation
``theta = omega tau_c / 2 = J r tau_c / (2 I)`` (constant-force
approximation; the closed form the tests gate), with ``tau_c`` the
documented ~0.5 ms putter contact window (``swing_sim.putting.impact``
docs; consistent with ``impact_coupling``'s integrated contact times).
Signs: toe strike opens the face (+, matching face-angle-positive-
open); high strike adds dynamic loft (+). Speed loss couples into the
launch through P1's explicit ``strike(..., head_moi_kg_m2=...)`` hook:
:func:`head_moi_for_strike` collapses the tensor to the exact scalar
of the ``1/(1/M + r^2/I)`` reduction,
``1/I_eff = (r_t^2/I_yy + r_h^2/I_zz)/r^2``.

Serialization follows the package idiom (``sort_keys``,
``allow_nan=False``, unknown fields rejected — fail closed).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.python.swing_sim.impact import GOLF_BALL_MASS_KG
from shared.python.swing_sim.putting.impact import (
    DEFAULT_PUTTER_COR,
    DEFAULT_PUTTER_MOI_KG_M2,
    PutterSpec,
    PuttLaunch,
    strike,
)

from ._validation import (
    Matrix3,
    Vector3,
    reject_unknown_fields,
    require_finite_float,
    require_identifier,
    require_inertia,
    require_mapping,
    require_vector3,
)
from .mesh_mass_properties import mesh_inertia
from .stl_validation import read_binary_stl

PUTTER_HEAD_FORMAT = "golf_club.putter_head/1"

#: Documented putter-ball contact window [s] (~0.5 ms; see module docs).
PUTTER_CONTACT_TIME_S = 5.0e-4

_MM_TO_M = 1.0e-3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KINDS = frozenset({"mesh", "library"})

_DOCUMENT_FIELDS = frozenset(
    {"format", "name", "head_mass_kg", "loft_deg", "cor", "cg_m"}
    | {"inertia_at_cg_kg_m2", "provenance"}
)
_PROVENANCE_FIELDS = frozenset(
    {"source_kind", "mesh_sha256", "density_kg_m3", "target_mass_kg", "library_name"}
)

__all__ = [
    "PUTTER_CONTACT_TIME_S",
    "PUTTER_HEAD_FORMAT",
    "PutterHeadDocument",
    "PutterHeadProvenance",
    "PutterStrikeResult",
    "PutterTwist",
    "head_moi_for_strike",
    "putter_head_from_json",
    "putter_head_from_library",
    "putter_head_from_mesh",
    "putter_head_from_stl",
    "putter_head_to_json",
    "putter_spec",
    "strike_with_head",
    "twist_response",
]


@dataclass(frozen=True)
class PutterHeadProvenance:
    """Fail-closed per kind: ``"mesh"`` requires the STL SHA-256 plus
    exactly one of ``density_kg_m3`` / ``target_mass_kg`` (the C1
    scale selector, recorded so the document alone reproduces the
    tensor); ``"library"`` requires the club-library name only."""

    source_kind: str
    mesh_sha256: str | None = None
    density_kg_m3: float | None = None
    target_mass_kg: float | None = None
    library_name: str | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {sorted(_SOURCE_KINDS)}")
        if self.source_kind == "mesh":
            if self.library_name is not None:
                raise ValueError("mesh provenance must not carry library_name")
            if not isinstance(self.mesh_sha256, str) or not _SHA256.fullmatch(
                self.mesh_sha256
            ):
                raise ValueError("mesh_sha256 must be 64 lowercase hex characters")
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
        else:
            for name in ("mesh_sha256", "density_kg_m3", "target_mass_kg"):
                if getattr(self, name) is not None:
                    raise ValueError(f"library provenance must not carry {name}")
            object.__setattr__(
                self,
                "library_name",
                require_identifier(self.library_name, "library_name"),
            )


@dataclass(frozen=True)
class PutterHeadDocument:
    """PutterSpec v2: the v1 spec fields plus CG, tensor, provenance.

    A mesh-derived head carries ``cg_m`` and the CG inertia tensor
    (head frame); a library-fallback head carries neither, and that
    absence *is* the fallback — strike and twist then use P1's
    documented catalogue default."""

    name: str
    head_mass_kg: float
    loft_deg: float
    cor: float
    provenance: PutterHeadProvenance
    cg_m: Vector3 | None = None
    inertia_at_cg_kg_m2: Matrix3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, PutterHeadProvenance):
            raise TypeError("provenance must be PutterHeadProvenance")
        # Building the v1 spec validates name/mass/loft/cor by its DbC.
        putter_spec(self)
        if self.provenance.source_kind == "mesh":
            if self.cg_m is None or self.inertia_at_cg_kg_m2 is None:
                raise ValueError(
                    "mesh-sourced heads must carry cg_m and inertia_at_cg_kg_m2"
                )
            object.__setattr__(self, "cg_m", require_vector3(self.cg_m, "cg_m"))
            tensor = require_inertia(self.inertia_at_cg_kg_m2)
            for axis in (1, 2):
                if tensor[axis][axis] <= 0.0:
                    raise ValueError(
                        "twist moments I_yy and I_zz must be > 0", tensor[axis][axis]
                    )
            object.__setattr__(self, "inertia_at_cg_kg_m2", tensor)
        elif self.cg_m is not None or self.inertia_at_cg_kg_m2 is not None:
            raise ValueError(
                "library-sourced heads must not carry cg_m or inertia_at_cg_kg_m2"
            )


def putter_spec(document: PutterHeadDocument) -> PutterSpec:
    """The v1 :class:`PutterSpec` a v2 document builds on (reconciliation)."""
    return PutterSpec(
        name=document.name,
        head_mass_kg=document.head_mass_kg,
        loft_deg=document.loft_deg,
        cor=document.cor,
    )


def putter_head_from_mesh(
    name: str,
    triangles: np.ndarray,
    *,
    mesh_sha256: str,
    loft_deg: float,
    cor: float = DEFAULT_PUTTER_COR,
    density_kg_m3: float | None = None,
    target_mass_kg: float | None = None,
) -> PutterHeadDocument:
    """Build a v2 document from a watertight mesh via the C1 authority.

    Exactly one of ``density_kg_m3`` / ``target_mass_kg`` must be
    given (:func:`mesh_inertia` enforces the selector); the mesh must
    be authored in the documented head frame. Raises
    ``PreconditionError``/``ValueError`` on any invalid input."""
    report = mesh_inertia(
        triangles, density_kg_m3=density_kg_m3, mass_kg=target_mass_kg
    )
    tensor = report.inertia_at_cog_kg_m2
    return PutterHeadDocument(
        name=name,
        head_mass_kg=report.mass_kg,
        loft_deg=loft_deg,
        cor=cor,
        provenance=PutterHeadProvenance(
            source_kind="mesh",
            mesh_sha256=mesh_sha256,
            density_kg_m3=density_kg_m3,
            target_mass_kg=target_mass_kg,
        ),
        cg_m=report.centroid_m,
        inertia_at_cg_kg_m2=(tensor[0], tensor[1], tensor[2]),
    )


def putter_head_from_stl(
    name: str,
    stl_path: Path | str,
    *,
    loft_deg: float,
    cor: float = DEFAULT_PUTTER_COR,
    density_kg_m3: float | None = None,
    target_mass_kg: float | None = None,
) -> PutterHeadDocument:
    """Load a binary STL (shared reader) and build the mesh document;
    the provenance SHA-256 is the digest of the exact STL bytes."""
    path = Path(stl_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    triangles = np.asarray(read_binary_stl(path), dtype=np.float64)
    return putter_head_from_mesh(
        name,
        triangles,
        mesh_sha256=digest,
        loft_deg=loft_deg,
        cor=cor,
        density_kg_m3=density_kg_m3,
        target_mass_kg=target_mass_kg,
    )


def putter_head_from_library(
    library_name: str,
    *,
    head_mass_kg: float,
    loft_deg: float,
    cor: float = DEFAULT_PUTTER_COR,
) -> PutterHeadDocument:
    """Build the no-mesh fallback document from a club-library putter.

    Callers pass the library ``ClubSpec`` fields (shared-first: no
    tool-local imports here). The library's scalar MOI is about the
    *shaft* axis — not the CG twist axes, and the CG-to-shaft offset
    needed to convert is unmodeled — so the fallback carries no tensor
    and reproduces P1's catalogue-default behavior exactly."""
    return PutterHeadDocument(
        name=library_name,
        head_mass_kg=head_mass_kg,
        loft_deg=loft_deg,
        cor=cor,
        provenance=PutterHeadProvenance(
            source_kind="library", library_name=library_name
        ),
    )


def head_moi_for_strike(
    document: PutterHeadDocument,
    strike_offset_toe_mm: float = 0.0,
    strike_offset_high_mm: float = 0.0,
) -> float | None:
    """The scalar MOI for P1's ``strike(..., head_moi_kg_m2=...)`` hook.

    ``None`` for a library-fallback head (strike applies its catalogue
    default). Otherwise the exact directional scalar
    ``I_eff = r^2 / (r_t^2/I_yy + r_h^2/I_zz)``; a centered strike
    returns ``I_yy`` (strike ignores the value at zero offset)."""
    tensor = document.inertia_at_cg_kg_m2
    if tensor is None:
        return None
    r_t = require_finite_float(strike_offset_toe_mm, "strike_offset_toe_mm") * _MM_TO_M
    r_h = (
        require_finite_float(strike_offset_high_mm, "strike_offset_high_mm") * _MM_TO_M
    )
    moi_yy, moi_zz = float(tensor[1][1]), float(tensor[2][2])
    r_sq = r_t**2 + r_h**2
    return moi_yy if r_sq == 0.0 else r_sq / (r_t**2 / moi_yy + r_h**2 / moi_zz)


@dataclass(frozen=True)
class PutterTwist:
    """Quasi-static face rotation accumulated during contact.

    ``face_twist_open_deg``: about the vertical axis, + = face opens
    (toe strike). ``loft_twist_add_deg``: about the heel-toe axis,
    + = dynamic loft added (high strike). ``normal_impulse_n_s``: the
    contact impulse ``J``. ``head_moi_kg_m2``: the scalar fed to P1's
    strike hook (``None`` = library fallback, catalogue default).
    """

    face_twist_open_deg: float
    loft_twist_add_deg: float
    normal_impulse_n_s: float
    head_moi_kg_m2: float | None


def twist_response(
    document: PutterHeadDocument,
    clubhead_speed_mps: float,
    *,
    shaft_lean_deg: float = 0.0,
    attack_angle_deg: float = 0.0,
    strike_offset_toe_mm: float = 0.0,
    strike_offset_high_mm: float = 0.0,
) -> PutterTwist:
    """Quasi-static twist at an off-center strike (module-docs model).

    ``theta = J r tau_c / (2 I)`` per axis, with the normal impulse
    ``J = (1 + e) mu v cos(beta)`` from the same effective-mass COR
    model as :func:`strike`; library-fallback heads use the catalogue
    default MOI on both axes. Raises ``ValueError`` outside the strike
    model's input ranges."""
    speed = require_finite_float(clubhead_speed_mps, "clubhead_speed_mps")
    if not 0.0 < speed <= 10.0:
        raise ValueError("clubhead speed must be in (0, 10] m/s")
    for name, value, bound in (
        ("shaft_lean_deg", shaft_lean_deg, 10.0),
        ("attack_angle_deg", attack_angle_deg, 10.0),
        ("strike_offset_toe_mm", strike_offset_toe_mm, 40.0),
        ("strike_offset_high_mm", strike_offset_high_mm, 20.0),
    ):
        if abs(require_finite_float(value, name)) > bound:
            raise ValueError(f"{name} must be within +/-{bound}")
    effective_loft_deg = document.loft_deg + shaft_lean_deg
    if not -2.0 <= effective_loft_deg <= 15.0:
        raise ValueError("effective loft must stay in [-2, 15] deg")

    tensor = document.inertia_at_cg_kg_m2
    moi_yy = DEFAULT_PUTTER_MOI_KG_M2 if tensor is None else tensor[1][1]
    moi_zz = DEFAULT_PUTTER_MOI_KG_M2 if tensor is None else tensor[2][2]
    moi_hook = head_moi_for_strike(
        document, strike_offset_toe_mm, strike_offset_high_mm
    )

    r_t = strike_offset_toe_mm * _MM_TO_M
    r_h = strike_offset_high_mm * _MM_TO_M
    offset_r = math.hypot(r_t, r_h)
    mass = document.head_mass_kg
    scalar_moi = DEFAULT_PUTTER_MOI_KG_M2 if moi_hook is None else moi_hook
    mass_eff = (
        mass if offset_r == 0.0 else 1.0 / (1.0 / mass + offset_r**2 / scalar_moi)
    )
    beta = math.radians(effective_loft_deg - attack_angle_deg)
    reduced_mass = mass_eff * GOLF_BALL_MASS_KG / (mass_eff + GOLF_BALL_MASS_KG)
    impulse = (1.0 + document.cor) * reduced_mass * speed * math.cos(beta)

    half_window = PUTTER_CONTACT_TIME_S / 2.0
    face_twist = math.degrees(impulse * r_t / moi_yy * half_window)
    loft_twist = math.degrees(impulse * r_h / moi_zz * half_window)
    return PutterTwist(
        face_twist_open_deg=face_twist,
        loft_twist_add_deg=loft_twist,
        normal_impulse_n_s=impulse,
        head_moi_kg_m2=moi_hook,
    )


@dataclass(frozen=True)
class PutterStrikeResult:
    """The P1 launch plus this module's twist diagnostic."""

    launch: PuttLaunch
    twist: PutterTwist


def strike_with_head(
    document: PutterHeadDocument,
    clubhead_speed_mps: float,
    shaft_lean_deg: float = 0.0,
    *,
    aim_deg: float = 0.0,
    face_angle_deg: float = 0.0,
    path_angle_deg: float = 0.0,
    attack_angle_deg: float = 0.0,
    strike_offset_toe_mm: float = 0.0,
    strike_offset_high_mm: float = 0.0,
) -> PutterStrikeResult:
    """Solve the P1 impact with this head's MOI feeding the explicit
    hook; a library-fallback head passes ``None`` and reproduces P1's
    catalogue-default results field-for-field (a test gate)."""
    launch = strike(
        putter_spec(document),
        clubhead_speed_mps,
        shaft_lean_deg,
        aim_deg=aim_deg,
        face_angle_deg=face_angle_deg,
        path_angle_deg=path_angle_deg,
        attack_angle_deg=attack_angle_deg,
        strike_offset_toe_mm=strike_offset_toe_mm,
        strike_offset_high_mm=strike_offset_high_mm,
        head_moi_kg_m2=head_moi_for_strike(
            document, strike_offset_toe_mm, strike_offset_high_mm
        ),
    )
    twist = twist_response(
        document,
        clubhead_speed_mps,
        shaft_lean_deg=shaft_lean_deg,
        attack_angle_deg=attack_angle_deg,
        strike_offset_toe_mm=strike_offset_toe_mm,
        strike_offset_high_mm=strike_offset_high_mm,
    )
    return PutterStrikeResult(launch=launch, twist=twist)


def putter_head_to_json(document: PutterHeadDocument) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
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
