"""Club modeling: specs, library, inertia, and parametric head geometry.

The package owns everything "club" for the rate-of-closure explorer:

* :mod:`.types` — the frozen SI :class:`~rate_of_closure.club.types.ClubSpec`.
* :mod:`.library` — a 15-club database of typical published specs.
* :mod:`.inertia` — composite head+shaft+grip inertial model.
* :mod:`.geometry` — shared superellipse-loft mesh helpers.
* :mod:`.head_profiles` — type-specific cross-sections, hosel, face center.
* :mod:`.parametric_head` — representative head meshes with bulge & roll.
* :mod:`.stl_export` — deterministic selected-spec binary STL serialization.
* :mod:`.engineering_sidecar` — strict mass-property capability/provenance JSON.
* :mod:`.volumetrics` — divergence-theorem volume and geometric COG.

The TypeScript twin lives in ``web/src/model/club.ts`` and is pinned
test-for-test against the numbers asserted by ``tests/rate_of_closure/
test_club.py``.
"""

from __future__ import annotations

from .assembly_binding import (
    CLUB_ASSEMBLY_BINDING_FORMAT,
    CLUB_SPEC_IDENTITY_FORMAT,
    ClubAssemblyBinding,
    ClubAssemblySourceAuthority,
    MassPropertyAuthorityKind,
    build_club_assembly_binding,
    club_assembly_identity,
    club_assembly_identity_payload,
    club_spec_identity,
    club_spec_identity_payload,
    parse_club_assembly_binding,
    serialize_club_assembly_binding,
)
from .engineering_sidecar import (
    CLUBHEAD_ENGINEERING_FORMAT,
    CLUBHEAD_ENGINEERING_MEDIA_TYPE,
    build_clubhead_engineering_sidecar,
    default_clubhead_engineering_filename,
    serialize_clubhead_engineering_sidecar,
    write_clubhead_engineering_sidecar_atomic,
)
from .head_profiles import face_center_point, hosel_point
from .inertia import ClubInertia, club_inertia
from .library import CLUB_LIBRARY, club_names, get_club
from .parametric_head import (
    REFERENCE_HEAD_MASS_KG,
    build_parametric_head,
    face_normal_at_offset,
    face_sagitta,
    parametric_head_mesh,
)
from .simulation_adapter import (
    APP_FRAME_ID,
    ClubAssemblyImpactInputs,
    SimulationCapabilityUse,
    WorldFromHeadAttitude,
    adapt_club_assembly_for_impact,
)
from .stl_export import (
    default_clubhead_stl_filename,
    serialize_clubhead_stl,
    write_clubhead_stl_atomic,
)
from .types import ClubSpec, ClubType, HeadStyle
from .volumetrics import CogReport, head_cog, is_watertight, mesh_volume_centroid

__all__ = [
    "APP_FRAME_ID",
    "CLUB_ASSEMBLY_BINDING_FORMAT",
    "CLUB_SPEC_IDENTITY_FORMAT",
    "CLUBHEAD_ENGINEERING_FORMAT",
    "CLUBHEAD_ENGINEERING_MEDIA_TYPE",
    "CLUB_LIBRARY",
    "REFERENCE_HEAD_MASS_KG",
    "ClubAssemblyBinding",
    "ClubAssemblyImpactInputs",
    "ClubAssemblySourceAuthority",
    "ClubInertia",
    "ClubSpec",
    "ClubType",
    "CogReport",
    "HeadStyle",
    "MassPropertyAuthorityKind",
    "SimulationCapabilityUse",
    "WorldFromHeadAttitude",
    "adapt_club_assembly_for_impact",
    "build_club_assembly_binding",
    "build_clubhead_engineering_sidecar",
    "build_parametric_head",
    "club_assembly_identity",
    "club_assembly_identity_payload",
    "club_inertia",
    "club_names",
    "club_spec_identity",
    "club_spec_identity_payload",
    "default_clubhead_engineering_filename",
    "default_clubhead_stl_filename",
    "face_center_point",
    "face_normal_at_offset",
    "face_sagitta",
    "get_club",
    "head_cog",
    "hosel_point",
    "is_watertight",
    "mesh_volume_centroid",
    "parametric_head_mesh",
    "parse_club_assembly_binding",
    "serialize_club_assembly_binding",
    "serialize_clubhead_engineering_sidecar",
    "serialize_clubhead_stl",
    "write_clubhead_engineering_sidecar_atomic",
    "write_clubhead_stl_atomic",
]
