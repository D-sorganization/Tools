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

from importlib import import_module
from typing import Any

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
from .stl_export import (
    default_clubhead_stl_filename,
    serialize_clubhead_stl,
    write_clubhead_stl_atomic,
)
from .types import ClubSpec, ClubType, HeadStyle
from .volumetrics import CogReport, head_cog, is_watertight, mesh_volume_centroid

# `assembly_binding` reaches `shared.python.golf_club`, which transitively pulls
# `swing_sim.variation -> solver -> flight -> scipy.integrate`. Importing it at
# module scope meant even `rate_of_closure.club.types` — a leaf module of frozen
# specs — dragged SciPy in, because Python runs this `__init__` first. That
# breaks the Morris UI contract, which must import without SciPy or any optional
# server. Same lazy-export shape as `swing_sim.ground.__init__`.
_LAZY_EXPORTS = {
    "CLUBHEAD_ENGINEERING_FORMAT": "engineering_sidecar",
    "CLUBHEAD_ENGINEERING_MEDIA_TYPE": "engineering_sidecar",
    "build_clubhead_engineering_sidecar": "engineering_sidecar",
    "default_clubhead_engineering_filename": "engineering_sidecar",
    "serialize_clubhead_engineering_sidecar": "engineering_sidecar",
    "write_clubhead_engineering_sidecar_atomic": "engineering_sidecar",
    "APP_FRAME_ID": "simulation_adapter",
    "ClubAssemblyImpactInputs": "simulation_adapter",
    "SimulationCapabilityUse": "simulation_adapter",
    "WorldFromHeadAttitude": "simulation_adapter",
    "adapt_club_assembly_for_impact": "simulation_adapter",
    "CLUB_ASSEMBLY_BINDING_FORMAT": "assembly_binding",
    "CLUB_SPEC_IDENTITY_FORMAT": "assembly_binding",
    "ClubAssemblyBinding": "assembly_binding",
    "ClubAssemblySourceAuthority": "assembly_binding",
    "MassPropertyAuthorityKind": "assembly_binding",
    "build_club_assembly_binding": "assembly_binding",
    "club_assembly_identity": "assembly_binding",
    "club_assembly_identity_payload": "assembly_binding",
    "club_spec_identity": "assembly_binding",
    "club_spec_identity_payload": "assembly_binding",
    "parse_club_assembly_binding": "assembly_binding",
    "serialize_club_assembly_binding": "assembly_binding",
}


def __getattr__(name: str) -> Any:
    """Load the assembly-binding exports without importing SciPy eagerly."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


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
