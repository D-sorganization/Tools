"""Club modeling: specs, library, inertia, and parametric head geometry.

The package owns everything "club" for the rate-of-closure explorer:

* :mod:`.types` — the frozen SI :class:`~rate_of_closure.club.types.ClubSpec`.
* :mod:`.library` — a 15-club database of typical published specs.
* :mod:`.inertia` — composite head+shaft+grip inertial model.
* :mod:`.geometry` — shared superellipse-loft mesh helpers.
* :mod:`.head_profiles` — type-specific cross-sections, hosel, face center.
* :mod:`.parametric_head` — representative head meshes with bulge & roll.
* :mod:`.volumetrics` — divergence-theorem volume and geometric COG.

The TypeScript twin lives in ``web/src/model/club.ts`` and is pinned
test-for-test against the numbers asserted by ``tests/rate_of_closure/
test_club.py``.
"""

from __future__ import annotations

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
from .types import ClubSpec, ClubType, HeadStyle
from .volumetrics import CogReport, head_cog, is_watertight, mesh_volume_centroid

__all__ = [
    "CLUB_LIBRARY",
    "REFERENCE_HEAD_MASS_KG",
    "ClubInertia",
    "ClubSpec",
    "ClubType",
    "CogReport",
    "HeadStyle",
    "build_parametric_head",
    "club_inertia",
    "club_names",
    "face_center_point",
    "face_normal_at_offset",
    "face_sagitta",
    "get_club",
    "head_cog",
    "hosel_point",
    "is_watertight",
    "mesh_volume_centroid",
    "parametric_head_mesh",
]
