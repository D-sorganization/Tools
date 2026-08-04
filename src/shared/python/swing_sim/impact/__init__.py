"""Impact physics subpackage for swing_sim (epic #4103, issue #4106).

Self-façaded: downstream code imports from
``shared.python.swing_sim.impact`` only. The rigid-body COR impulse model
(with the 2/7 rolling-cap friction spin derivation), spring-damper model,
energy-balance validator, and recorder are ported from UpstreamDrift's
``src/shared/python/physics/impact_model`` package; the delivery
front-end (:mod:`.delivery`) and the physics-based gear effect
(:mod:`.gear_effect`) are new in Tools.

The parent ``swing_sim/__init__.py`` façade is wired up during epic
integration; do not add impact exports there from this subpackage.
"""

from __future__ import annotations

from .constants import (
    DRIVER_CG_DEPTH_M,
    DRIVER_COR,
    DRIVER_MASS_KG,
    DRIVER_MOI_KG_M2,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    TYPICAL_CONTACT_DURATION_S,
)
from .delivery import (
    DeliveryDerived,
    DeliveryParameters,
    derive_delivery,
    to_pre_impact_state,
)
from .gear_effect import (
    FaceNormalAtOffset,
    GearEffectResult,
    compute_gear_effect,
    resolve_contact_normal,
)
from .models import (
    SPHERE_ROLLING_CAP_FACTOR,
    FiniteTimeImpactModel,
    ImpactModel,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    create_impact_model,
    face_basis,
    offset_to_face_vector,
)
from .solver import ImpactRecorder, ImpactSolverAPI
from .types import (
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
)
from .utils import validate_energy_balance

__all__ = [
    "DRIVER_CG_DEPTH_M",
    "DRIVER_COR",
    "DRIVER_MASS_KG",
    "DRIVER_MOI_KG_M2",
    "GOLF_BALL_MASS_KG",
    "GOLF_BALL_MOMENT_OF_INERTIA_KG_M2",
    "GOLF_BALL_RADIUS_M",
    "SPHERE_ROLLING_CAP_FACTOR",
    "TYPICAL_CONTACT_DURATION_S",
    "DeliveryDerived",
    "DeliveryParameters",
    "FaceNormalAtOffset",
    "FiniteTimeImpactModel",
    "GearEffectResult",
    "ImpactEvent",
    "ImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "ImpactRecorder",
    "ImpactSolverAPI",
    "PostImpactState",
    "PreImpactState",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "compute_gear_effect",
    "create_impact_model",
    "derive_delivery",
    "face_basis",
    "offset_to_face_vector",
    "resolve_contact_normal",
    "to_pre_impact_state",
    "validate_energy_balance",
]
