"""Golfer-model interchange for the heavy-hit epic (H2, #4564).

Imports body-chain structure (masses, inertias, joint stiffness/damping)
from the multibody engines UpstreamDrift features — MuJoCo (MJCF), Drake
and Pinocchio (URDF), OpenSim (.osim) — through runtime-free XML parsing,
and reduces a named hand-side selection to the :class:`GripBoundary`
record the coupled impact model consumes.
"""

from .body_chain import (
    BODY_CHAIN_FORMAT,
    BodyChain,
    ChainBody,
    ChainJoint,
    body_chain_from_json,
    body_chain_to_json,
    grip_boundary_reduction,
)
from .parsers import (
    chain_from_mjcf,
    chain_from_osim,
    chain_from_urdf,
)

__all__ = [
    "BODY_CHAIN_FORMAT",
    "BodyChain",
    "ChainBody",
    "ChainJoint",
    "body_chain_from_json",
    "body_chain_to_json",
    "chain_from_mjcf",
    "chain_from_osim",
    "chain_from_urdf",
    "grip_boundary_reduction",
]
