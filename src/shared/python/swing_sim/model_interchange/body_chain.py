"""Neutral body-chain contract and grip-boundary reduction (H2, #4564).

Wire ``swing_sim.body_chain/1``: an engine-agnostic description of the
bodies and joints of a golfer model, carrying exactly what the
impact-timescale coupling analysis needs — masses, diagonal inertias,
and per-joint stiffness/damping — with the same fail-closed, deterministic
posture as the delivery-trajectory wire.

:func:`grip_boundary_reduction` collapses a **named** hand-side selection
into the ``{effective mass, stiffness, damping}`` boundary the coupled
impact model consumes. The selection is explicit — the caller lists the
bodies that move with the grip and names the boundary joint; nothing is
inferred from body names. Stiffness/damping come from the boundary joint,
with explicit overrides for engines whose formats do not carry joint
stiffness (URDF, OpenSim).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from shared.python.contracts import require

BODY_CHAIN_FORMAT = "swing_sim.body_chain/1"

_JOINT_TYPES = frozenset({"revolute", "prismatic", "ball", "free", "fixed"})
_BODY_FIELDS = frozenset({"name", "mass_kg", "inertia_diag_kg_m2", "parent", "joint"})
_JOINT_FIELDS = frozenset({"name", "type", "stiffness", "damping"})
_CHAIN_FIELDS = frozenset({"format", "source_id", "bodies"})

__all__ = [
    "BODY_CHAIN_FORMAT",
    "BodyChain",
    "ChainBody",
    "ChainJoint",
    "body_chain_from_json",
    "body_chain_to_json",
    "grip_boundary_reduction",
]


def _nonempty(value: object, name: str) -> str:
    require(
        isinstance(value, str) and value != "" and value.strip() == value,
        f"{name} must be a trimmed nonempty string",
    )
    return str(value)


def _nonnegative(value: object, name: str) -> float:
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number")
    require(
        math.isfinite(value) and float(value) >= 0.0,
        f"{name} must be finite and >= 0",
    )
    return float(value)


@dataclass(frozen=True)
class ChainJoint:
    """The joint connecting a body to its parent.

    ``stiffness`` is N·m/rad for rotational types and N/m for prismatic;
    the wire keeps the engine's own convention and the reduction consumes
    it as-declared. Engines whose formats carry no stiffness export 0.
    """

    name: str
    type: str
    stiffness: float = 0.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "joint name"))
        require(
            self.type in _JOINT_TYPES,
            f"joint type must be one of {sorted(_JOINT_TYPES)}",
        )
        object.__setattr__(self, "stiffness", _nonnegative(self.stiffness, "stiffness"))
        object.__setattr__(self, "damping", _nonnegative(self.damping, "damping"))


@dataclass(frozen=True)
class ChainBody:
    """One body: mass, diagonal inertia, and its parent link."""

    name: str
    mass_kg: float
    inertia_diag_kg_m2: tuple[float, float, float]
    parent: str | None = None
    joint: ChainJoint | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "body name"))
        require(
            isinstance(self.mass_kg, (float, int))
            and math.isfinite(self.mass_kg)
            and float(self.mass_kg) >= 0.0,
            "mass_kg must be finite and >= 0",
        )
        object.__setattr__(self, "mass_kg", float(self.mass_kg))
        inertia = self.inertia_diag_kg_m2
        require(
            isinstance(inertia, (tuple, list)) and len(inertia) == 3,
            "inertia_diag_kg_m2 must have three components",
        )
        object.__setattr__(
            self,
            "inertia_diag_kg_m2",
            tuple(_nonnegative(item, "inertia component") for item in inertia),
        )
        if self.parent is not None:
            object.__setattr__(self, "parent", _nonempty(self.parent, "parent"))
        if self.joint is not None and not isinstance(self.joint, ChainJoint):
            raise TypeError("joint must be ChainJoint or None")


@dataclass(frozen=True)
class BodyChain:
    """A validated golfer-model body chain with unique names."""

    source_id: str
    bodies: tuple[ChainBody, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        require(
            isinstance(self.bodies, tuple)
            and len(self.bodies) >= 1
            and all(isinstance(item, ChainBody) for item in self.bodies),
            "bodies must be a nonempty tuple of ChainBody records",
        )
        names = [body.name for body in self.bodies]
        require(len(set(names)) == len(names), "body names must be unique")
        known = set(names)
        for body in self.bodies:
            require(
                body.parent is None or body.parent in known,
                f"parent {body.parent!r} of {body.name!r} is not in the chain",
            )

    def body(self, name: str) -> ChainBody:
        """The named body; refuses unknown names."""
        for candidate in self.bodies:
            if candidate.name == name:
                return candidate
        require(False, f"unknown body {name!r}")
        raise AssertionError  # pragma: no cover - require raised above


def body_chain_to_json(chain: BodyChain) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
    require(isinstance(chain, BodyChain), "chain must be BodyChain")
    payload: dict[str, Any] = {
        "format": BODY_CHAIN_FORMAT,
        "source_id": chain.source_id,
        "bodies": [
            {
                "name": body.name,
                "mass_kg": body.mass_kg,
                "inertia_diag_kg_m2": list(body.inertia_diag_kg_m2),
                "parent": body.parent,
                "joint": (
                    None
                    if body.joint is None
                    else {
                        "name": body.joint.name,
                        "type": body.joint.type,
                        "stiffness": body.joint.stiffness,
                        "damping": body.joint.damping,
                    }
                ),
            }
            for body in chain.bodies
        ],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def body_chain_from_json(text: str) -> BodyChain:
    """Parse and validate; unknown fields and wrong formats are refused."""
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "body chain must be an object")
    unknown = set(data) - _CHAIN_FIELDS
    require(not unknown, f"unknown body-chain fields: {sorted(unknown)}")
    require(
        data.get("format") == BODY_CHAIN_FORMAT,
        f"format must be {BODY_CHAIN_FORMAT!r}",
    )
    raw_bodies = data.get("bodies")
    require(isinstance(raw_bodies, list), "bodies must be a list")
    bodies = []
    for raw in raw_bodies:
        require(isinstance(raw, dict), "each body must be an object")
        unknown_body = set(raw) - _BODY_FIELDS
        require(not unknown_body, f"unknown body fields: {sorted(unknown_body)}")
        joint = None
        raw_joint = raw.get("joint")
        if raw_joint is not None:
            require(isinstance(raw_joint, dict), "joint must be an object")
            unknown_joint = set(raw_joint) - _JOINT_FIELDS
            require(not unknown_joint, f"unknown joint fields: {sorted(unknown_joint)}")
            joint = ChainJoint(
                name=raw_joint.get("name"),
                type=raw_joint.get("type"),
                stiffness=raw_joint.get("stiffness", 0.0),
                damping=raw_joint.get("damping", 0.0),
            )
        bodies.append(
            ChainBody(
                name=raw.get("name"),
                mass_kg=raw.get("mass_kg"),
                inertia_diag_kg_m2=tuple(raw.get("inertia_diag_kg_m2", ())),
                parent=raw.get("parent"),
                joint=joint,
            )
        )
    return BodyChain(source_id=data.get("source_id"), bodies=tuple(bodies))


def grip_boundary_reduction(
    chain: BodyChain,
    *,
    hand_bodies: tuple[str, ...],
    boundary_joint_of: str,
    stiffness_override_n_m: float | None = None,
    damping_override_n_s_m: float | None = None,
) -> dict[str, object]:
    """Reduce a named hand-side selection to grip-boundary parameters.

    ``hand_bodies`` are the bodies that move with the grip during the
    contact window (their masses sum into the effective mass);
    ``boundary_joint_of`` names the body whose parent joint is the
    grip-to-body boundary. Overrides exist because URDF and ``.osim``
    carry no joint stiffness — supplying one is an explicit, provenance-
    recorded modeling decision, never a guess.

    Returns a plain dict (``effective_mass_kg``, ``stiffness_n_m``,
    ``damping_n_s_m``, ``provenance``) so this shared package does not
    import golf_club; callers construct ``golf_club.GripBoundary(**out)``.
    """
    require(isinstance(chain, BodyChain), "chain must be BodyChain")
    require(
        isinstance(hand_bodies, tuple) and len(hand_bodies) >= 1,
        "hand_bodies must be a nonempty tuple of body names",
    )
    total_mass = 0.0
    for name in hand_bodies:
        total_mass += chain.body(_nonempty(name, "hand body name")).mass_kg
    require(total_mass > 0.0, "selected hand bodies must carry positive mass")

    boundary_body = chain.body(_nonempty(boundary_joint_of, "boundary_joint_of"))
    joint = boundary_body.joint
    if joint is None:
        require(
            False,
            f"body {boundary_body.name!r} has no parent joint to use as the boundary",
        )
        raise AssertionError  # pragma: no cover - require raised above
    stiffness = (
        _nonnegative(stiffness_override_n_m, "stiffness_override_n_m")
        if stiffness_override_n_m is not None
        else joint.stiffness
    )
    damping = (
        _nonnegative(damping_override_n_s_m, "damping_override_n_s_m")
        if damping_override_n_s_m is not None
        else joint.damping
    )
    override_note = (
        " (stiffness/damping overridden by caller)"
        if stiffness_override_n_m is not None or damping_override_n_s_m is not None
        else ""
    )
    provenance = (
        f"{chain.source_id}: bodies {list(hand_bodies)} via joint "
        f"{joint.name!r}{override_note}"
    )
    return {
        "effective_mass_kg": total_mass,
        "stiffness_n_m": stiffness,
        "damping_n_s_m": damping,
        "provenance": provenance,
    }
