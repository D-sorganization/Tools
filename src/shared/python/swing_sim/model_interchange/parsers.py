"""Runtime-free engine-model parsers for the body-chain wire (H2, #4564).

Each parser reads one engine's native model format with the standard
library's XML parser — **no engine runtime is imported** — and produces a
validated :class:`~.body_chain.BodyChain`. Coverage:

- :func:`chain_from_mjcf` — MuJoCo MJCF: nested ``<body>`` tree,
  ``<inertial mass diaginertia>``, ``<joint stiffness damping>`` (both
  native MJCF attributes).
- :func:`chain_from_urdf` — URDF, consumed natively by **Drake and
  Pinocchio**: flat ``<link>`` + ``<joint>`` elements;
  ``<dynamics damping>`` is native, URDF has **no** joint stiffness — it
  parses as 0 and the reduction's explicit override is the sanctioned way
  to supply one.
- :func:`chain_from_osim` — OpenSim ``.osim``: ``<BodySet>`` bodies with
  ``<mass>`` and ``<inertia>`` (six values; the diagonal is taken);
  ``.osim`` joints carry no stiffness/damping in the base schema, so
  bodies join with zeroed ``fixed``-type placeholders and the override
  path applies.

All parsers fail closed on missing structure with named reasons. XML
security note: parsing uses ``xml.etree.ElementTree`` on **local model
files the caller supplies**; these formats do not use DTDs and entity
expansion is not processed by ElementTree.
"""

from __future__ import annotations

import math
from xml.etree import ElementTree

from shared.python.contracts import require

from .body_chain import BodyChain, ChainBody, ChainJoint

__all__ = [
    "chain_from_mjcf",
    "chain_from_osim",
    "chain_from_urdf",
]

_MJCF_JOINT_TYPES = {
    "hinge": "revolute",
    "slide": "prismatic",
    "ball": "ball",
    "free": "free",
}
_URDF_JOINT_TYPES = {
    "revolute": "revolute",
    "continuous": "revolute",
    "prismatic": "prismatic",
    "floating": "free",
    "fixed": "fixed",
    "planar": "free",
}


def _parse_xml(text: str, kind: str) -> ElementTree.Element:
    require(isinstance(text, str) and text.strip() != "", f"{kind} must be nonempty")
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"{kind} is not well-formed XML: {exc}") from exc


def _floats(raw: str, count: int, name: str) -> tuple[float, ...]:
    parts = raw.split()
    require(len(parts) == count, f"{name} must have {count} values")
    values = tuple(float(part) for part in parts)
    require(all(math.isfinite(v) for v in values), f"{name} must be finite")
    return values


def chain_from_mjcf(text: str) -> BodyChain:
    """Parse a MuJoCo MJCF model into a body chain."""
    root = _parse_xml(text, "MJCF")
    require(root.tag == "mujoco", "MJCF root element must be <mujoco>")
    model_name = root.get("model", "mujoco-model")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF must contain a <worldbody>")

    bodies: list[ChainBody] = []

    def visit(element: ElementTree.Element, parent: str | None) -> None:
        for child in element.findall("body"):
            name = child.get("name")
            if name is None or name == "":
                raise ValueError("every MJCF body needs a name")
            inertial = child.find("inertial")
            if inertial is None:
                raise ValueError(
                    f"MJCF body {name!r} needs an explicit <inertial> element"
                )
            mass = float(inertial.get("mass", "nan"))
            require(
                math.isfinite(mass) and mass >= 0.0,
                f"MJCF body {name!r} mass must be finite and >= 0",
            )
            diag_raw = inertial.get("diaginertia")
            if diag_raw is None:
                raise ValueError(
                    f"MJCF body {name!r} needs diaginertia (fullinertia unsupported)"
                )
            values = _floats(diag_raw, 3, f"{name} diaginertia")
            inertia = (values[0], values[1], values[2])
            joint_element = child.find("joint")
            joint = None
            if joint_element is not None:
                mjcf_type = joint_element.get("type", "hinge")
                require(
                    mjcf_type in _MJCF_JOINT_TYPES,
                    f"unsupported MJCF joint type {mjcf_type!r}",
                )
                joint = ChainJoint(
                    name=joint_element.get("name", f"{name}-joint"),
                    type=_MJCF_JOINT_TYPES[mjcf_type],
                    stiffness=float(joint_element.get("stiffness", "0")),
                    damping=float(joint_element.get("damping", "0")),
                )
            bodies.append(
                ChainBody(
                    name=str(name),
                    mass_kg=mass,
                    inertia_diag_kg_m2=inertia,
                    parent=parent,
                    joint=joint,
                )
            )
            visit(child, str(name))

    visit(worldbody, None)
    require(bool(bodies), "MJCF model contains no bodies")
    return BodyChain(source_id=f"mjcf:{model_name}", bodies=tuple(bodies))


def chain_from_urdf(text: str) -> BodyChain:
    """Parse a URDF model (Drake and Pinocchio both consume URDF natively)."""
    root = _parse_xml(text, "URDF")
    require(root.tag == "robot", "URDF root element must be <robot>")
    model_name = root.get("name", "urdf-model")

    masses: dict[str, tuple[float, tuple[float, float, float]]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name is None or name == "":
            raise ValueError("every URDF link needs a name")
        inertial = link.find("inertial")
        if inertial is None:
            # Massless frame links are legal URDF; they carry zero mass.
            masses[str(name)] = (0.0, (0.0, 0.0, 0.0))
            continue
        mass_element = inertial.find("mass")
        inertia_element = inertial.find("inertia")
        if mass_element is None or inertia_element is None:
            raise ValueError(f"URDF link {name!r} inertial needs <mass> and <inertia>")
        mass = float(mass_element.get("value", "nan"))
        require(
            math.isfinite(mass) and mass >= 0.0,
            f"URDF link {name!r} mass must be finite and >= 0",
        )
        diag = tuple(
            float(inertia_element.get(axis, "nan")) for axis in ("ixx", "iyy", "izz")
        )
        require(
            all(math.isfinite(v) and v >= 0.0 for v in diag),
            f"URDF link {name!r} inertia diagonal must be finite and >= 0",
        )
        masses[str(name)] = (mass, (diag[0], diag[1], diag[2]))

    parent_of: dict[str, tuple[str, ChainJoint]] = {}
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        require(
            joint_name is not None and joint_type in _URDF_JOINT_TYPES,
            f"URDF joint {joint_name!r} has unsupported type {joint_type!r}",
        )
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {joint_name!r} needs <parent> and <child>")
        dynamics = joint.find("dynamics")
        damping = float(dynamics.get("damping", "0")) if dynamics is not None else 0.0
        child_link = str(child.get("link"))
        parent_of[child_link] = (
            str(parent.get("link")),
            ChainJoint(
                name=str(joint_name),
                type=_URDF_JOINT_TYPES[str(joint_type)],
                stiffness=0.0,  # URDF has no joint stiffness; override explicitly.
                damping=damping,
            ),
        )

    require(bool(masses), "URDF model contains no links")
    bodies = []
    for name, (mass, inertia) in masses.items():
        parent_joint = parent_of.get(name)
        bodies.append(
            ChainBody(
                name=name,
                mass_kg=mass,
                inertia_diag_kg_m2=inertia,
                parent=parent_joint[0] if parent_joint else None,
                joint=parent_joint[1] if parent_joint else None,
            )
        )
    return BodyChain(source_id=f"urdf:{model_name}", bodies=tuple(bodies))


def chain_from_osim(text: str) -> BodyChain:
    """Parse an OpenSim ``.osim`` model's BodySet into a body chain."""
    root = _parse_xml(text, "OSIM")
    require(root.tag == "OpenSimDocument", "OSIM root must be <OpenSimDocument>")
    model = root.find("Model")
    if model is None:
        raise ValueError("OSIM document must contain a <Model>")
    model_name = model.get("name", "osim-model")
    body_set = model.find("BodySet")
    objects = body_set.find("objects") if body_set is not None else None
    if objects is None:
        raise ValueError("OSIM model must contain BodySet/objects")

    bodies = []
    previous: str | None = None
    for element in objects.findall("Body"):
        name = element.get("name")
        if name is None or name == "":
            raise ValueError("every OSIM Body needs a name")
        mass_element = element.find("mass")
        if mass_element is None or mass_element.text is None:
            raise ValueError(f"OSIM body {name!r} needs a <mass>")
        mass = float(mass_element.text.strip())
        require(
            math.isfinite(mass) and mass >= 0.0,
            f"OSIM body {name!r} mass must be finite and >= 0",
        )
        inertia_element = element.find("inertia")
        if inertia_element is not None and inertia_element.text:
            six = _floats(inertia_element.text.strip(), 6, f"{name} inertia")
            diag = (six[0], six[1], six[2])
        else:
            diag = (0.0, 0.0, 0.0)
        # .osim joints carry no stiffness/damping in the base schema; bodies
        # chain in document order with zeroed placeholders, and the
        # grip-boundary reduction's explicit override supplies real values.
        joint = (
            ChainJoint(name=f"{name}-joint", type="fixed")
            if previous is not None
            else None
        )
        bodies.append(
            ChainBody(
                name=str(name),
                mass_kg=mass,
                inertia_diag_kg_m2=diag,
                parent=previous,
                joint=joint,
            )
        )
        previous = str(name)
    require(bool(bodies), "OSIM model contains no bodies")
    return BodyChain(source_id=f"osim:{model_name}", bodies=tuple(bodies))
