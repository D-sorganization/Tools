"""Portable STL serialization for selected parametric club specifications."""

from __future__ import annotations

import re

from rate_of_closure._contracts import ensure, require
from rate_of_closure.mesh import write_binary_stl

from .parametric_head import build_parametric_head
from .types import ClubSpec

__all__ = ["default_clubhead_stl_filename", "serialize_clubhead_stl"]

_UNSAFE_FILENAME = re.compile(r"[^a-z0-9]+")


def default_clubhead_stl_filename(spec: ClubSpec) -> str:
    """Return a portable default filename for ``spec``.

    Preconditions:
        ``spec`` is a validated :class:`ClubSpec`.
    Postconditions:
        The result is a non-empty, lowercase ``.stl`` filename.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    stem = _UNSAFE_FILENAME.sub("-", spec.name.lower()).strip("-")
    ensure(bool(stem), "club name must yield a filename")
    return f"{stem}.stl"


def serialize_clubhead_stl(spec: ClubSpec) -> bytes:
    """Serialize the deterministic parametric head for ``spec`` as binary STL.

    The triangle coordinates remain in the canonical head frame and SI metres.
    The binary STL format itself carries no unit metadata, so consumers must
    preserve that documented metre convention when importing the file.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    payload: bytes = write_binary_stl(
        build_parametric_head(spec),
        header=f"rate_of_closure {spec.name} metres",
    )
    ensure(bool(payload), "serialized STL must not be empty")
    return payload
