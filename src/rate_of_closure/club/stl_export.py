"""Portable STL serialization for selected parametric club specifications."""

from __future__ import annotations

import re
from pathlib import Path

from rate_of_closure._contracts import ensure, require
from rate_of_closure.mesh import write_binary_stl

from ._atomic_file import write_bytes_atomic
from .parametric_head import build_parametric_head
from .types import ClubSpec

__all__ = [
    "default_clubhead_stl_filename",
    "serialize_clubhead_stl",
    "write_clubhead_stl_atomic",
]

_UNSAFE_FILENAME = re.compile(r"[^a-z0-9]+")
_MAX_FILENAME_STEM_LENGTH = 80
_MM_PER_M = 1000.0
_STL_HEADER = "ROC;units=mm;frame=head;axes=x=target,y=up,z=toe;mesh=parametric"
_WINDOWS_RESERVED_STEMS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def default_clubhead_stl_filename(spec: ClubSpec) -> str:
    """Return a portable default filename for ``spec``.

    Preconditions:
        ``spec`` is a validated :class:`ClubSpec`.
    Postconditions:
        The result is a non-empty, lowercase ``.stl`` filename.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    stem = _UNSAFE_FILENAME.sub("-", spec.name.lower()).strip("-")
    stem = stem[:_MAX_FILENAME_STEM_LENGTH].rstrip("-") or "clubhead"
    if stem in _WINDOWS_RESERVED_STEMS:
        stem = f"clubhead-{stem}"
    ensure(bool(stem), "club name must yield a filename")
    return f"{stem}.stl"


def serialize_clubhead_stl(spec: ClubSpec) -> bytes:
    """Serialize the deterministic parametric head for ``spec`` as binary STL.

    The generator computes internally in SI metres. Because STL coordinates are
    unitless and golf/CAD workflows conventionally interpret them as
    millimetres, this boundary scales vertices to millimetres and records the
    convention plus canonical axes in the fixed binary header.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    payload: bytes = write_binary_stl(
        build_parametric_head(spec) * _MM_PER_M,
        header=_STL_HEADER,
    )
    ensure(bool(payload), "serialized STL must not be empty")
    return payload


def write_clubhead_stl_atomic(spec: ClubSpec, path: str | Path) -> Path:
    """Atomically replace ``path`` with the selected head's serialized STL.

    The temporary file is created beside the destination, so the final replace
    stays on one filesystem. If serialization, writing, flushing, or replacing
    fails, an existing destination remains intact and the temporary file is
    removed where the operating system permits it.
    """
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    require(isinstance(path, (str, Path)), "path must be a string or Path")
    payload = serialize_clubhead_stl(spec)
    return write_bytes_atomic(payload, path)
