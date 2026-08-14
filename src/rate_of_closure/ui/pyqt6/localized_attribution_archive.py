"""Bounded authority-only JSON persistence for localized attribution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rate_of_closure.variation.localized_attribution import (
    AttributionAuthority,
    attribution_authority_from_dict,
    attribution_authority_to_dict,
)
from shared.python.contracts import require

MAX_AUTHORITY_JSON_BYTES = 8 * 1024 * 1024
ARCHIVED_AUTHORITY_DISCLAIMER = (
    "Loaded archived paired authority. Schema, pair matrix, units, and typed "
    "availability were validated; the originating execution was not rerun or "
    "provenance-verified in this session."
)


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def authority_to_json(authority: AttributionAuthority) -> str:
    """Serialize one finite authority document in canonical compact form."""
    return json.dumps(
        attribution_authority_to_dict(authority),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def authority_from_json(text: str) -> AttributionAuthority:
    """Parse strict authority JSON after enforcing its UTF-8 byte cap."""
    require(isinstance(text, str), "authority JSON must be text")
    require(
        len(text.encode("utf-8")) <= MAX_AUTHORITY_JSON_BYTES,
        "authority JSON exceeds byte cap",
    )
    try:
        raw: Any = json.loads(text, parse_constant=_reject_nonfinite_constant)
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError("invalid localized attribution authority JSON") from error
    return attribution_authority_from_dict(raw)


def read_authority_json(path: str | Path) -> AttributionAuthority:
    """Read at most the bounded authority document plus one sentinel byte."""
    source = Path(path)
    with source.open("rb") as handle:
        payload = handle.read(MAX_AUTHORITY_JSON_BYTES + 1)
    require(
        len(payload) <= MAX_AUTHORITY_JSON_BYTES,
        "authority JSON exceeds byte cap",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("authority JSON must be valid UTF-8") from error
    return authority_from_json(text)


def write_authority_json(path: str | Path, authority: AttributionAuthority) -> None:
    """Atomically write bounded authority JSON without claiming provenance."""
    payload = authority_to_json(authority).encode("utf-8")
    require(
        len(payload) <= MAX_AUTHORITY_JSON_BYTES,
        "authority JSON exceeds byte cap",
    )
    destination = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "ARCHIVED_AUTHORITY_DISCLAIMER",
    "MAX_AUTHORITY_JSON_BYTES",
    "authority_from_json",
    "authority_to_json",
    "read_authority_json",
    "write_authority_json",
]
