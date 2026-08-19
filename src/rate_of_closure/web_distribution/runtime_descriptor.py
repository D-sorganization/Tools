"""Strict packaged-web runtime descriptor contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

WEB_RUNTIME_SCHEMA: Final = "rate-of-closure/web-runtime/v1"
WEB_RUNTIME_NAME: Final = "rate-of-closure-runtime.v1.json"
WEB_RUNTIME_ELEMENT_ID: Final = "rate-of-closure-web-runtime"
WEB_AUTHORITY_PATH: Final = "/api/rate-of-closure/v1"
MAX_WEB_RUNTIME_BYTES: Final = 4_096
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STATIC_FIELDS = {"schema_version", "mode", "release_revision"}
_COMPANION_FIELDS = _STATIC_FIELDS | {"authority_path"}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate web runtime field: {key}")
        result[key] = value
    return result


def _strict_utf8(source: bytes) -> str:
    if (
        not source
        or len(source) > MAX_WEB_RUNTIME_BYTES
        or source.startswith(b"\xef\xbb\xbf")
    ):
        raise ValueError("web runtime descriptor violates its UTF-8 byte contract")
    try:
        return source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("web runtime descriptor must be strict UTF-8") from exc


def _revision(value: object) -> str:
    if type(value) is not str:
        raise ValueError("web runtime release_revision must be text")
    if value != "development" and _COMMIT.fullmatch(value) is None:
        raise ValueError("web runtime release_revision is not an exact commit")
    return value


@dataclass(frozen=True, slots=True)
class WebRuntimeDescriptor:
    """One exact public web mode and source identity."""

    mode: str
    release_revision: str
    authority_path: str | None


def parse_web_runtime_descriptor(source: bytes) -> WebRuntimeDescriptor:
    """Parse a duplicate-safe, bounded, exact v1 runtime descriptor."""
    try:
        value = json.loads(_strict_utf8(source), object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("web runtime descriptor must be valid JSON") from exc
    if type(value) is not dict or value.get("schema_version") != WEB_RUNTIME_SCHEMA:
        raise ValueError("unsupported web runtime descriptor")
    mode = value.get("mode")
    fields = _STATIC_FIELDS if mode == "static_inspection" else _COMPANION_FIELDS
    if mode not in {"static_inspection", "local_companion"} or set(value) != fields:
        raise ValueError("web runtime descriptor has unsupported mode or fields")
    authority_path = value.get("authority_path")
    if mode == "local_companion" and authority_path != WEB_AUTHORITY_PATH:
        raise ValueError("web runtime authority path must be fixed and same-origin")
    return WebRuntimeDescriptor(
        mode, _revision(value["release_revision"]), authority_path
    )
