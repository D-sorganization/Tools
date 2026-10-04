"""Strict calculation-runtime manifest ``calculation-runtime-manifest/v1`` parser.

One-for-one Python port of ``parseRuntimeManifest`` in
``src/rate_of_closure/web/src/model/runtimeManifest.ts``. Both runtimes are
pinned by ``runtime_manifest_parity_v1.json``; error messages deliberately
carry the same key phrases as the TypeScript errors. The contract is specified
in ``docs/specs/active/CALCULATION_RUNTIME_MANIFEST.md``.

``TypeError`` replaces the TypeScript ``TypeError`` (wrong JSON type) and
``ValueError`` replaces ``RangeError`` (right type, invalid value).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

RUNTIME_MANIFEST_SCHEMA = "calculation-runtime-manifest/v1"

_SURFACES = ("tools.pyqt6", "tools.react", "upstreamdrift.pyqt6", "upstreamdrift.react")
_DOMAINS = ("impact", "flight", "ground")
_SOURCE_KINDS = (
    "installed_package",
    "source_checkout",
    "embedded_web_build",
    "test_fixture",
)
_AVAILABILITY = ("available", "unavailable")
_MAX_SAFE_INTEGER = 9_007_199_254_740_991

_STABLE_ID = re.compile(r"[a-z0-9][a-z0-9._/-]*\Z")
_SEMVER_IDENTIFIER = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?\Z"
)
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_PLACEHOLDER = re.compile(
    r"(?:\A|[^A-Za-z0-9])(?:fixme|placeholder|tbd|todo|unknown)(?=\Z|[^A-Za-z0-9])",
    re.IGNORECASE | re.ASCII,
)
_REASON_SENTINELS = frozenset(
    {"x", "na", "none", "nodata", "notavailable", "notapplicable", "unavailable"}
)
_REASON_WHITESPACE = frozenset(
    "\u0009\u000a\u000b\u000c\u000d \u0085                 　"
)
_REASON_NORMALIZATION_SEPARATORS = _REASON_WHITESPACE | frozenset("./_-")
# ECMAScript String.prototype.trim() set, so "nonempty text" matches TypeScript.
_JS_TRIM = "".join("\u0009\u000a\u000b\u000c\u000d                  　﻿")
_WORD = re.compile(r"[A-Za-z]{2,}")

_AUTHORITY_FIELDS = (
    "model_id",
    "model_version",
    "implementation_authority",
    "backend",
    "integrator",
    "request_schema",
    "result_schema",
    "frame_id",
    "unit_system_id",
)


def _record(value: object, fields: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value) or sorted(value) != sorted(
        fields
    ):
        raise ValueError(f"{name} fields do not match v1 schema")
    return value


def _has_unpaired_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _text(value: object, name: str, *, stable: bool = False) -> str:
    if not isinstance(value, str) or not value.strip(_JS_TRIM):
        raise TypeError(f"{name} must be nonempty text")
    if _has_unpaired_surrogate(value):
        raise ValueError(f"{name} must not contain unpaired surrogates")
    if _PLACEHOLDER.search(value):
        raise ValueError(f"{name} must not contain a placeholder")
    if stable and not _STABLE_ID.match(value):
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _nullable_stable_id(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name, stable=True)


def _unavailable_reason(value: object) -> str:
    reason = _text(value, "reason")
    if reason[0] in _REASON_WHITESPACE or reason[-1] in _REASON_WHITESPACE:
        raise ValueError("reason must not contain surrounding whitespace")
    normalized = "".join(
        character.lower() if "A" <= character <= "Z" else character
        for character in reason
        if character not in _REASON_NORMALIZATION_SEPARATORS
    ).rstrip("!?")
    if normalized in _REASON_SENTINELS:
        raise ValueError("reason must not be a sentinel value")
    if not 16 <= len(reason) <= 500:
        raise ValueError("reason must contain 16 to 500 Unicode scalar values")
    if len(_WORD.findall(reason)) < 3:
        raise ValueError("reason must contain at least three explanatory words")
    return reason


def _member(value: object, values: Sequence[str], name: str) -> str:
    if not isinstance(value, str) or value not in values:
        raise ValueError(f"{name} is unsupported")
    return value


def _option_value(value: object) -> bool | int | float | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("option value must be finite")
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(
                "option value exceeds the cross-runtime safe numeric magnitude"
            )
        return value
    return _text(value, "option value")


def _parse_option(value: object) -> dict[str, Any]:
    item = _record(value, ("option_id", "value", "unit"), "runtime option")
    parsed_value = _option_value(item["value"])
    unit = None if item["unit"] is None else _text(item["unit"], "option unit")
    numeric = not isinstance(parsed_value, (bool, str))
    if numeric != (unit is not None):
        raise ValueError(
            "numeric options require a unit; text/bool options require null"
        )
    return {
        "option_id": _text(item["option_id"], "option_id", stable=True),
        "value": parsed_value,
        "unit": unit,
    }


def _parse_build(value: object) -> dict[str, str]:
    item = _record(
        value,
        ("package_name", "package_version", "tools_commit", "build_id"),
        "build",
    )
    version = _text(item["package_version"], "package_version")
    commit = _text(item["tools_commit"], "tools_commit")
    if not _SEMVER.match(version):
        raise ValueError("package_version must use semantic version text")
    if not _SHA.match(commit):
        raise ValueError("tools_commit must be an exact lowercase SHA")
    return {
        "package_name": _text(item["package_name"], "package_name", stable=True),
        "package_version": version,
        "tools_commit": commit,
        "build_id": _text(item["build_id"], "build_id", stable=True),
    }


def _validate_availability(
    status: str,
    reason: str | None,
    identities: Mapping[str, str | None],
    options: Sequence[object],
) -> None:
    values = list(identities.values())
    if status == "available" and (reason is not None or None in values):
        raise ValueError(
            "available calculation requires all identities and null reason"
        )
    if status == "unavailable" and (
        reason is None or any(v is not None for v in values) or options
    ):
        raise ValueError(
            "unavailable calculation requires reason, null identities, and no options"
        )


def _parse_authority(value: object) -> dict[str, Any]:
    fields = ("domain", "status", "reason", *_AUTHORITY_FIELDS, "numerical_options")
    item = _record(value, fields, "calculation authority")
    domain = _member(item["domain"], _DOMAINS, "calculation domain")
    status = _member(item["status"], _AVAILABILITY, "availability")
    reason = None if item["reason"] is None else _unavailable_reason(item["reason"])
    identities = {
        field: _nullable_stable_id(item[field], field) for field in _AUTHORITY_FIELDS
    }
    raw_options = item["numerical_options"]
    if not isinstance(raw_options, list):
        raise TypeError("numerical_options must be an array")
    options = [_parse_option(option) for option in raw_options]
    if len({option["option_id"] for option in options}) != len(options):
        raise ValueError("numerical option IDs must be unique")
    _validate_availability(status, reason, identities, options)
    return {
        "domain": domain,
        "status": status,
        "reason": reason,
        **identities,
        "numerical_options": options,
    }


def _parse_provenance(value: object) -> dict[str, Any]:
    item = _record(
        value, ("source_kind", "source_reference", "evidence_ids"), "provenance"
    )
    raw_evidence = item["evidence_ids"]
    if not isinstance(raw_evidence, list):
        raise TypeError("evidence_ids must be an array")
    evidence = [_text(entry, "evidence_id", stable=True) for entry in raw_evidence]
    if not evidence or len(set(evidence)) != len(evidence):
        raise ValueError("evidence_ids must be nonempty and unique")
    return {
        "source_kind": _member(item["source_kind"], _SOURCE_KINDS, "source_kind"),
        "source_reference": _text(
            item["source_reference"], "source_reference", stable=True
        ),
        "evidence_ids": evidence,
    }


def parse_runtime_manifest(value: object) -> dict[str, Any]:
    """Validate one exact v1 manifest and return a new, independent dict.

    Raises:
        TypeError: a value has the wrong JSON type.
        ValueError: a value has the right type but violates the v1 contract.
    """
    item = _record(
        value,
        ("schema_version", "surface_id", "build", "calculations", "provenance"),
        "runtime manifest",
    )
    if item["schema_version"] != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("unsupported runtime manifest schema")
    raw_calculations = item["calculations"]
    if not isinstance(raw_calculations, list):
        raise TypeError("calculations must be an array")
    calculations = [_parse_authority(entry) for entry in raw_calculations]
    if [entry["domain"] for entry in calculations] != list(_DOMAINS):
        raise ValueError("calculations must contain impact, flight, ground in order")
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA,
        "surface_id": _member(item["surface_id"], _SURFACES, "surface_id"),
        "build": _parse_build(item["build"]),
        "calculations": calculations,
        "provenance": _parse_provenance(item["provenance"]),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, entry in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = entry
    return result


def runtime_manifest_from_json(text: str) -> dict[str, Any]:
    """Parse JSON with duplicate-field rejection, then validate the contract."""
    if not isinstance(text, str):
        raise TypeError("runtime manifest JSON source must be text")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid runtime manifest JSON: {exc}") from exc
    return parse_runtime_manifest(value)


def stable_runtime_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize with stable keys and the shared 11-decimal numeric policy."""
    return str(canonical_numeric_json(manifest))


__all__ = [
    "RUNTIME_MANIFEST_SCHEMA",
    "parse_runtime_manifest",
    "runtime_manifest_from_json",
    "stable_runtime_manifest_json",
]
