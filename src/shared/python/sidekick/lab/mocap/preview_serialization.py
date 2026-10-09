"""Strict JSON interchange for path-free preview manifests."""

from __future__ import annotations

import json
from typing import Any

from .preview_artifacts import (
    PREVIEW_MANIFEST_SCHEMA_VERSION,
    PreviewArtifact,
    PreviewManifest,
    _manifest_payload,
)


def dumps_preview_manifest(manifest: PreviewManifest) -> str:
    """Serialize a validated manifest using stable keys and no filesystem paths."""
    if not isinstance(manifest, PreviewManifest):
        raise TypeError("manifest must be a PreviewManifest")
    payload = _manifest_payload(manifest)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def load_preview_manifest(text: str) -> PreviewManifest:
    """Load a path-free preview manifest and reject unknown or missing fields."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("preview manifest must contain valid JSON") from exc
    data = _require_fields(
        payload,
        {
            "schema_version",
            "experiment_id",
            "cache_key_sha256",
            "provider_sha256",
            "artifacts",
        },
        "manifest",
    )
    if data["schema_version"] != PREVIEW_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported preview manifest schema version")
    artifacts = tuple(_load_artifact(item) for item in data["artifacts"])
    return PreviewManifest(
        data["experiment_id"],
        data["cache_key_sha256"],
        data["provider_sha256"],
        artifacts,
        data["schema_version"],
    )


def _load_artifact(value: Any) -> PreviewArtifact:
    data = _require_fields(
        value, {"artifact_id", "sha256", "byte_size", "media_type"}, "artifact"
    )
    return PreviewArtifact(
        data["artifact_id"], data["sha256"], data["byte_size"], data["media_type"]
    )


def _require_fields(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{field} fields differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value
