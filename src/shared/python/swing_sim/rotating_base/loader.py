"""Qualified loading for the source-pinned rotating-base study."""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path

from .contract import (
    EXPECTED_UPSTREAM_SOURCE_REVISION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    RotatingBaseProviderResult,
)

EXPECTED_STUDY_SHA256 = (
    "e6a55e6cf91e51f21fe3eb8bcb07b990a7798f18abcaf5ca73f5214cb6c5f9ec"
)
QUALIFIED_STUDY_RESOURCE_NAME = "rotating_base_torso_velocity_study_v1.json"


def load_qualified_study(path: str | Path) -> RotatingBaseProviderResult:
    """Load and verify the exact governed UpstreamDrift study.

    Preconditions
    -------------
    ``path`` identifies UTF-8 JSON with the registered v1 study payload.

    Postconditions
    --------------
    The returned result matches the immutable scientific content digest and
    retains all valid and adverse design rows.
    """
    source = Path(path)
    try:
        study = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"qualified rotating-base study is unreadable: {exc}") from exc
    result = RotatingBaseProviderResult.from_mapping(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "source_revision": EXPECTED_UPSTREAM_SOURCE_REVISION,
            "study": study,
        }
    )
    if result.study_sha256 != EXPECTED_STUDY_SHA256:
        raise ValueError("qualified rotating-base study digest does not match")
    return result


def load_embedded_qualified_study() -> RotatingBaseProviderResult:
    """Load the packaged immutable authority for desktop and web consumers."""
    resource = files(__package__).joinpath("resources", QUALIFIED_STUDY_RESOURCE_NAME)
    with as_file(resource) as path:
        return load_qualified_study(path)


__all__ = [
    "EXPECTED_STUDY_SHA256",
    "QUALIFIED_STUDY_RESOURCE_NAME",
    "load_embedded_qualified_study",
    "load_qualified_study",
]
