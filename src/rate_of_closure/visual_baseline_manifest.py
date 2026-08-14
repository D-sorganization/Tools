"""Strict immutable authority for approved cross-toolkit visual baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import PurePath
from typing import Any

from rate_of_closure.visualization_tab_manifest import load_visualization_tab_manifest

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_POLICY = "proposed-off-default-branch-approved-after-protected-merge"


class VisualBaselineManifestError(ValueError):
    """Raised when baseline identity, coverage, or tolerances are malformed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualBaselineManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise VisualBaselineManifestError(f"non-finite JSON value: {value}")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VisualBaselineManifestError(f"{context} fields must be exact")
    return value


def _text(value: object, context: str, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise VisualBaselineManifestError(f"{context} must be bounded text")
    return value


def _integer(value: object, context: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise VisualBaselineManifestError(f"{context} is outside its domain")
    return value


def _filename(value: object) -> str:
    text = _text(value, "baseline filename")
    if PurePath(text).name != text or not text.endswith(".png"):
        raise VisualBaselineManifestError("baseline filename must be one PNG basename")
    return text


@dataclass(frozen=True)
class VisualBaselineTolerance:
    """Bounded image-difference limits represented without floating JSON."""

    changed_channel_threshold: int
    max_mean_channel_delta_microunits: int
    max_changed_pixel_fraction_microunits: int


@dataclass(frozen=True)
class VisualBaselineEntry:
    """One exact surface/tab reference image and its drift envelope."""

    surface: str
    tab_id: str
    environment: str
    filename: str
    sha256: str
    tolerance: VisualBaselineTolerance


@dataclass(frozen=True)
class VisualBaselineManifest:
    """Versioned baseline set whose merge is the approval event."""

    schema_id: str
    schema_version: int
    approval_policy: str
    source_artifact_commit: str
    baselines: tuple[VisualBaselineEntry, ...]

    def for_surface(self, surface: str) -> tuple[VisualBaselineEntry, ...]:
        return tuple(entry for entry in self.baselines if entry.surface == surface)


def _tolerance(value: object) -> VisualBaselineTolerance:
    document = _object(
        value,
        {
            "changed_channel_threshold",
            "max_mean_channel_delta_microunits",
            "max_changed_pixel_fraction_microunits",
        },
        "tolerance",
    )
    return VisualBaselineTolerance(
        _integer(document["changed_channel_threshold"], "channel threshold", 0, 255),
        _integer(
            document["max_mean_channel_delta_microunits"],
            "mean channel delta",
            0,
            1_000_000,
        ),
        _integer(
            document["max_changed_pixel_fraction_microunits"],
            "changed pixel fraction",
            0,
            1_000_000,
        ),
    )


def _entry(value: object) -> VisualBaselineEntry:
    document = _object(
        value,
        {"surface", "tab_id", "environment", "file", "sha256", "tolerance"},
        "baseline",
    )
    surface = _text(document["surface"], "surface")
    if surface not in {"react", "pyqt"}:
        raise VisualBaselineManifestError("unknown baseline surface")
    digest = _text(document["sha256"], "baseline SHA-256")
    if _HEX_SHA256.fullmatch(digest) is None:
        raise VisualBaselineManifestError("baseline SHA-256 must be lowercase hex")
    return VisualBaselineEntry(
        surface,
        _text(document["tab_id"], "tab id"),
        _text(document["environment"], "environment"),
        _filename(document["file"]),
        digest,
        _tolerance(document["tolerance"]),
    )


def load_visual_baseline_manifest() -> VisualBaselineManifest:
    """Load the packaged references and bind them to exact tab coverage."""

    resource = files("rate_of_closure").joinpath("visual_baselines.v1.json")
    try:
        raw = json.loads(
            resource.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualBaselineManifestError("baseline manifest is unreadable") from exc
    document = _object(
        raw,
        {
            "schema_id",
            "schema_version",
            "approval_policy",
            "source_artifact_commit",
            "baselines",
        },
        "manifest",
    )
    raw_entries = document["baselines"]
    if not isinstance(raw_entries, list):
        raise VisualBaselineManifestError("baselines must be an array")
    manifest = VisualBaselineManifest(
        _text(document["schema_id"], "schema id"),
        _integer(document["schema_version"], "schema version", 1, 1),
        _text(document["approval_policy"], "approval policy"),
        _text(document["source_artifact_commit"], "source artifact commit"),
        tuple(_entry(value) for value in raw_entries),
    )
    if (
        manifest.schema_id != "rate-of-closure/visual-baselines"
        or manifest.approval_policy != _POLICY
        or _HEX_COMMIT.fullmatch(manifest.source_artifact_commit) is None
    ):
        raise VisualBaselineManifestError("unsupported baseline manifest")
    identities = tuple((entry.surface, entry.tab_id) for entry in manifest.baselines)
    expected = tuple(
        (entry.surface, entry.tab_id)
        for entry in load_visualization_tab_manifest().tabs
    )
    if identities != expected or len(set(identities)) != len(identities):
        raise VisualBaselineManifestError(
            "baselines must exactly match visibility authority"
        )
    filenames = tuple((entry.surface, entry.filename) for entry in manifest.baselines)
    if len(filenames) != len(set(filenames)):
        raise VisualBaselineManifestError("duplicate baseline filename")
    return manifest


__all__ = [
    "VisualBaselineEntry",
    "VisualBaselineManifest",
    "VisualBaselineManifestError",
    "VisualBaselineTolerance",
    "load_visual_baseline_manifest",
]
