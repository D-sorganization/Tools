"""Protected comparison of hosted candidates with approved visual references."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from importlib.resources import files
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from rate_of_closure.visual_baseline_manifest import (
    VisualBaselineEntry,
    load_visual_baseline_manifest,
)

_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_SCHEMA = "rate-of-closure/visual-baseline-candidates"
_CANDIDATE_POLICY = "candidate-diagnostic-not-approved-until-protected-merge"
_MAX_PNG_BYTES = 10 * 1024 * 1024
_MAX_DIMENSION = 4096
_MAX_PIXELS = 16_777_216


class VisualBaselineComparisonError(ValueError):
    """Raised when candidate identity, bytes, or pixels violate authority."""


@dataclass(frozen=True)
class VisualBaselineComparison:
    """Measured difference for one verified reference/candidate pair."""

    surface: str
    tab_id: str
    mean_channel_delta_microunits: int
    changed_pixel_fraction_microunits: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualBaselineComparisonError(f"duplicate candidate field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise VisualBaselineComparisonError(f"non-finite candidate value: {value}")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VisualBaselineComparisonError(f"{context} fields must be exact")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise VisualBaselineComparisonError(f"{context} must be bounded text")
    return value


def _filename(value: object) -> str:
    text = _text(value, "candidate filename")
    if PurePath(text).name != text or not text.endswith(".png"):
        raise VisualBaselineComparisonError("candidate filename must be one PNG")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualBaselineComparisonError("candidate manifest is unreadable") from exc
    if not isinstance(raw, dict):
        raise VisualBaselineComparisonError("candidate manifest must be an object")
    return raw


def _candidate_manifest(
    root: Path,
    surface: str,
    references: tuple[VisualBaselineEntry, ...],
    expected_source_commit: str,
) -> dict[str, bytes]:
    document = _read_json(root / surface / "manifest.json")
    camel = surface == "react"
    schema_key = "schemaId" if camel else "schema_id"
    version_key = "schemaVersion" if camel else "schema_version"
    policy_key = "artifactPolicy" if camel else "artifact_policy"
    commit_key = "sourceCommit" if camel else "source_commit"
    tab_key = "tabId" if camel else "tab_id"
    _object(
        document,
        {
            schema_key,
            version_key,
            policy_key,
            commit_key,
            "surface",
            "environment",
            "captures",
        },
        "candidate manifest",
    )
    if (
        document[schema_key] != _CANDIDATE_SCHEMA
        or type(document[version_key]) is not int
        or document[version_key] != 1
        or document[policy_key] != _CANDIDATE_POLICY
        or document["surface"] != surface
        or _COMMIT.fullmatch(_text(document[commit_key], "candidate commit")) is None
        or document[commit_key] != expected_source_commit
    ):
        raise VisualBaselineComparisonError("candidate identity is unsupported")
    captures = document["captures"]
    if not isinstance(captures, list):
        raise VisualBaselineComparisonError("candidate captures must be an array")
    expected = tuple((entry.tab_id, entry.filename) for entry in references)
    decoded: list[tuple[str, str, str]] = []
    for value in captures:
        capture = _object(value, {tab_key, "file", "sha256"}, "candidate capture")
        digest = _text(capture["sha256"], "candidate SHA-256")
        if _HEX.fullmatch(digest) is None:
            raise VisualBaselineComparisonError(
                "candidate SHA-256 must be lowercase hex"
            )
        decoded.append(
            (
                _text(capture[tab_key], "candidate tab id"),
                _filename(capture["file"]),
                digest,
            )
        )
    if tuple((tab, name) for tab, name, _digest in decoded) != expected:
        raise VisualBaselineComparisonError(
            "candidate coverage differs from references"
        )
    if any(document["environment"] != entry.environment for entry in references):
        raise VisualBaselineComparisonError(
            "candidate environment differs from reference"
        )
    result: dict[str, bytes] = {}
    for tab_id, filename, digest in decoded:
        path = root / surface / filename
        data = _read_png_bytes(path)
        if hashlib.sha256(data).hexdigest() != digest:
            raise VisualBaselineComparisonError(
                "candidate digest differs from manifest"
            )
        result[tab_id] = data
    return result


def _read_png_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if not 1 <= size <= _MAX_PNG_BYTES:
            raise VisualBaselineComparisonError("PNG byte size is outside its domain")
        return path.read_bytes()
    except OSError as exc:
        raise VisualBaselineComparisonError("PNG is unreadable") from exc


def _pixels(data: bytes) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if (
                image.format != "PNG"
                or width < 1
                or height < 1
                or width > _MAX_DIMENSION
                or height > _MAX_DIMENSION
                or width * height > _MAX_PIXELS
            ):
                raise VisualBaselineComparisonError("PNG geometry is unsupported")
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            return cast(np.ndarray[Any, np.dtype[np.uint8]], pixels)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise VisualBaselineComparisonError("PNG decoding failed") from exc


def _compare(
    reference: VisualBaselineEntry, reference_bytes: bytes, candidate_bytes: bytes
) -> VisualBaselineComparison:
    if hashlib.sha256(reference_bytes).hexdigest() != reference.sha256:
        raise VisualBaselineComparisonError("reference digest differs from manifest")
    expected = _pixels(reference_bytes)
    observed = _pixels(candidate_bytes)
    if expected.shape != observed.shape:
        raise VisualBaselineComparisonError(
            "candidate dimensions differ from reference"
        )
    difference = np.abs(expected.astype(np.int16) - observed.astype(np.int16))
    mean = int(round(float(np.mean(difference)) / 255 * 1_000_000))
    changed = int(
        round(
            float(
                np.mean(
                    np.any(
                        difference > reference.tolerance.changed_channel_threshold,
                        axis=2,
                    )
                )
            )
            * 1_000_000
        )
    )
    if (
        mean > reference.tolerance.max_mean_channel_delta_microunits
        or changed > reference.tolerance.max_changed_pixel_fraction_microunits
    ):
        raise VisualBaselineComparisonError(
            f"visual drift exceeds limits for {reference.surface}/{reference.tab_id}: "
            f"mean={mean}, changed={changed}"
        )
    return VisualBaselineComparison(reference.surface, reference.tab_id, mean, changed)


def _reference_bytes(surface: str, reference: VisualBaselineEntry) -> bytes:
    resource = files("rate_of_closure").joinpath(
        "visual_baselines", "v1", surface, reference.filename
    )
    try:
        data = resource.read_bytes()
    except OSError as exc:
        raise VisualBaselineComparisonError("reference PNG is unreadable") from exc
    if not 1 <= len(data) <= _MAX_PNG_BYTES:
        raise VisualBaselineComparisonError("reference PNG byte size is unsupported")
    return data


def compare_visual_baselines(
    candidate_root: Path, expected_source_commit: str
) -> tuple[VisualBaselineComparison, ...]:
    """Verify all candidate identities, reference bytes, and pixel tolerances.

    Every entry is evaluated even when earlier entries drift, so the report
    names every offender instead of masking the tabs behind the first one
    (issue #4844: the trusted run named only ``pyqt/clubhead`` and eight
    further drifting tabs were never evaluated).
    """

    if _COMMIT.fullmatch(expected_source_commit) is None:
        raise VisualBaselineComparisonError("expected candidate commit is invalid")
    manifest = load_visual_baseline_manifest()
    results: list[VisualBaselineComparison] = []
    offenders: list[str] = []
    for surface in ("react", "pyqt"):
        references = manifest.for_surface(surface)
        candidates = _candidate_manifest(
            candidate_root, surface, references, expected_source_commit
        )
        for reference in references:
            reference_bytes = _reference_bytes(surface, reference)
            candidate_bytes = candidates[reference.tab_id]
            try:
                results.append(_compare(reference, reference_bytes, candidate_bytes))
            except VisualBaselineComparisonError as exc:
                offenders.append(str(exc))
    if offenders:
        raise VisualBaselineComparisonError("\n".join(offenders))
    return tuple(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--candidate-commit", required=True)
    args = parser.parse_args(argv)
    results = compare_visual_baselines(args.candidate_root, args.candidate_commit)
    sys.stdout.write(
        json.dumps([asdict(result) for result in results], indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VisualBaselineComparison",
    "VisualBaselineComparisonError",
    "compare_visual_baselines",
    "main",
]
