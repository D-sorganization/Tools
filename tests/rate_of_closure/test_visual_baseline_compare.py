"""Immutable visual-reference identity and pixel-drift enforcement."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from rate_of_closure.visual_baseline_compare import (
    VisualBaselineComparisonError,
    compare_visual_baselines,
)
from rate_of_closure.visual_baseline_manifest import (
    VisualBaselineEntry,
    VisualBaselineManifest,
    VisualBaselineTolerance,
    load_visual_baseline_manifest,
)

pytestmark = pytest.mark.unit


def test_packaged_manifest_binds_exact_reviewed_bytes() -> None:
    manifest = load_visual_baseline_manifest()

    assert manifest.source_artifact_commit == (
        "8b9935afece7296409dd77c473806f3e8414220e"  # pragma: allowlist secret
    )
    assert len(manifest.baselines) == 20
    package = files("rate_of_closure")
    tolerances = {
        "react": VisualBaselineTolerance(1, 4_000, 50_000),
        "pyqt": VisualBaselineTolerance(1, 200, 250),
    }
    calibrated_tolerances = {
        ("pyqt", "simulation"): VisualBaselineTolerance(1, 10_000, 10_000),
    }
    for entry in manifest.baselines:
        data = package.joinpath(
            "visual_baselines", "v1", entry.surface, entry.filename
        ).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry.sha256
        expected_tolerance = calibrated_tolerances.get(
            (entry.surface, entry.tab_id), tolerances[entry.surface]
        )
        assert entry.tolerance == expected_tolerance


def test_calibration_authority_bounds_host_drift_and_rejects_stale_controls() -> None:
    package = files("rate_of_closure")
    document = json.loads(
        package.joinpath("visual_baseline_calibration.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(document) == {
        "schema_id",
        "schema_version",
        "approval_policy",
        "reviewed_candidate_commit",
        "runs",
        "surface_envelopes",
    }
    assert document["schema_id"] == "rate-of-closure/visual-baseline-calibration"
    assert document["schema_version"] == 1
    assert document["approval_policy"] == "two-run-bounded-renderer-envelope"
    assert document["reviewed_candidate_commit"] == (
        "1214008e9dbf06b583ef44a4c821dc0567efdf8b"  # pragma: allowlist secret
    )
    assert [run["run_id"] for run in document["runs"]] == [
        32685823741,
        32686727162,
    ]
    for envelope in document["surface_envelopes"]:
        observed = envelope["observed_repeatability_microunits"]
        tolerance = envelope["selected_tolerance_microunits"]
        rejection = envelope["stale_control_microunits"]
        assert observed["max_mean_channel_delta"] < tolerance["max_mean_channel_delta"]
        assert (
            observed["max_changed_pixel_fraction"]
            < tolerance["max_changed_pixel_fraction"]
        )
        assert (
            rejection["min_material_mean_channel_delta"]
            > tolerance["max_mean_channel_delta"]
        )
        assert (
            rejection["min_material_changed_pixel_fraction"]
            > tolerance["max_changed_pixel_fraction"]
        )


def _png(path: Path, color: tuple[int, int, int], changed: int = 0) -> str:
    image = Image.new("RGB", (20, 20), color)
    for index in range(changed):
        image.putpixel((index % 20, index // 20), (255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(reference_root: Path) -> VisualBaselineManifest:
    tolerance = VisualBaselineTolerance(8, 10_000, 100_000)
    entries = []
    for surface in ("react", "pyqt"):
        filename = f"initial-{surface}.png"
        digest = _png(reference_root / surface / filename, (24, 24, 24))
        entries.append(
            VisualBaselineEntry(
                surface,
                surface,
                f"hosted-{surface}",
                filename,
                digest,
                tolerance,
            )
        )
    return VisualBaselineManifest(
        "rate-of-closure/visual-baselines",
        1,
        "proposed-off-default-branch-approved-after-protected-merge",
        "1" * 40,
        tuple(entries),
    )


def _candidate_document(
    surface: str, filename: str, digest: str, environment: str
) -> dict[str, object]:
    if surface == "react":
        return {
            "schemaId": "rate-of-closure/visual-baseline-candidates",
            "schemaVersion": 1,
            "artifactPolicy": (
                "candidate-diagnostic-not-approved-until-protected-merge"
            ),
            "sourceCommit": "2" * 40,
            "surface": surface,
            "environment": environment,
            "captures": [{"tabId": surface, "file": filename, "sha256": digest}],
        }
    return {
        "schema_id": "rate-of-closure/visual-baseline-candidates",
        "schema_version": 1,
        "artifact_policy": "candidate-diagnostic-not-approved-until-protected-merge",
        "source_commit": "2" * 40,
        "surface": surface,
        "environment": environment,
        "captures": [{"tab_id": surface, "file": filename, "sha256": digest}],
    }


def _candidates(root: Path, changed: int = 0) -> None:
    for surface in ("react", "pyqt"):
        filename = f"initial-{surface}.png"
        directory = root / surface
        digest = _png(directory / filename, (24, 24, 24), changed)
        document = _candidate_document(surface, filename, digest, f"hosted-{surface}")
        (directory / "manifest.json").write_text(json.dumps(document), encoding="utf-8")


def _compare(tmp_path: Path, changed: int = 0) -> tuple[object, ...]:
    package_root = tmp_path / "package"
    reference_root = package_root / "visual_baselines" / "v1"
    candidate_root = tmp_path / "candidates"
    manifest = _manifest(reference_root)
    _candidates(candidate_root, changed)
    with (
        patch(
            "rate_of_closure.visual_baseline_compare.load_visual_baseline_manifest",
            return_value=manifest,
        ),
        patch(
            "rate_of_closure.visual_baseline_compare.files",
            return_value=package_root,
        ),
    ):
        return compare_visual_baselines(candidate_root, "2" * 40)


def test_exact_and_bounded_small_drift_pass(tmp_path: Path) -> None:
    exact = _compare(tmp_path / "exact")
    assert len(exact) == 2
    assert all(result.mean_channel_delta_microunits == 0 for result in exact)

    changed = _compare(tmp_path / "changed", changed=4)
    assert all(result.mean_channel_delta_microunits > 0 for result in changed)
    assert all(result.changed_pixel_fraction_microunits == 10_000 for result in changed)


def test_material_drift_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(VisualBaselineComparisonError, match="exceeds limits"):
        _compare(tmp_path, changed=80)


@pytest.mark.parametrize(
    "mutation", ["environment", "digest", "coverage", "source_commit"]
)
def test_candidate_identity_tampering_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    package_root = tmp_path / "package"
    reference_root = package_root / "visual_baselines" / "v1"
    candidate_root = tmp_path / "candidates"
    manifest = _manifest(reference_root)
    _candidates(candidate_root)
    path = candidate_root / "react" / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "environment":
        document["environment"] = "foreign"
    elif mutation == "digest":
        document["captures"][0]["sha256"] = "f" * 64
    elif mutation == "source_commit":
        document["sourceCommit"] = "3" * 40
    else:
        document["captures"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    with (
        patch(
            "rate_of_closure.visual_baseline_compare.load_visual_baseline_manifest",
            return_value=manifest,
        ),
        patch(
            "rate_of_closure.visual_baseline_compare.files",
            return_value=package_root,
        ),
        pytest.raises(VisualBaselineComparisonError),
    ):
        compare_visual_baselines(candidate_root, "2" * 40)
