"""Synthetic contracts for bounded replay resources and preview artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from sidekick.lab.mocap import (
    CleanupEvidence,
    ExperimentResourceBudget,
    PreviewArtifact,
    PreviewManifest,
    PreviewReleaseAuthorization,
    RemoteResourceReceipt,
    build_preview_root,
    dumps_preview_manifest,
    evaluate_cleanup_eligibility,
    load_preview_manifest,
    require_preview_release,
    verify_preview_artifacts,
)


def _artifact(name: str, payload: bytes) -> PreviewArtifact:
    return PreviewArtifact(
        artifact_id=name,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        media_type="video/mp4",
    )


def test_preview_manifest_is_bounded_and_contains_no_local_paths() -> None:
    budget = ExperimentResourceBudget(1, 1024, 4096, 1, 8)
    artifact = _artifact("opaque:preview-0001", b"syntheti")
    manifest = PreviewManifest.create(
        "opaque:run-0001",
        (artifact,),
        budget,
        cache_key_sha256="c" * 64,
        provider_sha256="d" * 64,
    )

    assert manifest.artifacts == (artifact,)
    serialized = dumps_preview_manifest(manifest)
    assert load_preview_manifest(serialized) == manifest
    assert "Motion_Matching_Previews" not in serialized
    assert "C:\\" not in serialized
    schema_path = (
        Path(__file__).resolve().parents[6]
        / "schemas"
        / "mocap"
        / "preview-manifest-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(json.loads(serialized))
    payload = json.loads(serialized)
    payload["path"] = "C:/private/subject.mp4"
    assert list(Draft202012Validator(schema).iter_errors(payload))
    payload.pop("path")
    payload["artifacts"][0]["source_path"] = "C:/private/subject.mp4"
    with pytest.raises(ValueError, match="unknown"):
        load_preview_manifest(json.dumps(payload))
    assert "C:\\" not in repr(manifest)
    assert "/home/" not in repr(manifest)
    with pytest.raises(ValueError, match="preview artifact count"):
        PreviewManifest.create(
            "opaque:run-0001",
            (artifact, artifact),
            budget,
            cache_key_sha256="c" * 64,
            provider_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="completed preview artifacts"):
        PreviewManifest.create(
            "opaque:run-0001",
            (),
            budget,
            cache_key_sha256="c" * 64,
            provider_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="byte budget"):
        PreviewManifest.create(
            "opaque:run-0001",
            (artifact,),
            ExperimentResourceBudget(1, 1024, 4096, 1, 4),
            cache_key_sha256="c" * 64,
            provider_sha256="d" * 64,
        )


def test_preview_root_uses_parameterized_desktop_and_env_override(
    tmp_path: Path,
) -> None:
    desktop = tmp_path / "Redirected Desktop"
    expected = desktop / "Motion_Matching_Previews"

    assert build_preview_root({}, desktop_directory=desktop) == expected
    assert not expected.exists()
    assert (
        build_preview_root(
            {"MOTION_MATCHING_PREVIEW_ROOT": str(tmp_path / "custom")},
            desktop_directory=desktop,
        )
        == tmp_path / "custom"
    )


def test_cleanup_fails_closed_until_owned_merged_clean_and_receipts_retained() -> None:
    evidence = CleanupEvidence(
        task_owned=True,
        verified_merged=True,
        clean_worktree=True,
        receipts_retained=True,
        contains_user_data=False,
        contains_raw_capture=False,
        shared_directory=False,
    )
    assert evaluate_cleanup_eligibility(evidence).eligible
    assert not evaluate_cleanup_eligibility(
        replace(evidence, contains_raw_capture=True)
    ).eligible
    assert not evaluate_cleanup_eligibility(
        replace(evidence, verified_merged=False)
    ).eligible


def test_public_preview_helpers_validate_contract_types() -> None:
    with pytest.raises(TypeError, match="CleanupEvidence"):
        evaluate_cleanup_eligibility(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PreviewManifest"):
        require_preview_release(
            None,  # type: ignore[arg-type]
            source_is_private=True,
            authorization=None,
        )


def test_remote_receipt_requires_capacity_access_and_license_identity() -> None:
    receipt = RemoteResourceReceipt(
        host_id="host:test-runner",
        provider_id="test-provider",
        provider_sha256="a" * 64,
        license_identity_sha256="b" * 64,
        preview_root_identity="opaque:preview-root-01",
        access_confirmed=True,
        available_workers=2,
        available_memory_bytes=4096,
        available_disk_bytes=8192,
    )
    receipt.validate_for_dispatch(
        required_host_id="host:test-runner",
        required_provider_id="test-provider",
        required_provider_sha256="a" * 64,
        required_license_identity_sha256="b" * 64,
        required_preview_root_identity="opaque:preview-root-01",
        workers=1,
        memory_bytes=2048,
        disk_bytes=4096,
    )
    with pytest.raises(ValueError, match="provider"):
        receipt.validate_for_dispatch(
            required_host_id="host:test-runner",
            required_provider_id="different-provider",
            required_provider_sha256="a" * 64,
            required_license_identity_sha256="b" * 64,
            required_preview_root_identity="opaque:preview-root-01",
            workers=1,
            memory_bytes=2048,
            disk_bytes=4096,
        )


def test_remote_preflight_rejects_wrong_host_root_access_license_and_capacity() -> None:
    receipt = RemoteResourceReceipt(
        host_id="host:test-runner",
        provider_id="test-provider",
        provider_sha256="a" * 64,
        license_identity_sha256="b" * 64,
        preview_root_identity="opaque:preview-root-01",
        access_confirmed=True,
        available_workers=2,
        available_memory_bytes=4096,
        available_disk_bytes=8192,
    )
    request = {
        "required_host_id": "host:test-runner",
        "required_provider_id": "test-provider",
        "required_provider_sha256": "a" * 64,
        "required_license_identity_sha256": "b" * 64,
        "required_preview_root_identity": "opaque:preview-root-01",
        "workers": 1,
        "memory_bytes": 1024,
        "disk_bytes": 2048,
    }
    cases = (
        (replace(receipt, host_id="host:other"), "host"),
        (replace(receipt, preview_root_identity="opaque:other-root-01"), "root"),
        (replace(receipt, access_confirmed=False), "access"),
        (replace(receipt, license_identity_sha256="e" * 64), "license"),
        (replace(receipt, available_workers=0), "worker capacity"),
    )
    for changed, message in cases:
        with pytest.raises(ValueError, match=message):
            changed.validate_for_dispatch(**request)


def test_preview_file_verification_checks_digest_and_root_containment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "previews"
    root.mkdir()
    output = root / "sample.mp4"
    output.write_bytes(b"synthetic")
    artifact = _artifact("opaque:preview-0001", b"synthetic")
    manifest = PreviewManifest.create(
        "opaque:run-0001",
        (artifact,),
        ExperimentResourceBudget(1, 1024, 4096, 1, 32),
        cache_key_sha256="c" * 64,
        provider_sha256="d" * 64,
    )

    verify_preview_artifacts(manifest, root, {artifact.artifact_id: output})
    verify_preview_artifacts(manifest, root, {artifact.artifact_id: Path("sample.mp4")})
    with pytest.raises(ValueError, match="digest"):
        output.write_bytes(b"tampered!")
        verify_preview_artifacts(manifest, root, {artifact.artifact_id: output})
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="escapes"):
        verify_preview_artifacts(manifest, root, {artifact.artifact_id: outside})


def test_private_preview_cannot_be_released_without_manifest_bound_approval() -> None:
    manifest = PreviewManifest.create(
        "opaque:run-0001",
        (_artifact("opaque:preview-0001", b"synthetic"),),
        ExperimentResourceBudget(1, 1024, 4096, 1, 32),
        cache_key_sha256="c" * 64,
        provider_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="private preview"):
        require_preview_release(manifest, source_is_private=True, authorization=None)
    with pytest.raises(ValueError, match="does not match"):
        require_preview_release(
            manifest,
            source_is_private=True,
            authorization=PreviewReleaseAuthorization("opaque:approved-01", "e" * 64),
        )
    authorization = PreviewReleaseAuthorization("opaque:approved-01", manifest.sha256)
    require_preview_release(
        manifest, source_is_private=True, authorization=authorization
    )


def test_remote_result_binds_provider_cache_and_returned_files(tmp_path: Path) -> None:
    receipt = RemoteResourceReceipt(
        host_id="host:test-runner",
        provider_id="test-provider",
        provider_sha256="d" * 64,
        license_identity_sha256="b" * 64,
        preview_root_identity="opaque:preview-root-01",
        access_confirmed=True,
        available_workers=1,
        available_memory_bytes=1024,
        available_disk_bytes=1024,
    )
    manifest = PreviewManifest.create(
        "opaque:run-0001",
        (_artifact("opaque:preview-0001", b"synthetic"),),
        ExperimentResourceBudget(1, 1024, 4096, 1, 32),
        cache_key_sha256="c" * 64,
        provider_sha256="d" * 64,
    )
    root = tmp_path / "remote-preview"
    root.mkdir()
    output = root / "preview.mp4"
    output.write_bytes(b"synthetic")
    paths = {manifest.artifacts[0].artifact_id: output}

    receipt.validate_returned_artifacts(
        manifest,
        expected_cache_key_sha256="c" * 64,
        expected_provider_sha256="d" * 64,
        preview_root=root,
        artifact_paths=paths,
    )
    with pytest.raises(ValueError, match="cache identity"):
        receipt.validate_returned_artifacts(
            manifest,
            expected_cache_key_sha256="e" * 64,
            expected_provider_sha256="d" * 64,
            preview_root=root,
            artifact_paths=paths,
        )
    output.write_bytes(b"tampered!")
    with pytest.raises(ValueError, match="digest"):
        receipt.validate_returned_artifacts(
            manifest,
            expected_cache_key_sha256="c" * 64,
            expected_provider_sha256="d" * 64,
            preview_root=root,
            artifact_paths=paths,
        )
