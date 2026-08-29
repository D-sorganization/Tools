from __future__ import annotations

import json

import pytest
from sidekick.lab.mocap import (
    CameraIdentity,
    ClockDomain,
    ClockKind,
    CoordinateFrame,
    MethodDescriptor,
    MocapSessionManifest,
    RecordingPolicy,
    SessionState,
    dumps_canonical,
    load_session_manifest,
)


def _manifest() -> MocapSessionManifest:
    return MocapSessionManifest(
        session_id="session-2026-08-25-001",
        created_at_utc="2026-08-25T20:00:00Z",
        state=SessionState.FINALIZED,
        world_frame=CoordinateFrame.affinedrift_world_v1(),
        cameras=(
            CameraIdentity(
                provider_id="synthetic",
                device_id="camera-01",
                transport="memory",
                vendor="D-sorganization",
                model="reference-camera",
                serial_number="SYN-001",
            ),
        ),
        clocks=(
            ClockDomain(
                clock_id="synthetic-clock",
                kind=ClockKind.DEVICE_HARDWARE,
                tick_period_seconds=1e-9,
                monotonic=True,
            ),
        ),
        methods=(
            MethodDescriptor(
                method_id="synthetic-capture",
                version="1.0.0",
                implementation="sidekick.lab.mocap",
                license_spdx="MIT",
            ),
        ),
        recording_policy=RecordingPolicy(
            consent_recorded=True,
            raw_video_retained=False,
            retention_days=0,
            no_store=True,
        ),
        calibration_ids=("calibration-reference-v1",),
        warnings=("synthetic fixture; not physical qualification",),
    )


def test_manifest_serialization_is_canonical_and_round_trips() -> None:
    manifest = _manifest()
    first = dumps_canonical(manifest)
    second = dumps_canonical(manifest)
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["schema_version"] == "mocap-session/1.0.0"
    assert load_session_manifest(first) == manifest


def test_manifest_loader_rejects_unknown_fields() -> None:
    payload = json.loads(dumps_canonical(_manifest()))
    payload["invented"] = "silently ignored"
    with pytest.raises(ValueError, match="unknown fields"):
        load_session_manifest(json.dumps(payload))


def test_manifest_loader_rejects_incompatible_schema_version() -> None:
    payload = json.loads(dumps_canonical(_manifest()))
    payload["schema_version"] = "mocap-session/2.0.0"
    with pytest.raises(ValueError, match="schema_version"):
        load_session_manifest(json.dumps(payload))


def test_no_store_policy_cannot_claim_raw_video_retention() -> None:
    with pytest.raises(ValueError, match="no_store"):
        RecordingPolicy(
            consent_recorded=True,
            raw_video_retained=True,
            retention_days=1,
            no_store=True,
        )


def test_finalized_session_requires_consent_and_method_provenance() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="consent"):
        MocapSessionManifest(
            session_id=manifest.session_id,
            created_at_utc=manifest.created_at_utc,
            state=SessionState.FINALIZED,
            world_frame=manifest.world_frame,
            cameras=manifest.cameras,
            clocks=manifest.clocks,
            methods=manifest.methods,
            recording_policy=RecordingPolicy(
                consent_recorded=False,
                raw_video_retained=False,
                retention_days=0,
                no_store=True,
            ),
            calibration_ids=manifest.calibration_ids,
            warnings=manifest.warnings,
        )
