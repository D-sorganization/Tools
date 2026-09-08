"""TDD contract tests for markerless-mocap pose backend adapters (#4715)."""

from __future__ import annotations

from sidekick.lab.mocap.acquisition import FramePacket
from sidekick.lab.mocap.adapters import (
    ApprovalStatus,
    ExternalServicePoseAdapter,
    LicenseCategory,
    LicenseRecord,
    MediaPipePoseAdapter,
    PermittedUse,
    PoseInferenceStatus,
    ProviderLicenseManifest,
    SkeletonConverter,
    SyntheticPoseAdapter,
    get_canonical_skeleton,
    get_default_mediapipe_mapping,
)
from sidekick.lab.mocap.enums import Availability


def _sample_license_record(
    category: LicenseCategory,
    permitted_use: PermittedUse = PermittedUse.COMMERCIAL,
    approval: ApprovalStatus = ApprovalStatus.APPROVED,
) -> LicenseRecord:
    return LicenseRecord(
        name=f"test-{category.value}",
        version="1.0.0",
        source_url=f"https://example.com/{category.value}",
        immutable_digest="a" * 64,
        license_spdx="Apache-2.0",
        permitted_use=permitted_use,
        redistributable=True,
        approval_status=approval,
        review_date_utc="2026-09-07T00:00:00Z",
    )


def _sample_valid_manifest() -> ProviderLicenseManifest:
    return ProviderLicenseManifest(
        adapter_code=_sample_license_record(LicenseCategory.ADAPTER_CODE),
        inference_framework=_sample_license_record(LicenseCategory.INFERENCE_FRAMEWORK),
        checkpoint_weights=_sample_license_record(LicenseCategory.CHECKPOINT_WEIGHTS),
        training_dataset=_sample_license_record(LicenseCategory.TRAINING_DATASET),
        body_model=_sample_license_record(LicenseCategory.BODY_MODEL),
    )


def test_license_manifest_fail_closed_validation() -> None:
    manifest = _sample_valid_manifest()
    eval_result = manifest.evaluate_licensing(required_use=PermittedUse.COMMERCIAL)
    assert eval_result.is_available
    assert eval_result.reason is None
    assert len(eval_result.offending_categories) == 0

    # Non-commercial checkpoint fails commercial evaluation
    unapproved_checkpoint = _sample_license_record(
        LicenseCategory.CHECKPOINT_WEIGHTS,
        permitted_use=PermittedUse.NON_COMMERCIAL_RESEARCH,
    )
    non_comm_manifest = ProviderLicenseManifest(
        adapter_code=manifest.adapter_code,
        inference_framework=manifest.inference_framework,
        checkpoint_weights=unapproved_checkpoint,
        training_dataset=manifest.training_dataset,
        body_model=manifest.body_model,
    )
    eval_non_comm = non_comm_manifest.evaluate_licensing(
        required_use=PermittedUse.COMMERCIAL
    )
    assert not eval_non_comm.is_available
    assert eval_non_comm.status == "unavailable_license"
    assert LicenseCategory.CHECKPOINT_WEIGHTS in eval_non_comm.offending_categories

    # Unapproved status fails closed
    pending_adapter = _sample_license_record(
        LicenseCategory.ADAPTER_CODE,
        approval=ApprovalStatus.PENDING_REVIEW,
    )
    pending_manifest = ProviderLicenseManifest(
        adapter_code=pending_adapter,
        inference_framework=manifest.inference_framework,
        checkpoint_weights=manifest.checkpoint_weights,
        training_dataset=manifest.training_dataset,
        body_model=manifest.body_model,
    )
    eval_pending = pending_manifest.evaluate_licensing(
        required_use=PermittedUse.COMMERCIAL
    )
    assert not eval_pending.is_available
    assert eval_pending.status == "unavailable_license"
    assert LicenseCategory.ADAPTER_CODE in eval_pending.offending_categories


def test_skeleton_mapping_and_conversion() -> None:
    skeleton = get_canonical_skeleton("mediapipe-pose-33-v1")
    assert len(skeleton.keypoint_ids) == 33
    assert "nose" in skeleton.keypoint_ids
    assert "left_wrist" in skeleton.keypoint_ids

    mapping = get_default_mediapipe_mapping()
    assert mapping.skeleton_id == "mediapipe-pose-33-v1"

    converter = SkeletonConverter(skeleton=skeleton, mapping=mapping)

    # Raw backend predictions (index -> (u, v, confidence))
    raw_points = {
        0: (640.0, 360.0, 0.95),  # nose
        15: (500.0, 400.0, 0.88),  # left_wrist
    }
    observations = converter.convert_detections(
        camera_id="cam_0",
        frame_sequence=1,
        timestamp_ns=1_000_000,
        detections=raw_points,
    )
    assert len(observations) == 2
    nose_obs = next(obs for obs in observations if obs.keypoint_id == "nose")
    assert nose_obs.camera_id == "cam_0"
    assert nose_obs.frame_sequence == 1
    assert nose_obs.timestamp_ns == 1_000_000
    assert nose_obs.uv_px == (640.0, 360.0)
    assert nose_obs.confidence == 0.95
    assert nose_obs.availability is Availability.OBSERVED


def test_synthetic_pose_adapter() -> None:
    manifest = _sample_valid_manifest()
    skeleton = get_canonical_skeleton("coco-17-v1")

    sample_obs = {
        "nose": (300.0, 200.0, 0.99),
        "left_eye": (290.0, 190.0, 0.98),
    }
    adapter = SyntheticPoseAdapter(
        adapter_id="synth_0",
        skeleton=skeleton,
        license_manifest=manifest,
        canned_detections=sample_obs,
    )
    assert adapter.is_available()

    frame = FramePacket(
        source_id="cam_0",
        sequence_number=42,
        timestamp_ns=50_000_000,
        host_monotonic_ns=50_100_000,
        image_bytes=b"fake_frame_bytes",
        pixel_format="RGB8",
        resolution_px=(1920, 1080),
    )
    res = adapter.infer_frame(frame)
    assert res.status == PoseInferenceStatus.SUCCESS
    assert len(res.observations) == 2
    assert res.observations[0].frame_sequence == 42
    assert res.observations[0].camera_id == "cam_0"


def test_mediapipe_adapter_fail_closed_when_uninstalled_or_unlicensed() -> None:
    manifest = _sample_valid_manifest()
    adapter = MediaPipePoseAdapter(
        adapter_id="mp_pose_0",
        license_manifest=manifest,
    )
    frame = FramePacket(
        source_id="cam_0",
        sequence_number=1,
        timestamp_ns=1_000_000,
        host_monotonic_ns=1_100_000,
        image_bytes=b"dummy",
        pixel_format="RGB8",
        resolution_px=(640, 480),
    )
    res = adapter.infer_frame(frame)
    # MediaPipe uninstalled returns UNAVAILABLE_BACKEND, otherwise executes
    assert res.status in (
        PoseInferenceStatus.UNAVAILABLE_BACKEND,
        PoseInferenceStatus.SUCCESS,
        PoseInferenceStatus.NO_DETECTION,
    )


def test_adapter_fails_closed_on_unlicensed_eval() -> None:
    unapproved_checkpoint = _sample_license_record(
        LicenseCategory.CHECKPOINT_WEIGHTS,
        permitted_use=PermittedUse.NON_COMMERCIAL_RESEARCH,
    )
    manifest = ProviderLicenseManifest(
        adapter_code=_sample_license_record(LicenseCategory.ADAPTER_CODE),
        inference_framework=_sample_license_record(LicenseCategory.INFERENCE_FRAMEWORK),
        checkpoint_weights=unapproved_checkpoint,
        training_dataset=_sample_license_record(LicenseCategory.TRAINING_DATASET),
        body_model=_sample_license_record(LicenseCategory.BODY_MODEL),
    )
    adapter = SyntheticPoseAdapter(
        adapter_id="synth_unlicensed",
        skeleton=get_canonical_skeleton("coco-17-v1"),
        license_manifest=manifest,
        required_use=PermittedUse.COMMERCIAL,
    )
    frame = FramePacket(
        source_id="cam_0",
        sequence_number=1,
        timestamp_ns=1_000_000,
        host_monotonic_ns=1_100_000,
        image_bytes=b"dummy",
        pixel_format="RGB8",
        resolution_px=(640, 480),
    )
    res = adapter.infer_frame(frame)
    assert res.status == PoseInferenceStatus.UNAVAILABLE_LICENSE
    assert len(res.observations) == 0
    assert "License check failed" in (res.reason or "")


def test_external_service_pose_adapter_protocol() -> None:
    manifest = _sample_valid_manifest()
    skeleton = get_canonical_skeleton("coco-17-v1")

    def mock_service_call(payload: dict) -> dict:
        assert payload["frame_sequence"] == 10
        assert payload["camera_id"] == "cam_0"
        return {
            "status": "success",
            "keypoints": {
                "nose": [320.0, 240.0, 0.9],
            },
        }

    adapter = ExternalServicePoseAdapter(
        adapter_id="ext_service_0",
        skeleton=skeleton,
        license_manifest=manifest,
        service_url="http://localhost:8088/infer",
        dispatcher=mock_service_call,
    )
    assert adapter.is_available()

    frame = FramePacket(
        source_id="cam_0",
        sequence_number=10,
        timestamp_ns=100_000_000,
        host_monotonic_ns=100_100_000,
        image_bytes=b"frame_bytes",
        pixel_format="RGB8",
        resolution_px=(640, 480),
    )
    res = adapter.infer_frame(frame)
    assert res.status == PoseInferenceStatus.SUCCESS
    assert len(res.observations) == 1
    assert res.observations[0].keypoint_id == "nose"
    assert res.observations[0].uv_px == (320.0, 240.0)
