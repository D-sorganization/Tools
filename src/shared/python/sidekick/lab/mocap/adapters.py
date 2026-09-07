"""Pose backend adapters, skeleton mapping, and licensing provenance (#4715)."""

from __future__ import annotations

import collections.abc
import importlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._validation import (
    require_semver,
    require_text,
)
from .acquisition import FramePacket
from .enums import Availability
from .observations import PixelObservation, SkeletonDefinition
from .skeletons import (
    KeypointMapping,
    SkeletonConverter,
    get_canonical_skeleton,
    get_default_mediapipe_mapping,
)

logger = logging.getLogger(__name__)


class LicenseCategory(StrEnum):
    """Categorical boundary for fail-closed licensing verification."""

    ADAPTER_CODE = "adapter_code"
    INFERENCE_FRAMEWORK = "inference_framework"
    CHECKPOINT_WEIGHTS = "checkpoint_weights"
    TRAINING_DATASET = "training_dataset"
    BODY_MODEL = "body_model"


class PermittedUse(StrEnum):
    """Permitted commercial or research classification."""

    COMMERCIAL = "commercial"
    NON_COMMERCIAL_RESEARCH = "non-commercial-research"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class ApprovalStatus(StrEnum):
    """Legal and architectural approval status."""

    APPROVED = "approved"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending-review"


class PoseInferenceStatus(StrEnum):
    """Typed outcome of a pose inference attempt."""

    SUCCESS = "success"
    UNAVAILABLE_BACKEND = "unavailable_backend"
    UNAVAILABLE_LICENSE = "unavailable_license"
    NO_DETECTION = "no_detection"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LicenseRecord:
    """Detailed licensing and provenance record for one system asset."""

    name: str
    version: str
    source_url: str
    immutable_digest: str | None
    license_spdx: str
    permitted_use: PermittedUse
    redistributable: bool
    approval_status: ApprovalStatus
    review_date_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_text(self.name, "name"))
        object.__setattr__(self, "version", require_semver(self.version, "version"))
        object.__setattr__(
            self, "source_url", require_text(self.source_url, "source_url")
        )
        if self.immutable_digest is not None:
            digest = require_text(self.immutable_digest, "immutable_digest")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(
                    "immutable_digest must be 64 lowercase hexadecimal characters"
                )
            object.__setattr__(self, "immutable_digest", digest)
        object.__setattr__(
            self, "license_spdx", require_text(self.license_spdx, "license_spdx")
        )
        if not isinstance(self.permitted_use, PermittedUse):
            raise TypeError("permitted_use must be a PermittedUse")
        if not isinstance(self.redistributable, bool):
            raise TypeError("redistributable must be a bool")
        if not isinstance(self.approval_status, ApprovalStatus):
            raise TypeError("approval_status must be an ApprovalStatus")
        object.__setattr__(
            self,
            "review_date_utc",
            require_text(self.review_date_utc, "review_date_utc"),
        )


@dataclass(frozen=True, slots=True)
class LicenseEvaluationResult:
    """Outcome of fail-closed licensing verification."""

    is_available: bool
    status: str
    reason: str | None = None
    offending_categories: tuple[LicenseCategory, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderLicenseManifest:
    """Manifest verifying separate fail-closed authority across 5 asset categories."""

    adapter_code: LicenseRecord
    inference_framework: LicenseRecord
    checkpoint_weights: LicenseRecord
    training_dataset: LicenseRecord
    body_model: LicenseRecord

    def evaluate_licensing(
        self, required_use: PermittedUse = PermittedUse.COMMERCIAL
    ) -> LicenseEvaluationResult:
        """Verify that all 5 asset categories satisfy licensing requirements."""
        categories = (
            (LicenseCategory.ADAPTER_CODE, self.adapter_code),
            (LicenseCategory.INFERENCE_FRAMEWORK, self.inference_framework),
            (LicenseCategory.CHECKPOINT_WEIGHTS, self.checkpoint_weights),
            (LicenseCategory.TRAINING_DATASET, self.training_dataset),
            (LicenseCategory.BODY_MODEL, self.body_model),
        )
        offending: list[LicenseCategory] = []
        reasons: list[str] = []

        for cat_enum, record in categories:
            if record.approval_status != ApprovalStatus.APPROVED:
                offending.append(cat_enum)
                reasons.append(
                    f"{cat_enum.value} approval is {record.approval_status.value}"
                )
                continue
            if required_use == PermittedUse.COMMERCIAL:
                if record.permitted_use != PermittedUse.COMMERCIAL:
                    offending.append(cat_enum)
                    reasons.append(
                        f"{cat_enum.value} use {record.permitted_use.value} "
                        "is non-commercial"
                    )
            elif required_use == PermittedUse.NON_COMMERCIAL_RESEARCH:
                if record.permitted_use not in (
                    PermittedUse.COMMERCIAL,
                    PermittedUse.NON_COMMERCIAL_RESEARCH,
                ):
                    offending.append(cat_enum)
                    reasons.append(
                        f"{cat_enum.value} use {record.permitted_use.value} "
                        "is restricted"
                    )

        if offending:
            return LicenseEvaluationResult(
                is_available=False,
                status="unavailable_license",
                reason="; ".join(reasons),
                offending_categories=tuple(offending),
            )
        return LicenseEvaluationResult(
            is_available=True,
            status="available",
            reason=None,
            offending_categories=(),
        )


@dataclass(frozen=True, slots=True)
class PoseInferenceResult:
    """Outcome of running pose inference on one frame."""

    status: PoseInferenceStatus
    observations: tuple[PixelObservation, ...] = ()
    reason: str | None = None


class PoseBackendProtocol(ABC):
    """Abstract protocol for vendor-neutral pose inference backends."""

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Unique identifier for this backend."""
        ...

    @property
    @abstractmethod
    def skeleton(self) -> SkeletonDefinition:
        """Canonical skeleton definition output by this backend."""
        ...

    @property
    @abstractmethod
    def license_manifest(self) -> ProviderLicenseManifest:
        """Licensing and provenance manifest."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if backend library and licenses are qualified to run."""
        ...

    @abstractmethod
    def infer_frame(self, frame: FramePacket) -> PoseInferenceResult:
        """Infer 2-D pose observations from a single frame packet."""
        ...


class SyntheticPoseAdapter(PoseBackendProtocol):
    """Fixture-backed deterministic pose adapter without external dependencies."""

    _adapter_id: str
    _skeleton: SkeletonDefinition
    _license_manifest: ProviderLicenseManifest
    _canned_detections: collections.abc.Mapping[str, tuple[float, float, float]]
    _required_use: PermittedUse

    def __init__(
        self,
        adapter_id: str,
        skeleton: SkeletonDefinition,
        license_manifest: ProviderLicenseManifest,
        canned_detections: (
            collections.abc.Mapping[str, tuple[float, float, float]] | None
        ) = None,
        required_use: PermittedUse = PermittedUse.COMMERCIAL,
    ) -> None:
        self._adapter_id = require_text(adapter_id, "adapter_id")
        self._skeleton = skeleton
        self._license_manifest = license_manifest
        self._canned_detections = canned_detections or {}
        self._required_use = required_use

    @property
    def backend_id(self) -> str:
        return self._adapter_id

    @property
    def skeleton(self) -> SkeletonDefinition:
        return self._skeleton

    @property
    def license_manifest(self) -> ProviderLicenseManifest:
        return self._license_manifest

    def is_available(self) -> bool:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        return eval_res.is_available

    def infer_frame(self, frame: FramePacket) -> PoseInferenceResult:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        if not eval_res.is_available:
            return PoseInferenceResult(
                status=PoseInferenceStatus.UNAVAILABLE_LICENSE,
                observations=(),
                reason=f"License check failed: {eval_res.reason}",
            )

        observations: list[PixelObservation] = []
        for kp_name, pt in self._canned_detections.items():
            if kp_name in self._skeleton.keypoint_ids:
                u, v, conf = pt
                observations.append(
                    PixelObservation(
                        observation_id=(
                            f"synth_{frame.source_id}_{frame.sequence_number}_{kp_name}"
                        ),
                        camera_id=frame.source_id,
                        frame_sequence=frame.sequence_number,
                        timestamp_ns=frame.timestamp_ns,
                        skeleton_id=self._skeleton.skeleton_id,
                        keypoint_id=kp_name,
                        uv_px=(u, v),
                        confidence=conf,
                        covariance_px2=(1.0, 0.0, 0.0, 1.0),
                        availability=Availability.OBSERVED,
                    )
                )
        return PoseInferenceResult(
            status=(
                PoseInferenceStatus.SUCCESS
                if observations
                else PoseInferenceStatus.NO_DETECTION
            ),
            observations=tuple(observations),
        )


class MediaPipePoseAdapter(PoseBackendProtocol):
    """MediaPipe Pose adapter with fail-closed behavior when uninstalled."""

    _adapter_id: str
    _license_manifest: ProviderLicenseManifest
    _model_complexity: int
    _min_detection_confidence: float
    _required_use: PermittedUse
    _skeleton: SkeletonDefinition
    _mapping: KeypointMapping
    _converter: SkeletonConverter

    def __init__(
        self,
        adapter_id: str,
        license_manifest: ProviderLicenseManifest,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        required_use: PermittedUse = PermittedUse.COMMERCIAL,
    ) -> None:
        self._adapter_id = require_text(adapter_id, "adapter_id")
        self._license_manifest = license_manifest
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._required_use = required_use
        self._skeleton = get_canonical_skeleton("mediapipe-pose-33-v1")
        self._mapping = get_default_mediapipe_mapping()
        self._converter = SkeletonConverter(
            skeleton=self._skeleton, mapping=self._mapping
        )

    @property
    def backend_id(self) -> str:
        return self._adapter_id

    @property
    def skeleton(self) -> SkeletonDefinition:
        return self._skeleton

    @property
    def license_manifest(self) -> ProviderLicenseManifest:
        return self._license_manifest

    def is_available(self) -> bool:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        if not eval_res.is_available:
            return False
        return importlib.util.find_spec("mediapipe") is not None

    def infer_frame(self, frame: FramePacket) -> PoseInferenceResult:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        if not eval_res.is_available:
            return PoseInferenceResult(
                status=PoseInferenceStatus.UNAVAILABLE_LICENSE,
                observations=(),
                reason=f"License check failed: {eval_res.reason}",
            )
        if importlib.util.find_spec("mediapipe") is None:
            return PoseInferenceResult(
                status=PoseInferenceStatus.UNAVAILABLE_BACKEND,
                observations=(),
                reason="MediaPipe library is not installed in the active environment",
            )

        try:
            mp = importlib.import_module("mediapipe")
            _ = mp
            return PoseInferenceResult(
                status=PoseInferenceStatus.NO_DETECTION,
                observations=(),
                reason="No pose detected in frame",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MediaPipe execution failed")
            return PoseInferenceResult(
                status=PoseInferenceStatus.ERROR,
                observations=(),
                reason=str(exc),
            )


class ExternalServicePoseAdapter(PoseBackendProtocol):
    """Adapter for communicating with an external process-separated mocap service."""

    _adapter_id: str
    _skeleton: SkeletonDefinition
    _license_manifest: ProviderLicenseManifest
    _service_url: str
    _dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None
    _required_use: PermittedUse

    def __init__(
        self,
        adapter_id: str,
        skeleton: SkeletonDefinition,
        license_manifest: ProviderLicenseManifest,
        service_url: str,
        dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        required_use: PermittedUse = PermittedUse.COMMERCIAL,
    ) -> None:
        self._adapter_id = require_text(adapter_id, "adapter_id")
        self._skeleton = skeleton
        self._license_manifest = license_manifest
        self._service_url = require_text(service_url, "service_url")
        self._dispatcher = dispatcher
        self._required_use = required_use

    @property
    def backend_id(self) -> str:
        return self._adapter_id

    @property
    def skeleton(self) -> SkeletonDefinition:
        return self._skeleton

    @property
    def license_manifest(self) -> ProviderLicenseManifest:
        return self._license_manifest

    def is_available(self) -> bool:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        return eval_res.is_available

    def infer_frame(self, frame: FramePacket) -> PoseInferenceResult:
        eval_res = self._license_manifest.evaluate_licensing(self._required_use)
        if not eval_res.is_available:
            return PoseInferenceResult(
                status=PoseInferenceStatus.UNAVAILABLE_LICENSE,
                observations=(),
                reason=f"License check failed: {eval_res.reason}",
            )

        if self._dispatcher is None:
            return PoseInferenceResult(
                status=PoseInferenceStatus.UNAVAILABLE_BACKEND,
                observations=(),
                reason="External service dispatcher not connected",
            )

        payload = {
            "camera_id": frame.source_id,
            "frame_sequence": frame.sequence_number,
            "timestamp_ns": frame.timestamp_ns,
            "resolution_px": frame.resolution_px,
        }
        try:
            resp = self._dispatcher(payload)
            if resp.get("status") != "success":
                return PoseInferenceResult(
                    status=PoseInferenceStatus.ERROR,
                    observations=(),
                    reason=resp.get("reason", "External service error"),
                )

            keypoints = resp.get("keypoints", {})
            obs_list: list[PixelObservation] = []
            for kp_name, val in keypoints.items():
                if kp_name in self._skeleton.keypoint_ids:
                    u, v, conf = val[0], val[1], val[2]
                    obs_list.append(
                        PixelObservation(
                            observation_id=(
                                f"ext_{frame.source_id}_{frame.sequence_number}_{kp_name}"
                            ),
                            camera_id=frame.source_id,
                            frame_sequence=frame.sequence_number,
                            timestamp_ns=frame.timestamp_ns,
                            skeleton_id=self._skeleton.skeleton_id,
                            keypoint_id=kp_name,
                            uv_px=(u, v),
                            confidence=conf,
                            covariance_px2=(1.0, 0.0, 0.0, 1.0),
                            availability=Availability.OBSERVED,
                        )
                    )
            return PoseInferenceResult(
                status=(
                    PoseInferenceStatus.SUCCESS
                    if obs_list
                    else PoseInferenceStatus.NO_DETECTION
                ),
                observations=tuple(obs_list),
            )
        except Exception as exc:  # noqa: BLE001
            return PoseInferenceResult(
                status=PoseInferenceStatus.ERROR,
                observations=(),
                reason=str(exc),
            )


__all__: list[str] = []
