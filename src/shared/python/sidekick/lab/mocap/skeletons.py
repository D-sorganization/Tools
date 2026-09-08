"""Canonical skeletons, keypoint mapping, and observation conversion (#4715)."""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field

from ._validation import require_text
from .enums import Availability
from .observations import PixelObservation, SkeletonDefinition


@dataclass(frozen=True, slots=True)
class KeypointMapping:
    """Mapping from backend keypoint names or indices to canonical skeleton names."""

    skeleton_id: str
    backend_name_to_canonical: dict[str, str] = field(default_factory=dict)
    backend_index_to_canonical: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "skeleton_id", require_text(self.skeleton_id, "skeleton_id")
        )


@dataclass(frozen=True, slots=True)
class SkeletonConverter:
    """Converts backend detection outputs into canonical PixelObservations."""

    skeleton: SkeletonDefinition
    mapping: KeypointMapping

    def convert_detections(
        self,
        camera_id: str,
        frame_sequence: int,
        timestamp_ns: int,
        detections: collections.abc.Mapping[str | int, tuple[float, float, float]],
        observation_id_prefix: str = "obs",
    ) -> tuple[PixelObservation, ...]:
        """Convert detections map into canonical PixelObservations."""
        obs_list: list[PixelObservation] = []
        for raw_key, pt in detections.items():
            canonical_name: str | None = None
            if isinstance(raw_key, int):
                canonical_name = self.mapping.backend_index_to_canonical.get(raw_key)
            elif isinstance(raw_key, str):
                canonical_name = self.mapping.backend_name_to_canonical.get(raw_key)
            if not canonical_name:
                continue
            if canonical_name not in self.skeleton.keypoint_ids:
                continue

            u, v, conf = pt
            obs = PixelObservation(
                observation_id=(
                    f"{observation_id_prefix}_{camera_id}_{frame_sequence}_{canonical_name}"
                ),
                camera_id=camera_id,
                frame_sequence=frame_sequence,
                timestamp_ns=timestamp_ns,
                skeleton_id=self.skeleton.skeleton_id,
                keypoint_id=canonical_name,
                uv_px=(u, v),
                confidence=conf,
                covariance_px2=(1.0, 0.0, 0.0, 1.0),
                availability=Availability.OBSERVED,
            )
            obs_list.append(obs)
        return tuple(obs_list)


# Built-in canonical skeletons
_CANONICAL_SKELETONS: dict[str, SkeletonDefinition] = {
    "coco-17-v1": SkeletonDefinition(
        skeleton_id="coco-17-v1",
        version="1.0.0",
        keypoint_ids=(
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ),
    ),
    "mediapipe-pose-33-v1": SkeletonDefinition(
        skeleton_id="mediapipe-pose-33-v1",
        version="1.0.0",
        keypoint_ids=(
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ),
    ),
}

_MEDIAPIPE_INDEX_TO_CANONICAL: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


def get_canonical_skeleton(skeleton_id: str) -> SkeletonDefinition:
    """Return a built-in canonical skeleton definition or raise KeyError."""
    return _CANONICAL_SKELETONS[skeleton_id]


def get_default_mediapipe_mapping() -> KeypointMapping:
    """Return the default index mapping for MediaPipe Pose 33."""
    return KeypointMapping(
        skeleton_id="mediapipe-pose-33-v1",
        backend_index_to_canonical=dict(_MEDIAPIPE_INDEX_TO_CANONICAL),
    )


__all__: list[str] = []
