"""Camera-agnostic markerless-mocap contracts and deterministic interchange."""

from .devices import CameraCapabilities, CameraIdentity, FeatureSupport, NumericRange
from .enums import Availability, ClockKind, SessionState, ShutterKind, SupportLevel
from .geometry import CoordinateFrame, RigidTransform
from .observations import Landmark3D, PixelObservation, SkeletonDefinition
from .serialization import dumps_canonical, load_session_manifest
from .session import (
    MOCAP_SESSION_SCHEMA_VERSION,
    MethodDescriptor,
    MocapSessionManifest,
    RecordingPolicy,
)
from .timebase import ClockDomain, FrameStamp

__all__ = [
    "MOCAP_SESSION_SCHEMA_VERSION",
    "Availability",
    "CameraCapabilities",
    "CameraIdentity",
    "ClockDomain",
    "ClockKind",
    "CoordinateFrame",
    "FeatureSupport",
    "FrameStamp",
    "Landmark3D",
    "MethodDescriptor",
    "MocapSessionManifest",
    "NumericRange",
    "PixelObservation",
    "RecordingPolicy",
    "RigidTransform",
    "SessionState",
    "ShutterKind",
    "SkeletonDefinition",
    "SupportLevel",
    "dumps_canonical",
    "load_session_manifest",
]
