"""Camera-agnostic markerless-mocap contracts and deterministic interchange."""

from .acquisition import (
    AcquisitionError,
    CaptureGroup,
    DropPolicy,
    FramePacket,
    FrameSource,
    PrerecordedFrameSource,
    QueueFullError,
    SourceState,
    SyntheticFrameSource,
)
from .devices import CameraCapabilities, CameraIdentity, FeatureSupport, NumericRange
from .enums import Availability, ClockKind, SessionState, ShutterKind, SupportLevel
from .geometry import CoordinateFrame, RigidTransform
from .observations import Landmark3D, PixelObservation, SkeletonDefinition
from .recording import (
    FrameIndexEntry,
    RecordingIntegrityReport,
    RecordingReader,
    RecordingWriter,
)
from .serialization import dumps_canonical, load_session_manifest
from .session import (
    MOCAP_SESSION_SCHEMA_VERSION,
    MethodDescriptor,
    MocapSessionManifest,
    RecordingPolicy,
)
from .sync import (
    ClockSkewEstimate,
    SyncAnomaly,
    SyncAnomalyType,
    SyncMonitor,
    SyncQuality,
)
from .timebase import ClockDomain, FrameStamp

__all__ = [
    "MOCAP_SESSION_SCHEMA_VERSION",
    "AcquisitionError",
    "Availability",
    "CameraCapabilities",
    "CameraIdentity",
    "CaptureGroup",
    "ClockDomain",
    "ClockKind",
    "ClockSkewEstimate",
    "CoordinateFrame",
    "DropPolicy",
    "FeatureSupport",
    "FrameIndexEntry",
    "FramePacket",
    "FrameSource",
    "FrameStamp",
    "Landmark3D",
    "MethodDescriptor",
    "MocapSessionManifest",
    "NumericRange",
    "PixelObservation",
    "PrerecordedFrameSource",
    "QueueFullError",
    "RecordingIntegrityReport",
    "RecordingPolicy",
    "RecordingReader",
    "RecordingWriter",
    "RigidTransform",
    "SessionState",
    "ShutterKind",
    "SkeletonDefinition",
    "SourceState",
    "SupportLevel",
    "SyncAnomaly",
    "SyncAnomalyType",
    "SyncMonitor",
    "SyncQuality",
    "SyntheticFrameSource",
    "dumps_canonical",
    "load_session_manifest",
]
