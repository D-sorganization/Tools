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
from .calibration import (
    CalibrationDegeneracyKind,
    CalibrationObservation,
    CalibrationPatternKind,
    CalibrationQuality,
    CalibrationTarget,
    DistortionCoefficients,
    DistortionModel,
    FisheyeIntrinsics,
    IntrinsicCalibrationResult,
    PinholeIntrinsics,
    ReprojectionResidual,
    check_coverage_and_degeneracy,
    evaluate_intrinsic_quality,
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
    "CalibrationDegeneracyKind",
    "CalibrationObservation",
    "CalibrationPatternKind",
    "CalibrationQuality",
    "CalibrationTarget",
    "CameraCapabilities",
    "CameraIdentity",
    "CaptureGroup",
    "ClockDomain",
    "ClockKind",
    "ClockSkewEstimate",
    "CoordinateFrame",
    "DistortionCoefficients",
    "DistortionModel",
    "DropPolicy",
    "FeatureSupport",
    "FisheyeIntrinsics",
    "FrameIndexEntry",
    "FramePacket",
    "FrameSource",
    "FrameStamp",
    "IntrinsicCalibrationResult",
    "Landmark3D",
    "MethodDescriptor",
    "MocapSessionManifest",
    "NumericRange",
    "PinholeIntrinsics",
    "PixelObservation",
    "PrerecordedFrameSource",
    "QueueFullError",
    "RecordingIntegrityReport",
    "RecordingPolicy",
    "RecordingReader",
    "RecordingWriter",
    "ReprojectionResidual",
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
    "check_coverage_and_degeneracy",
    "dumps_canonical",
    "evaluate_intrinsic_quality",
    "load_session_manifest",
]
