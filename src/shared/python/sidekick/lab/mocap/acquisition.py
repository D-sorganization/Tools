"""Camera acquisition protocol and reference drivers for markerless-mocap."""

from __future__ import annotations

import collections
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_text,
)
from .devices import CameraCapabilities, CameraIdentity
from .timebase import FrameStamp


class SourceState(StrEnum):
    """Lifecycle state of an acquisition frame source."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    CAPTURING = "capturing"
    STOPPED = "stopped"
    ERROR = "error"
    CLOSED = "closed"


class DropPolicy(StrEnum):
    """Policy when capture queue buffer exceeds maximum capacity."""

    DROP_OLDEST = "drop-oldest"
    DROP_NEWEST = "drop-newest"
    FAIL_CLOSED = "fail-closed"


class AcquisitionError(RuntimeError):
    """Base error for camera acquisition failures."""


class QueueFullError(AcquisitionError):
    """Raised when an acquisition queue overflows under FAIL_CLOSED policy."""


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One captured frame packet with timing, raw payload, and metadata."""

    source_id: str
    sequence_number: int
    timestamp_ns: int
    host_monotonic_ns: int
    image_bytes: bytes
    pixel_format: str
    resolution_px: tuple[int, int]
    exposure_start_ns: int | None = None
    exposure_end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "sequence_number",
            require_nonnegative_integer(self.sequence_number, "sequence_number"),
        )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(
            self,
            "host_monotonic_ns",
            require_nonnegative_integer(self.host_monotonic_ns, "host_monotonic_ns"),
        )
        if not isinstance(self.image_bytes, bytes):
            raise TypeError("image_bytes must be bytes")
        object.__setattr__(
            self, "pixel_format", require_text(self.pixel_format, "pixel_format")
        )
        if (
            len(self.resolution_px) != 2
            or self.resolution_px[0] <= 0
            or self.resolution_px[1] <= 0
        ):
            raise ValueError("resolution_px must contain two positive integers")
        if (self.exposure_start_ns is None) != (self.exposure_end_ns is None):
            raise ValueError("exposure start and end must be provided together")
        if self.exposure_start_ns is not None and self.exposure_end_ns is not None:
            start = require_nonnegative_integer(
                self.exposure_start_ns, "exposure_start_ns"
            )
            end = require_nonnegative_integer(self.exposure_end_ns, "exposure_end_ns")
            if start > end:
                raise ValueError("exposure start must not exceed exposure end")

    def to_frame_stamp(self, stream_id: str, clock_id: str) -> FrameStamp:
        """Derive a formal FrameStamp evidence record."""
        return FrameStamp(
            source_id=self.source_id,
            stream_id=stream_id,
            sequence_number=self.sequence_number,
            clock_id=clock_id,
            capture_timestamp_ns=self.timestamp_ns,
            host_monotonic_ns=self.host_monotonic_ns,
            timing_uncertainty_ns=0,
            exposure_start_ns=self.exposure_start_ns,
            exposure_end_ns=self.exposure_end_ns,
        )


class FrameSource(ABC):
    """Abstract contract for camera and virtual frame acquisition sources."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Stable local identifier for this source."""

    @property
    @abstractmethod
    def identity(self) -> CameraIdentity:
        """Physical or virtual provider identity."""

    @property
    @abstractmethod
    def capabilities(self) -> CameraCapabilities:
        """Advertised camera capabilities."""

    @property
    @abstractmethod
    def state(self) -> SourceState:
        """Current lifecycle state."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize provider hardware, resources, and connections."""

    @abstractmethod
    def start_capture(self) -> None:
        """Begin streaming frames."""

    @abstractmethod
    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        """Read the next available frame within the specified timeout."""

    @abstractmethod
    def stop_capture(self) -> None:
        """Halt streaming frames."""

    @abstractmethod
    def close(self) -> None:
        """Release all allocated hardware and memory resources."""


class SyntheticFrameSource(FrameSource):
    """Deterministic synthetic frame generator for testing and virtual capture."""

    def __init__(
        self,
        source_id: str,
        identity: CameraIdentity,
        capabilities: CameraCapabilities,
        resolution_px: tuple[int, int],
        frame_rate_hz: float,
        pixel_format: str,
    ) -> None:
        self._source_id: str = require_text(source_id, "source_id")
        self._identity = identity
        self._capabilities = capabilities
        self._resolution_px = resolution_px
        self._frame_rate_hz = require_finite(frame_rate_hz, "frame_rate_hz")
        self._pixel_format = require_text(pixel_format, "pixel_format")

        if not self._capabilities.supports_mode(
            resolution_px, frame_rate_hz, pixel_format
        ):
            raise ValueError(
                f"unsupported mode: {resolution_px}@{frame_rate_hz}Hz ({pixel_format})"
            )

        self._state = SourceState.UNINITIALIZED
        self._sequence_number = 0
        self._frame_interval_ns = int(1_000_000_000 / self._frame_rate_hz)
        self._start_time_ns: int | None = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def identity(self) -> CameraIdentity:
        return self._identity

    @property
    def capabilities(self) -> CameraCapabilities:
        return self._capabilities

    @property
    def state(self) -> SourceState:
        return self._state

    def initialize(self) -> None:
        if self._state is SourceState.CLOSED:
            raise AcquisitionError("cannot initialize a closed source")
        self._state = SourceState.INITIALIZED

    def start_capture(self) -> None:
        if self._state not in {SourceState.INITIALIZED, SourceState.STOPPED}:
            raise AcquisitionError(f"cannot start capture in state {self._state.value}")
        self._state = SourceState.CAPTURING
        if self._start_time_ns is None:
            self._start_time_ns = time.monotonic_ns()

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        if self._state is not SourceState.CAPTURING:
            raise AcquisitionError("source is not currently capturing")

        self._sequence_number += 1
        now_ns = time.monotonic_ns()
        timestamp_ns = (
            self._start_time_ns or now_ns
        ) + self._sequence_number * self._frame_interval_ns

        width, height = self._resolution_px
        payload_size = width * height * (3 if self._pixel_format == "RGB8" else 1)
        synthetic_payload = b"\x00" * payload_size

        return FramePacket(
            source_id=self._source_id,
            sequence_number=self._sequence_number,
            timestamp_ns=timestamp_ns,
            host_monotonic_ns=now_ns,
            image_bytes=synthetic_payload,
            pixel_format=self._pixel_format,
            resolution_px=self._resolution_px,
        )

    def stop_capture(self) -> None:
        if self._state is SourceState.CAPTURING:
            self._state = SourceState.STOPPED

    def close(self) -> None:
        self._state = SourceState.CLOSED


class PrerecordedFrameSource(FrameSource):
    """Replays an explicit sequence of recorded FramePackets."""

    def __init__(
        self,
        source_id: str,
        identity: CameraIdentity,
        capabilities: CameraCapabilities,
        frames: list[FramePacket],
        loop: bool = False,
    ) -> None:
        self._source_id: str = require_text(source_id, "source_id")
        self._identity = identity
        self._capabilities = capabilities
        self._frames = list(frames)
        self._loop = loop
        self._index = 0
        self._state = SourceState.UNINITIALIZED

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def identity(self) -> CameraIdentity:
        return self._identity

    @property
    def capabilities(self) -> CameraCapabilities:
        return self._capabilities

    @property
    def state(self) -> SourceState:
        return self._state

    def initialize(self) -> None:
        if self._state is SourceState.CLOSED:
            raise AcquisitionError("cannot initialize a closed source")
        self._state = SourceState.INITIALIZED

    def start_capture(self) -> None:
        if self._state not in {SourceState.INITIALIZED, SourceState.STOPPED}:
            raise AcquisitionError(f"cannot start capture in state {self._state.value}")
        self._state = SourceState.CAPTURING

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        if self._state is not SourceState.CAPTURING:
            raise AcquisitionError("source is not currently capturing")

        if not self._frames or self._index >= len(self._frames):
            if self._loop and self._frames:
                self._index = 0
            else:
                raise AcquisitionError("end of prerecorded stream")

        frame = self._frames[self._index]
        self._index += 1
        return frame

    def stop_capture(self) -> None:
        if self._state is SourceState.CAPTURING:
            self._state = SourceState.STOPPED

    def close(self) -> None:
        self._state = SourceState.CLOSED


class CaptureGroup:
    """Synchronized multi-camera capture manager with bounded queues."""

    def __init__(
        self,
        sources: tuple[FrameSource, ...],
        max_queue_size: int = 100,
        drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
    ) -> None:
        if not sources:
            raise ValueError("sources must be non-empty")
        ids = tuple(s.source_id for s in sources)
        if len(set(ids)) != len(ids):
            raise ValueError("source IDs must be unique")

        self._sources = {s.source_id: s for s in sources}
        self._max_queue_size = require_nonnegative_integer(
            max_queue_size, "max_queue_size"
        )
        if self._max_queue_size == 0:
            raise ValueError("max_queue_size must be positive")
        self._drop_policy = drop_policy
        self._queues: dict[str, collections.deque[FramePacket]] = {
            s_id: collections.deque() for s_id in self._sources
        }
        self._dropped_count: dict[str, int] = {s_id: 0 for s_id in self._sources}

    @property
    def sources(self) -> Mapping[str, FrameSource]:
        return self._sources

    def initialize(self) -> None:
        for source in self._sources.values():
            source.initialize()

    def start_capture(self) -> None:
        for source in self._sources.values():
            source.start_capture()

    def push_frame_direct(self, source_id: str, frame: FramePacket) -> None:
        """Push a frame directly into the internal buffer, applying drop policy."""
        queue = self._queues[source_id]
        if len(queue) >= self._max_queue_size:
            if self._drop_policy is DropPolicy.FAIL_CLOSED:
                raise QueueFullError(
                    f"Queue overflow on {source_id} under FAIL_CLOSED policy"
                )
            if self._drop_policy is DropPolicy.DROP_OLDEST:
                queue.popleft()
                self._dropped_count[source_id] += 1
            elif self._drop_policy is DropPolicy.DROP_NEWEST:
                self._dropped_count[source_id] += 1
                return
        queue.append(frame)

    def read_frames(self, timeout_seconds: float = 5.0) -> dict[str, FramePacket]:
        """Read a synchronized or paired frame across all managed sources."""
        result: dict[str, FramePacket] = {}
        for s_id, source in self._sources.items():
            queue = self._queues[s_id]
            if queue:
                result[s_id] = queue.popleft()
            else:
                result[s_id] = source.read_frame(timeout_seconds=timeout_seconds)
        return result

    def stop_capture(self) -> None:
        for source in self._sources.values():
            source.stop_capture()

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
        for q in self._queues.values():
            q.clear()


__all__ = [
    "AcquisitionError",
    "CaptureGroup",
    "DropPolicy",
    "FramePacket",
    "FrameSource",
    "PrerecordedFrameSource",
    "QueueFullError",
    "SourceState",
    "SyntheticFrameSource",
]
