"""Enumerations used by camera-agnostic markerless-mocap contracts."""

from __future__ import annotations

from enum import StrEnum


class Availability(StrEnum):
    """Evidence class for an observation or derived quantity."""

    OBSERVED = "observed"
    DERIVED = "derived"
    MODEL_CONDITIONED = "model-conditioned"
    PROVISIONAL = "provisional"
    UNAVAILABLE = "unavailable"


class SupportLevel(StrEnum):
    """Negotiated provider support level for one capability."""

    SUPPORTED = "supported"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"


class ShutterKind(StrEnum):
    """Declared image-sensor readout behavior."""

    GLOBAL = "global"
    ROLLING = "rolling"
    GLOBAL_RESET = "global-reset"
    UNKNOWN = "unknown"


class ClockKind(StrEnum):
    """Origin and authority of a timestamp clock domain."""

    DEVICE_HARDWARE = "device-hardware"
    TRIGGER = "trigger"
    HOST_MONOTONIC = "host-monotonic"
    UTC_PRESENTATION = "utc-presentation"


class SessionState(StrEnum):
    """Lifecycle state of a markerless-mocap session manifest."""

    DRAFT = "draft"
    RECORDING = "recording"
    FINALIZED = "finalized"
    INCOMPLETE = "incomplete"


__all__: list[str] = []
