"""Bounded resource and resumability contracts for replay experiments."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass

from ._validation import require_finite, require_text
from .experiment_replay import ExperimentReplayBundle

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPERIMENT_CACHE_KEY_VERSION = "experiment-cache-key/1.0.0"


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_digest(value: str, field: str) -> str:
    digest = require_text(value, field)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return str(digest)


@dataclass(frozen=True, slots=True)
class ExperimentResourceBudget:
    """Hard per-job ceilings for workers, memory, disk, and preview output."""

    max_workers: int
    max_memory_bytes: int
    max_disk_bytes: int
    max_preview_artifacts: int
    max_preview_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "max_workers",
            "max_memory_bytes",
            "max_disk_bytes",
            "max_preview_artifacts",
            "max_preview_bytes",
        ):
            _positive_integer(getattr(self, field), field)
        if self.max_preview_bytes > self.max_disk_bytes:
            raise ValueError("preview byte budget cannot exceed the disk budget")

    def validate_request(
        self,
        *,
        workers: int,
        memory_bytes: int,
        disk_bytes: int,
        available_disk_bytes: int,
    ) -> None:
        """Reject work that exceeds configured limits or current free disk."""
        requests = (
            (workers, self.max_workers, "workers"),
            (memory_bytes, self.max_memory_bytes, "memory"),
            (disk_bytes, self.max_disk_bytes, "disk"),
        )
        for amount, limit, name in requests:
            _positive_integer(amount, f"requested {name}")
            if amount > limit:
                raise ValueError(f"requested {name} exceeds configured budget")
        if (
            isinstance(available_disk_bytes, bool)
            or not isinstance(available_disk_bytes, int)
            or available_disk_bytes < disk_bytes
        ):
            raise ValueError("requested disk exceeds available disk")


def make_experiment_cache_key(bundle: ExperimentReplayBundle) -> str:
    """Hash model/provider and every input, state, and executed-policy digest."""
    if not isinstance(bundle, ExperimentReplayBundle):
        raise TypeError("bundle must be an ExperimentReplayBundle")
    payload = {
        "cache_key_version": EXPERIMENT_CACHE_KEY_VERSION,
        "schema": bundle.schema_version,
        "engine": bundle.model.engine_id,
        "model": bundle.model.model_id,
        "variant": bundle.model.variant_id,
        "model_version": bundle.model.model_version,
        "provider": bundle.model.provider_id,
        "provider_version": bundle.model.provider_version,
        "provider_sha256": bundle.model.provider_sha256,
        "integrity": {
            name: getattr(bundle.integrity, name)
            for name in (
                "model_identity_sha256",
                "state_schema_sha256",
                "capability_declarations_sha256",
                "input_channel_schema_sha256",
                "time_grid_sha256",
                "initial_state_sha256",
                "applied_input_sha256",
                "execution_policy_sha256",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ExperimentCancelledError(RuntimeError):
    """Raised when a cooperative experiment worker observes cancellation."""


class ExperimentCancellationToken:
    """Thread-safe cancellation signal checked at safe worker boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def request_cancel(self) -> None:
        """Request cooperative cancellation without interrupting native code."""
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise at the next worker boundary after cancellation is requested."""
        if self.cancelled:
            raise ExperimentCancelledError("experiment cancellation requested")


@dataclass(frozen=True, slots=True)
class ExperimentTiming:
    """Measured elapsed time for a cold run or cache hit without host paths."""

    cache_key_sha256: str
    elapsed_seconds: float
    cache_hit: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cache_key_sha256",
            _require_digest(self.cache_key_sha256, "cache_key_sha256"),
        )
        elapsed = require_finite(self.elapsed_seconds, "elapsed_seconds")
        if elapsed < 0.0:
            raise ValueError("elapsed_seconds cannot be negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a boolean")
