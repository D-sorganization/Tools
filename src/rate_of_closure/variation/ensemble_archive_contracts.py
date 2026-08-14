"""Public lifecycle values and strict JSON primitives for chunk archives."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shared.python.contracts import ContractViolationError, require

ARCHIVE_SCHEMA_ID = "rate-of-closure/ensemble-chunk-archive"
ARCHIVE_SCHEMA_VERSION = 1
ZERO_SHA256 = "0" * 64


def require_sha256(value: object, name: str) -> str:
    """Return an exact lowercase SHA-256 string."""
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} must be lowercase SHA-256",
        value,
    )
    return cast(str, value)


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite deterministic UTF-8 JSON without insignificant whitespace."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as error:
        require(False, "archive metadata must contain strict finite JSON", str(error))
        raise AssertionError from error


def strict_json_bytes(data: bytes, *, maximum_bytes: int) -> object:
    """Decode bounded UTF-8 JSON while rejecting duplicate and nonfinite values."""
    require(len(data) <= maximum_bytes, "archive descriptor byte limit exceeded")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "duplicate archive JSON field", key)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        require(False, "archive JSON numbers must be finite", value)

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ContractViolationError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as error:
        require(False, "archive descriptor must be valid JSON", str(error))
        raise AssertionError from error


@dataclass(frozen=True)
class EnsembleResumeCursor:
    """Verified provisional prefix returned to a resume-aware executor."""

    next_trial_index: int
    failure_count: int
    previous_chunk_sha256: str
    verified_chunk_count: int

    def __post_init__(self) -> None:
        for name in ("next_trial_index", "failure_count", "verified_chunk_count"):
            value = getattr(self, name)
            require(type(value) is int and value >= 0, f"{name} must be non-negative")
        require_sha256(self.previous_chunk_sha256, "previous_chunk_sha256")


@dataclass(frozen=True)
class CommittedEnsembleArchive:
    """Small completed-archive handle that never materializes result tensors."""

    path: Path
    scientific_root_sha256: str
    trial_count: int
    chunk_count: int
    elapsed_s: float

    def __post_init__(self) -> None:
        require(isinstance(self.path, Path), "path must be a Path")
        require_sha256(self.scientific_root_sha256, "scientific_root_sha256")
        require(
            type(self.trial_count) is int and self.trial_count >= 0, "invalid trials"
        )
        require(
            type(self.chunk_count) is int and self.chunk_count >= 0, "invalid chunks"
        )
        require(
            isinstance(self.elapsed_s, (int, float))
            and not isinstance(self.elapsed_s, bool)
            and math.isfinite(float(self.elapsed_s))
            and float(self.elapsed_s) >= 0.0,
            "elapsed_s must be finite and non-negative",
        )


__all__ = [
    "ARCHIVE_SCHEMA_ID",
    "ARCHIVE_SCHEMA_VERSION",
    "CommittedEnsembleArchive",
    "EnsembleResumeCursor",
    "ZERO_SHA256",
    "canonical_json_bytes",
    "require_sha256",
    "strict_json_bytes",
]
