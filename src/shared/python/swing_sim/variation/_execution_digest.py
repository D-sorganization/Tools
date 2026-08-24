"""Cross-runtime canonical numeric digests for variation documents."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import cast

from shared.python.contracts import require

from .spec import MAX_SAFE_INTEGER


def normalized_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _digest_value(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        require(abs(value) <= MAX_SAFE_INTEGER, "digest integer must be safe", value)
        return {"$f64": struct.pack(">d", float(value)).hex()}
    if isinstance(value, float):
        numeric = normalized_float(value)
        require(math.isfinite(numeric), "digest numbers must be finite", value)
        return {"$f64": struct.pack(">d", numeric).hex()}
    if isinstance(value, Mapping):
        require(
            all(isinstance(key, str) for key in value),
            "digest mapping keys must be strings",
        )
        item = cast(Mapping[str, object], value)
        return {key: _digest_value(item[key]) for key in sorted(item)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_digest_value(item) for item in value]
    raise TypeError(f"unsupported canonical digest value: {type(value).__name__}")


def canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        _digest_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["canonical_sha256", "normalized_float"]
