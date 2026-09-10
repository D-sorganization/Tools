"""Versioned exact content identity, independent of legacy hash conventions."""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np

from shared.python.swing_sim.variation._execution_digest import canonical_sha256


def _identity_value(value: object) -> object:
    """Encode immutable record inputs and exact little-endian sample content."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _identity_value(getattr(value, f.name))
            for f in fields(value)
            if f.init
        }
    if isinstance(value, np.ndarray):
        samples = np.asarray(value, dtype="<f8")
        return {
            "encoding": "float64-le",
            "count": int(samples.size),
            "sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        }
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError("unsupported calibration identity component")


def calibrated_identity(record: object, values: np.ndarray) -> str:
    """Bind all declared inputs, uncertainty model and converted sample bytes.

    Metadata numbers use the existing exact binary64 digest (signed zero is
    normalized). Raw sample bytes retain their actual signed-zero representation.
    Digest equality is content identity, never certificate or source authority.
    """
    return str(
        canonical_sha256(
            {
                "format": "swing_sim.calibrated_waveform/1",
                "uncertainty_model": "independent-indications-shared-gain-offset/1",
                "inputs": _identity_value(record),
                "converted_samples": _identity_value(values),
            }
        )
    )


__all__ = ()
