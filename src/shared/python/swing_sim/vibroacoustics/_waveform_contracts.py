"""Private real-sample boundary shared by recordings and alignment."""

from __future__ import annotations

from numbers import Real

import numpy as np


def real_samples(value: object) -> np.ndarray:
    """Return finite float64 samples without discarding or coercing information.

    Require nonempty one-dimensional real numeric arrays. Python sequences may
    contain real numbers, but never booleans; object arrays are unsupported.
    The returned array may share storage: ownership belongs to its caller.
    """
    if not isinstance(value, np.ndarray):
        original = np.asarray(value, dtype=object)
        if any(
            not isinstance(item, Real) or isinstance(item, (bool, np.bool_))
            for item in original.flat
        ):
            raise ValueError("samples must contain real non-boolean numbers")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("samples must have a real numeric dtype")
    if array.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if array.size == 0:
        raise ValueError("samples must be nonempty")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(array, dtype=np.float64)
    except (OverflowError, FloatingPointError) as error:
        raise ValueError("samples must be finite float64 values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("samples must be finite")
    return result


def owned_real_samples(value: object) -> np.ndarray:
    """Own validated samples over immutable bytes, including the write flag."""
    samples = real_samples(value)
    return np.frombuffer(samples.tobytes(), dtype=np.float64)


__all__ = ()
