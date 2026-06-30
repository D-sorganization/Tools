"""Pure-numpy signal filters, transforms, resampling and a dispatcher.

Every kernel here maps a 1-D series ``y`` to another array, with no SciPy /
pandas dependency. The Pi backend ships only numpy, so frequency-domain work is
done with ``rfft``/``irfft`` and smoothing with hand-rolled kernels.

This module is deliberately decoupled from the Data Explorer enums/models: the
dispatcher :func:`apply_filter` takes the *string value* of a ``FilterType``
(e.g. ``"moving_average"``, ``"fft_lowpass"``), not the enum, so the numeric
layer has no import edge to the model layer.

Conventions
-----------
* Inputs are coerced with ``np.asarray(x, dtype=float)``.
* Empty input -> :class:`ValueError`.
* Wrong types -> :class:`TypeError`; out-of-range params -> :class:`ValueError`.
* Non-finite (NaN/inf) inputs are rejected where they would silently corrupt
  the maths; :func:`zscore_filter` deliberately *introduces* NaN internally then
  interpolates it away. Each function documents its own stance.
* Edge handling (reflect / shrink) is documented per function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

_F64: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "moving_average",
    "exponential_smoothing",
    "median_filter",
    "gaussian_filter",
    "savgol_filter",
    "hampel_filter",
    "zscore_filter",
    "fft_filter",
    "integrate",
    "differentiate",
    "resample_series",
    "apply_filter",
]

_ArrayLike: TypeAlias = "Sequence[float] | np.ndarray"


def _as_1d_finite(y: object, name: str = "y") -> _F64:
    """Coerce ``y`` to a 1-D finite float array (DbC helper).

    Raises
    ------
    TypeError
        If ``y`` is not array-like (string scalars are rejected too).
    ValueError
        If empty, not 1-D, or contains NaN/inf.
    """
    if isinstance(y, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of floats, not a string")
    try:
        arr: _F64 = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like of floats") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _check_int(value: object, name: str) -> int:
    """Validate that ``value`` is an int (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _check_float(value: object, name: str) -> float:
    """Validate that ``value`` is a real number (int or float, not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _reflect_pad(arr: _F64, pad: int) -> _F64:
    """Reflect-pad ``arr`` by ``pad`` samples on each side."""
    if pad <= 0:
        return arr
    return cast(_F64, np.pad(arr, pad, mode="reflect"))


def moving_average(y: _ArrayLike, window: int) -> _F64:
    """Centered moving average with edge-shrinking windows.

    At the borders the window shrinks to the available samples (no padding), so
    the output has the same length as ``y`` and endpoints are means of fewer
    points.

    Preconditions: ``window`` is an int >= 1; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    _check_int(window, "window")
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return arr.copy()

    n = arr.size
    half = window // 2
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    out: _F64 = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def exponential_smoothing(y: _ArrayLike, alpha: float) -> _F64:
    """First-order exponential smoothing, ``s0 = y0``.

    ``s[i] = alpha * y[i] + (1 - alpha) * s[i-1]``.

    Preconditions: ``0 < alpha <= 1``; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    a = _check_float(alpha, "alpha")
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
    return out


def median_filter(y: _ArrayLike, window: int) -> _F64:
    """Sliding median with an odd window and reflect-padded edges.

    Preconditions: ``window`` is an odd int >= 1; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    _check_int(window, "window")
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if window == 1:
        return arr.copy()

    half = window // 2
    padded = _reflect_pad(arr, half)
    n = arr.size
    out: _F64 = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.median(padded[i : i + window])
    return out


def gaussian_filter(y: _ArrayLike, sigma: float) -> _F64:
    """Gaussian smoothing; kernel truncated at 4 sigma, reflect-padded edges.

    The kernel is normalized so a constant signal is preserved.

    Preconditions: ``sigma > 0``; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    s = _check_float(sigma, "sigma")
    if s <= 0.0:
        raise ValueError("sigma must be > 0")

    radius = int(np.ceil(4.0 * s))
    offsets: _F64 = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / s) ** 2)
    kernel /= kernel.sum()

    padded = _reflect_pad(arr, radius)
    return cast(_F64, np.convolve(padded, kernel, mode="valid"))


def savgol_filter(y: _ArrayLike, window: int, polyorder: int) -> _F64:
    """Savitzky-Golay smoothing via least-squares (Vandermonde + pinv).

    Interior points use the centered window. Edge points (the first/last
    ``window // 2`` samples) fit the polynomial over the first/last full window
    and evaluate it at the point's true offset, so a polynomial of degree
    <= ``polyorder`` is recovered **exactly everywhere** (no edge distortion).

    Preconditions: ``window`` odd, ``window > polyorder >= 0``, at least
    ``window`` samples; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    _check_int(window, "window")
    _check_int(polyorder, "polyorder")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd int")
    if window <= polyorder:
        raise ValueError("window must be greater than polyorder")
    if arr.size < window:
        raise ValueError("y must have at least `window` samples")

    half = window // 2
    positions: _F64 = np.arange(-half, half + 1, dtype=np.float64)
    vander = np.vander(positions, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(vander)
    # Row 0 of pinv evaluates the centered fitted polynomial at offset 0.
    center_coeffs = pinv[0]

    n = arr.size
    out: _F64 = np.empty(n, dtype=np.float64)

    # Interior: centered window convolution.
    for i in range(half, n - half):
        out[i] = float(center_coeffs @ arr[i - half : i + half + 1])

    # Leading edge: fit the first full window, evaluate at each offset.
    first = arr[:window]
    first_fit = pinv @ first  # polynomial coefficients (increasing powers)
    for i in range(half):
        # offset of point i relative to the first window's center (index half).
        out[i] = float(first_fit @ _offset_powers(i - half, polyorder))
    # Trailing edge: fit the last full window, evaluate at each offset.
    last = arr[-window:]
    last_fit = pinv @ last
    last_center = n - 1 - half  # index of the last window's center
    for i in range(n - half, n):
        out[i] = float(last_fit @ _offset_powers(i - last_center, polyorder))
    return out


def _offset_powers(offset: int, polyorder: int) -> _F64:
    """Return ``[offset**0, offset**1, ..., offset**polyorder]`` as floats."""
    return np.array(
        [float(offset) ** k for k in range(polyorder + 1)], dtype=np.float64
    )


def hampel_filter(y: _ArrayLike, window: int, n_sigma: float) -> _F64:
    """Hampel outlier replacement using a rolling median and scaled MAD.

    A point deviating from its window median by more than
    ``n_sigma * 1.4826 * MAD`` is replaced by that median. Reflect-padded edges.
    When ``MAD == 0`` the threshold is zero, so any point differing from the
    window median is treated as an outlier (a point equal to the median is
    never modified).

    Preconditions: ``window`` odd >= 1; ``n_sigma > 0``; ``y`` finite.
    """
    arr = _as_1d_finite(y)
    _check_int(window, "window")
    ns = _check_float(n_sigma, "n_sigma")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd int")
    if ns <= 0.0:
        raise ValueError("n_sigma must be > 0")

    half = window // 2
    padded = _reflect_pad(arr, half)
    n = arr.size
    out = arr.copy()
    scale = 1.4826
    for i in range(n):
        seg = padded[i : i + window]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        threshold = ns * scale * mad
        if abs(arr[i] - med) > threshold:
            out[i] = med
    return out


def zscore_filter(y: _ArrayLike, threshold: float) -> _F64:
    """Replace ``|z| > threshold`` samples with linear interpolation.

    z-scores use the sample mean/std (population, ddof=0). Flagged points become
    NaN and are linearly interpolated from the surviving neighbours; endpoints
    fall back to nearest-value (numpy ``interp`` edge behaviour).

    Preconditions: ``threshold > 0``; ``y`` finite, non-empty.
    """
    arr = _as_1d_finite(y)
    thr = _check_float(threshold, "threshold")
    if thr <= 0.0:
        raise ValueError("threshold must be > 0")

    std = arr.std()
    out = arr.copy()
    if std == 0.0:
        return out
    z = (arr - arr.mean()) / std
    mask = np.abs(z) > thr
    if not mask.any():
        return out
    good = ~mask
    if not good.any():
        return out
    idx: _F64 = np.arange(arr.size, dtype=np.float64)
    out[mask] = np.interp(idx[mask], idx[good], arr[good])
    return out


def fft_filter(
    y: _ArrayLike,
    sample_rate_hz: float,
    low: float | None,
    high: float | None,
) -> _F64:
    """Zero-phase brick-wall frequency filter via ``rfft``/``irfft``.

    * ``low=None, high=f``  -> lowpass, keep ``freq <= f``.
    * ``low=f, high=None``  -> highpass, keep ``freq >= f``.
    * ``low=a, high=b``     -> bandpass, keep ``a <= freq <= b``.

    Preconditions: ``sample_rate_hz > 0``; at least one of ``low``/``high`` set;
    each given cutoff finite and >= 0; if both given, ``low < high``.
    """
    arr = _as_1d_finite(y)
    sr = _check_float(sample_rate_hz, "sample_rate_hz")
    if sr <= 0.0:
        raise ValueError("sample_rate_hz must be > 0")
    if low is None and high is None:
        raise ValueError("at least one of low/high must be provided")
    lo = None if low is None else _check_float(low, "low")
    hi = None if high is None else _check_float(high, "high")
    if lo is not None and lo < 0.0:
        raise ValueError("low must be >= 0")
    if hi is not None and hi < 0.0:
        raise ValueError("high must be >= 0")
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("low must be < high for a bandpass")

    n = arr.size
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(arr)
    keep = np.ones(freqs.shape, dtype=bool)
    if lo is not None:
        keep &= freqs >= lo
    if hi is not None:
        keep &= freqs <= hi
    spectrum = np.where(keep, spectrum, 0.0)
    return cast(_F64, np.fft.irfft(spectrum, n=n))


def integrate(
    y: _ArrayLike,
    x: _ArrayLike | None = None,
    *,
    initial: float = 0.0,
) -> _F64:
    """Cumulative trapezoidal integral, same length as ``y``.

    Output[0] = ``initial``; output[i] adds the trapezoid over ``[x[i-1],
    x[i]]``. ``x`` defaults to ``arange(len(y))`` (unit spacing).

    Preconditions: ``y`` finite, non-empty; if given, ``x`` finite, same length.
    """
    arr = _as_1d_finite(y)
    init = _check_float(initial, "initial")
    if x is None:
        xs: _F64 = np.arange(arr.size, dtype=np.float64)
    else:
        xs = _as_1d_finite(x, "x")
        if xs.size != arr.size:
            raise ValueError("x and y must have the same length")

    out: _F64 = np.empty_like(arr)
    out[0] = init
    if arr.size > 1:
        dx = np.diff(xs)
        trap = 0.5 * (arr[1:] + arr[:-1]) * dx
        out[1:] = init + np.cumsum(trap)
    return out


def differentiate(y: _ArrayLike, x: _ArrayLike | None = None) -> _F64:
    """Numerical derivative via ``np.gradient`` (default unit spacing).

    Preconditions: ``y`` finite, non-empty; if given, ``x`` finite, same length.
    """
    arr = _as_1d_finite(y)
    if arr.size == 1:
        return np.zeros(1, dtype=np.float64)
    if x is None:
        return cast(_F64, np.gradient(arr))
    xs = _as_1d_finite(x, "x")
    if xs.size != arr.size:
        raise ValueError("x and y must have the same length")
    return cast(_F64, np.gradient(arr, xs))


_AGG_FUNCS: dict[str, Callable[[_F64], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
    "first": lambda v: float(v[0]),
    "last": lambda v: float(v[-1]),
    "min": lambda v: float(np.min(v)),
    "max": lambda v: float(np.max(v)),
    "sum": lambda v: float(np.sum(v)),
}


def resample_series(
    t: _ArrayLike,
    y: _ArrayLike,
    interval_s: float,
    agg: str,
    interpolate: bool,
) -> tuple[_F64, _F64]:
    """Bin ``(t, y)`` onto a uniform ``interval_s`` grid and aggregate.

    ``t`` is epoch SECONDS (float), ascending. Bins start at ``t.min()`` and
    step by ``interval_s``; a sample falls in bin ``floor((t - t0)/interval)``.
    The returned x-values are bin **centers**.

    ``agg`` is one of ``{mean, median, first, last, min, max, sum}``.

    * ``interpolate=True``  -> empty bins are linearly interpolated (over a full
      regular grid spanning the data) from neighbouring filled bins.
    * ``interpolate=False`` -> empty bins are dropped.

    Preconditions: ``t``/``y`` finite, equal length, non-empty; ``t`` ascending;
    ``interval_s > 0``; ``agg`` recognised; ``interpolate`` a bool.
    """
    ts = _as_1d_finite(t, "t")
    ys = _as_1d_finite(y, "y")
    step = _check_float(interval_s, "interval_s")
    if not isinstance(agg, str):
        raise TypeError("agg must be a str")
    if not isinstance(interpolate, bool):
        raise TypeError("interpolate must be a bool")
    if ts.size != ys.size:
        raise ValueError("t and y must have the same length")
    if step <= 0.0:
        raise ValueError("interval_s must be > 0")
    if np.any(np.diff(ts) < 0):
        raise ValueError("t must be ascending")
    func = _AGG_FUNCS.get(agg)
    if func is None:
        raise ValueError(f"unknown agg method: {agg!r}")

    t0 = ts[0]
    span = ts[-1] - t0
    n_bins = int(np.floor(span / step)) + 1
    bin_idx = np.floor((ts - t0) / step).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    centers: _F64 = t0 + (np.arange(n_bins) + 0.5) * step
    filled: npt.NDArray[np.bool_] = np.zeros(n_bins, dtype=bool)
    values: _F64 = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        members = ys[bin_idx == b]
        if members.size:
            values[b] = func(members)
            filled[b] = True

    if interpolate:
        if filled.any() and not filled.all():
            grid: _F64 = np.arange(n_bins, dtype=np.float64)
            values = np.interp(grid, grid[filled], values[filled])
        return centers, values

    return centers[filled], values[filled]


def _require_param(params: Mapping[str, float], key: str) -> float:
    """Fetch a required numeric filter parameter or raise ValueError."""
    if key not in params:
        raise ValueError(f"missing required parameter: {key!r}")
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"parameter {key!r} must be a number")
    return float(value)


def _sample_rate_from_t(params: Mapping[str, float], t: _ArrayLike | None) -> float:
    """Resolve a sample rate from params['sample_rate_hz'] or median dt of t."""
    if "sample_rate_hz" in params:
        return _require_param(params, "sample_rate_hz")
    if t is None:
        raise ValueError("sample rate requires params['sample_rate_hz'] or t")
    ts = _as_1d_finite(t, "t")
    if ts.size < 2:
        raise ValueError("t must have >= 2 samples to derive a sample rate")
    dt = float(np.median(np.diff(ts)))
    if dt <= 0.0:
        raise ValueError("cannot derive a positive sample rate from t")
    return 1.0 / dt


def apply_filter(
    y: _ArrayLike,
    ftype: str,
    params: Mapping[str, float],
    t: _ArrayLike | None = None,
) -> _F64:
    """Dispatch on a ``FilterType`` string value and apply the kernel.

    Recognised ``ftype`` values (the ``FilterType`` enum's string values):
    ``moving_average``, ``exponential``, ``median``, ``gaussian``, ``savgol``,
    ``hampel``, ``zscore``, ``fft_lowpass``, ``fft_highpass``,
    ``fft_bandpass``, ``integrate``, ``differentiate``.

    ``integrate``/``differentiate`` use ``t`` (if provided) as the x-axis. The
    FFT filters derive their sample rate from ``params['sample_rate_hz']`` or
    the median spacing of ``t``.

    Preconditions: ``ftype`` is a str; ``params`` is a mapping. Unknown type or
    bad params -> :class:`ValueError`.
    """
    if not isinstance(ftype, str):
        raise TypeError("ftype must be a str")
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    if ftype == "moving_average":
        return moving_average(y, int(_require_param(params, "window")))
    if ftype == "exponential":
        return exponential_smoothing(y, _require_param(params, "alpha"))
    if ftype == "median":
        return median_filter(y, int(_require_param(params, "window")))
    if ftype == "gaussian":
        return gaussian_filter(y, _require_param(params, "sigma"))
    if ftype == "savgol":
        return savgol_filter(
            y,
            int(_require_param(params, "window")),
            int(_require_param(params, "polyorder")),
        )
    if ftype == "hampel":
        return hampel_filter(
            y,
            int(_require_param(params, "window")),
            _require_param(params, "n_sigma"),
        )
    if ftype == "zscore":
        return zscore_filter(y, _require_param(params, "threshold"))
    if ftype == "fft_lowpass":
        sr = _sample_rate_from_t(params, t)
        return fft_filter(y, sr, None, _require_param(params, "high"))
    if ftype == "fft_highpass":
        sr = _sample_rate_from_t(params, t)
        return fft_filter(y, sr, _require_param(params, "low"), None)
    if ftype == "fft_bandpass":
        sr = _sample_rate_from_t(params, t)
        return fft_filter(
            y,
            sr,
            _require_param(params, "low"),
            _require_param(params, "high"),
        )
    if ftype == "integrate":
        initial = float(params.get("initial", 0.0))
        return integrate(y, t, initial=initial)
    if ftype == "differentiate":
        return differentiate(y, t)

    raise ValueError(f"unknown filter type: {ftype!r}")
