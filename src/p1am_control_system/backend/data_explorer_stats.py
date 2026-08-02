"""Pure-numpy statistical kernels for the Data Explorer analysis suite.

This module hosts the numeric heart of the descriptive/spectral/correlation
analyses offered by the Data Explorer. It is intentionally a *shallow* module
(LOD): it imports nothing from the DB, PLC, FastAPI, or sibling analysis
packages — only ``numpy`` — so it is unit-testable in isolation and light
enough to run on a Raspberry Pi.

Design notes:
    - numpy ONLY. No scipy/sklearn/pandas. Welch PSD is implemented by
      segmenting + windowing + averaging periodograms; PCA via the numpy SVD;
      Spearman correlation via rank-then-Pearson; exponential/power trendlines
      via log/linear fits with the reported R^2 computed in the ORIGINAL space.
    - DbC: every public entry point validates its inputs (``TypeError`` for
      wrong types, ``ValueError`` for empty/NaN/out-of-range/length-mismatch)
      and documents its pre/postconditions.
    - The kernels take *plain* ``str`` for ``method``/``kind``/``window`` (they
      do not import the enum module) and return raw ``numpy`` arrays / plain
      dicts; the service layer adapts those to the response models.

Arrays are coerced with ``np.asarray(x, dtype=float)``. Non-finite inputs
(NaN/inf) are rejected wherever they would corrupt the math.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

__all__ = [
    "describe",
    "correlation_matrix",
    "cross_correlation",
    "spectrum",
    "fit_trendline",
    "pca",
    "histogram",
]

# --------------------------------------------------------------------------- #
# Shared private helpers
# --------------------------------------------------------------------------- #

_CORRELATION_METHODS = ("pearson", "spearman")
_SPECTRUM_METHODS = ("fft", "welch")
_WINDOW_KINDS = ("none", "hanning", "hamming", "blackman")
_TRENDLINE_KINDS = ("linear", "polynomial", "exponential", "power")


def _as_float_1d(value: object, name: str) -> np.ndarray:
    """Coerce ``value`` to a finite 1-D float ``np.ndarray`` (DbC checked).

    Raises:
        TypeError: If ``value`` is not array-like (e.g. a bare ``str``) or
            holds non-numeric elements.
        ValueError: If the result is not 1-D, is empty, or contains NaN/inf.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be array-like, got {type(value).__name__}")
    try:
        array: np.ndarray = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be coercible to a float array: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {array.ndim}-D")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf)")
    return array


def _check_positive_float(value: object, name: str) -> float:
    """Validate ``value`` is a finite ``> 0`` real number; return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def _check_int(value: object, name: str) -> int:
    """Validate ``value`` is an ``int`` (bool rejected); return it."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _check_str_choice(value: object, name: str, choices: Sequence[str]) -> str:
    """Validate ``value`` is one of ``choices`` (string); return it."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if value not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"{name} must be one of [{valid}], got {value!r}")
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to ``values`` (ties share the mean of their ranks).

    Equivalent to ``scipy.stats.rankdata(values, method='average')`` but pure
    numpy. Ranks are 1-based.
    """
    order = np.argsort(values, kind="mergesort")
    ranks: np.ndarray = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    # Average ties.
    sorted_vals = values[order]
    n = values.shape[0]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            mean_rank = (start + end + 1) / 2.0  # mean of 1-based ranks
            ranks[order[start:end]] = mean_rank
        start = end
    return ranks


def _window_values(kind: str, length: int) -> np.ndarray:
    """Return a length-``length`` taper window of the named ``kind``."""
    if kind == "none":
        return np.ones(length, dtype=float)
    if kind == "hanning":
        return cast("np.ndarray", np.hanning(length))
    if kind == "hamming":
        return cast("np.ndarray", np.hamming(length))
    # blackman
    return cast("np.ndarray", np.blackman(length))


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def describe(y: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Compute a descriptive-statistics summary of a 1-D series.

    Args:
        y: A non-empty 1-D array-like of finite floats.

    Returns:
        A dict with keys ``count``, ``mean``, ``std``, ``min``, ``max``,
        ``median``, ``p25``, ``p75``, ``rms``. ``std`` is the *sample* standard
        deviation (ddof=1) when ``n > 1``, else ``0.0``.

    Raises:
        TypeError: If ``y`` is not array-like / holds non-numerics.
        ValueError: If ``y`` is empty, not 1-D, or contains NaN/inf.

    Preconditions:
        - ``y`` is a non-empty 1-D sequence of finite real numbers.
    """
    array = _as_float_1d(y, "y")
    n = int(array.size)
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    return {
        "count": float(n),
        "mean": float(np.mean(array)),
        "std": std,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


# --------------------------------------------------------------------------- #
# correlation_matrix
# --------------------------------------------------------------------------- #


def correlation_matrix(
    columns: Mapping[str, np.ndarray], method: str
) -> tuple[list[str], np.ndarray]:
    """Compute an NxN correlation matrix over named equal-length columns.

    Args:
        columns: Mapping of column label -> 1-D array-like. At least two
            columns are required; all must be the same length (>= 2 samples).
        method: ``"pearson"`` (``np.corrcoef``) or ``"spearman"``
            (rank-transform then Pearson).

    Returns:
        ``(labels, matrix)`` where ``labels`` is the list of column names in
        insertion order and ``matrix`` is the symmetric NxN correlation matrix.

    Raises:
        TypeError: If ``columns`` is not a mapping of str->array-like, or
            ``method`` is not a str.
        ValueError: If fewer than two columns, unequal lengths, fewer than two
            samples, an empty/NaN column, or an unknown ``method``.

    Preconditions:
        - ``columns`` has >= 2 entries, each finite, 1-D, equal length >= 2.
    """
    if not isinstance(columns, Mapping):
        raise TypeError(f"columns must be a mapping, got {type(columns).__name__}")
    method = _check_str_choice(method, "method", _CORRELATION_METHODS)

    labels = list(columns.keys())
    if len(labels) < 2:
        raise ValueError("columns must contain at least two columns")
    for label in labels:
        if not isinstance(label, str):
            raise TypeError(f"column keys must be str, got {type(label).__name__}")

    arrays = [_as_float_1d(columns[label], f"columns[{label!r}]") for label in labels]
    length = arrays[0].size
    if length < 2:
        raise ValueError("each column must have at least two samples")
    for label, array in zip(labels, arrays, strict=True):
        if array.size != length:
            raise ValueError(
                f"all columns must be equal length; {label!r} has {array.size}, "
                f"expected {length}"
            )

    matrix = np.vstack(arrays)
    if method == "spearman":
        matrix = np.vstack([_rankdata(row) for row in matrix])

    corr = np.corrcoef(matrix)
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    return labels, corr


# --------------------------------------------------------------------------- #
# cross_correlation
# --------------------------------------------------------------------------- #


def cross_correlation(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalized cross-correlation of ``a`` vs ``b`` over a range of lags.

    The correlation is computed on the mean-subtracted, unit-energy normalized
    signals so the result lies (numerically) in ``[-1, 1]``. A positive lag
    ``k`` measures the similarity of ``a[n]`` with ``b[n + k]`` — i.e. how well
    ``b`` lines up with ``a`` once ``b`` is advanced by ``k`` samples. When
    ``b`` is ``a`` delayed by ``d`` samples (``b[n] == a[n - d]``), the peak
    occurs at ``best_lag == d``.

    Args:
        a: First 1-D finite series.
        b: Second 1-D finite series (same length as ``a``).
        max_lag: Maximum absolute lag (int, ``1 <= max_lag <= len(a) - 1``).

    Returns:
        ``(lags, values, best_lag)`` where ``lags`` is
        ``[-max_lag .. max_lag]``, ``values`` the normalized correlation at
        each lag, and ``best_lag`` the lag of the maximum value.

    Raises:
        TypeError: If inputs are not array-like / ``max_lag`` not an int.
        ValueError: If lengths differ, a series is empty/NaN/constant, or
            ``max_lag`` is out of ``[1, len(a) - 1]``.

    Preconditions:
        - ``a`` and ``b`` are equal-length finite 1-D series, each non-constant.
    """
    array_a = _as_float_1d(a, "a")
    array_b = _as_float_1d(b, "b")
    max_lag = _check_int(max_lag, "max_lag")

    if array_a.size != array_b.size:
        raise ValueError(
            f"a and b must be equal length; got {array_a.size} and {array_b.size}"
        )
    n = array_a.size
    if max_lag < 1 or max_lag > n - 1:
        raise ValueError(f"max_lag must be in [1, {n - 1}], got {max_lag}")

    da = array_a - array_a.mean()
    db = array_b - array_b.mean()
    norm = cast(float, np.sqrt(float(np.sum(da**2)) * float(np.sum(db**2))))
    if norm == 0.0:
        raise ValueError("cross_correlation requires non-constant inputs")

    lags: np.ndarray = np.arange(-max_lag, max_lag + 1, dtype=int)
    values: np.ndarray = np.empty(lags.size, dtype=float)
    for index, lag in enumerate(lags):
        if lag >= 0:
            # a[n] * b[n + lag]: overlap a[:n-lag], b[lag:]
            prod = da[: n - lag] * db[lag:]
        else:
            prod = da[-lag:] * db[: n + lag]
        values[index] = float(np.sum(prod) / norm)

    best_lag = int(lags[int(np.argmax(values))])
    return lags, values, best_lag


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #


def spectrum(
    y: Sequence[float] | np.ndarray,
    sample_rate_hz: float,
    method: str,
    window: str,
    segment_size: int | None,
    detrend: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the single-sided spectrum of a uniformly sampled series.

    Args:
        y: 1-D finite series, sampled at ``sample_rate_hz``.
        sample_rate_hz: Positive sampling rate in Hz.
        method: ``"fft"`` (single-sided magnitude via ``rfft``) or ``"welch"``
            (mean of 50%-overlap, windowed segment periodograms scaled to PSD).
        window: Taper window in ``{"none","hanning","hamming","blackman"}``.
        segment_size: Welch segment length; ``None`` -> ``min(len(y), 256)``.
            Ignored by the ``"fft"`` method.
        detrend: If True, subtract the mean before transforming.

    Returns:
        ``(freqs, values)``. For ``"fft"`` ``values`` is amplitude-scaled
        single-sided magnitude; for ``"welch"`` it is power spectral density.

    Raises:
        TypeError: For wrong-typed inputs.
        ValueError: For empty/NaN ``y``, non-positive ``sample_rate_hz``,
            unknown ``method``/``window``, or a non-positive ``segment_size``.

    Preconditions:
        - ``y`` is a finite 1-D series; ``sample_rate_hz > 0``.
    """
    array = _as_float_1d(y, "y")
    sample_rate_hz = _check_positive_float(sample_rate_hz, "sample_rate_hz")
    method = _check_str_choice(method, "method", _SPECTRUM_METHODS)
    window = _check_str_choice(window, "window", _WINDOW_KINDS)
    if not isinstance(detrend, bool):
        raise TypeError(f"detrend must be a bool, got {type(detrend).__name__}")

    if detrend:
        array = array - array.mean()

    if method == "fft":
        return _spectrum_fft(array, sample_rate_hz, window)
    return _spectrum_welch(array, sample_rate_hz, window, segment_size)


def _spectrum_fft(
    array: np.ndarray, sample_rate_hz: float, window: str
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum via ``rfft`` with a taper window."""
    n = array.size
    win = _window_values(window, n)
    win_sum = win.sum()
    if win_sum == 0.0:
        win_sum = 1.0
    spec = np.fft.rfft(array * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    # Amplitude-correct single-sided magnitude.
    mag = np.abs(spec) / win_sum * 2.0
    if mag.size > 0:
        mag[0] = np.abs(spec[0]) / win_sum
        if n % 2 == 0:
            mag[-1] = np.abs(spec[-1]) / win_sum
    return freqs, mag


def _spectrum_welch(
    array: np.ndarray,
    sample_rate_hz: float,
    window: str,
    segment_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD: averaged periodograms over 50%-overlap windowed segments."""
    n = array.size
    if segment_size is None:
        seg = min(n, 256)
    else:
        seg = _check_int(segment_size, "segment_size")
        if seg < 1:
            raise ValueError(f"segment_size must be >= 1, got {seg}")
        seg = min(seg, n)

    win = _window_values(window, seg)
    win_power: float = float(np.sum(win**2))
    if win_power == 0.0:
        win_power = 1.0
    scale = 1.0 / (sample_rate_hz * win_power)

    step = max(seg // 2, 1)
    starts = list(range(0, n - seg + 1, step))
    if not starts:
        starts = [0]

    freqs = np.fft.rfftfreq(seg, d=1.0 / sample_rate_hz)
    accum = np.zeros(freqs.size, dtype=float)
    for start in starts:
        segment = array[start : start + seg]
        spec = np.fft.rfft(segment * win)
        psd = (np.abs(spec) ** 2) * scale
        # Single-sided: double interior bins (not DC / Nyquist).
        if psd.size > 2:
            psd[1:-1] *= 2.0
            if seg % 2 != 0:
                psd[-1] *= 2.0
        accum += psd
    accum /= len(starts)
    return freqs, accum


# --------------------------------------------------------------------------- #
# fit_trendline
# --------------------------------------------------------------------------- #


def fit_trendline(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    kind: str,
    degree: int,
    num_points: int = 200,
) -> dict:
    """Fit a trendline of the requested family and report fit quality.

    Args:
        x: 1-D finite independent variable.
        y: 1-D finite dependent variable (same length as ``x``).
        kind: ``"linear"``, ``"polynomial"``, ``"exponential"`` (``y=a*e^{bx}``,
            requires ``y > 0``), or ``"power"`` (``y=a*x^b``, requires
            ``x > 0`` and ``y > 0``).
        degree: Polynomial degree (used only for ``"polynomial"``; ``>= 1``).
        num_points: Number of points on the returned smooth fit curve (>= 2).

    Returns:
        ``{"coefficients", "equation", "r_squared", "x_fit", "y_fit"}``.
        ``coefficients`` is a list of floats; for exponential/power it is
        ``[a, b]``. ``r_squared`` is computed in the ORIGINAL (un-logged) space.
        ``x_fit`` is ``linspace(min(x), max(x), num_points)``.

    Raises:
        TypeError: For wrong-typed inputs.
        ValueError: For length mismatch, empty/NaN data, invalid ``degree``,
            unknown ``kind``, ``num_points < 2``, fewer points than the fit
            needs, or domain violations (non-positive values for log fits).

    Preconditions:
        - ``x`` and ``y`` are equal-length finite 1-D series.
    """
    array_x = _as_float_1d(x, "x")
    array_y = _as_float_1d(y, "y")
    kind = _check_str_choice(kind, "kind", _TRENDLINE_KINDS)
    degree = _check_int(degree, "degree")
    num_points = _check_int(num_points, "num_points")

    if array_x.size != array_y.size:
        raise ValueError(
            f"x and y must be equal length; got {array_x.size} and {array_y.size}"
        )
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")

    n = array_x.size
    x_fit = np.linspace(float(array_x.min()), float(array_x.max()), num_points)

    if kind == "linear":
        coeffs = np.polyfit(array_x, array_y, 1)
        y_pred = np.polyval(coeffs, array_x)
        y_fit = np.polyval(coeffs, x_fit)
        equation = _poly_equation(coeffs)
        coefficients = [float(c) for c in coeffs]
    elif kind == "polynomial":
        if degree < 1:
            raise ValueError(f"degree must be >= 1 for polynomial, got {degree}")
        if n < degree + 1:
            raise ValueError(
                f"polynomial of degree {degree} needs >= {degree + 1} points, got {n}"
            )
        coeffs = np.polyfit(array_x, array_y, degree)
        y_pred = np.polyval(coeffs, array_x)
        y_fit = np.polyval(coeffs, x_fit)
        equation = _poly_equation(coeffs)
        coefficients = [float(c) for c in coeffs]
    elif kind == "exponential":
        if np.any(array_y <= 0.0):
            raise ValueError("exponential fit requires all y > 0")
        # ln(y) = ln(a) + b x
        b_ln_a = np.polyfit(array_x, np.log(array_y), 1)
        b = float(b_ln_a[0])
        a = float(np.exp(b_ln_a[1]))
        y_pred = a * np.exp(b * array_x)
        y_fit = a * np.exp(b * x_fit)
        equation = f"y = {a:.6g} * exp({b:.6g} * x)"
        coefficients = [a, b]
    else:  # power
        if np.any(array_x <= 0.0):
            raise ValueError("power fit requires all x > 0")
        if np.any(array_y <= 0.0):
            raise ValueError("power fit requires all y > 0")
        # ln(y) = ln(a) + b ln(x)
        b_ln_a = np.polyfit(np.log(array_x), np.log(array_y), 1)
        b = float(b_ln_a[0])
        a = float(np.exp(b_ln_a[1]))
        y_pred = a * np.power(array_x, b)
        y_fit = a * np.power(x_fit, b)
        equation = f"y = {a:.6g} * x^{b:.6g}"
        coefficients = [a, b]

    r_squared = _r_squared(array_y, y_pred)
    return {
        "coefficients": coefficients,
        "equation": equation,
        "r_squared": r_squared,
        "x_fit": x_fit,
        "y_fit": y_fit,
    }


def _poly_equation(coeffs: np.ndarray) -> str:
    """Render polynomial coefficients (highest power first) as ``y = ...``."""
    degree = len(coeffs) - 1
    terms: list[str] = []
    for index, coeff in enumerate(coeffs):
        power = degree - index
        if power == 0:
            terms.append(f"{coeff:.6g}")
        elif power == 1:
            terms.append(f"{coeff:.6g}*x")
        else:
            terms.append(f"{coeff:.6g}*x^{power}")
    return "y = " + " + ".join(terms)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination in the ORIGINAL space (clamped at >= 0)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# pca
# --------------------------------------------------------------------------- #


def pca(
    columns: Mapping[str, np.ndarray], standardize: bool, n_components: int
) -> dict:
    """Principal-component analysis of named variables via the numpy SVD.

    Builds a samples-by-variables matrix from ``columns``, mean-centers it (and
    divides each variable by its sample std when ``standardize``), then takes
    the SVD. Explained-variance ratios come from the squared singular values.

    Args:
        columns: Mapping of variable label -> 1-D array-like. At least two
            variables, each finite, equal length, >= 2 samples.
        standardize: If True, scale each variable to unit sample variance
            (correlation-based PCA) after centering.
        n_components: Number of components to report; ``0`` -> all available
            (``min(n_samples, n_variables)``).

    Returns:
        ``{"explained_variance_ratio", "cumulative_variance",
        "singular_values", "component_labels", "loadings", "scores_pc1",
        "scores_pc2"}``. ``loadings`` is ``(n_components x n_variables)``.
        ``scores_pc2`` is an empty array when only one component is available.

    Raises:
        TypeError: For non-mapping ``columns`` / wrong-typed flags.
        ValueError: For < 2 variables, unequal lengths, < 2 samples,
            empty/NaN columns, a zero-variance variable under ``standardize``,
            or ``n_components`` out of range.

    Preconditions:
        - ``columns`` has >= 2 finite equal-length variables, >= 2 samples each.
    """
    if not isinstance(columns, Mapping):
        raise TypeError(f"columns must be a mapping, got {type(columns).__name__}")
    if not isinstance(standardize, bool):
        raise TypeError(f"standardize must be a bool, got {type(standardize).__name__}")
    n_components = _check_int(n_components, "n_components")
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")

    labels = list(columns.keys())
    if len(labels) < 2:
        raise ValueError("pca requires at least two variables")
    for label in labels:
        if not isinstance(label, str):
            raise TypeError(f"column keys must be str, got {type(label).__name__}")

    arrays = [_as_float_1d(columns[label], f"columns[{label!r}]") for label in labels]
    n_samples = arrays[0].size
    if n_samples < 2:
        raise ValueError("pca requires at least two samples")
    for label, array in zip(labels, arrays, strict=True):
        if array.size != n_samples:
            raise ValueError(
                f"all variables must be equal length; {label!r} has "
                f"{array.size}, expected {n_samples}"
            )

    matrix = np.column_stack(arrays)  # (n_samples, n_variables)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if standardize:
        stds = matrix.std(axis=0, ddof=1)
        if np.any(stds == 0.0):
            raise ValueError(
                "standardize=True requires every variable to have non-zero variance"
            )
        matrix = matrix / stds

    n_variables = len(labels)
    max_components = min(n_samples, n_variables)
    if n_components == 0:
        keep = max_components
    else:
        if n_components > max_components:
            raise ValueError(
                f"n_components must be <= {max_components}, got {n_components}"
            )
        keep = n_components

    # SVD of the centered matrix; columns of Vt are the principal axes.
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    variances = singular_values**2
    total = float(variances.sum())
    if total == 0.0:
        ratio = np.zeros_like(variances)
    else:
        ratio = variances / total

    ratio_keep = ratio[:keep]
    sv_keep = singular_values[:keep]
    loadings = vt[:keep, :]  # (keep, n_variables)
    scores = u * singular_values  # (n_samples, max_components)

    scores_pc1 = scores[:, 0] if keep >= 1 else np.empty(0, dtype=float)
    scores_pc2 = scores[:, 1] if keep >= 2 else np.empty(0, dtype=float)

    component_labels = [f"PC{i + 1}" for i in range(keep)]
    return {
        "explained_variance_ratio": np.asarray(ratio_keep, dtype=float),
        "cumulative_variance": np.cumsum(ratio_keep).astype(float),
        "singular_values": np.asarray(sv_keep, dtype=float),
        "component_labels": component_labels,
        "loadings": np.asarray(loadings, dtype=float),
        "scores_pc1": np.asarray(scores_pc1, dtype=float),
        "scores_pc2": np.asarray(scores_pc2, dtype=float),
    }


# --------------------------------------------------------------------------- #
# histogram
# --------------------------------------------------------------------------- #


def histogram(
    y: Sequence[float] | np.ndarray, bins: int, density: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a 1-D series into a histogram.

    Args:
        y: 1-D finite series.
        bins: Number of equal-width bins (int, ``>= 1``).
        density: If True, normalize so the integral over the range is 1
            (``np.histogram(density=True)``); else raw counts.

    Returns:
        ``(edges, counts)`` where ``edges`` has length ``bins + 1`` and
        ``counts`` has length ``bins``. When ``density`` is False the counts
        sum to ``len(y)``.

    Raises:
        TypeError: For wrong-typed inputs.
        ValueError: For empty/NaN ``y`` or ``bins < 1``.

    Preconditions:
        - ``y`` is a finite 1-D series; ``bins >= 1``.
    """
    array = _as_float_1d(y, "y")
    bins = _check_int(bins, "bins")
    if not isinstance(density, bool):
        raise TypeError(f"density must be a bool, got {type(density).__name__}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    counts, edges = np.histogram(array, bins=bins, density=density)
    return np.asarray(edges, dtype=float), np.asarray(counts, dtype=float)
