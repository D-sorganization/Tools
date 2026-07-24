"""Orchestration service for the Data Explorer analysis suite.

This module sits between the FastAPI router (:mod:`data_explorer_router`) and the
pure-numpy kernels (:mod:`data_explorer_signals`, :mod:`data_explorer_stats`,
:mod:`data_explorer_expression`). It loads raw series (from the SQLite historian
or inline client data), runs the deterministic build pipeline
(align -> resample -> filters -> derived -> trim -> downsample), and adapts the
kernels' raw ``numpy`` outputs into the pydantic response models
(:mod:`data_explorer_models`).

Boundary conventions
---------------------
* The wire contract uses ``None`` to mark a gap / non-finite sample so the data
  round-trips through strict JSON. This service maps ``None`` -> ``numpy.nan`` on
  the way IN (inline columns) and ``nan``/``inf`` -> ``None`` on the way OUT
  (every :class:`Column` it emits).
* Analysis wrappers drop ``NaN`` appropriately (listwise for correlation/PCA,
  per-series otherwise) before calling the strict-finite kernels.
* The kernels raise ``TypeError``/``ValueError``; the router maps those to 400.

LOD: this is a shallow module. It imports only the sibling Data Explorer
modules, ``numpy``, and the historian ``TagLog`` model — no cross-package edges
to ``data_processing``/``sidekick``/etc.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from data_explorer_expression import evaluate_expression
from data_explorer_models import (
    Column,
    ColumnsRequest,
    ColumnStatistics,
    CorrelationRequest,
    CorrelationResponse,
    DatasetRequest,
    DatasetResponse,
    HistogramRequest,
    HistogramResponse,
    PcaRequest,
    PcaResponse,
    SignalInfo,
    SignalListResponse,
    SpectrumRequest,
    SpectrumResponse,
    StatisticsResponse,
    TrendlineRequest,
    TrendlineResponse,
)
from data_explorer_signals import apply_filter, resample_series
from data_explorer_stats import (
    correlation_matrix,
    describe,
    fit_trendline,
    histogram,
    pca,
    spectrum,
)

_F64: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "list_signals",
    "build_dataset",
    "compute_statistics",
    "compute_correlation",
    "compute_spectrum",
    "compute_trendline",
    "compute_pca",
    "compute_histogram",
    "dataset_to_csv_rows",
    "dataset_to_json",
    "validate_export",
]


try:  # UTC alias (Python 3.10 lacks datetime.UTC)
    from datetime import UTC as _UTC
except ImportError:  # pragma: no cover - depends on interpreter version
    _UTC = timezone.utc  # noqa: UP017


# --------------------------------------------------------------------------- #
# Boundary helpers (None <-> NaN, finite cleaning)
# --------------------------------------------------------------------------- #


def _values_to_array(values: Sequence[float | None]) -> _F64:
    """Map a wire column (``None`` = gap) to a 1-D float array with ``NaN`` gaps."""
    out: _F64 = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        out[i] = np.nan if v is None else float(v)
    return out


def _array_to_values(array: np.ndarray) -> list[float | None]:
    """Map a float array to a wire column, sending non-finite samples to ``None``."""
    flat = np.asarray(array, dtype=float).ravel()
    return [None if not np.isfinite(v) else float(v) for v in flat]


def _make_column(name: str, array: np.ndarray) -> Column:
    """Build a :class:`Column` from an array, cleaning non-finite -> ``None``."""
    return Column(name=name, values=_array_to_values(array))


def _drop_nan(array: _F64) -> _F64:
    """Return the finite samples of ``array`` (drops ``NaN``/``inf``)."""
    finite: _F64 = array[np.isfinite(array)]
    return finite


# datetime can only represent up to ~year 9999; ~8.64e15 ms is comfortably past
# any real timestamp while staying inside the representable range.
_MAX_EPOCH_MS = 8.64e15

# Memory ceiling for a historian dataset build: index samples x tag count. At
# ~8 bytes/cell, 20M cells is ~160 MB of float64 — a safe ceiling for the Pi.
_MAX_HISTORIAN_CELLS = 20_000_000


def _epoch_ms_to_iso(ms: float) -> str:
    """Render epoch milliseconds as an ISO-8601 UTC string.

    Defensive: a non-finite or out-of-range ``ms`` yields ``""`` rather than
    raising, so a streaming CSV export can never crash mid-body.
    """
    if not np.isfinite(ms) or abs(ms) > _MAX_EPOCH_MS:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=_UTC).isoformat()


def validate_export(index: Sequence[float] | None, columns: Sequence[Column]) -> None:
    """DbC precheck for an export so bad input is a 400, not a corrupt 200.

    The CSV export streams lazily, so an exception raised *during* iteration
    cannot be turned into an error status — the response has already begun. This
    eager check lets the router reject a non-finite / out-of-range index up front.

    Raises:
        ValueError: if ``index`` contains a ``None``, non-finite, or
            out-of-representable-range epoch-ms value.
    """
    if index is None:
        return
    for v in index:
        if v is None or not np.isfinite(v) or abs(float(v)) > _MAX_EPOCH_MS:
            raise ValueError(
                "export index contains a non-finite or out-of-range epoch-ms value"
            )


def _to_epoch_ms(value: datetime) -> float:
    """Convert a (possibly naive) datetime to epoch milliseconds."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return value.timestamp() * 1000.0


# --------------------------------------------------------------------------- #
# Historian queries (list_signals + raw load)
# --------------------------------------------------------------------------- #


def list_signals(session: object) -> SignalListResponse:
    """Summarise historian tag availability (distinct tag + count + time span).

    Args:
        session: A SQLModel ``Session`` bound to the historian database.

    Returns:
        A :class:`SignalListResponse` with one :class:`SignalInfo` per distinct
        ``TagLog.tag_name`` (count, ISO start/end timestamps), sorted by name.
    """
    from models import TagLog
    from sqlmodel import col, func, select

    statement = (
        select(
            TagLog.tag_name,
            func.count(col(TagLog.id)),
            func.min(col(TagLog.timestamp)),
            func.max(col(TagLog.timestamp)),
        )
        .group_by(col(TagLog.tag_name))
        .order_by(col(TagLog.tag_name).asc())
    )
    signals: list[SignalInfo] = []
    for name, count, start, end in session.exec(statement):  # type: ignore[attr-defined]
        signals.append(
            SignalInfo(
                name=name,
                count=int(count),
                start_time=_iso_of(start),
                end_time=_iso_of(end),
            )
        )
    return SignalListResponse(signals=signals)


def _iso_of(value: object) -> str | None:
    """Render a historian timestamp (datetime or str) as ISO, or ``None``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_UTC)
        return value.isoformat()
    return str(value)


def _load_historian(
    session: object, tags: Sequence[str], start: str, end: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load and align tag series from the historian onto a common index.

    Each tag's ``(timestamp, value)`` rows within ``[start, end]`` are pulled in
    ascending time order. All series are linearly interpolated (``np.interp``,
    edge-held out of range) onto the sorted union of every series' epoch-ms
    timestamps, producing a rectangular dataset.

    Returns:
        ``(index_ms, columns)`` where ``index_ms`` is the common epoch-ms index
        and ``columns`` maps each tag name to its aligned values.
    """
    from models import TagLog
    from sqlmodel import col, select

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    raw: dict[str, tuple[_F64, _F64]] = {}
    union: list[float] = []
    for tag in tags:
        statement = (
            select(TagLog.timestamp, TagLog.value)
            .where(col(TagLog.tag_name) == tag)
            .where(col(TagLog.timestamp) >= start_dt)
            .where(col(TagLog.timestamp) <= end_dt)
            .order_by(col(TagLog.timestamp).asc())
        )
        times: list[float] = []
        vals: list[float] = []
        for ts, value in session.exec(statement):  # type: ignore[attr-defined]
            times.append(_to_epoch_ms(ts))
            vals.append(float(value))
        raw[tag] = (np.asarray(times, dtype=float), np.asarray(vals, dtype=float))
        union.extend(times)

    index: _F64 = np.unique(np.asarray(union, dtype=np.float64))
    # Bound the materialized matrix so a wide range x many tags cannot exhaust
    # the Pi's RAM. Reject up front (router -> 400) so the operator narrows the
    # range or resamples, rather than OOM-killing the backend.
    cells = int(index.size) * max(1, len(list(tags)))
    if cells > _MAX_HISTORIAN_CELLS:
        raise ValueError(
            f"historian selection too large: {index.size} samples x "
            f"{len(list(tags))} tags = {cells} cells; narrow the time range "
            f"or select fewer tags"
        )
    columns: dict[str, _F64] = {}
    for tag in tags:
        tag_times, tag_vals = raw[tag]
        if tag_times.size == 0:
            columns[tag] = np.full(index.size, np.nan, dtype=np.float64)
        elif index.size == 0:
            columns[tag] = np.empty(0, dtype=np.float64)
        else:
            columns[tag] = np.interp(index, tag_times, tag_vals)
    return index, columns


# --------------------------------------------------------------------------- #
# build_dataset pipeline
# --------------------------------------------------------------------------- #


def _load_raw(
    session: object, req: DatasetRequest
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Resolve the single configured source to ``(index_ms, columns)``."""
    if req.inline is not None:
        index = np.asarray(req.inline.index, dtype=float)
        columns = {c.name: _values_to_array(c.values) for c in req.inline.columns}
        return index, columns
    assert req.historian is not None  # guaranteed by the model validator
    src = req.historian
    return _load_historian(session, src.tags, src.start_time, src.end_time)


def _resample(
    index_ms: np.ndarray, columns: dict[str, np.ndarray], req: DatasetRequest
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Re-bin every column onto a uniform grid; rebuild the index from centers."""
    spec = req.resample
    if spec is None or index_ms.size == 0:
        return index_ms, columns
    index_s = index_ms / 1000.0
    new_columns: dict[str, np.ndarray] = {}
    centers_s: np.ndarray | None = None
    for name, values in columns.items():
        centers, agg = resample_series(
            index_s, values, spec.interval_s, str(spec.agg), bool(spec.interpolate)
        )
        new_columns[name] = agg
        centers_s = centers
    new_index = (
        index_ms if centers_s is None else np.asarray(centers_s, dtype=float) * 1000.0
    )
    return new_index, new_columns


def _apply_filters(
    index_ms: np.ndarray, columns: dict[str, np.ndarray], req: DatasetRequest
) -> dict[str, np.ndarray]:
    """Apply each :class:`FilterSpec` in order, writing to ``output`` or target."""
    index_s = index_ms / 1000.0
    for spec in req.filters:
        if spec.target not in columns:
            raise ValueError(f"filter target column not found: {spec.target!r}")
        result = apply_filter(
            columns[spec.target], str(spec.type), dict(spec.params), index_s
        )
        columns[spec.output or spec.target] = np.asarray(result, dtype=float)
    return columns


def _apply_derived(
    columns: dict[str, np.ndarray], req: DatasetRequest
) -> dict[str, np.ndarray]:
    """Evaluate each derived expression over the current columns and append it."""
    for derived in req.derived:
        result = evaluate_expression(derived.expression, columns)
        columns[derived.name] = np.asarray(result, dtype=float)
    return columns


def _trim(
    index_ms: np.ndarray, columns: dict[str, np.ndarray], req: DatasetRequest
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mask the index to ``[start_ms, end_ms]`` (inclusive), keeping columns aligned."""
    spec = req.trim
    if spec is None or index_ms.size == 0:
        return index_ms, columns
    mask: npt.NDArray[np.bool_] = np.ones(index_ms.size, dtype=bool)
    if spec.start_ms is not None:
        mask &= index_ms >= float(spec.start_ms)
    if spec.end_ms is not None:
        mask &= index_ms <= float(spec.end_ms)
    return index_ms[mask], {name: vals[mask] for name, vals in columns.items()}


def _downsample(
    index_ms: np.ndarray, columns: dict[str, np.ndarray], max_points: int
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """Uniform-stride downsample to ``max_points`` (endpoints kept)."""
    n = index_ms.size
    if n <= max_points or n <= 2:
        return index_ms, columns, False
    keep = np.linspace(0, n - 1, max_points).round().astype(int)
    keep = np.unique(keep)
    if keep[-1] != n - 1:
        keep = np.append(keep, n - 1)
    return (
        index_ms[keep],
        {name: vals[keep] for name, vals in columns.items()},
        True,
    )


def _assert_columns_aligned(
    index_ms: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    """DbC: every column must match the index length before masking.

    Raises:
        ValueError: if any column's length differs from the index length (e.g. a
            reducer-derived scalar broadcast to the wrong length, or a ragged
            resample). The router maps this to HTTP 400.
    """
    n = int(index_ms.size)
    for name, values in columns.items():
        if int(np.asarray(values).size) != n:
            raise ValueError(
                f"column {name!r} length {np.asarray(values).size} != index "
                f"length {n}; pipeline produced a ragged dataset"
            )


def _sample_rate_hz(index_ms: np.ndarray) -> float | None:
    """Median sample rate (Hz) of the epoch-ms index, or ``None`` if underivable."""
    if index_ms.size < 2:
        return None
    diffs = np.diff(index_ms)
    median_dt_ms = float(np.median(diffs))
    if median_dt_ms <= 0.0 or not np.isfinite(median_dt_ms):
        return None
    return 1000.0 / median_dt_ms


def build_dataset(session: object, req: DatasetRequest) -> DatasetResponse:
    """Build a processed dataset from one source and run the build pipeline.

    Pipeline order is deterministic: load+align -> resample -> filters ->
    derived -> trim -> downsample-to ``req.max_points``.

    Args:
        session: SQLModel ``Session`` (used only for the historian source).
        req: The validated :class:`DatasetRequest` (exactly one source set).

    Returns:
        A :class:`DatasetResponse` with python-native index/columns, the row
        count, a ``truncated`` flag, and the derived ``sample_rate_hz``.
        Non-finite samples are emitted as ``None`` in each :class:`Column`.

    Raises:
        ValueError: For an unknown filter target or a kernel precondition breach.
        TypeError: For a kernel type-precondition breach.
    """
    index_ms, columns = _load_raw(session, req)
    index_ms, columns = _resample(index_ms, columns, req)
    columns = _apply_filters(index_ms, columns, req)
    columns = _apply_derived(columns, req)
    # Every column must stay aligned to the index before trim/downsample mask it;
    # a filter/derived/ragged-resample length mismatch is a client-input error
    # (clean 400) rather than an IndexError-500 deeper in the pipeline.
    _assert_columns_aligned(index_ms, columns)
    index_ms, columns = _trim(index_ms, columns, req)
    index_ms, columns, truncated = _downsample(index_ms, columns, req.max_points)

    return DatasetResponse(
        index=[float(v) for v in index_ms.tolist()],
        columns=[_make_column(name, vals) for name, vals in columns.items()],
        row_count=int(index_ms.size),
        truncated=truncated,
        sample_rate_hz=_sample_rate_hz(index_ms),
    )


# --------------------------------------------------------------------------- #
# Analysis wrappers
# --------------------------------------------------------------------------- #


def compute_statistics(req: ColumnsRequest) -> StatisticsResponse:
    """Per-column descriptive statistics (NaN dropped per series).

    A column with no finite samples yields zero-valued stats with ``count = 0``.
    """
    stats: list[ColumnStatistics] = []
    for column in req.columns:
        finite = _drop_nan(_values_to_array(column.values))
        if finite.size == 0:
            stats.append(
                ColumnStatistics(
                    name=column.name,
                    count=0,
                    mean=0.0,
                    std=0.0,
                    min=0.0,
                    max=0.0,
                    median=0.0,
                    p25=0.0,
                    p75=0.0,
                    rms=0.0,
                )
            )
            continue
        summary = describe(finite)
        stats.append(
            ColumnStatistics(
                name=column.name,
                count=int(summary["count"]),
                mean=summary["mean"],
                std=summary["std"],
                min=summary["min"],
                max=summary["max"],
                median=summary["median"],
                p25=summary["p25"],
                p75=summary["p75"],
                rms=summary["rms"],
            )
        )
    return StatisticsResponse(stats=stats)


def _complete_case(columns: list[Column]) -> dict[str, np.ndarray]:
    """Listwise-drop rows where any column is non-finite; return finite columns."""
    arrays = {c.name: _values_to_array(c.values) for c in columns}
    if not arrays:
        return {}
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have equal length")
    mask: npt.NDArray[np.bool_] = np.ones(next(iter(arrays.values())).size, dtype=bool)
    for arr in arrays.values():
        mask &= np.isfinite(arr)
    return {name: arr[mask] for name, arr in arrays.items()}


def compute_correlation(req: CorrelationRequest) -> CorrelationResponse:
    """Correlation matrix over the request columns (listwise complete-case)."""
    cleaned = _complete_case(list(req.columns))
    labels, matrix = correlation_matrix(cleaned, str(req.method))
    return CorrelationResponse(
        labels=labels,
        matrix=[[float(v) for v in row] for row in np.asarray(matrix).tolist()],
        method=req.method,
    )


def compute_spectrum(req: SpectrumRequest) -> SpectrumResponse:
    """Single-sided spectrum / Welch PSD of one series (NaN dropped)."""
    finite = _drop_nan(_values_to_array(req.values))
    freqs, power = spectrum(
        finite,
        req.sample_rate_hz,
        str(req.method),
        str(req.window),
        req.segment_size,
        bool(req.detrend),
    )
    return SpectrumResponse(
        freqs=[float(v) for v in np.asarray(freqs).tolist()],
        power=[float(v) for v in np.asarray(power).tolist()],
        method=req.method,
    )


def compute_trendline(req: TrendlineRequest) -> TrendlineResponse:
    """Fit a trendline of the requested family (pairwise NaN drop on x/y)."""
    x = _values_to_array(req.x)
    y = _values_to_array(req.y)
    mask = np.isfinite(x) & np.isfinite(y)
    result = fit_trendline(x[mask], y[mask], str(req.kind), req.degree, req.num_points)
    return TrendlineResponse(
        kind=req.kind,
        coefficients=[float(c) for c in result["coefficients"]],
        equation=str(result["equation"]),
        r_squared=float(result["r_squared"]),
        x_fit=[float(v) for v in np.asarray(result["x_fit"]).tolist()],
        y_fit=[float(v) for v in np.asarray(result["y_fit"]).tolist()],
    )


def compute_pca(req: PcaRequest) -> PcaResponse:
    """Principal-component analysis over the request columns (listwise drop)."""
    cleaned = _complete_case(list(req.columns))
    result = pca(cleaned, bool(req.standardize), req.n_components)
    return PcaResponse(
        explained_variance_ratio=[
            float(v) for v in np.asarray(result["explained_variance_ratio"]).tolist()
        ],
        cumulative_variance=[
            float(v) for v in np.asarray(result["cumulative_variance"]).tolist()
        ],
        singular_values=[
            float(v) for v in np.asarray(result["singular_values"]).tolist()
        ],
        component_labels=list(result["component_labels"]),
        loadings=[
            [float(v) for v in row] for row in np.asarray(result["loadings"]).tolist()
        ],
        scores_pc1=[float(v) for v in np.asarray(result["scores_pc1"]).tolist()],
        scores_pc2=[float(v) for v in np.asarray(result["scores_pc2"]).tolist()],
    )


def compute_histogram(req: HistogramRequest) -> HistogramResponse:
    """Histogram of one series (NaN dropped)."""
    finite = _drop_nan(_values_to_array(req.values))
    edges, counts = histogram(finite, req.bins, bool(req.density))
    return HistogramResponse(
        bin_edges=[float(v) for v in np.asarray(edges).tolist()],
        counts=[float(v) for v in np.asarray(counts).tolist()],
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _csv_cell(value: float | None) -> str:
    """Render one numeric cell; ``None``/non-finite -> empty string."""
    if value is None:
        return ""
    if not np.isfinite(value):
        return ""
    return repr(float(value))


def dataset_to_csv_rows(
    index: Sequence[float] | None, columns: Sequence[Column]
) -> Iterator[str]:
    """Stream CSV rows for a dataset (header + one row per sample).

    The first column is ``timestamp`` (ISO-8601 from epoch ms) when ``index`` is
    given; otherwise it is a 0-based integer row counter. Gaps (``None`` /
    non-finite) are emitted as empty cells.

    Args:
        index: Optional epoch-ms index aligned to every column.
        columns: The named columns to export (equal length, matching ``index``).

    Yields:
        CSV-formatted lines terminated by ``\\n``.
    """
    names = [c.name for c in columns]
    header = "timestamp," + ",".join(names) if names else "timestamp"
    yield header + "\n"

    n = len(index) if index is not None else (len(columns[0].values) if columns else 0)
    for i in range(n):
        if index is not None:
            stamp = _epoch_ms_to_iso(float(index[i]))
        else:
            stamp = str(i)
        cells = [_csv_cell(c.values[i]) for c in columns]
        yield stamp + ("," + ",".join(cells) if cells else "") + "\n"


def dataset_to_json(
    index: Sequence[float] | None, columns: Sequence[Column]
) -> dict[str, object]:
    """Serialise a dataset to a JSON-ready dict (``index`` + named columns)."""
    return {
        "index": list(index) if index is not None else None,
        "columns": [{"name": c.name, "values": list(c.values)} for c in columns],
    }
