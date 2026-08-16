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
modules, ``numpy``, the historian models, and (lazily, inside the historian
load) the historian read helpers in :mod:`data_capture` — no cross-package edges
to ``data_processing``/``sidekick``/etc. Reusing ``data_capture`` rather than
re-implementing bound parsing and streamed decimation keeps one definition of
the historian read contract (DRY).
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
    require_aligned_columns,
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


def _ensure_utc(value: datetime) -> datetime:
    """Normalize to aware UTC via the single fleet-wide rule in ``models``.

    Imported lazily so this module keeps importing cleanly without SQLModel /
    the hardware tag table present (the numeric kernels are unit-tested alone).
    """
    from models import ensure_utc

    aware: datetime = ensure_utc(value)
    return aware


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

# Per-tag sample budget when a caller omits ``HistorianSource.max_points``;
# matches that field's own default so the two cannot drift apart.
_DEFAULT_HISTORIAN_MAX_POINTS = 5_000


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
    cannot be turned into an error status — the response has already begun. Both
    known ways to blow up mid-body are therefore checked eagerly:

    * a non-finite / out-of-range epoch-ms index value, and
    * ragged columns — :func:`dataset_to_csv_rows` takes its row count from the
      first column and indexes every other column at ``i``, so a short column
      raised ``IndexError`` after the 200 and the header had already been sent
      (issue #4040). The old check skipped this entirely when ``index`` was
      ``None``, which is exactly the case the router hits for an unindexed
      export.

    Raises:
        ValueError: if the columns are ragged, or if ``index`` contains a
            ``None``, non-finite, or out-of-representable-range epoch-ms value.
    """
    require_aligned_columns(index, columns)
    if index is None:
        return
    for v in index:
        if v is None or not np.isfinite(v) or abs(float(v)) > _MAX_EPOCH_MS:
            raise ValueError(
                "export index contains a non-finite or out-of-range epoch-ms value"
            )


def _to_epoch_ms(value: datetime) -> float:
    """Convert a (possibly naive) datetime to epoch milliseconds, treating a
    naive value as UTC — the one interpretation the historian ever writes."""
    return _ensure_utc(value).timestamp() * 1000.0


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
    """Render a historian timestamp (datetime or str) as ISO-UTC, or ``None``.

    Always emits an explicit offset: an offset-less string is re-parsed by the
    browser as *local* time (issue #4025).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value).isoformat()
    return str(value)


def _historian_row_counts(
    session: object, tags: Sequence[str], start_dt: datetime, end_dt: datetime
) -> dict[str, int]:
    """Rows per tag inside the window, in one grouped indexed COUNT."""
    from models import TagLog
    from sqlmodel import col, func, select

    statement = (
        select(TagLog.tag_name, func.count(col(TagLog.id)))
        .where(col(TagLog.tag_name).in_(list(tags)))
        .where(col(TagLog.timestamp) >= start_dt)
        .where(col(TagLog.timestamp) <= end_dt)
        .group_by(col(TagLog.tag_name))
    )
    counts = dict.fromkeys(tags, 0)
    for name, count in session.exec(statement):  # type: ignore[attr-defined]
        counts[str(name)] = int(count or 0)
    return counts


def _load_historian(
    session: object,
    tags: Sequence[str],
    start: str,
    end: str,
    max_points: int = _DEFAULT_HISTORIAN_MAX_POINTS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load, decimate and align tag series from the historian onto one index.

    Each tag's ``(timestamp, value)`` rows within ``[start, end]`` are streamed
    in ascending time order and decimated server-side to at most ``max_points``
    samples spanning the *whole* window (see
    :func:`data_capture.query_trend_series`). The decimated series are then
    linearly interpolated (``np.interp``, edge-held out of range) onto the sorted
    union of their epoch-ms timestamps, producing a rectangular dataset.

    Two defects are addressed here (issue #4026):

    * The memory guard used to append every row of every tag into Python lists
      and only *then* compare the cell count against the ceiling — it could not
      prevent the OOM it existed to prevent (64 tags x 30 days at 5 s is ~33 M
      rows, >1.3 GB resident, before the 400 was raised). The budget is now
      decided from row ``COUNT``s before a single row is read.
    * ``HistorianSource.max_points`` was accepted by the model and sent by the
      HMI but never read anywhere in the backend. It now bounds the load itself,
      so peak memory is proportional to the *output* size, not the range.

    ``start``/``end`` are parsed with :func:`data_capture.parse_query_bound`, so
    an offset-less bound means UTC and an explicit offset is honoured rather
    than silently compared against a different clock (issue #4025).

    Args:
        session: A SQLModel ``Session`` bound to the historian database.
        tags: Tag names to load (duplicates collapse to one column).
        start: ISO-8601 inclusive lower bound.
        end: ISO-8601 inclusive upper bound.
        max_points: Per-tag sample budget; clamped into the trend-read range.

    Returns:
        ``(index_ms, columns)`` where ``index_ms`` is the common epoch-ms index
        and ``columns`` maps each tag name to its aligned values.

    Raises:
        TypeError: If ``tags`` is not a sequence of str or ``max_points`` is not
            an int.
        ValueError: If the selection would exceed ``_MAX_HISTORIAN_CELLS``, or
            if a bound is not a valid ISO datetime.
    """
    from data_capture import (
        TRENDS_MAX_MAX_POINTS,
        TRENDS_MIN_MAX_POINTS,
        parse_query_bound,
        query_trend_series,
    )

    if isinstance(tags, str) or not isinstance(tags, Sequence):
        raise TypeError(f"tags must be a sequence of str, got {type(tags).__name__}")
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError(f"max_points must be an int, got {type(max_points).__name__}")

    # Preserve request order while collapsing duplicates: a repeated tag would
    # otherwise be counted twice against the cell budget for one column.
    unique_tags = list(dict.fromkeys(tags))
    start_dt = parse_query_bound(start)
    end_dt = parse_query_bound(end)
    # The trend reader owns the decimation contract, including its bounds.
    per_tag_cap = max(
        TRENDS_MIN_MAX_POINTS, min(int(max_points), TRENDS_MAX_MAX_POINTS)
    )

    counts = _historian_row_counts(session, unique_tags, start_dt, end_dt)
    total_rows = sum(counts.values())
    n_tags = max(1, len(unique_tags))
    # Upper bound on the union index: each tag contributes at most its own row
    # count, and at most per_tag_cap + 1 after decimation (the +1 is the forced
    # final sample). Reject BEFORE reading a single row.
    est_index = min(total_rows, n_tags * (per_tag_cap + 1))
    est_cells = est_index * n_tags
    if est_cells > _MAX_HISTORIAN_CELLS:
        raise ValueError(
            f"historian selection too large: ~{est_index} samples x {n_tags} "
            f"tags = ~{est_cells} cells (limit {_MAX_HISTORIAN_CELLS}); narrow "
            f"the time range, select fewer tags, or lower max_points"
        )

    raw: dict[str, tuple[_F64, _F64]] = {}
    union: list[float] = []
    for tag in unique_tags:
        timestamps, values, _ = query_trend_series(
            session,
            tag_name=tag,
            start=start_dt,
            end=end_dt,
            max_points=per_tag_cap,
        )
        times = [_to_epoch_ms(ts) for ts in timestamps]
        raw[tag] = (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        union.extend(times)

    index: _F64 = np.unique(np.asarray(union, dtype=np.float64))
    # Belt and braces: the pre-check bounds the estimate, this bounds the fact.
    cells = int(index.size) * n_tags
    if cells > _MAX_HISTORIAN_CELLS:
        raise ValueError(
            f"historian selection too large: {index.size} samples x {n_tags} "
            f"tags = {cells} cells; narrow the time range or select fewer tags"
        )
    columns: dict[str, _F64] = {}
    for tag in unique_tags:
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
    return _load_historian(
        session, src.tags, src.start_time, src.end_time, src.max_points
    )


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
