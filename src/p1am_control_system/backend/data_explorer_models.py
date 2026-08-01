"""Pydantic request/response models for the Data Explorer API.

These are plain data contracts shared by the numeric kernels
(:mod:`data_explorer_signals`, :mod:`data_explorer_stats`), the orchestration
service (:mod:`data_explorer_service`) and the FastAPI router
(:mod:`data_explorer_router`). They carry no business logic and import no heavy
numeric libraries, so they are safe to import anywhere.

Design notes:
- A *dataset* is a shared epoch-millisecond ``index`` plus one or more equal
  length :class:`Column` s. Every analysis endpoint consumes columns in this
  exact shape, whether the data originated in the historian or was parsed from a
  CSV in the browser — one uniform contract for "flexible data types".
- ``Field`` bounds encode the cheap, declarative half of Design-by-Contract;
  the numeric kernels enforce the richer preconditions (finite, monotone time,
  odd window, …) and raise ``ValueError``/``TypeError`` the router maps to 4xx.
"""

from __future__ import annotations

from collections.abc import Sequence

from data_explorer_enums import (
    AggMethod,
    CorrelationMethod,
    ExportFormat,
    FilterType,
    SpectrumMethod,
    TrendlineKind,
    WindowKind,
)
from pydantic import BaseModel, Field, model_validator

# --- Core dataset shape ------------------------------------------------------


class Column(BaseModel):
    """A named numeric series. Length must match its dataset's index.

    ``None`` marks a gap/non-finite sample so it round-trips through strict JSON;
    the service maps ``None`` <-> ``numpy.nan`` at the numeric boundary.
    """

    name: str = Field(min_length=1, max_length=200)
    values: list[float | None]


def require_aligned_columns(
    index: Sequence[float] | None, columns: Sequence[Column]
) -> None:
    """DbC: every column must be the same length (and match ``index`` if given).

    The single definition of "aligned" for every dataset-shaped payload. It is
    deliberately unconditional on ``index``: the CSV export walks ``n`` from the
    first column and indexes all the others at ``i``, so a *ragged column set
    with no index* raised ``IndexError`` partway through a response that had
    already been sent with HTTP 200 — the client received a silently truncated
    CSV (issue #4040). Validating up front makes that a clean 400.

    Args:
        index: The shared epoch-ms index, or None when rows are numbered.
        columns: The columns to check.

    Raises:
        ValueError: If any column's length differs from ``index`` (when given)
            or from the first column's length.
    """
    if not columns:
        return
    expected = len(index) if index is not None else len(columns[0].values)
    label = (
        "index length" if index is not None else f"column {columns[0].name!r} length"
    )
    for column in columns:
        if len(column.values) != expected:
            raise ValueError(
                f"column {column.name!r} length {len(column.values)} != "
                f"{label} {expected}"
            )


class InlineData(BaseModel):
    """A dataset supplied directly by the client (e.g. a browser-parsed CSV).

    ``index`` is epoch milliseconds; every column must be the same length.
    """

    index: list[float]
    columns: list[Column] = Field(min_length=1)

    @model_validator(mode="after")
    def _columns_match_index(self) -> InlineData:
        require_aligned_columns(self.index, self.columns)
        return self


class HistorianSource(BaseModel):
    """Selector for pulling raw tag series out of the SQLite historian."""

    tags: list[str] = Field(min_length=1, max_length=64)
    start_time: str = Field(description="ISO-8601 inclusive lower bound")
    end_time: str = Field(description="ISO-8601 inclusive upper bound")
    max_points: int = Field(default=5000, ge=10, le=500_000)


# --- Pipeline steps ----------------------------------------------------------


class FilterSpec(BaseModel):
    """One filter/transform applied to a single named column, in list order.

    ``params`` are filter-specific (validated by the numeric kernel):
    moving_average/median/savgol/hampel → ``window`` (and ``polyorder`` for
    savgol, ``n_sigma`` for hampel); exponential → ``alpha``; gaussian → ``sigma``;
    zscore → ``threshold``; fft_* → ``low``/``high`` (Hz); integrate/differentiate
    → ``method`` is implicit. ``output`` optionally writes a new column instead of
    overwriting ``target``.
    """

    target: str = Field(min_length=1)
    type: FilterType
    params: dict[str, float] = Field(default_factory=dict)
    output: str | None = Field(
        default=None, description="New column name, or None to overwrite"
    )


class DerivedColumn(BaseModel):
    """A new column computed from a safe arithmetic expression over columns.

    Expression grammar (see :mod:`data_explorer_expression`): ``+ - * / **``,
    parentheses, numeric literals, other column names, and the whitelisted
    functions ``sin cos tan sqrt abs log log10 exp min max mean clip`` plus the
    constants ``pi`` and ``e``. No attribute access, calls to arbitrary names,
    comprehensions or statements are permitted.
    """

    name: str = Field(min_length=1, max_length=200)
    expression: str = Field(min_length=1)


class ResampleSpec(BaseModel):
    """Re-bin the dataset onto a uniform ``interval_s`` grid."""

    interval_s: float = Field(gt=0.0, le=86_400.0)
    agg: AggMethod = AggMethod.MEAN
    interpolate: bool = Field(
        default=False, description="Linearly fill empty bins instead of dropping them"
    )


class TrimSpec(BaseModel):
    """Keep only samples whose epoch-ms index falls within [start, end]."""

    start_ms: float | None = None
    end_ms: float | None = None


# --- Dataset build -----------------------------------------------------------


class DatasetRequest(BaseModel):
    """Build a processed dataset from exactly one source, then run the pipeline.

    Pipeline order is deterministic: align → resample → filters → derived → trim
    → downsample-to ``max_points``.
    """

    historian: HistorianSource | None = None
    inline: InlineData | None = None
    resample: ResampleSpec | None = None
    filters: list[FilterSpec] = Field(default_factory=list)
    derived: list[DerivedColumn] = Field(default_factory=list)
    trim: TrimSpec | None = None
    max_points: int = Field(default=5000, ge=10, le=200_000)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DatasetRequest:
        if (self.historian is None) == (self.inline is None):
            raise ValueError("exactly one of 'historian' or 'inline' must be set")
        return self


class DatasetResponse(BaseModel):
    """A processed dataset ready to plot or feed to an analysis endpoint."""

    index: list[float]
    columns: list[Column]
    row_count: int
    truncated: bool = Field(description="True if downsampled to fit max_points")
    sample_rate_hz: float | None = Field(
        default=None, description="Median sample rate of the index, if derivable"
    )


class SignalInfo(BaseModel):
    """Historian availability summary for one tag."""

    name: str
    count: int
    start_time: str | None
    end_time: str | None


class SignalListResponse(BaseModel):
    signals: list[SignalInfo]


# --- Analysis: statistics ----------------------------------------------------


class ColumnsRequest(BaseModel):
    """Base request carrying a set of aligned columns to analyze."""

    columns: list[Column] = Field(min_length=1)


class ColumnStatistics(BaseModel):
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p25: float
    p75: float
    rms: float


class StatisticsResponse(BaseModel):
    stats: list[ColumnStatistics]


# --- Analysis: correlation ---------------------------------------------------


class CorrelationRequest(ColumnsRequest):
    method: CorrelationMethod = CorrelationMethod.PEARSON


class CorrelationResponse(BaseModel):
    labels: list[str]
    matrix: list[list[float]]
    method: CorrelationMethod


# --- Analysis: spectrum ------------------------------------------------------


class SpectrumRequest(BaseModel):
    values: list[float | None] = Field(min_length=2)
    sample_rate_hz: float = Field(gt=0.0)
    method: SpectrumMethod = SpectrumMethod.FFT
    window: WindowKind = WindowKind.HANNING
    segment_size: int | None = Field(default=None, ge=8, le=1_048_576)
    detrend: bool = True


class SpectrumResponse(BaseModel):
    freqs: list[float]
    power: list[float]
    method: SpectrumMethod


# --- Analysis: trendline -----------------------------------------------------


class TrendlineRequest(BaseModel):
    x: list[float | None] = Field(min_length=2)
    y: list[float | None] = Field(min_length=2)
    kind: TrendlineKind = TrendlineKind.LINEAR
    degree: int = Field(default=2, ge=1, le=10)
    num_points: int = Field(default=200, ge=2, le=10_000)

    @model_validator(mode="after")
    def _xy_same_length(self) -> TrendlineRequest:
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have equal length")
        return self


class TrendlineResponse(BaseModel):
    kind: TrendlineKind
    coefficients: list[float]
    equation: str
    r_squared: float
    x_fit: list[float]
    y_fit: list[float]


# --- Analysis: PCA -----------------------------------------------------------


class PcaRequest(ColumnsRequest):
    standardize: bool = True
    n_components: int = Field(default=0, ge=0, le=1000, description="0 = all")


class PcaResponse(BaseModel):
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    singular_values: list[float]
    component_labels: list[str]
    loadings: list[list[float]]
    scores_pc1: list[float]
    scores_pc2: list[float]


# --- Analysis: histogram -----------------------------------------------------


class HistogramRequest(BaseModel):
    values: list[float | None] = Field(min_length=1)
    bins: int = Field(default=30, ge=1, le=1000)
    density: bool = False


class HistogramResponse(BaseModel):
    bin_edges: list[float]
    counts: list[float]


# --- Export ------------------------------------------------------------------


class ExportRequest(BaseModel):
    index: list[float] | None = None
    columns: list[Column] = Field(min_length=1)
    format: ExportFormat = ExportFormat.CSV
    filename: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _columns_match_index(self) -> ExportRequest:
        # Unconditional: ragged columns break the streaming CSV writer even when
        # no index was supplied (issue #4040).
        require_aligned_columns(self.index, self.columns)
        return self
