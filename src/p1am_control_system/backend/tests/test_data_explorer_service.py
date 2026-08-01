"""Tests for the Data Explorer orchestration service.

Builds datasets from inline data and from an in-memory SQLModel session seeded
with ``TagLog`` rows, exercises the build pipeline (resample/filter/derived/
trim/downsample), the ``None`` <-> ``NaN`` boundary mapping, the ``compute_*``
analysis wrappers (numeric correctness asserted against hand/analytic values),
and the CSV/JSON export helpers.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_explorer_enums import (  # noqa: E402
    AggMethod,
    CorrelationMethod,
    FilterType,
    SpectrumMethod,
    TrendlineKind,
    WindowKind,
)
from data_explorer_models import (  # noqa: E402
    Column,
    ColumnsRequest,
    CorrelationRequest,
    DatasetRequest,
    DerivedColumn,
    FilterSpec,
    HistogramRequest,
    HistorianSource,
    InlineData,
    PcaRequest,
    ResampleSpec,
    SpectrumRequest,
    TrendlineRequest,
    TrimSpec,
)
from data_explorer_service import (  # noqa: E402
    build_dataset,
    compute_correlation,
    compute_histogram,
    compute_pca,
    compute_spectrum,
    compute_statistics,
    compute_trendline,
    dataset_to_csv_rows,
    dataset_to_json,
    list_signals,
)
from models import TagLog  # noqa: E402
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # noqa: UP017


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed(session: Session, rows: list[tuple[str, float, datetime]]) -> None:
    for name, value, ts in rows:
        session.add(TagLog(tag_name=name, value=value, timestamp=ts))
    session.commit()


def _inline(index: list[float], **cols: list[float | None]) -> DatasetRequest:
    return DatasetRequest(
        inline=InlineData(
            index=index,
            columns=[Column(name=n, values=v) for n, v in cols.items()],
        )
    )


# --------------------------------------------------------------------------- #
# build_dataset — inline
# --------------------------------------------------------------------------- #


def test_build_dataset_inline_happy_path() -> None:
    req = _inline([0.0, 1000.0, 2000.0], a=[1.0, 2.0, 3.0])
    resp = build_dataset(object(), req)
    assert resp.row_count == 3
    assert resp.index == [0.0, 1000.0, 2000.0]
    assert resp.columns[0].name == "a"
    assert resp.columns[0].values == [1.0, 2.0, 3.0]
    assert resp.truncated is False
    # 1000 ms spacing -> 1 Hz.
    assert resp.sample_rate_hz == pytest.approx(1.0)


def test_build_dataset_inline_none_maps_to_none_out() -> None:
    # None on the way in -> NaN internally -> None on the way out.
    req = _inline([0.0, 1000.0, 2000.0], a=[1.0, None, 3.0])
    resp = build_dataset(object(), req)
    assert resp.columns[0].values == [1.0, None, 3.0]


def test_build_dataset_derived_column() -> None:
    req = DatasetRequest(
        inline=InlineData(
            index=[0.0, 1000.0],
            columns=[
                Column(name="v", values=[2.0, 4.0]),
                Column(name="i", values=[3.0, 5.0]),
            ],
        ),
        derived=[DerivedColumn(name="p", expression="v * i")],
    )
    resp = build_dataset(object(), req)
    p = next(c for c in resp.columns if c.name == "p")
    assert p.values == [6.0, 20.0]


def test_build_dataset_filter_to_new_output_column() -> None:
    req = DatasetRequest(
        inline=InlineData(
            index=[0.0, 1.0, 2.0, 3.0, 4.0],
            columns=[Column(name="x", values=[1.0, 2.0, 3.0, 4.0, 5.0])],
        ),
        filters=[
            FilterSpec(
                target="x",
                type=FilterType.MOVING_AVERAGE,
                params={"window": 3.0},
                output="x_ma",
            )
        ],
    )
    resp = build_dataset(object(), req)
    names = {c.name for c in resp.columns}
    assert names == {"x", "x_ma"}
    # Original preserved.
    x = next(c for c in resp.columns if c.name == "x")
    assert x.values == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_build_dataset_unknown_filter_target_raises() -> None:
    req = DatasetRequest(
        inline=InlineData(
            index=[0.0, 1.0],
            columns=[Column(name="x", values=[1.0, 2.0])],
        ),
        filters=[
            FilterSpec(target="missing", type=FilterType.MEDIAN, params={"window": 1.0})
        ],
    )
    with pytest.raises(ValueError, match="target column not found"):
        build_dataset(object(), req)


def test_build_dataset_trim() -> None:
    req = DatasetRequest(
        inline=InlineData(
            index=[0.0, 1000.0, 2000.0, 3000.0],
            columns=[Column(name="a", values=[1.0, 2.0, 3.0, 4.0])],
        ),
        trim=TrimSpec(start_ms=1000.0, end_ms=2000.0),
    )
    resp = build_dataset(object(), req)
    assert resp.index == [1000.0, 2000.0]
    assert resp.columns[0].values == [2.0, 3.0]


def test_build_dataset_downsample_sets_truncated_and_keeps_endpoints() -> None:
    n = 100
    index = [float(i) * 1000.0 for i in range(n)]
    req = DatasetRequest(
        inline=InlineData(
            index=index,
            columns=[Column(name="a", values=[float(i) for i in range(n)])],
        ),
        max_points=10,
    )
    resp = build_dataset(object(), req)
    assert resp.truncated is True
    assert resp.row_count <= 11
    assert resp.index[0] == 0.0
    assert resp.index[-1] == index[-1]


def test_build_dataset_resample_mean() -> None:
    # Two 1s samples per 2s bin, mean aggregated.
    index = [0.0, 1000.0, 2000.0, 3000.0]
    req = DatasetRequest(
        inline=InlineData(
            index=index,
            columns=[Column(name="a", values=[1.0, 3.0, 5.0, 7.0])],
        ),
        resample=ResampleSpec(interval_s=2.0, agg=AggMethod.MEAN, interpolate=False),
    )
    resp = build_dataset(object(), req)
    # Bin 0 covers t in [0,2): mean(1,3)=2; bin 1 covers [2,4): mean(5,7)=6.
    assert resp.columns[0].values == [2.0, 6.0]


def test_build_dataset_full_pipeline() -> None:
    n = 50
    index = [float(i) * 1000.0 for i in range(n)]
    req = DatasetRequest(
        inline=InlineData(
            index=index,
            columns=[Column(name="a", values=[float(i) for i in range(n)])],
        ),
        resample=ResampleSpec(interval_s=1.0, agg=AggMethod.MEAN, interpolate=True),
        filters=[
            FilterSpec(
                target="a", type=FilterType.MOVING_AVERAGE, params={"window": 3.0}
            )
        ],
        derived=[DerivedColumn(name="b", expression="a + 1")],
        trim=TrimSpec(start_ms=0.0, end_ms=index[-1]),
        max_points=20,
    )
    resp = build_dataset(object(), req)
    assert resp.truncated is True
    names = {c.name for c in resp.columns}
    assert names == {"a", "b"}


# --------------------------------------------------------------------------- #
# build_dataset — historian
# --------------------------------------------------------------------------- #


def test_build_dataset_historian_aligns_on_union(session: Session) -> None:
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    _seed(
        session,
        [
            ("TAG_0", 0.0, t0),
            ("TAG_0", 2.0, t2),
            ("TAG_1", 10.0, t1),
        ],
    )
    req = DatasetRequest(
        historian=HistorianSource(
            tags=["TAG_0", "TAG_1"],
            start_time=t0.isoformat(),
            end_time=t2.isoformat(),
        )
    )
    resp = build_dataset(session, req)
    # Union of {t0,t2} and {t1} -> three timestamps.
    assert resp.row_count == 3
    tag0 = next(c for c in resp.columns if c.name == "TAG_0")
    # TAG_0 interpolated at t1 between 0 and 2 -> 1.0.
    assert tag0.values == [0.0, 1.0, 2.0]
    tag1 = next(c for c in resp.columns if c.name == "TAG_1")
    # TAG_1 only at t1; edge-held at t0 and t2.
    assert tag1.values == [10.0, 10.0, 10.0]


def test_list_signals(session: Session) -> None:
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    _seed(
        session,
        [("TAG_0", 1.0, t0), ("TAG_0", 2.0, t1), ("TAG_1", 3.0, t0)],
    )
    resp = list_signals(session)
    by_name = {s.name: s for s in resp.signals}
    assert by_name["TAG_0"].count == 2
    assert by_name["TAG_1"].count == 1
    assert by_name["TAG_0"].start_time is not None
    assert by_name["TAG_0"].end_time is not None


# --------------------------------------------------------------------------- #
# compute_* wrappers
# --------------------------------------------------------------------------- #


def test_compute_statistics_drops_nan() -> None:
    req = ColumnsRequest(columns=[Column(name="a", values=[1.0, None, 3.0, 5.0])])
    resp = compute_statistics(req)
    s = resp.stats[0]
    assert s.count == 3
    assert s.mean == pytest.approx(3.0)
    assert s.min == pytest.approx(1.0)
    assert s.max == pytest.approx(5.0)


def test_compute_statistics_all_nan_column() -> None:
    req = ColumnsRequest(columns=[Column(name="a", values=[None, None])])
    resp = compute_statistics(req)
    assert resp.stats[0].count == 0
    assert resp.stats[0].mean == 0.0


def test_compute_correlation_perfect_positive() -> None:
    req = CorrelationRequest(
        columns=[
            Column(name="x", values=[1.0, 2.0, 3.0, 4.0]),
            Column(name="y", values=[2.0, 4.0, 6.0, 8.0]),
        ],
        method=CorrelationMethod.PEARSON,
    )
    resp = compute_correlation(req)
    assert resp.labels == ["x", "y"]
    assert resp.matrix[0][1] == pytest.approx(1.0)


def test_compute_correlation_listwise_drops_nan_rows() -> None:
    # The NaN row is dropped pairwise; remaining points are perfectly correlated.
    req = CorrelationRequest(
        columns=[
            Column(name="x", values=[1.0, 2.0, None, 4.0]),
            Column(name="y", values=[2.0, 4.0, 100.0, 8.0]),
        ],
        method=CorrelationMethod.PEARSON,
    )
    resp = compute_correlation(req)
    assert resp.matrix[0][1] == pytest.approx(1.0)


def test_compute_spectrum_peak_at_sine_frequency() -> None:
    fs = 64.0
    n = 256
    t = np.arange(n) / fs
    freq = 8.0
    y = np.sin(2.0 * np.pi * freq * t)
    req = SpectrumRequest(
        values=[float(v) for v in y],
        sample_rate_hz=fs,
        method=SpectrumMethod.FFT,
        window=WindowKind.NONE,
        detrend=True,
    )
    resp = compute_spectrum(req)
    peak_idx = int(np.argmax(resp.power))
    assert resp.freqs[peak_idx] == pytest.approx(freq, abs=0.5)


def test_compute_trendline_linear_exact() -> None:
    req = TrendlineRequest(
        x=[0.0, 1.0, 2.0, 3.0],
        y=[1.0, 3.0, 5.0, 7.0],
        kind=TrendlineKind.LINEAR,
    )
    resp = compute_trendline(req)
    assert resp.r_squared == pytest.approx(1.0)
    # slope 2, intercept 1.
    assert resp.coefficients[0] == pytest.approx(2.0)
    assert resp.coefficients[1] == pytest.approx(1.0)


def test_compute_trendline_pairwise_nan_drop() -> None:
    req = TrendlineRequest(
        x=[0.0, 1.0, None, 3.0],
        y=[1.0, 3.0, 100.0, 7.0],
        kind=TrendlineKind.LINEAR,
    )
    resp = compute_trendline(req)
    assert resp.r_squared == pytest.approx(1.0)


def test_compute_pca_two_correlated_vars() -> None:
    x = np.linspace(0.0, 10.0, 50)
    req = PcaRequest(
        columns=[
            Column(name="a", values=[float(v) for v in x]),
            Column(name="b", values=[float(v) for v in (2.0 * x + 1.0)]),
        ],
        standardize=True,
        n_components=0,
    )
    resp = compute_pca(req)
    assert resp.explained_variance_ratio[0] == pytest.approx(1.0, abs=1e-6)


def test_compute_histogram_counts_sum_to_n() -> None:
    req = HistogramRequest(
        values=[1.0, 2.0, 2.0, 3.0, None, 3.0, 3.0], bins=3, density=False
    )
    resp = compute_histogram(req)
    assert sum(resp.counts) == pytest.approx(6.0)
    assert len(resp.bin_edges) == 4


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #


def test_dataset_to_csv_rows_header_and_content() -> None:
    cols = [Column(name="a", values=[1.0, None]), Column(name="b", values=[3.0, 4.0])]
    rows = list(dataset_to_csv_rows([0.0, 1000.0], cols))
    assert rows[0] == "timestamp,a,b\n"
    assert len(rows) == 3
    # Second data row has an empty cell for the None.
    assert rows[2].count(",") == 2
    body = rows[2].split(",")
    assert body[1] == ""  # gap rendered empty


def test_dataset_to_csv_rows_no_index_uses_counter() -> None:
    cols = [Column(name="a", values=[1.0, 2.0])]
    rows = list(dataset_to_csv_rows(None, cols))
    assert rows[1].startswith("0,")
    assert rows[2].startswith("1,")


def test_dataset_to_json_shape() -> None:
    cols = [Column(name="a", values=[1.0, None])]
    payload = dataset_to_json([0.0, 1.0], cols)
    assert payload["index"] == [0.0, 1.0]
    assert payload["columns"][0]["name"] == "a"
    assert payload["columns"][0]["values"] == [1.0, None]


# --------------------------------------------------------------------------- #
# Regression: export validation + safe ISO rendering                          #
# --------------------------------------------------------------------------- #
import data_explorer_service as _svc  # noqa: E402
from data_explorer_models import Column as _Column  # noqa: E402


def test_validate_export_rejects_nonfinite_index() -> None:
    with pytest.raises(ValueError, match="non-finite or out-of-range"):
        _svc.validate_export(
            [0.0, float("inf")], [_Column(name="a", values=[1.0, 2.0])]
        )


def test_validate_export_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        _svc.validate_export([1e19], [_Column(name="a", values=[1.0])])


def test_validate_export_accepts_none_or_finite_index() -> None:
    _svc.validate_export(None, [_Column(name="a", values=[1.0])])
    _svc.validate_export([0.0, 1000.0], [_Column(name="a", values=[1.0, 2.0])])


def test_epoch_ms_to_iso_safe_on_bad_input() -> None:
    assert _svc._epoch_ms_to_iso(float("nan")) == ""
    assert _svc._epoch_ms_to_iso(float("inf")) == ""
    assert _svc._epoch_ms_to_iso(1e18) == ""
    assert _svc._epoch_ms_to_iso(0.0).startswith("1970-01-01")


def test_assert_columns_aligned_rejects_ragged() -> None:
    import data_explorer_service as _svc

    with pytest.raises(ValueError, match="ragged"):
        _svc._assert_columns_aligned(
            np.array([0.0, 1.0, 2.0]), {"a": np.array([1.0, 2.0])}
        )


def test_assert_columns_aligned_accepts_matched() -> None:
    import data_explorer_service as _svc

    _svc._assert_columns_aligned(
        np.array([0.0, 1.0]), {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    )


# --------------------------------------------------------------------------- #
# Regression #4026: historian load must respect its budget BEFORE materializing #
# --------------------------------------------------------------------------- #


def test_load_historian_rejects_oversized_selection_before_reading_rows(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The old guard appended every row for every tag into Python lists and only
    # then compared `cells` against the ceiling — it could not prevent the OOM it
    # existed to prevent. The budget check must fire off row COUNTs alone.
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    _seed(
        session,
        [("TAG_0", float(i), t0 + timedelta(seconds=i)) for i in range(50)]
        + [("TAG_1", float(i), t0 + timedelta(seconds=i)) for i in range(50)],
    )
    monkeypatch.setattr(_svc, "_MAX_HISTORIAN_CELLS", 4)

    def _must_not_load(*_a: object, **_kw: object) -> None:
        raise AssertionError("rows were streamed before the budget was checked")

    import data_capture

    monkeypatch.setattr(data_capture, "query_trend_series", _must_not_load)

    with pytest.raises(ValueError, match="too large"):
        _svc._load_historian(
            session,
            ["TAG_0", "TAG_1"],
            t0.isoformat(),
            (t0 + timedelta(seconds=60)).isoformat(),
            max_points=5000,
        )


def test_build_dataset_historian_honours_max_points(session: Session) -> None:
    # #4026: HistorianSource.max_points was accepted by the model and sent by the
    # HMI but never read anywhere in the backend.
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    _seed(session, [("TAG_0", float(i), t0 + timedelta(seconds=i)) for i in range(500)])
    req = DatasetRequest(
        historian=HistorianSource(
            tags=["TAG_0"],
            start_time=t0.isoformat(),
            end_time=(t0 + timedelta(seconds=500)).isoformat(),
            max_points=10,
        ),
        max_points=200_000,  # do not let the post-pipeline downsample do the work
    )
    resp = build_dataset(session, req)
    assert resp.row_count <= 11  # decimated server-side at load time
    # The whole window is still covered (endpoints preserved).
    assert resp.columns[0].values[0] == 0.0
    assert resp.columns[0].values[-1] == 499.0


def test_build_dataset_historian_small_range_is_not_decimated(
    session: Session,
) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    _seed(session, [("TAG_0", float(i), t0 + timedelta(seconds=i)) for i in range(20)])
    req = DatasetRequest(
        historian=HistorianSource(
            tags=["TAG_0"],
            start_time=t0.isoformat(),
            end_time=(t0 + timedelta(seconds=60)).isoformat(),
            max_points=5000,
        )
    )
    resp = build_dataset(session, req)
    assert resp.row_count == 20


def test_load_historian_window_bounds_are_utc_normalized(session: Session) -> None:
    # #4025: an offset-less bound must be read as UTC, and an explicit offset
    # must be honoured — otherwise "export everything" starts hours late.
    t0 = datetime(2026, 1, 1, 12, tzinfo=UTC)
    _seed(session, [("TAG_0", 1.0, t0), ("TAG_0", 2.0, t0 + timedelta(hours=8))])

    naive = _svc._load_historian(
        session, ["TAG_0"], "2026-01-01T00:00:00", "2026-01-02T00:00:00"
    )
    assert naive[0].size == 2

    # 13:00-07:00 == 20:00Z, so only the later sample is in range.
    offset = _svc._load_historian(
        session, ["TAG_0"], "2026-01-01T13:00:00-07:00", "2026-01-02T00:00:00Z"
    )
    assert offset[0].size == 1


def test_load_historian_rejects_bad_types(session: Session) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    with pytest.raises(TypeError):
        _svc._load_historian(session, "TAG_0", t0, t0)
    with pytest.raises(TypeError):
        _svc._load_historian(session, ["TAG_0"], t0, t0, max_points="lots")


def test_list_signals_emits_explicit_offset(session: Session) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    _seed(session, [("TAG_0", 1.0, t0)])
    info = list_signals(session).signals[0]
    assert info.start_time is not None and info.start_time.endswith("+00:00")
    assert info.end_time is not None and info.end_time.endswith("+00:00")


# --------------------------------------------------------------------------- #
# Regression #4040: ragged columns must 400 before the CSV body starts         #
# --------------------------------------------------------------------------- #


def test_validate_export_rejects_ragged_columns_without_index() -> None:
    # dataset_to_csv_rows takes n from columns[0] and indexes every other column
    # at i -> IndexError mid-stream, i.e. a TRUNCATED body behind an HTTP 200.
    with pytest.raises(ValueError, match="length"):
        _svc.validate_export(
            None,
            [
                _Column(name="a", values=[1.0, 2.0, 3.0]),
                _Column(name="b", values=[1.0]),
            ],
        )


def test_validate_export_rejects_ragged_columns_with_index() -> None:
    with pytest.raises(ValueError, match="length"):
        _svc.validate_export(
            [0.0, 1.0],
            [
                _Column(name="a", values=[1.0, 2.0]),
                _Column(name="b", values=[1.0]),
            ],
        )


def test_validate_export_accepts_equal_length_columns_without_index() -> None:
    _svc.validate_export(
        None,
        [
            _Column(name="a", values=[1.0, 2.0]),
            _Column(name="b", values=[3.0, 4.0]),
        ],
    )


def test_export_request_model_rejects_ragged_columns_without_index() -> None:
    from data_explorer_models import ExportRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExportRequest(
            index=None,
            columns=[
                Column(name="a", values=[1.0, 2.0]),
                Column(name="b", values=[1.0]),
            ],
        )
