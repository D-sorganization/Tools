"""Tests for the Data Explorer FastAPI router.

A tiny FastAPI app includes ``create_data_explorer_router`` with a session
dependency overridden to an in-memory SQLModel engine seeded with ``TagLog``
rows. Covers the historian-backed routes, every analysis endpoint, the export
formats, and the ``ValueError``/``TypeError`` -> ``HTTPException(400)`` mapping.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_explorer_router import create_data_explorer_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from models import TagLog  # noqa: E402
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # noqa: UP017


@pytest.fixture
def client() -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    with Session(engine) as seed:
        seed.add_all(
            [
                TagLog(tag_name="TAG_0", value=0.0, timestamp=t0),
                TagLog(tag_name="TAG_0", value=2.0, timestamp=t2),
                TagLog(tag_name="TAG_1", value=10.0, timestamp=t1),
            ]
        )
        seed.commit()

    def get_session() -> Iterator[Session]:
        with Session(engine) as s:
            yield s

    app = FastAPI()
    app.include_router(create_data_explorer_router(get_session))
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Historian-backed routes
# --------------------------------------------------------------------------- #


def test_get_signals(client: TestClient) -> None:
    resp = client.get("/api/explorer/signals")
    assert resp.status_code == 200
    names = {s["name"]: s for s in resp.json()["signals"]}
    assert names["TAG_0"]["count"] == 2
    assert names["TAG_1"]["count"] == 1


def test_post_dataset_historian(client: TestClient) -> None:
    body = {
        "historian": {
            "tags": ["TAG_0", "TAG_1"],
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:02+00:00",
        }
    }
    resp = client.post("/api/explorer/dataset", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["row_count"] == 3
    tag0 = next(c for c in data["columns"] if c["name"] == "TAG_0")
    assert tag0["values"] == [0.0, 1.0, 2.0]


def test_post_dataset_inline(client: TestClient) -> None:
    body = {
        "inline": {
            "index": [0.0, 1000.0, 2000.0],
            "columns": [{"name": "a", "values": [1.0, None, 3.0]}],
        }
    }
    resp = client.post("/api/explorer/dataset", json=body)
    assert resp.status_code == 200
    assert resp.json()["columns"][0]["values"] == [1.0, None, 3.0]


def test_post_dataset_bad_filter_target_is_400(client: TestClient) -> None:
    body = {
        "inline": {
            "index": [0.0, 1.0],
            "columns": [{"name": "a", "values": [1.0, 2.0]}],
        },
        "filters": [{"target": "missing", "type": "median", "params": {"window": 1}}],
    }
    resp = client.post("/api/explorer/dataset", json=body)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Analysis routes
# --------------------------------------------------------------------------- #


def test_post_statistics(client: TestClient) -> None:
    body = {"columns": [{"name": "a", "values": [1.0, 2.0, 3.0, 4.0, 5.0]}]}
    resp = client.post("/api/explorer/statistics", json=body)
    assert resp.status_code == 200
    s = resp.json()["stats"][0]
    assert s["count"] == 5
    assert s["mean"] == pytest.approx(3.0)


def test_post_correlation(client: TestClient) -> None:
    body = {
        "columns": [
            {"name": "x", "values": [1.0, 2.0, 3.0, 4.0]},
            {"name": "y", "values": [2.0, 4.0, 6.0, 8.0]},
        ],
        "method": "pearson",
    }
    resp = client.post("/api/explorer/correlation", json=body)
    assert resp.status_code == 200
    assert resp.json()["matrix"][0][1] == pytest.approx(1.0)


def test_post_spectrum(client: TestClient) -> None:
    fs = 64.0
    n = 256
    t = np.arange(n) / fs
    y = np.sin(2.0 * np.pi * 8.0 * t)
    body = {
        "values": [float(v) for v in y],
        "sample_rate_hz": fs,
        "method": "fft",
        "window": "none",
        "detrend": True,
    }
    resp = client.post("/api/explorer/spectrum", json=body)
    assert resp.status_code == 200
    power = resp.json()["power"]
    freqs = resp.json()["freqs"]
    assert freqs[int(np.argmax(power))] == pytest.approx(8.0, abs=0.5)


def test_post_trendline(client: TestClient) -> None:
    body = {
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [1.0, 3.0, 5.0, 7.0],
        "kind": "linear",
    }
    resp = client.post("/api/explorer/trendline", json=body)
    assert resp.status_code == 200
    assert resp.json()["r_squared"] == pytest.approx(1.0)


def test_post_pca(client: TestClient) -> None:
    x = np.linspace(0.0, 10.0, 40)
    body = {
        "columns": [
            {"name": "a", "values": [float(v) for v in x]},
            {"name": "b", "values": [float(v) for v in (2.0 * x + 1.0)]},
        ],
        "standardize": True,
        "n_components": 0,
    }
    resp = client.post("/api/explorer/pca", json=body)
    assert resp.status_code == 200
    assert resp.json()["explained_variance_ratio"][0] == pytest.approx(1.0, abs=1e-5)


def test_post_histogram(client: TestClient) -> None:
    body = {"values": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0], "bins": 3, "density": False}
    resp = client.post("/api/explorer/histogram", json=body)
    assert resp.status_code == 200
    assert sum(resp.json()["counts"]) == pytest.approx(6.0)


def test_post_correlation_bad_method_is_422(client: TestClient) -> None:
    # An unknown enum value is rejected by pydantic request validation (422)
    # before the handler runs.
    body = {
        "columns": [
            {"name": "x", "values": [1.0, 2.0]},
            {"name": "y", "values": [1.0, 2.0]},
        ],
        "method": "bogus",
    }
    resp = client.post("/api/explorer/correlation", json=body)
    # Unknown enum value -> 422 from pydantic validation before our handler.
    assert resp.status_code == 422


def test_post_correlation_single_column_is_400(client: TestClient) -> None:
    body = {
        "columns": [{"name": "x", "values": [1.0, 2.0, 3.0]}],
        "method": "pearson",
    }
    resp = client.post("/api/explorer/correlation", json=body)
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_post_export_csv(client: TestClient) -> None:
    body = {
        "index": [0.0, 1000.0],
        "columns": [{"name": "a", "values": [1.0, 2.0]}],
        "format": "csv",
    }
    resp = client.post("/api/explorer/export", json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.text
    assert text.splitlines()[0] == "timestamp,a"


def test_post_export_json(client: TestClient) -> None:
    body = {
        "index": [0.0, 1000.0],
        "columns": [{"name": "a", "values": [1.0, None]}],
        "format": "json",
    }
    resp = client.post("/api/explorer/export", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["columns"][0]["values"] == [1.0, None]
