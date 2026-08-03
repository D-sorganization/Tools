"""REST integration for investigations, asset health, and shift handover."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from asset_health import AssetHealthPolicy, AssetHealthService, AssetObservation
from fastapi import FastAPI
from fastapi.testclient import TestClient
from identity import Principal, Role
from operations_router import create_operations_router
from saved_investigation import InvestigationService, SqliteInvestigationRepository
from shift_log import ShiftLogService
from shift_log_repository import SqliteShiftLogRepository
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


def _client() -> TestClient:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def factory() -> Session:
        return Session(engine)

    investigations = InvestigationService(
        SqliteInvestigationRepository(factory), now=lambda: now
    )
    shifts = ShiftLogService(SqliteShiftLogRepository(factory), now=lambda: now)
    health = AssetHealthService(AssetHealthPolicy(), now=lambda: now)
    observations = (
        AssetObservation(
            observed_at=now - timedelta(minutes=10),
            value=10,
            reference=10,
            command=True,
            feedback=True,
            running=True,
        ),
        AssetObservation(
            observed_at=now,
            value=10,
            reference=10,
            command=True,
            feedback=True,
            running=True,
        ),
    )
    app = FastAPI()
    app.include_router(
        create_operations_router(
            investigations,
            shifts,
            asset_report_provider=lambda: health.assess(
                "SYNTHETIC.FEED.PUMP",
                observations,
                calibration_due_at=now + timedelta(days=1),
            ),
            operator_dependency=lambda: Principal(
                "operator.one", "Operator One", Role.OPERATOR
            ),
        )
    )
    return TestClient(app)


def _investigation() -> dict[str, object]:
    start = datetime(2026, 8, 3, 19, 0, tzinfo=UTC)
    return {
        "title": "Synthetic feed review",
        "query": {
            "tags": ["SYNTHETIC.FEED.FLOW"],
            "start": start.isoformat(),
            "end": (start + timedelta(hours=1)).isoformat(),
            "max_points": 1000,
        },
        "tag_metadata": [
            {
                "tag": "SYNTHETIC.FEED.FLOW",
                "description": "Representative flow",
                "unit": "%",
                "source": "synthetic_driver",
            }
        ],
        "charts": [
            {
                "chart_id": "flow",
                "kind": "trend",
                "tags": ["SYNTHETIC.FEED.FLOW"],
            }
        ],
        "bad_data_policy": "preserve",
        "context": "Synthetic only",
    }


def test_investigation_create_fetch_and_checksum_export() -> None:
    client = _client()

    created = client.post("/api/operator/investigations", json=_investigation())
    investigation_id = created.json()["investigation_id"]
    fetched = client.get(f"/api/operator/investigations/{investigation_id}")
    exported = client.get(f"/api/operator/investigations/{investigation_id}/export")

    assert created.status_code == 200
    assert fetched.json() == created.json()
    assert len(exported.headers["X-Artifact-SHA256"]) == 64
    assert exported.headers["X-Investigation-ID"] == investigation_id
    assert exported.content.startswith(b"PK")


def test_asset_health_report_is_advisory_not_trip() -> None:
    response = _client().get("/api/operator/assets/health/representative")

    assert response.status_code == 200
    assert response.json()["asset_id"] == "SYNTHETIC.FEED.PUMP"
    assert response.json()["data_classification"] == "synthetic"


def test_shift_entry_signoff_and_handover_workflow() -> None:
    client = _client()
    created = client.post(
        "/api/operator/shift-log",
        json={
            "shift_id": "SYNTHETIC.SHIFT.NIGHT",
            "run_id": "SYNTHETIC.RUN.0042",
            "summary": "Synthetic handover entry",
            "unresolved_actions": ["Review representative calibration"],
            "event_references": [],
            "trend_references": [],
        },
    )
    entry_id = created.json()["entry_id"]
    signoff = client.post(f"/api/operator/shift-log/{entry_id}/signoff")
    handover = client.post(
        f"/api/operator/shift-log/{entry_id}/handover",
        json={"note": "Accepted by receiving synthetic shift"},
    )
    search = client.get("/api/operator/shift-log", params={"query": "handover"})

    assert created.status_code == 200
    assert len(signoff.json()["content_sha256"]) == 64
    assert handover.json()["acknowledged_by"] == "operator.one"
    assert search.json()[0]["entry_id"] == entry_id
