"""Backend auth-gate + import-hardening regression tests (issues #3289, #3292).

These verify that state-mutating endpoints and the live WebSocket reject requests
without a valid server-side credential, that destructive/admin operations require
the elevated key, and that ``/api/project/import`` enforces upload/zip-bomb limits
and replaces the plant DB atomically.

The backend uses a flat-``sys.path`` import style; ``conftest.py`` in this
directory adds the backend dir to ``sys.path``. If the backend's runtime
dependencies (fastapi/sqlmodel/httpx) are unavailable the whole module is
skipped rather than erroring at collection.
"""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Generator

import pytest

os.environ["PLC_DRIVER"] = "modbus"
# Import the app WITHOUT dev-no-auth so the real gate is active for this suite.
os.environ.pop("P1AM_DEV_NO_AUTH", None)

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

try:
    import main  # type: ignore[import-not-found]
    import project_import  # type: ignore[import-not-found]
    from models import (  # type: ignore[import-not-found]
        PlantArea,
        PlantEquipment,
        PlantUnit,
        TagDefinitionDb,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    # Only a genuinely missing module skips this suite. A NameError /
    # SyntaxError / ImportError inside main.py or models.py is a real defect
    # in the security-critical backend and must fail loudly, not silently
    # skip the entire auth/zip-bomb suite (issue #3745).
    pytest.skip(
        f"P1AM backend not importable in this environment: {exc}",
        allow_module_level=True,
    )

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

TestClient = fastapi_testclient.TestClient
app = main.app
get_session = main.get_session

OPERATOR_KEY = "operator-secret-key"
ADMIN_KEY = "admin-secret-key"

_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def _override_get_session() -> Generator[Session, None, None]:
    with Session(_test_engine) as session:
        yield session


app.dependency_overrides[get_session] = _override_get_session
client = TestClient(app)


@pytest.fixture(autouse=True)
def _setup() -> Generator[None, None, None]:
    SQLModel.metadata.create_all(_test_engine)
    app.dependency_overrides[get_session] = _override_get_session
    yield
    SQLModel.metadata.drop_all(_test_engine)


@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P1AM_DEV_NO_AUTH", raising=False)
    monkeypatch.delenv("P1AM_API_KEY", raising=False)
    monkeypatch.delenv("P1AM_ADMIN_API_KEY", raising=False)


MUTATING_OPERATOR_ROUTES = [
    ("post", "/api/events", {"event_type": "x", "description": "y"}),
    ("post", "/api/alarms/TAG_1/acknowledge", None),
]

MUTATING_ADMIN_ROUTES = [
    ("post", "/api/estop/clear", None),
    ("post", "/api/tags/5", {"value": 1.0}),
    ("post", "/api/alicats/A/setpoint", {"setpoint": 1.0}),
    ("post", "/api/alicats/A/gas", {"gas": "O2"}),
]


def _call(method: str, path: str, json_body, headers=None):
    return client.request(method, path, json=json_body, headers=headers)


def test_fails_closed_when_no_key_configured() -> None:
    """No P1AM_API_KEY and no dev opt-out -> 503 on mutating routes (#3289)."""
    for method, path, body in MUTATING_OPERATOR_ROUTES + MUTATING_ADMIN_ROUTES:
        resp = _call(method, path, body)
        assert resp.status_code == 503, (path, resp.status_code)


def test_mutating_routes_reject_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    for method, path, body in MUTATING_OPERATOR_ROUTES:
        resp = _call(method, path, body)
        assert resp.status_code == 401, (path, resp.status_code)


def test_mutating_routes_reject_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    headers = {"X-API-Key": "not-the-key"}
    for method, path, body in MUTATING_OPERATOR_ROUTES:
        resp = _call(method, path, body, headers)
        assert resp.status_code == 401, (path, resp.status_code)


def test_operator_route_accepts_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    headers = {"X-API-Key": OPERATOR_KEY}
    resp = _call(
        "post", "/api/events", {"event_type": "x", "description": "y"}, headers
    )
    assert resp.status_code == 200


def test_admin_routes_require_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When an admin key is set, the operator key must NOT open admin routes."""
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", ADMIN_KEY)

    op_headers = {"X-API-Key": OPERATOR_KEY}
    admin_headers = {"X-API-Key": ADMIN_KEY}

    resp = _call("post", "/api/estop/clear", None, op_headers)
    assert resp.status_code == 403

    resp = _call("post", "/api/estop/clear", None, admin_headers)
    assert resp.status_code == 200


def test_single_key_deployment_admin_uses_operator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a separate admin key, the operator key opens admin routes."""
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    headers = {"X-API-Key": OPERATOR_KEY}
    resp = _call("post", "/api/estop/clear", None, headers)
    assert resp.status_code == 200


def test_estop_activation_remains_unauthenticated() -> None:
    """E-stop activation is intentionally always reachable (documented choice)."""
    resp = client.post("/api/estop")
    assert resp.status_code == 200


def test_websocket_rejects_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    with pytest.raises(Exception):  # noqa: B017,PT011 close(1008) surfaces as error
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text("not-the-key")
            ws.receive_text()


def test_websocket_accepts_with_query_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    with client.websocket_connect(f"/api/stream?api_key={OPERATOR_KEY}") as ws:
        ws.send_text("ping")


# --------------------------- project import (#3292) ---------------------------


def _make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _admin_headers() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}


def test_import_requires_admin_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", ADMIN_KEY)
    zip_bytes = _make_zip({"tagl.json": b"[]"})
    resp = client.post(
        "/api/project/import",
        files={"file": ("p.zip", zip_bytes, "application/zip")},
        headers={"X-API-Key": OPERATOR_KEY},
    )
    assert resp.status_code == 403


def test_import_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setattr(project_import, "MAX_IMPORT_UPLOAD_BYTES", 1024)
    big = b"x" * 5000
    zip_bytes = _make_zip({"tagl.json": big})
    resp = client.post(
        "/api/project/import",
        files={"file": ("p.zip", zip_bytes, "application/zip")},
        headers=_admin_headers(),
    )
    assert resp.status_code == 413


def test_import_rejects_zip_bomb_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setattr(project_import, "MAX_IMPORT_UPLOAD_BYTES", 50 * 1024 * 1024)
    bomb = b"\x00" * (5 * 1024 * 1024)  # compresses tiny -> huge ratio
    zip_bytes = _make_zip({"tagl.json": b"[]", "bomb.bin": bomb})
    resp = client.post(
        "/api/project/import",
        files={"file": ("p.zip", zip_bytes, "application/zip")},
        headers=_admin_headers(),
    )
    assert resp.status_code == 413


def test_import_failure_leaves_db_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse failure during import must not wipe the existing plant config."""
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setattr(project_import, "MAX_IMPORT_UPLOAD_BYTES", 50 * 1024 * 1024)

    with Session(_test_engine) as s:
        area = PlantArea(name="ExistingArea")
        s.add(area)
        s.commit()
        s.refresh(area)
        unit = PlantUnit(name="U", area_id=area.id)
        s.add(unit)
        s.commit()
        s.refresh(unit)
        equip = PlantEquipment(name="E", unit_id=unit.id)
        s.add(equip)
        s.commit()
        s.refresh(equip)
        s.add(
            TagDefinitionDb(name="EXISTING_TAG", tag_type="Real", equipment_id=equip.id)
        )
        s.commit()

    # tagl.json with content the parser cannot read -> failure after delete/flush.
    zip_bytes = _make_zip({"tagl.json": b"this is not valid json"})
    resp = client.post(
        "/api/project/import",
        files={"file": ("p.zip", zip_bytes, "application/zip")},
        headers=_admin_headers(),
    )
    assert resp.status_code in (400, 500)

    with Session(_test_engine) as s:
        tags = s.exec(select(TagDefinitionDb)).all()
        assert any(t.name == "EXISTING_TAG" for t in tags), (
            "import failure wiped the existing plant DB"
        )


def test_safe_extract_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "traversal.zip"
    destination = tmp_path / "extract"
    outside = tmp_path / "evil.txt"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "owned")

    with (
        pytest.raises(main.HTTPException) as exc_info,
        zipfile.ZipFile(archive, "r") as zf,
    ):
        project_import._safe_extract_zip(zf, destination)

    assert exc_info.value.status_code == 400
    assert not outside.exists()
