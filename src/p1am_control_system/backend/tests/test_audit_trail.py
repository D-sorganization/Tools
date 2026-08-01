"""Tests for the append-only audit trail (#4029).

Before this, no hardware-mutating endpoint wrote a durable record. Tag forces,
setpoint changes, permissive enables, E-stop trip/clear, routing deploys and
project imports produced only an unstructured ``logger.info``; the only rows in
``EventLog`` came from alarm processing, alarm acknowledgement, and the
*client-supplied* ``POST /api/events``. So the "history" was both forgeable (any
operator key could POST arbitrary events) and erasable (``POST
/api/capture/clear {"include_events": true}`` drops the table).

The audit trail closes that with a **separate** table:

- Written by one shared ASGI helper, so a newly added mutating route is covered
  by default rather than by remembering to instrument it.
- Records route, method, redacted payload, resolved credential tier, a
  non-reversible credential fingerprint, the client IP, and the response status.
- Never reachable from ``POST /api/events`` (which can only write ``EventLog``)
  and never touched by ``clear_capture`` — so neither forging nor erasure works.
- Mirrored to the ``dcs_backend.audit`` logger, which systemd captures into
  journald.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from audit import (  # noqa: E402
    AUDITED_METHODS,
    AuditEvent,
    AuditMiddleware,
    credential_fingerprint,
    redact_payload,
    resolve_actor,
)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

_OPERATOR_KEY = "operator-secret"  # pragma: allowlist secret
_ADMIN_KEY = "admin-secret"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in ("P1AM_DEV_NO_AUTH", "P1AM_API_KEY", "P1AM_ADMIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def audit_session_factory(tmp_path: Path) -> Iterator[object]:
    """An isolated SQLite engine so audit rows never touch the real historian."""
    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    SQLModel.metadata.create_all(engine, tables=[AuditEvent.__table__])

    def factory() -> Session:
        return Session(engine)

    factory.engine = engine  # type: ignore[attr-defined]
    yield factory
    engine.dispose()


def _rows(factory: object) -> list[AuditEvent]:
    with factory() as session:  # type: ignore[operator]
        return list(session.exec(select(AuditEvent)).all())


def _audited_app(factory: object) -> FastAPI:
    app = FastAPI()

    @app.post("/api/tags/{tag_id}")
    async def write_tag(tag_id: str, payload: dict[str, float]) -> dict[str, str]:
        return {"status": "ok", "tag": tag_id}

    @app.post("/api/estop")
    async def estop() -> dict[str, str]:
        return {"status": "tripped"}

    @app.get("/api/routing")
    async def routing() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/boom")
    async def boom() -> dict[str, str]:
        raise ValueError("kaboom")

    app.add_middleware(AuditMiddleware, session_factory=factory)
    return app


# --------------------------------------------------------------------------- #
# Credential fingerprinting and actor resolution                               #
# --------------------------------------------------------------------------- #


def test_fingerprint_is_not_the_key() -> None:
    fingerprint = credential_fingerprint(_ADMIN_KEY)
    assert _ADMIN_KEY not in fingerprint
    assert fingerprint != _ADMIN_KEY


def test_fingerprint_is_stable_and_distinguishing() -> None:
    assert credential_fingerprint(_ADMIN_KEY) == credential_fingerprint(_ADMIN_KEY)
    assert credential_fingerprint(_ADMIN_KEY) != credential_fingerprint(_OPERATOR_KEY)


def test_fingerprint_of_missing_key_is_none() -> None:
    assert credential_fingerprint(None) is None
    assert credential_fingerprint("") is None


def test_resolve_actor_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    assert resolve_actor(_ADMIN_KEY).tier == "admin"


def test_resolve_actor_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    assert resolve_actor(_OPERATOR_KEY).tier == "operator"


def test_resolve_actor_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    assert resolve_actor(None).tier == "anonymous"


def test_resolve_actor_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    assert resolve_actor("bogus").tier == "invalid"


def test_resolve_actor_dev_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bench box must be *identifiable as such* in the trail."""
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    assert resolve_actor(None).tier == "dev-no-auth"


# --------------------------------------------------------------------------- #
# Payload redaction                                                            #
# --------------------------------------------------------------------------- #


def test_redact_payload_masks_credential_like_fields() -> None:
    redacted = redact_payload({"value": 42.0, "api_key": _ADMIN_KEY})
    assert redacted["value"] == 42.0
    assert _ADMIN_KEY not in str(redacted)


def test_redact_payload_recurses() -> None:
    redacted = redact_payload({"outer": {"password": "hunter2", "ok": 1}})
    assert "hunter2" not in str(redacted)
    assert redacted["outer"]["ok"] == 1


def test_redact_payload_passes_through_plain_values() -> None:
    assert redact_payload({"setpoint": 3.5}) == {"setpoint": 3.5}


# --------------------------------------------------------------------------- #
# Middleware behaviour                                                         #
# --------------------------------------------------------------------------- #


def test_mutating_request_is_recorded(audit_session_factory: object) -> None:
    client = TestClient(_audited_app(audit_session_factory))
    resp = client.post("/api/tags/TAG_3", json={"value": 12.5})
    assert resp.status_code == 200

    rows = _rows(audit_session_factory)
    assert len(rows) == 1
    assert rows[0].method == "POST"
    assert rows[0].route == "/api/tags/TAG_3"
    assert rows[0].status_code == 200
    assert "12.5" in (rows[0].payload or "")


def test_read_request_is_not_recorded(audit_session_factory: object) -> None:
    client = TestClient(_audited_app(audit_session_factory))
    assert client.get("/api/routing").status_code == 200
    assert _rows(audit_session_factory) == []


def test_every_mutating_method_is_audited() -> None:
    assert {"POST", "PUT", "PATCH", "DELETE"} <= set(AUDITED_METHODS)
    assert "GET" not in AUDITED_METHODS


def test_bodyless_mutation_is_recorded(audit_session_factory: object) -> None:
    client = TestClient(_audited_app(audit_session_factory))
    client.post("/api/estop")
    rows = _rows(audit_session_factory)
    assert len(rows) == 1
    assert rows[0].route == "/api/estop"


def test_actor_tier_and_fingerprint_are_recorded(
    audit_session_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    client = TestClient(_audited_app(audit_session_factory))
    client.post(
        "/api/tags/TAG_1",
        json={"value": 1.0},
        headers={"X-API-Key": _ADMIN_KEY},  # pragma: allowlist secret
    )
    row = _rows(audit_session_factory)[0]
    assert row.actor_tier == "admin"
    assert row.actor_fingerprint
    assert _ADMIN_KEY not in (row.actor_fingerprint or "")


def test_client_ip_is_recorded(audit_session_factory: object) -> None:
    client = TestClient(_audited_app(audit_session_factory))
    client.post("/api/estop")
    assert _rows(audit_session_factory)[0].client_ip


def test_credential_never_lands_in_the_row(
    audit_session_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    client = TestClient(_audited_app(audit_session_factory))
    client.post(
        "/api/tags/TAG_1",
        json={"value": 1.0, "api_key": _ADMIN_KEY},
        headers={"X-API-Key": _ADMIN_KEY},  # pragma: allowlist secret
    )
    row = _rows(audit_session_factory)[0]
    assert _ADMIN_KEY not in "".join(
        str(v) for v in (row.payload, row.actor_fingerprint, row.route)
    )


def test_rejected_request_is_still_recorded(audit_session_factory: object) -> None:
    """A denied control attempt is exactly what an audit trail is for."""
    client = TestClient(_audited_app(audit_session_factory))
    client.post("/api/tags/TAG_1", json={"nope": "bad"})  # 422 from validation
    rows = _rows(audit_session_factory)
    assert len(rows) == 1
    assert rows[0].status_code >= 400


def test_middleware_failure_never_breaks_the_control_path(tmp_path: Path) -> None:
    """DbC: auditing is best-effort. A broken sink must not fail a command."""

    def exploding_factory() -> Session:
        raise RuntimeError("audit sink is down")

    client = TestClient(_audited_app(exploding_factory))
    assert client.post("/api/estop").status_code == 200


def test_audit_is_mirrored_to_the_journald_logger(
    audit_session_factory: object, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    client = TestClient(_audited_app(audit_session_factory))
    with caplog.at_level(logging.INFO, logger="dcs_backend.audit"):
        client.post("/api/estop")
    assert any("/api/estop" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# Append-only: unforgeable and un-erasable                                     #
# --------------------------------------------------------------------------- #


def test_audit_table_is_not_the_client_writable_event_log() -> None:
    """``POST /api/events`` writes ``EventLog``; it can never reach the trail."""
    from models import EventLog

    assert AuditEvent.__tablename__ != EventLog.__tablename__


def test_clear_capture_cannot_erase_the_audit_trail(tmp_path: Path) -> None:
    """``POST /api/capture/clear {"include_events": true}`` must not touch it."""
    import data_capture
    from models import EventLog, TagLog

    engine = create_engine(f"sqlite:///{tmp_path / 'historian.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(TagLog(tag_name="TAG_0", value=1.0))
        session.add(EventLog(event_type="SYSTEM", description="forged"))
        session.add(
            AuditEvent(
                route="/api/estop/clear",
                method="POST",
                actor_tier="admin",
                status_code=200,
            )
        )
        session.commit()

    with Session(engine) as session:
        data_capture.clear_capture(session, include_events=True)
        session.commit()
        assert session.exec(select(EventLog)).all() == []
        assert len(session.exec(select(AuditEvent)).all()) == 1
    engine.dispose()
