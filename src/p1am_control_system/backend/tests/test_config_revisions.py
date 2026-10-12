"""Append-only config revision history (SCADA F04 slice, issue #5527)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).parent.parent))

import config_revisions  # noqa: E402
import config_store  # noqa: E402
from audit import credential_fingerprint  # noqa: E402
from auth_config import CREDENTIAL_HEADER_NAME, require_admin_key  # noqa: E402
from config_revisions import (  # noqa: E402
    ConfigRevision,
    canonical_json,
    create_config_revision_router,
    get_revision,
    list_revisions,
)
from config_store import PersistedConfig, load_config, save_config  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import DBAPIError  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

_ADMIN = "rev-admin-key"  # pragma: allowlist secret
_OPERATOR = "rev-operator-key"  # pragma: allowlist secret


@pytest.fixture
def engine(tmp_path: Path):
    eng = create_engine(
        f"sqlite:///{(tmp_path / 'rev.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


# --- acceptance 1: two writes -> two revisions, old value readable ----------


def test_two_writes_leave_two_revisions_and_old_value_readable(
    session: Session,
) -> None:
    rid1 = save_config(session, "routing", {"a": 1})
    rid2 = save_config(session, "routing", {"a": 2})
    assert rid2 > rid1
    revs = list_revisions(session, "routing")
    assert [r.id for r in revs] == [rid2, rid1]  # newest first
    assert json.loads(get_revision(session, rid1).value_json) == {"a": 1}
    assert load_config(session, "routing") == {"a": 2}


def test_list_is_scoped_to_key(session: Session) -> None:
    save_config(session, "x", {"v": 1})
    save_config(session, "y", {"v": 1})
    assert len(list_revisions(session, "x")) == 1


def test_get_missing_revision_returns_none(session: Session) -> None:
    assert get_revision(session, 999) is None


# --- acceptance 4: hash determinism ----------------------------------------


def test_identical_canonical_value_yields_identical_hash(session: Session) -> None:
    r1 = save_config(session, "k", {"b": 2, "a": [1, 2]})
    r2 = save_config(session, "k", {"a": [1, 2], "b": 2})
    assert get_revision(session, r1).sha256 == get_revision(session, r2).sha256


def test_hash_is_sha256_of_canonical_json(session: Session) -> None:
    rid = save_config(session, "k", {"b": 1, "a": "x"})
    canon = '{"a":"x","b":1}'
    assert canonical_json({"b": 1, "a": "x"}) == canon
    assert (
        get_revision(session, rid).sha256 == hashlib.sha256(canon.encode()).hexdigest()
    )


def test_different_value_different_hash(session: Session) -> None:
    r1 = save_config(session, "k", {"a": 1})
    r2 = save_config(session, "k", {"a": 2})
    assert get_revision(session, r1).sha256 != get_revision(session, r2).sha256


# --- fingerprint never raw --------------------------------------------------


def test_fingerprint_is_hashed_never_raw(session: Session) -> None:
    fp = credential_fingerprint(_ADMIN)
    rid = save_config(session, "k", {"a": 1}, credential_fingerprint=fp)
    rev = get_revision(session, rid)
    assert rev.credential_fingerprint == fp
    assert _ADMIN not in (rev.credential_fingerprint or "")
    assert _ADMIN not in rev.value_json


def test_raw_looking_fingerprint_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError):
        save_config(session, "k", {"a": 1}, credential_fingerprint=_ADMIN)
    assert list_revisions(session, "k") == []


def test_save_contract_still_enforced(session: Session) -> None:
    with pytest.raises(ValueError):
        save_config(session, "", {"a": 1})
    with pytest.raises(ValueError):
        save_config(session, "k", {"a": object()})  # type: ignore[dict-item]
    assert session.exec(select(ConfigRevision)).all() == []


# --- acceptance 2: atomicity -----------------------------------------------


def test_fault_between_append_and_upsert_rolls_back_both(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_config(session, "k", {"v": 1})

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("injected fault")

    monkeypatch.setattr(config_store, "_upsert_current", boom)
    with pytest.raises(RuntimeError):
        save_config(session, "k", {"v": 2})
    monkeypatch.undo()

    assert len(list_revisions(session, "k")) == 1
    assert load_config(session, "k") == {"v": 1}
    row = session.get(PersistedConfig, "k")
    assert row is not None and json.loads(row.value_json) == {"v": 1}


def test_fault_on_first_write_leaves_nothing(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("injected fault")

    monkeypatch.setattr(config_store, "_upsert_current", boom)
    with pytest.raises(RuntimeError):
        save_config(session, "new", {"v": 1})
    monkeypatch.undo()
    assert list_revisions(session, "new") == []
    assert load_config(session, "new") is None


# --- acceptance 3: append-only ---------------------------------------------


def test_module_exposes_no_mutator() -> None:
    public = {n for n in dir(config_revisions) if not n.startswith("_")}
    assert not {n for n in public if n.startswith(("update_", "delete_", "remove_"))}


def test_db_rejects_update_and_delete_of_a_revision(engine, session: Session) -> None:
    rid = save_config(session, "k", {"v": 1})
    with engine.begin() as conn:
        with pytest.raises(DBAPIError):
            conn.exec_driver_sql(
                "UPDATE configrevision SET value_json='{}' WHERE id=?", (rid,)
            )
    with engine.begin() as conn:
        with pytest.raises(DBAPIError):
            conn.exec_driver_sql("DELETE FROM configrevision WHERE id=?", (rid,))
    assert json.loads(get_revision(session, rid).value_json) == {"v": 1}


# --- routes -----------------------------------------------------------------


@pytest.fixture
def client(engine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("P1AM_DEV_NO_AUTH", raising=False)
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN)

    def _get_session():
        with Session(engine) as s:
            yield s

    app = FastAPI()
    app.include_router(create_config_revision_router(_get_session, require_admin_key))
    with Session(engine) as s:
        save_config(s, "routing", {"secret_marker": "TOPSECRET"})
    return TestClient(app, raise_server_exceptions=False)


def _h(key: str | None) -> dict[str, str]:
    return {CREDENTIAL_HEADER_NAME: key} if key else {}


_PATHS = ["/api/config/revisions?key=routing", "/api/config/revisions/1"]


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("key", [None, "wrong-key", _OPERATOR])
def test_routes_reject_unauthenticated_and_non_admin(
    client: TestClient, path: str, key: str | None
) -> None:
    resp = client.get(path, headers=_h(key))
    assert resp.status_code in (401, 403)
    assert "TOPSECRET" not in resp.text


def test_admin_lists_newest_first_and_gets_one(client: TestClient) -> None:
    resp = client.get("/api/config/revisions?key=routing", headers=_h(_ADMIN))
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == [1]
    assert "value" not in body[0]  # list is metadata only
    one = client.get("/api/config/revisions/1", headers=_h(_ADMIN))
    assert one.status_code == 200
    assert one.json()["value"] == {"secret_marker": "TOPSECRET"}
    assert len(one.json()["sha256"]) == 64
    assert client.get("/api/config/revisions/99", headers=_h(_ADMIN)).status_code == 404


def test_list_requires_nonempty_key(client: TestClient) -> None:
    assert client.get("/api/config/revisions", headers=_h(_ADMIN)).status_code == 422
    resp = client.get("/api/config/revisions?key=", headers=_h(_ADMIN))
    assert resp.status_code == 422


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path", ["/api/config/revisions", "/api/config/revisions/1"])
def test_no_mutating_route_exists(client: TestClient, method: str, path: str) -> None:
    resp = client.request(method, path, headers=_h(_ADMIN), json={})
    assert resp.status_code in (404, 405)
