"""Append-only configuration revision history (SCADA F04 slice, issue #5527).

Every write of a persisted operator setting also appends an immutable
:class:`ConfigRevision` row, so a previous value is never lost and every change
carries a content hash and the (hashed) credential that made it.

Append-only is enforced twice:

- this module exposes no update/delete function and the HTTP surface is
  read-only (``GET`` list / ``GET`` one, both admin-gated);
- SQLite triggers abort any ``UPDATE`` or ``DELETE`` on the table, so even a
  stray ORM statement or ad-hoc SQL cannot rewrite history.

Atomicity is owned by the caller (:func:`config_store.save_config`): it appends
through :func:`append_revision` (flush only, no commit) and commits the
revision and the ``PersistedConfig`` upsert together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from models import utc_now
from sqlalchemy import DDL, event
from sqlmodel import Field, Session, SQLModel, col, select

logger = logging.getLogger("dcs_backend.config_revisions")

__all__ = [
    "ConfigRevision",
    "append_revision",
    "canonical_json",
    "create_config_revision_router",
    "get_revision",
    "list_revisions",
]

#: ``audit.credential_fingerprint`` output: first 16 hex chars of a SHA-256.
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16}$")
_MAX_LIST = 500


class ConfigRevision(SQLModel, table=True):
    """One immutable revision of a persisted setting (never updated/deleted)."""

    __table_args__ = {"sqlite_autoincrement": True}  # ids are never reused

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(index=True)
    value_json: str
    sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    credential_fingerprint: str | None = None


for _op in ("UPDATE", "DELETE"):
    event.listen(
        ConfigRevision.__table__,  # type: ignore[attr-defined]
        "after_create",
        DDL(
            f"CREATE TRIGGER IF NOT EXISTS configrevision_no_{_op.lower()} "
            f"BEFORE {_op} ON configrevision "
            "BEGIN SELECT RAISE(ABORT, 'configrevision is append-only'); END"
        ).execute_if(dialect="sqlite"),
    )


def canonical_json(payload: dict[str, object]) -> str:
    """Return the canonical JSON text: sorted keys, compact separators.

    Raises:
        TypeError: if ``payload`` is not JSON-serializable.
        ValueError: on circular references.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def append_revision(
    session: Session,
    key: str,
    value_json: str,
    credential_fingerprint: str | None = None,
) -> int:
    """Add a revision to ``session`` and flush; the CALLER commits.

    Args:
        value_json: Canonical JSON text (see :func:`canonical_json`).
        credential_fingerprint: Output of ``audit.credential_fingerprint`` or
            ``None``. A raw key is rejected, never stored.

    Returns:
        The new monotonic revision id.

    Raises:
        ValueError: if ``key`` is empty or the fingerprint is not 16 hex chars.
    """
    if not key:
        raise ValueError("key must be non-empty")
    if credential_fingerprint is not None and not _FINGERPRINT_RE.match(
        credential_fingerprint
    ):
        raise ValueError("credential_fingerprint must be a 16-hex-char hash")
    revision = ConfigRevision(
        key=key,
        value_json=value_json,
        sha256=hashlib.sha256(value_json.encode("utf-8")).hexdigest(),
        credential_fingerprint=credential_fingerprint,
    )
    session.add(revision)
    session.flush()
    if revision.id is None:  # pragma: no cover - flush assigns the PK
        raise RuntimeError("revision id was not assigned")
    return revision.id


def list_revisions(
    session: Session, key: str, limit: int = _MAX_LIST
) -> list[ConfigRevision]:
    """Revisions of ``key``, newest first (at most ``limit``)."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    stmt = (
        select(ConfigRevision)
        .where(ConfigRevision.key == key)
        .order_by(col(ConfigRevision.id).desc())
        .limit(min(limit, _MAX_LIST))
    )
    return list(session.exec(stmt).all())


def get_revision(session: Session, revision_id: int) -> ConfigRevision | None:
    """One revision by id, or ``None``."""
    return session.get(ConfigRevision, revision_id)


def _summary(rev: ConfigRevision) -> dict[str, Any]:
    return {
        "id": rev.id,
        "key": rev.key,
        "sha256": rev.sha256,
        "created_at": rev.created_at.isoformat(),
        "credential_fingerprint": rev.credential_fingerprint,
    }


def create_config_revision_router(
    get_session: Callable[..., Iterator[Session]],
    require_admin: Callable[..., None],
) -> APIRouter:
    """Read-only, admin-gated revision routes (collaborators injected).

    ``require_admin`` is the same dependency the config write uses, so list/get
    can never be weaker than the write they audit.
    """
    router = APIRouter(dependencies=[Depends(require_admin)])

    @router.get("/api/config/revisions")
    def list_config_revisions(
        key: str = Query(min_length=1),
        db: Session = Depends(get_session),  # noqa: B008
    ) -> list[dict[str, Any]]:
        """List revisions of ``key`` newest first (metadata, no values)."""
        return [_summary(r) for r in list_revisions(db, key)]

    @router.get("/api/config/revisions/{revision_id}")
    def get_config_revision(
        revision_id: int,
        db: Session = Depends(get_session),  # noqa: B008
    ) -> dict[str, Any]:
        """Return one revision including its stored value."""
        rev = get_revision(db, revision_id)
        if rev is None:
            raise HTTPException(status_code=404, detail="Revision not found.")
        return {**_summary(rev), "value": json.loads(rev.value_json)}

    return router
